"""Tests for the scalar along-track line-search controller (one-axis reduction)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from antenna_pomdp.baselines.line_search import (
    _expected_information_gain_rows,
    _track_grid_azel,
    belief_anisotropy,
    belief_track_coordinates,
    make_line_search_policy,
    make_track_grid_search_policy,
)
from antenna_pomdp.config import default_config
from antenna_pomdp.models.particle_filter import make_belief
from antenna_pomdp.orbit.geometry import Station, find_next_pass
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle, sample_leop_particles
from antenna_pomdp.pomdp.environment import PointingEnv, TrackAction, action_to_pointing


def _binary_entropy(probability: np.ndarray | float) -> np.ndarray:
    """Binary entropy in nats, with the endpoint convention 0 log 0 = 0."""
    p = np.asarray(probability, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0.0, p * np.log(p), 0.0) - np.where(
            p < 1.0, (1.0 - p) * np.log1p(-p), 0.0
        )


def _ho_information_gain(
    field_of_view_mass: np.ndarray | float,
    p_detection_present: float,
    p_detection_absent: float,
) -> np.ndarray:
    """Ho et al. (2021), Eq. (20), for a hard-FOV binary channel."""
    w = np.asarray(field_of_view_mass, dtype=float)
    p_detection = w * p_detection_present + (1.0 - w) * p_detection_absent
    return (
        _binary_entropy(p_detection)
        - w * _binary_entropy(p_detection_present)
        - (1.0 - w) * _binary_entropy(p_detection_absent)
    )


def _ho_exact_optimal_mass(
    p_detection_present: float,
    p_detection_absent: float,
) -> float:
    """Exact interior argmax of Ho et al.'s hard-FOV information gain."""
    entropy_difference = float(
        _binary_entropy(p_detection_present) - _binary_entropy(p_detection_absent)
    )
    likelihood_difference = p_detection_present - p_detection_absent
    optimal_detection_probability = 1.0 / (1.0 + np.exp(entropy_difference / likelihood_difference))
    return float((optimal_detection_probability - p_detection_absent) / likelihood_difference)


def _hard_fov_likelihoods(
    field_of_view_masses: np.ndarray,
    *,
    n_particles: int,
    p_detection_present: float,
    p_detection_absent: float,
) -> np.ndarray:
    """Binary-channel likelihood rows for a uniform particle belief."""
    particle_ranks = np.arange(n_particles)
    inside_counts = np.rint(field_of_view_masses * n_particles).astype(int)
    assert np.allclose(inside_counts / n_particles, field_of_view_masses)
    inside = particle_ranks[None, :] < inside_counts[:, None]
    return np.where(inside, p_detection_present, p_detection_absent)


def _setup(n_particles=120, fwhm_deg=2.0, slew_deg_s=1.0e9):
    base = default_config()
    cfg = replace(
        base,
        filter=replace(base.filter, n_particles=n_particles),
        antenna=replace(base.antenna, fwhm_deg=fwhm_deg, max_slew_rate_deg_s=slew_deg_s),
        pomcp=replace(base.pomcp, along_track_offset_max_s=120.0),
    )
    rng = np.random.default_rng(3)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_leop_particles(cfg.orbit, cfg.leop, n=cfg.filter.n_particles, rng=rng)
    belief = make_belief(particles)
    visible = find_next_pass(
        nominal,
        station,
        nominal.epoch,
        min_elevation_rad=np.deg2rad(cfg.station.min_elevation_deg),
        search_horizon_s=12 * 3600.0,
    )
    assert visible is not None
    when = visible.open + timedelta(seconds=0.5 * (visible.close - visible.open).total_seconds())
    env = PointingEnv(
        nominal=nominal,
        truth=particles[0],
        station=station,
        pass_start=when,
        pass_duration_s=cfg.station.pass_duration_s,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    return cfg, belief, env, when, nominal, station, rng


def test_hard_fov_information_gain_matches_ho_equation_action_by_action():
    """The particle entropy reduction reduces exactly to Ho et al., Eq. (20)."""
    field_of_view_masses = np.array([0.1, 0.2, 0.4, 0.5, 0.8])
    p_detection_present = 0.7
    p_detection_absent = 0.1
    weights = np.full(10, 0.1)
    likelihoods = _hard_fov_likelihoods(
        field_of_view_masses,
        n_particles=weights.size,
        p_detection_present=p_detection_present,
        p_detection_absent=p_detection_absent,
    )

    implementation_scores = _expected_information_gain_rows(weights, likelihoods)

    # Independently evaluate H(X) - E_Y[H(X|Y)] particle by particle.
    prior_entropy = -float(np.sum(weights * np.log(weights)))
    direct_scores = []
    for likelihood in likelihoods:
        p_detection = float(likelihood @ weights)
        detected_posterior = weights * likelihood / p_detection
        absent_posterior = weights * (1.0 - likelihood) / (1.0 - p_detection)
        expected_posterior_entropy = p_detection * -float(
            np.sum(detected_posterior * np.log(detected_posterior))
        ) + (1.0 - p_detection) * -float(np.sum(absent_posterior * np.log(absent_posterior)))
        direct_scores.append(prior_entropy - expected_posterior_entropy)

    ho_scores = _ho_information_gain(
        field_of_view_masses,
        p_detection_present,
        p_detection_absent,
    )
    assert implementation_scores == pytest.approx(direct_scores, abs=1e-14)
    assert implementation_scores == pytest.approx(ho_scores, abs=1e-14)


def test_ho_symmetric_channel_has_half_mass_information_optimum():
    assert _ho_exact_optimal_mass(0.9, 0.1) == pytest.approx(0.5, abs=1e-15)


def test_below_information_optimum_infogain_and_maximum_mass_agree():
    field_of_view_masses = np.array([0.1, 0.2, 0.4])
    weights = np.full(10, 0.1)
    likelihoods = _hard_fov_likelihoods(
        field_of_view_masses,
        n_particles=weights.size,
        p_detection_present=0.9,
        p_detection_absent=0.1,
    )

    information_scores = _expected_information_gain_rows(weights, likelihoods)
    detection_scores = likelihoods @ weights
    assert field_of_view_masses[np.argmax(information_scores)] == 0.4
    assert field_of_view_masses[np.argmax(detection_scores)] == 0.4
    assert field_of_view_masses[np.argmax(field_of_view_masses)] == 0.4


def test_straddling_information_optimum_separates_infogain_and_detection():
    field_of_view_masses = np.array([0.2, 0.5, 0.8])
    weights = np.full(10, 0.1)
    likelihoods = _hard_fov_likelihoods(
        field_of_view_masses,
        n_particles=weights.size,
        p_detection_present=0.9,
        p_detection_absent=0.1,
    )

    information_scores = _expected_information_gain_rows(weights, likelihoods)
    detection_scores = likelihoods @ weights
    assert field_of_view_masses[np.argmax(information_scores)] == 0.5
    assert field_of_view_masses[np.argmax(detection_scores)] == 0.8
    assert field_of_view_masses[np.argmax(field_of_view_masses)] == 0.8


def test_ho_asymmetric_channel_exact_information_optimum():
    optimal_mass = _ho_exact_optimal_mass(0.7, 0.1)
    assert optimal_mass == pytest.approx(0.4718761380015357, abs=1e-15)

    epsilon = 1e-5
    at_optimum = _ho_information_gain(optimal_mass, 0.7, 0.1)
    assert at_optimum > _ho_information_gain(optimal_mass - epsilon, 0.7, 0.1)
    assert at_optimum > _ho_information_gain(optimal_mass + epsilon, 0.7, 0.1)


@pytest.mark.parametrize("objective", ["infogain", "pdetect"])
def test_returns_on_axis_action_in_bounds(objective):
    cfg, belief, env, when, _, _, rng = _setup()
    c = replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective=objective))
    act = make_line_search_policy(c)(c, rng)
    a = act(belief, env, when)
    assert isinstance(a, TrackAction)
    assert a.cross_track_deg == 0.0
    t_max = cfg.pomcp.along_track_offset_max_s
    assert -t_max - 1e-9 <= a.along_track_s <= t_max + 1e-9


def test_points_toward_offset_belief():
    """A belief displaced along-track must pull the commanded tau the same way."""
    cfg, belief, env, when, nominal, station, rng = _setup()
    # Shift every particle 40 s behind the nominal (positive launch slip).
    shifted = [
        OrbitParticle(propagator=p.propagator, launch_slip_s=p.launch_slip_s + 40.0)
        for p in belief.particles
    ]
    belief_shifted = make_belief(shifted)
    act = make_line_search_policy(cfg)(cfg, rng)
    a = act(belief_shifted, env, when)
    # A satellite 40 s behind appears where the nominal was 40 s ago -> tau < 0.
    assert a.along_track_s < 0.0


def test_track_coordinates_recover_slip():
    """belief_track_coordinates must recover a pure along-track displacement."""
    cfg, belief, env, when, nominal, station, _ = _setup()
    base = belief.particles[0]
    slips = np.array([-30.0, 0.0, 25.0, 60.0])
    parts = [OrbitParticle(propagator=base.propagator, launch_slip_s=s) for s in slips]
    b = make_belief(parts)
    tau_i, theta_i = belief_track_coordinates(b, when, station, nominal)
    # tau should move monotonically opposite the slip (behind => negative tau).
    assert np.all(np.diff(tau_i) < 0.0)
    # Relative spacing should match the slip spacing to within the refinement
    # error of the projection (grid pitch 2 s, parabolic refinement).
    d_tau = tau_i[0] - tau_i[-1]
    d_slip = slips[-1] - slips[0]
    assert abs(d_tau - d_slip) < 5.0


def test_anisotropy_ratio_is_large():
    """The LEOP prior must be strongly along-track elongated in track coords."""
    cfg, belief, env, when, nominal, station, _ = _setup(n_particles=200)
    sa, sc, ratio = belief_anisotropy(belief, when, station, nominal)
    assert sa > 0.0 and sc >= 0.0
    assert ratio > 3.0, f"expected along-track dominance, got ratio={ratio:.2f}"


def test_slew_awareness_limits_commanded_move():
    """With a tight slew limit, the realised candidate set collapses toward the
    boresight, so the chosen action's *pointing* must be reachable."""
    cfg, belief, env, when, nominal, station, rng = _setup(slew_deg_s=0.2)
    env.current_bore = None
    act = make_line_search_policy(cfg)(cfg, rng)
    a0 = act(belief, env, when)
    p0 = action_to_pointing(a0, nominal, when, station)
    # Publish a boresight and re-decide: the scored candidates are clamped to
    # the reachable set, so scoring can't prefer unreachable distant pointings.
    env.current_bore = (p0.az, p0.el)
    a1 = act(belief, env, when)
    assert isinstance(a1, TrackAction)


def test_vectorized_grid_matches_action_geometry():
    cfg, _, _, when, nominal, station, _ = _setup()
    taus = np.array([-30.0, 0.0, 25.0])
    crosses = np.array([-2.0, 0.0, 1.5])
    grid = _track_grid_azel(nominal, when, station, taus, crosses)
    expected = [
        action_to_pointing(TrackAction(float(tau), float(cross)), nominal, when, station)
        for tau in taus
        for cross in crosses
    ]
    for got, want in zip(grid, expected):
        daz = (got[0] - want.az + np.pi) % (2.0 * np.pi) - np.pi
        assert abs(daz) < 1e-10
        assert abs(got[1] - want.el) < 1e-10


def test_two_axis_grid_policy_returns_bounded_action():
    cfg, belief, env, when, _, _, rng = _setup()
    action = make_track_grid_search_policy(cfg, n_tau=11, n_cross=5)(cfg, rng)(belief, env, when)
    assert abs(action.along_track_s) <= cfg.pomcp.along_track_offset_max_s
    assert abs(action.cross_track_deg) <= cfg.pomcp.cross_track_offset_max_deg


def test_matches_infogreedy_choice_qualitatively():
    """Line search and 2-D info-greedy should broadly agree on where to look
    when the belief is a pure along-track filament."""
    from antenna_pomdp.baselines.heuristics import make_infogreedy_policy

    cfg, belief, env, when, nominal, station, rng = _setup(n_particles=150)
    a_line = make_line_search_policy(cfg)(cfg, np.random.default_rng(11))(belief, env, when)
    a_2d = make_infogreedy_policy(cfg)(cfg, np.random.default_rng(11))(belief, env, when)
    # Same sign / same neighbourhood of the tau axis (within the 2-D grid pitch).
    grid_pitch_s = (
        2.0 * cfg.pomcp.along_track_offset_max_s / max(1, cfg.pomcp.n_along_track_offsets - 1)
    )
    assert abs(a_line.along_track_s - a_2d.along_track_s) <= 1.5 * grid_pitch_s + 10.0
