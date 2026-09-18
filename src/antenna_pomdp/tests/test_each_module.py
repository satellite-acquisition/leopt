"""Coverage tests for each top-level module in antenna_pomdp.

These are smoke / contract tests, not exhaustive — the per-module `__main__`
blocks already verify numerical correctness. Here we make sure the pieces fit
together as documented and that a tiny end-to-end Monte Carlo runs.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from antenna_pomdp.baselines.scanner import (
    CenteredOneSidedTrackSweep,
    HierarchicalOneSidedTrackSweep,
    OneSidedTrackSweep,
    ProgressiveOneSidedTrackSweep,
    RandomScan,
    TrackLineSweep,
)
from antenna_pomdp.config import Config, beam_sigma_rad, default_config
from antenna_pomdp.eval.metrics import (
    TrialResult,
    acquisition_probability,
    summarize,
)
from antenna_pomdp.eval.runner import (
    make_pomcp_factory,
    make_random_factory,
    make_sweep_factory,
    run_monte_carlo,
)
from antenna_pomdp.models.observation import (
    Pointing,
    detection_probability,
    false_confirmation_probability,
    observation_likelihood,
    receiver_trigger_probability,
    true_acquisition_probability,
)
from antenna_pomdp.models.particle_filter import (
    belief_azel,
    make_belief,
    update,
)
from antenna_pomdp.orbit.geometry import (
    Station,
    eci_to_azel,
    find_next_pass,
    gmst_rad,
)
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import sample_particles, sample_truth
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
    build_action_grid,
)
from antenna_pomdp.pomdp.pomcp import PomcpSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> Config:
    base = default_config()
    return Config(
        orbit=base.orbit,
        station=base.station,
        antenna=base.antenna,
        filter=type(base.filter)(**{**base.filter.__dict__, "n_particles": 64}),
        pomcp=type(base.pomcp)(**{**base.pomcp.__dict__, "n_rollouts": 20, "max_depth": 3}),
        eval=type(base.eval)(n_trials=2, seed=0),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def station(cfg: Config) -> Station:
    return Station.from_config(cfg.station)


@pytest.fixture
def nominal(cfg: Config) -> SGP4Propagator:
    return SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_beam_sigma_from_fwhm(cfg):
    expected = np.deg2rad(cfg.antenna.fwhm_deg) / 2.355
    assert abs(beam_sigma_rad(cfg.antenna) - expected) < 1e-12


# ---------------------------------------------------------------------------
# orbit
# ---------------------------------------------------------------------------


def test_propagator_returns_finite_state(nominal):
    s = nominal.propagate(nominal.epoch + timedelta(minutes=10))
    assert np.all(np.isfinite(s.r_eci_m))
    assert np.linalg.norm(s.r_eci_m) > 6.5e6


def test_sampler_yields_distinct_particles(cfg, rng):
    parts = sample_particles(cfg.orbit, n=8, rng=rng)
    when = parts[0].propagator.epoch + timedelta(minutes=10)
    positions = np.array([p.propagator.propagate(when).r_eci_m for p in parts])
    assert np.std(positions, axis=0).sum() > 0.0


def test_gmst_is_in_range(nominal):
    g = gmst_rad(nominal.epoch)
    assert 0.0 <= g < 2.0 * np.pi


def test_azel_zenith(station, nominal):
    # Construct an ECI vector that lies along the local zenith.
    from antenna_pomdp.orbit.geometry import rot_eci_to_ecef

    R = rot_eci_to_ecef(nominal.epoch)
    up_ecef = station.ecef_m + station.ecef_m / np.linalg.norm(station.ecef_m) * 1e6
    eci = R.T @ up_ecef
    _, el, _ = eci_to_azel(eci, nominal.epoch, station)
    assert el > np.deg2rad(89.0)


def test_find_next_pass_returns_none_or_window(nominal, station, cfg):
    w = find_next_pass(
        nominal,
        station,
        nominal.epoch,
        min_elevation_rad=np.deg2rad(cfg.station.min_elevation_deg),
        search_horizon_s=3600.0,
    )
    assert w is None or w.close > w.open


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_detection_probability_monotone(cfg):
    thetas = np.linspace(0.0, np.deg2rad(45.0), 10)
    p = detection_probability(thetas, cfg.antenna)
    assert np.all(np.diff(p) <= 1e-9)  # non-increasing


def test_observation_likelihood_sums_to_one(cfg):
    for theta in (0.0, np.deg2rad(2.0), np.deg2rad(30.0)):
        pd = float(observation_likelihood(True, theta, cfg.antenna))
        pn = float(observation_likelihood(False, theta, cfg.antenna))
        assert abs(pd + pn - 1.0) < 1e-12


def test_false_trigger_is_not_confirmed_acquisition(cfg):
    """A raw off-target trigger must not be labeled spacecraft acquisition."""
    antenna = replace(
        cfg.antenna,
        false_alarm_rate=1.0,
        confirmation_false_accept_rate=0.0,
    )
    theta = np.deg2rad(60.0)
    assert float(receiver_trigger_probability(theta, antenna)) > 0.999
    assert float(detection_probability(theta, antenna)) < 1e-6


def test_terminal_confirmation_decomposes_into_true_and_false_events(cfg):
    antenna = replace(
        cfg.antenna,
        false_alarm_rate=0.25,
        confirmation_true_accept_rate=0.8,
        confirmation_false_accept_rate=0.2,
    )
    theta = np.deg2rad(3.0)
    p_true = float(true_acquisition_probability(theta, antenna))
    p_false = float(false_confirmation_probability(theta, antenna))
    p_terminal = float(detection_probability(theta, antenna))
    assert p_true > 0.0
    assert p_false > 0.0
    assert np.isclose(p_terminal, p_true + p_false)


def test_below_mask_signal_cannot_confirm(cfg):
    antenna = replace(
        cfg.antenna,
        min_signal_elevation_deg=5.0,
        false_alarm_rate=0.0,
    )
    p = detection_probability(
        0.0,
        antenna,
        elevation_rad=np.deg2rad(4.9),
        range_m=500e3,
    )
    assert float(p) == 0.0


def test_particle_filter_update_keeps_normalisation(cfg, rng, station, nominal):
    parts = sample_particles(cfg.orbit, n=32, rng=rng)
    belief = make_belief(parts)
    when = nominal.epoch + timedelta(minutes=5)
    az_arr = belief_azel(belief, when, station)
    pointing = Pointing(az=float(np.mean(az_arr[:, 0])), el=float(np.mean(az_arr[:, 1])))
    new = update(
        belief,
        pointing,
        detected=False,
        when=when,
        station=station,
        antenna_cfg=cfg.antenna,
        filter_cfg=cfg.filter,
        rng=rng,
    )
    assert abs(new.weights.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# pomdp
# ---------------------------------------------------------------------------


def test_action_grid_size(cfg):
    actions = build_action_grid(cfg.pomcp)
    assert len(actions) == cfg.pomcp.n_along_track_offsets * cfg.pomcp.n_cross_track_levels


def test_env_step_advances_time(cfg, rng, station, nominal):
    truth = sample_truth(cfg.orbit, rng)
    env = PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=nominal.epoch + timedelta(minutes=5),
        pass_duration_s=30.0,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    s0 = env.initial_state()
    s1, _, _, _ = env.step(s0, TrackAction(0.0, 0.0), rng)
    assert s1.step_idx == 1
    assert (s1.when - s0.when).total_seconds() == cfg.antenna.dwell_time_s


def test_action_to_pointing_within_pi(cfg, station, nominal):
    p = action_to_pointing(
        TrackAction(5.0, 3.0), nominal, nominal.epoch + timedelta(minutes=5), station
    )
    assert 0.0 <= p.az < 2.0 * np.pi
    assert -np.pi / 2 <= p.el <= np.pi / 2


def test_pomcp_plans_action(cfg, rng, station, nominal):
    parts = sample_particles(cfg.orbit, n=16, rng=rng)
    belief = make_belief(parts)
    env = PointingEnv(
        nominal=nominal,
        truth=sample_truth(cfg.orbit, rng),
        station=station,
        pass_start=nominal.epoch + timedelta(minutes=5),
        pass_duration_s=30.0,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    solver = PomcpSolver(
        pomcp_cfg=cfg.pomcp,
        antenna_cfg=cfg.antenna,
        station=station,
        nominal=nominal,
    )
    idx, root = solver.plan(belief, env.pass_start, env, rng)
    assert 0 <= idx < len(solver.actions)
    assert root.visits == cfg.pomcp.n_rollouts


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_sweep_wraps_around(cfg):
    s = TrackLineSweep(cfg.pomcp)
    n = len(s._sequence)
    first = s.act()
    for _ in range(n - 1):
        s.act()
    assert s.act() == first


def test_track_sweep_is_center_out_and_one_dimensional(cfg):
    s = TrackLineSweep(cfg.pomcp)
    assert all(a.cross_track_deg == 0.0 for a in s._sequence)
    magnitudes = [abs(a.along_track_s) for a in s._sequence]
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[0] < 1e-12


def test_one_sided_sweep_matches_negative_half_grid(cfg):
    sweep = OneSidedTrackSweep(cfg.pomcp)
    offsets = [action.along_track_s for action in sweep._sequence]
    full = np.linspace(
        -cfg.pomcp.along_track_offset_max_s,
        cfg.pomcp.along_track_offset_max_s,
        cfg.pomcp.n_along_track_offsets,
    )
    expected = list(full[full <= 1e-12][::-1])
    assert np.allclose(offsets, expected)
    assert offsets[0] == 0.0
    assert all(offset <= 0.0 for offset in offsets)


def test_centered_one_sided_sweep_orders_supported_grid_from_midpoint(cfg):
    sweep = CenteredOneSidedTrackSweep(cfg.pomcp)
    offsets = np.asarray([action.along_track_s for action in sweep._sequence])
    t_max = cfg.pomcp.along_track_offset_max_s
    expected = np.linspace(
        -t_max,
        t_max,
        cfg.pomcp.n_along_track_offsets,
    )
    supported = expected[expected <= 1e-12]

    assert offsets.size == supported.size
    assert set(offsets) == set(supported)
    assert offsets[0] == pytest.approx(-0.5 * t_max)
    assert np.all(np.diff(np.abs(offsets + 0.5 * t_max)) >= -1e-12)


def test_progressive_one_sided_sweep_fills_horizon_at_two_resolutions(cfg):
    sweep = ProgressiveOneSidedTrackSweep(cfg.pomcp, n_steps=90)
    offsets = np.asarray([action.along_track_s for action in sweep._sequence])
    t_max = cfg.pomcp.along_track_offset_max_s
    coarse = np.linspace(0.0, -t_max, 45)
    midpoints = 0.5 * (coarse[:-1] + coarse[1:])

    assert offsets.size == 90
    assert np.allclose(offsets[:45], coarse)
    assert np.allclose(offsets[45:89], midpoints)
    assert offsets[-1] == pytest.approx(-0.5 * t_max)
    assert all(action.cross_track_deg == 0.0 for action in sweep._sequence)
    assert np.all((-t_max <= offsets) & (offsets <= 0.0))


def test_hierarchical_one_sided_sweep_matches_frozen_90_dwell_rule(cfg):
    sweep = HierarchicalOneSidedTrackSweep(cfg.pomcp, n_steps=90)
    offsets = np.asarray([action.along_track_s for action in sweep._sequence])
    t_max = cfg.pomcp.along_track_offset_max_s
    anchors = np.linspace(0.0, -t_max, 30)
    first_midpoints = 0.5 * (anchors[:-1] + anchors[1:])
    children = np.empty(58)
    children[0::2] = 0.5 * (anchors[:-1] + first_midpoints)
    children[1::2] = 0.5 * (first_midpoints + anchors[1:])

    assert np.allclose(offsets[:30], anchors)
    assert np.allclose(offsets[30:59], first_midpoints)
    assert np.allclose(offsets[59:], children[:31])
    assert np.unique(offsets).size == 90
    assert np.all((-t_max <= offsets) & (offsets <= 0.0))
    assert all(action.cross_track_deg == 0.0 for action in sweep._sequence)


def test_random_scan_outputs_valid_action(cfg, rng):
    r = RandomScan(cfg.pomcp, rng)
    for _ in range(10):
        a = r.act()
        assert isinstance(a, TrackAction)


# ---------------------------------------------------------------------------
# eval + metrics
# ---------------------------------------------------------------------------


def test_metrics_summary_keys():
    s = summarize([TrialResult(True, 5.0, 1.0, 1), TrialResult(False, None, 2.0, 10)])
    assert s["n_trials"] == 2
    assert 0.0 <= s["p_acquire"] <= 1.0


def test_acquisition_probability_empty():
    assert acquisition_probability([]) == 0.0


def test_runner_end_to_end_sweep(cfg):
    results = run_monte_carlo(cfg, make_sweep_factory(), n_trials=2, seed=1)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, TrialResult)


def test_runner_end_to_end_random(cfg):
    results = run_monte_carlo(cfg, make_random_factory(), n_trials=2, seed=2)
    assert len(results) == 2


def test_runner_end_to_end_pomcp(cfg):
    results = run_monte_carlo(cfg, make_pomcp_factory(cfg), n_trials=2, seed=3)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# viz (smoke only — non-interactive backend)
# ---------------------------------------------------------------------------


def test_viz_belief_smoke(cfg, rng, station, nominal):
    import matplotlib

    matplotlib.use("Agg")
    from antenna_pomdp.viz.belief import plot_belief_azel

    belief = make_belief(sample_particles(cfg.orbit, n=16, rng=rng))
    ax = plot_belief_azel(belief, nominal.epoch + timedelta(minutes=5), station)
    assert ax is not None


def test_viz_heatmap_smoke(cfg):
    import matplotlib

    matplotlib.use("Agg")
    from antenna_pomdp.viz.heatmap import plot_action_visits

    n = cfg.pomcp.n_along_track_offsets * cfg.pomcp.n_cross_track_levels
    ax = plot_action_visits(np.ones(n), cfg.pomcp)
    assert ax is not None
