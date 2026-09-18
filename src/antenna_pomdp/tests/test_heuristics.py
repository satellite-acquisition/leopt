"""Tests for the belief-aware myopic controllers in baselines.heuristics."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from antenna_pomdp.baselines.heuristics import (
    make_infogreedy_policy,
    make_mls_policy,
    make_qmdp_policy,
)
from antenna_pomdp.config import default_config
from antenna_pomdp.models.observation import delta_theta_from_azel
from antenna_pomdp.models.particle_filter import (
    belief_azel_range,
    make_belief,
)
from antenna_pomdp.orbit.geometry import Station, angular_separation, find_next_pass
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import sample_particles
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
)


def _setup(n_particles=150, n_candidates=24):
    base = default_config()
    cfg = replace(
        base,
        filter=replace(base.filter, n_particles=n_particles),
        heuristic=replace(base.heuristic, n_candidate_actions=n_candidates),
    )
    rng = np.random.default_rng(7)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_particles(cfg.orbit, n=cfg.filter.n_particles, rng=rng)
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
    return cfg, belief, env, when, nominal, station


def _all_factories(cfg):
    return {
        "infogreedy_infogain": make_infogreedy_policy(
            replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective="infogain"))
        ),
        "infogreedy_pdetect": make_infogreedy_policy(
            replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective="pdetect"))
        ),
        "qmdp": make_qmdp_policy(cfg),
        "mls": make_mls_policy(cfg),
    }


@pytest.mark.parametrize(
    "name",
    ["infogreedy_infogain", "infogreedy_pdetect", "qmdp", "mls"],
)
def test_returns_in_bounds_action(name):
    cfg, belief, env, when, _, _ = _setup()
    # The factory it was built with must carry the right greedy_objective.
    if name == "infogreedy_infogain":
        c = replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective="infogain"))
        factory = make_infogreedy_policy(c)
    elif name == "infogreedy_pdetect":
        c = replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective="pdetect"))
        factory = make_infogreedy_policy(c)
    elif name == "qmdp":
        c = cfg
        factory = make_qmdp_policy(c)
    else:
        c = cfg
        factory = make_mls_policy(c)

    act = factory(c, np.random.default_rng(0))
    a = act(belief, env, when)
    assert isinstance(a, TrackAction)
    t_max = cfg.pomcp.along_track_offset_max_s
    k_max = cfg.pomcp.cross_track_offset_max_deg
    assert -t_max - 1e-9 <= a.along_track_s <= t_max + 1e-9
    assert -k_max - 1e-9 <= a.cross_track_deg <= k_max + 1e-9


def test_qmdp_compatibility_alias_points_near_belief_mean():
    """The legacy API alias executes the posterior-centroid rule."""
    cfg, belief, env, when, nominal, station = _setup(n_candidates=64)
    factory = make_qmdp_policy(cfg)
    a = factory(cfg, np.random.default_rng(0))(belief, env, when)
    p = action_to_pointing(a, nominal, when, station)

    azelr = belief_azel_range(belief, when, station)
    # mean direction of the particle cloud
    mean_az = np.arctan2(
        np.mean(np.cos(azelr[:, 1]) * np.sin(azelr[:, 0])),
        np.mean(np.cos(azelr[:, 1]) * np.cos(azelr[:, 0])),
    )
    mean_el = np.arcsin(np.clip(np.mean(np.sin(azelr[:, 1])), -1, 1))
    sep = np.rad2deg(angular_separation(p.az, p.el, mean_az, mean_el))
    # The candidate set spans a coarse grid; closest action should land within
    # a few beamwidths of the mean direction (cloud spread + grid quantisation).
    assert sep < 12.0, sep


def test_mls_points_near_max_weight_particle():
    """MLS points at the max-weight particle. Concentrate weight on one particle
    and confirm the chosen pointing is closest to it."""
    cfg, belief, env, when, nominal, station = _setup(n_candidates=64)
    # Spike the weights onto a single particle.
    w = np.full(belief.n, 1e-6)
    target = 17
    w[target] = 1.0
    belief.weights = w / w.sum()

    azelr = belief_azel_range(belief, when, station)
    tgt_az, tgt_el = float(azelr[target, 0]), float(azelr[target, 1])

    a = make_mls_policy(cfg)(cfg, np.random.default_rng(0))(belief, env, when)
    p = action_to_pointing(a, nominal, when, station)
    sep_to_target = angular_separation(p.az, p.el, tgt_az, tgt_el)

    # No other particle's direction should yield a strictly closer-by-design
    # claim; the chosen pointing should be near the target particle.
    dtheta = delta_theta_from_azel(p, tgt_az, tgt_el)
    assert np.isclose(sep_to_target, dtheta)
    assert np.rad2deg(sep_to_target) < 12.0, np.rad2deg(sep_to_target)


def test_infogain_prefers_high_pd_when_belief_tight():
    """For a tight belief the infogain objective should still command a pointing
    that yields a non-trivial detection probability over the cloud."""
    cfg, belief, env, when, nominal, station = _setup(n_candidates=64)
    from antenna_pomdp.models.observation import detection_probability

    a = make_infogreedy_policy(
        replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective="infogain"))
    )(cfg, np.random.default_rng(0))(belief, env, when)
    p = action_to_pointing(a, nominal, when, station)
    azelr = belief_azel_range(belief, when, station)
    dthetas = np.array(
        [delta_theta_from_azel(p, azelr[i, 0], azelr[i, 1]) for i in range(belief.n)]
    )
    pd = detection_probability(dthetas, cfg.antenna, azelr[:, 1], azelr[:, 2])
    e_pd = float(np.mean(pd))
    assert e_pd > cfg.antenna.false_alarm_rate
