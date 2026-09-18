"""Tests for the Koopman/Stone optimal open-loop search controller."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from antenna_pomdp.baselines.koopman import (
    allocate_dwells,
    along_track_cell_offsets,
    build_schedule,
    cell_prior_mass,
    make_koopman_policy,
    plan_pass,
    water_fill_effort,
)
from antenna_pomdp.config import default_config
from antenna_pomdp.models.particle_filter import make_belief
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import sample_leop_particles, sample_leop_truth
from antenna_pomdp.pomdp.environment import PointingEnv


# ---------------------------------------------------------------------------
# Water-filling allocation
# ---------------------------------------------------------------------------


def test_waterfill_sums_to_budget():
    p = np.array([0.5, 0.25, 0.15, 0.1])
    e = water_fill_effort(p, total_budget=7.0, detection_rate_lambda=1.5)
    assert e.sum() == pytest.approx(7.0, abs=1e-6)
    assert np.all(e >= 0.0)


def test_waterfill_uniform_prior_is_uniform():
    p = np.full(8, 1.0 / 8.0)
    e = water_fill_effort(p, total_budget=16.0, detection_rate_lambda=1.0)
    assert e.sum() == pytest.approx(16.0, abs=1e-6)
    assert np.allclose(e, e[0])


def test_waterfill_more_effort_on_higher_prior():
    p = np.array([0.6, 0.25, 0.1, 0.05])
    e = water_fill_effort(p, total_budget=12.0, detection_rate_lambda=1.0)
    # Effort must be monotone non-increasing in descending prior order.
    assert e[0] >= e[1] >= e[2] >= e[3]


def test_waterfill_starves_cells_below_threshold():
    # Heavily skewed prior + small budget => low-mass cells get zero effort.
    p = np.array([0.7, 0.2, 0.06, 0.03, 0.01])
    e = water_fill_effort(p, total_budget=1.5, detection_rate_lambda=2.0)
    assert e.sum() == pytest.approx(1.5, abs=1e-6)
    assert np.any(e == 0.0)
    # The starved cells must be the lowest-prior ones.
    assert e[-1] == 0.0


def test_waterfill_zero_budget():
    p = np.array([0.5, 0.5])
    e = water_fill_effort(p, total_budget=0.0, detection_rate_lambda=1.0)
    assert np.all(e == 0.0)


# ---------------------------------------------------------------------------
# Integer dwell allocation
# ---------------------------------------------------------------------------


def test_allocate_dwells_sums_to_budget():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    for budget in (1, 5, 20, 137):
        n = allocate_dwells(p, n_dwells_budget=budget, detection_rate_lambda=1.0)
        assert int(n.sum()) == budget
        assert np.all(n >= 0)


def test_allocate_dwells_prefers_high_prior():
    p = np.array([0.6, 0.25, 0.1, 0.05])
    n = allocate_dwells(p, n_dwells_budget=30, detection_rate_lambda=1.0)
    assert n[0] >= n[1] >= n[2] >= n[3]


def test_allocate_dwells_degenerate_dumps_on_top_cell():
    p = np.array([0.0, 1.0, 0.0])
    n = allocate_dwells(p, n_dwells_budget=10, detection_rate_lambda=1.0)
    assert int(n.sum()) == 10
    assert n[1] == 10


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------


def test_build_schedule_ordered_by_descending_prior():
    offsets = np.array([-10.0, 0.0, 10.0])
    prior = np.array([0.1, 0.7, 0.2])
    n_per_cell = np.array([1, 5, 2])
    sched = build_schedule(n_per_cell, offsets, prior)
    assert len(sched) == 8
    # Highest-prior cell (index 1, offset 0) comes first.
    assert sched[0].along_track_s == 0.0
    # Next is index 2 (offset 10), then index 0 (offset -10).
    assert sched[5].along_track_s == 10.0
    assert sched[-1].along_track_s == -10.0
    # All cross-track offsets are zero.
    assert all(a.cross_track_deg == 0.0 for a in sched)


# ---------------------------------------------------------------------------
# Cell offsets
# ---------------------------------------------------------------------------


def test_cell_offsets_symmetric_and_in_bounds():
    cfg = default_config()
    offsets = along_track_cell_offsets(cfg.koopman, along_track_max_s=120.0)
    assert len(offsets) == cfg.koopman.n_cells
    assert offsets.min() == pytest.approx(-120.0)
    assert offsets.max() == pytest.approx(120.0)
    # Symmetric about zero.
    assert np.allclose(offsets, -offsets[::-1])


# ---------------------------------------------------------------------------
# End-to-end planning + policy
# ---------------------------------------------------------------------------


def _make_env_and_belief(seed: int = 0):
    cfg = default_config()
    rng = np.random.default_rng(seed)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_leop_particles(cfg.orbit, cfg.leop, n=100, rng=rng)
    belief = make_belief(particles)
    truth = sample_leop_truth(cfg.orbit, cfg.leop, rng)
    when = nominal.epoch + timedelta(minutes=5)
    env = PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=when,
        pass_duration_s=cfg.station.pass_duration_s,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    return cfg, nominal, station, belief, env, when, rng


def test_cell_prior_mass_normalised():
    cfg, nominal, station, belief, env, when, _ = _make_env_and_belief()
    offsets = along_track_cell_offsets(cfg.koopman, cfg.pomcp.along_track_offset_max_s)
    mass = cell_prior_mass(belief, offsets, nominal, when, station)
    assert len(mass) == cfg.koopman.n_cells
    assert mass.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(mass >= 0.0)


def test_plan_pass_budget_matches_horizon():
    cfg, nominal, station, belief, env, when, _ = _make_env_and_belief()
    sched = plan_pass(
        belief, env, when, nominal, station, cfg.koopman, cfg.pomcp.along_track_offset_max_s
    )
    assert int(sched.n_dwells_per_cell.sum()) == env.horizon
    assert len(sched.actions) == env.horizon


def test_policy_returns_inbounds_actions():
    cfg, nominal, station, belief, env, when, rng = _make_env_and_belief()
    factory = make_koopman_policy()
    act = factory(cfg, rng)
    state = env.initial_state()
    n = 0
    while not state.done and n < env.horizon:
        a = act(belief, env, state.when)
        assert abs(a.along_track_s) <= cfg.pomcp.along_track_offset_max_s + 1e-6
        assert a.cross_track_deg == 0.0
        state, _, _, _ = env.step(state, a, rng)
        n += 1
    assert n >= 1
