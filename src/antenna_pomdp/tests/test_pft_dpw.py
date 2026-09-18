"""Tests for the PFT-DPW continuous-action POMDP solver."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np

from antenna_pomdp.config import default_config
from antenna_pomdp.models.particle_filter import make_belief
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle, sample_particles, sample_truth
from antenna_pomdp.pomdp.environment import PointingEnv, TrackAction, action_to_pointing
from antenna_pomdp.pomdp.pft_dpw import PftDpwSolver, make_pft_policy


def _build_env(cfg, nominal, station, truth, pass_start):
    return PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=pass_start,
        pass_duration_s=60.0,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )


def test_plan_returns_in_bounds_action():
    """Planner returns a valid TrackAction within the configured action box."""
    cfg = default_config()
    rng = np.random.default_rng(1)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_particles(cfg.orbit, n=64, rng=rng)
    belief = make_belief(particles)
    truth = sample_truth(cfg.orbit, rng)
    pass_start = nominal.epoch + timedelta(minutes=5)
    env = _build_env(cfg, nominal, station, truth, pass_start)

    small = replace(cfg.solver, n_iterations=60, max_depth=5, n_belief_particles=20)
    solver = PftDpwSolver(small, cfg.antenna, station, nominal)
    action = solver.plan(belief, pass_start, env, rng)

    assert isinstance(action, TrackAction)
    assert abs(action.along_track_s) <= small.along_track_offset_max_s + 1e-9
    assert abs(action.cross_track_deg) <= small.cross_track_offset_max_deg + 1e-9


def test_tight_belief_points_near_truth():
    """With a belief concentrated on the truth, the chosen pointing is near truth."""
    cfg = default_config()
    # Wide beam makes the off-axis detection gradient broad, so the planner can
    # reliably find an action that illuminates the (known) truth.
    cfg = replace(cfg, antenna=replace(cfg.antenna, fwhm_deg=6.0))
    rng = np.random.default_rng(7)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    truth = sample_truth(cfg.orbit, rng)
    pass_start = nominal.epoch + timedelta(minutes=5)
    env = _build_env(cfg, nominal, station, truth, pass_start)

    # Tight belief: many copies of the truth orbit (zero spread).
    tight = [
        OrbitParticle(propagator=truth.propagator, launch_slip_s=truth.launch_slip_s, weight=1.0)
        for _ in range(64)
    ]
    belief = make_belief(tight)

    small = replace(cfg.solver, n_iterations=200, max_depth=6, n_belief_particles=20)
    solver = PftDpwSolver(small, cfg.antenna, station, nominal)
    action = solver.plan(belief, pass_start, env, rng)

    pointing = action_to_pointing(action, nominal, pass_start, station)
    truth_az, truth_el, _ = env.truth_azel_range(pass_start)
    from antenna_pomdp.orbit.geometry import angular_separation

    sep_deg = np.rad2deg(angular_separation(pointing.az, pointing.el, truth_az, truth_el))
    # Chosen pointing should be within a couple of beam-widths of the truth.
    assert sep_deg < 2.0 * cfg.antenna.fwhm_deg


def test_policy_factory_runs():
    """make_pft_policy yields a callable that returns a TrackAction."""
    cfg = default_config()
    small = replace(cfg.solver, n_iterations=40, max_depth=4, n_belief_particles=16)
    cfg = replace(cfg, solver=small)
    rng = np.random.default_rng(3)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_particles(cfg.orbit, n=48, rng=rng)
    belief = make_belief(particles)
    truth = sample_truth(cfg.orbit, rng)
    pass_start = nominal.epoch + timedelta(minutes=5)
    env = _build_env(cfg, nominal, station, truth, pass_start)

    factory = make_pft_policy(cfg)
    act = factory(cfg, rng)
    action = act(belief, env, pass_start)
    assert isinstance(action, TrackAction)
