"""Tests for evaluator-consistent, cause-valued online execution."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from antenna_pomdp.baselines.line_search import make_line_search_policy
from antenna_pomdp.config import default_config
from antenna_pomdp.eval.compiled_schedule import (
    CompiledDwell,
    CompiledSchedule,
    compile_no_confirmation_schedule,
    evaluate_schedule,
)
from antenna_pomdp.eval.quadrature_executor import (
    QuadratureOutcome,
    QuadraturePolicyExecutor,
    dwell_cause_probabilities,
    sample_dwell_outcome,
)
from antenna_pomdp.models.particle_filter import make_belief
from antenna_pomdp.orbit.geometry import Station, find_next_pass
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle, sample_leop_particles
from antenna_pomdp.pomdp.environment import PointingEnv, TrackAction, action_to_pointing


def _case():
    base = default_config()
    cfg = replace(
        base,
        antenna=replace(
            base.antenna,
            dwell_time_s=2.0,
            temporal_quadrature_step_s=0.5,
            fwhm_deg=2.0,
            link_budget_enabled=False,
        ),
        filter=replace(base.filter, n_particles=24, ess_resample_threshold=0.0),
        station=replace(base.station, pass_duration_s=6.0),
    )
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    truth = OrbitParticle(propagator=nominal, launch_slip_s=-15.0)
    belief = make_belief(
        sample_leop_particles(
            cfg.orbit,
            cfg.leop,
            n=cfg.filter.n_particles,
            rng=np.random.default_rng(101),
        )
    )
    visible = find_next_pass(
        nominal,
        station,
        nominal.epoch,
        min_elevation_rad=np.deg2rad(cfg.station.min_elevation_deg),
        search_horizon_s=12 * 3600.0,
    )
    assert visible is not None
    pass_start = visible.open + timedelta(
        seconds=0.5 * (visible.close - visible.open).total_seconds()
    )
    env = PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=pass_start,
        pass_duration_s=cfg.station.pass_duration_s,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    return cfg, station, truth, belief, env


def test_forced_none_execution_matches_compiler_transition():
    cfg, _, _, belief, env = _case()
    factory = make_line_search_policy(cfg, n_tau=7)
    compiled = compile_no_confirmation_schedule(
        cfg,
        belief,
        env,
        factory,
        np.random.default_rng(202),
    )
    executor = QuadraturePolicyExecutor(
        cfg,
        belief,
        env,
        factory,
        policy_rng=np.random.default_rng(202),
        receiver_rng=np.random.default_rng(303),
    )
    result = executor.run(force_no_confirmation=True)
    online = executor.compiled_no_confirmation_prefix()

    assert result.state.done
    assert not result.state.acquired
    assert not result.state.false_confirmed
    assert all(step.outcome is QuadratureOutcome.NONE for step in result.steps)
    assert online.dwells == compiled.dwells
    assert np.allclose(
        online.final_no_confirmation_belief.weights,
        compiled.final_no_confirmation_belief.weights,
        atol=2e-13,
    )


def test_dwell_causes_are_bounded_and_conserve_probability():
    cfg, station, truth, belief, env = _case()
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            false_alarm_rate=0.8,
            confirmation_true_accept_rate=0.5,
            confirmation_false_accept_rate=0.8,
        ),
    )
    pointing = action_to_pointing(
        TrackAction(0.0, 0.0),
        env.nominal,
        env.pass_start,
        station,
    )
    dwell = CompiledDwell(
        start_time=env.pass_start,
        action=TrackAction(0.0, 0.0),
        pointing=pointing,
    )
    probabilities = dwell_cause_probabilities(
        dwell,
        truth,
        env.nominal,
        station,
        cfg,
    )

    assert np.all((probabilities.as_array() >= 0.0) & (probabilities.as_array() <= 1.0))
    assert probabilities.p_correct > 0.0
    assert probabilities.p_false > 0.0
    assert np.isclose(np.sum(probabilities.as_array()), 1.0, atol=1e-14)
    # The placeholder provenance belief used internally cannot leak into the
    # caller's supplied belief.
    assert belief.n == cfg.filter.n_particles


def test_false_confirmation_terminates_without_counting_as_acquisition():
    cfg, _, _, belief, env = _case()

    def fixed_policy_factory(_cfg, _rng):
        return lambda _belief, _env, _when: TrackAction(0.0, 0.0)

    executor = QuadraturePolicyExecutor(
        cfg,
        belief,
        env,
        fixed_policy_factory,
        policy_rng=np.random.default_rng(505),
        receiver_rng=np.random.default_rng(606),
    )
    step = executor.step(forced_outcome=QuadratureOutcome.FALSE)

    assert step.outcome is QuadratureOutcome.FALSE
    assert step.next_state.done
    assert step.next_state.false_confirmed
    assert not step.next_state.acquired
    with pytest.raises(RuntimeError, match="not a no-confirmation prefix"):
        executor.compiled_no_confirmation_prefix()


def test_sampled_causes_match_one_dwell_analytic_evaluator():
    cfg, station, _, belief, env = _case()
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            fwhm_deg=10.0,
            false_alarm_rate=0.8,
            confirmation_true_accept_rate=0.5,
            confirmation_false_accept_rate=0.8,
        ),
    )
    truth = OrbitParticle(propagator=env.nominal, launch_slip_s=0.0)
    action = TrackAction(0.0, 0.0)
    dwell = CompiledDwell(
        start_time=env.pass_start,
        action=action,
        pointing=action_to_pointing(action, env.nominal, env.pass_start, station),
    )
    schedule = CompiledSchedule(
        dwells=(dwell,),
        final_no_confirmation_belief=belief,
        nominal=env.nominal,
    )
    analytic = evaluate_schedule(schedule, truth, station, cfg)
    probabilities = dwell_cause_probabilities(
        dwell,
        truth,
        env.nominal,
        station,
        cfg,
    )

    n_draws = 50_000
    rng = np.random.default_rng(404)
    counts = {outcome: 0 for outcome in QuadratureOutcome}
    for _ in range(n_draws):
        counts[sample_dwell_outcome(probabilities, rng)] += 1
    observed = (
        np.asarray(
            (
                counts[QuadratureOutcome.CORRECT],
                counts[QuadratureOutcome.FALSE],
                counts[QuadratureOutcome.NONE],
            ),
            dtype=float,
        )
        / n_draws
    )
    expected = np.asarray(
        (
            analytic.p_acquire,
            analytic.p_false_confirmation,
            analytic.p_failure,
        )
    )
    standard_error = np.sqrt(expected * (1.0 - expected) / n_draws)

    assert expected[0] > 0.1
    assert expected[1] > 0.005
    assert np.allclose(probabilities.as_array(), expected, atol=2e-15)
    assert np.all(np.abs(observed - expected) <= 6.0 * standard_error + 5e-4)
