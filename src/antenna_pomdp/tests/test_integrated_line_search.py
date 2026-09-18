"""Focused contracts for exact integrated-dwell scalar action scoring."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np

from antenna_pomdp.baselines.line_search import make_line_search_policy
from antenna_pomdp.config import default_config
from antenna_pomdp.eval.compiled_schedule import (
    CompiledDwell,
    CompiledSchedule,
    IntegratedDwellLineSearchPolicy,
    _integrated_no_confirmation_update,
    compile_no_confirmation_schedule,
    evaluate_schedule,
    make_integrated_line_search_policy,
    update_no_confirmation_belief,
)
from antenna_pomdp.models.observation import (
    false_confirmation_probability,
    true_acquisition_probability,
)
from antenna_pomdp.models.particle_filter import (
    ParticleBelief,
    make_belief,
    particle_azel_range,
)
from antenna_pomdp.orbit.geometry import Station, angular_separation, find_next_pass
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import sample_leop_particles
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
    slew_limited_pointing,
)


def _case(*, temporal_step_s: float, n_particles: int = 7):
    base = default_config()
    cfg = replace(
        base,
        antenna=replace(
            base.antenna,
            dwell_time_s=5.0,
            temporal_quadrature_step_s=temporal_step_s,
            fwhm_deg=1.0,
            max_slew_rate_deg_s=1.0e9,
        ),
        filter=replace(
            base.filter,
            n_particles=n_particles,
            ess_resample_threshold=0.0,
        ),
        station=replace(base.station, pass_duration_s=5.0),
        pomcp=replace(
            base.pomcp,
            along_track_offset_max_s=30.0,
            n_along_track_offsets=7,
        ),
    )
    nominal = SGP4Propagator(
        cfg.orbit.nominal_tle_line1,
        cfg.orbit.nominal_tle_line2,
    )
    station = Station.from_config(cfg.station)
    visible = find_next_pass(
        nominal,
        station,
        nominal.epoch,
        min_elevation_rad=np.deg2rad(cfg.station.min_elevation_deg),
        search_horizon_s=12.0 * 3600.0,
    )
    assert visible is not None
    when = visible.open + timedelta(seconds=0.45 * (visible.close - visible.open).total_seconds())
    belief = make_belief(
        sample_leop_particles(
            cfg.orbit,
            cfg.leop,
            n=n_particles,
            rng=np.random.default_rng(1701),
        )
    )
    env = PointingEnv(
        nominal=nominal,
        truth=belief.particles[0],
        station=station,
        pass_start=when,
        pass_duration_s=cfg.station.pass_duration_s,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    return cfg, belief, env, when


def _brute_force_scores(
    policy: IntegratedDwellLineSearchPolicy,
    belief,
    env,
    when,
) -> np.ndarray:
    dwell_s = float(env.dwell_time_s)
    maximum_step_s = float(env.antenna_cfg.temporal_quadrature_step_s)
    n_subintervals = int(np.ceil(dwell_s / maximum_step_s))
    subinterval_s = dwell_s / n_subintervals
    offsets_s = (np.arange(n_subintervals) + 0.5) * subinterval_s
    sub_cfg = replace(env.antenna_cfg, dwell_time_s=subinterval_s)
    weights = np.asarray(belief.weights, dtype=float)
    weights /= weights.sum()

    scores = []
    for tau_s in policy.taus_s:
        particle_incidences = []
        action = TrackAction(float(tau_s), 0.0)
        requested_start = action_to_pointing(
            action,
            env.nominal,
            when,
            env.station,
        )
        uses_endpoint_fallback = False
        endpoint = requested_start
        if env.current_bore is not None:
            start_separation = angular_separation(
                env.current_bore[0],
                env.current_bore[1],
                requested_start.az,
                requested_start.el,
            )
            uses_endpoint_fallback = start_separation > env.max_slew_rad() + 1.0e-12
            if uses_endpoint_fallback:
                endpoint = slew_limited_pointing(
                    env.current_bore[0],
                    env.current_bore[1],
                    requested_start,
                    env.max_slew_rad(),
                )
        for particle in belief.particles:
            survival = 1.0
            correct_incidence = 0.0
            for offset_s in offsets_s:
                epoch = when + timedelta(seconds=float(offset_s))
                pointing = (
                    endpoint
                    if uses_endpoint_fallback
                    else action_to_pointing(
                        action,
                        env.nominal,
                        epoch,
                        env.station,
                    )
                )
                target_az, target_el, target_range_m = particle_azel_range(
                    particle,
                    epoch,
                    env.station,
                )
                separation = angular_separation(
                    pointing.az,
                    pointing.el,
                    target_az,
                    target_el,
                )
                p_true = float(
                    true_acquisition_probability(
                        separation,
                        sub_cfg,
                        target_el,
                        target_range_m,
                    )
                )
                p_false = float(
                    false_confirmation_probability(
                        separation,
                        sub_cfg,
                        target_el,
                        target_range_m,
                    )
                )
                correct_incidence += survival * p_true
                survival *= 1.0 - p_true - p_false
            particle_incidences.append(correct_incidence)
        scores.append(float(np.dot(weights, particle_incidences)))
    return np.asarray(scores)


def _single_dwell_evaluator_scores(
    policy: IntegratedDwellLineSearchPolicy,
    belief: ParticleBelief,
    env: PointingEnv,
    when,
    cfg,
) -> np.ndarray:
    """Independent policy objective assembled through the public evaluator."""
    weights = np.asarray(belief.weights, dtype=float)
    weights /= weights.sum()
    scores = []
    for tau_s in policy.taus_s:
        action = TrackAction(float(tau_s), 0.0)
        requested = action_to_pointing(action, env.nominal, when, env.station)
        realized = requested
        rate_clamped = False
        if env.current_bore is not None:
            separation = angular_separation(
                env.current_bore[0],
                env.current_bore[1],
                requested.az,
                requested.el,
            )
            rate_clamped = separation > env.max_slew_rad() + 1.0e-12
            if rate_clamped:
                realized = slew_limited_pointing(
                    env.current_bore[0],
                    env.current_bore[1],
                    requested,
                    env.max_slew_rad(),
                )
        schedule = CompiledSchedule(
            dwells=(
                CompiledDwell(
                    start_time=when,
                    action=action,
                    pointing=realized,
                    uses_rate_clamped_endpoint_fallback=rate_clamped,
                ),
            ),
            final_no_confirmation_belief=belief,
            nominal=env.nominal,
        )
        per_particle = np.asarray(
            [
                evaluate_schedule(schedule, particle, env.station, cfg).p_acquire
                for particle in belief.particles
            ]
        )
        scores.append(float(np.dot(weights, per_particle)))
    return np.asarray(scores)


def test_integrated_candidate_scores_match_independent_brute_force():
    cfg, belief, env, when = _case(temporal_step_s=1.25)
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            false_alarm_rate=0.2,
            confirmation_true_accept_rate=0.83,
            confirmation_false_accept_rate=0.17,
        ),
    )
    env.antenna_cfg = cfg.antenna
    policy = IntegratedDwellLineSearchPolicy(cfg, n_tau=5, candidate_batch_size=2)

    vectorized = policy.candidate_scores(belief, env, when)
    brute_force = _brute_force_scores(policy, belief, env, when)
    through_evaluator = _single_dwell_evaluator_scores(
        policy,
        belief,
        env,
        when,
        cfg,
    )

    assert np.allclose(vectorized, brute_force, rtol=0.0, atol=2.0e-13)
    assert np.allclose(vectorized, through_evaluator, rtol=0.0, atol=2.0e-13)


def test_one_subinterval_recovers_midpoint_policy_choice():
    cfg, belief, env, when = _case(temporal_step_s=5.0, n_particles=11)
    integrated = IntegratedDwellLineSearchPolicy(cfg, n_tau=9)
    midpoint = make_line_search_policy(cfg, n_tau=9)(
        cfg,
        np.random.default_rng(4),
    )

    integrated_action = integrated.act(belief, env, when)
    midpoint_action = midpoint(belief, env, when)

    assert integrated_action == midpoint_action


def test_finite_slew_scores_match_evaluator_endpoint_fallback():
    cfg, belief, env, when = _case(temporal_step_s=1.25)
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            max_slew_rate_deg_s=0.03,
            false_alarm_rate=0.12,
            confirmation_false_accept_rate=0.08,
        ),
    )
    env.antenna_cfg = cfg.antenna
    initial = action_to_pointing(
        TrackAction(0.0, 0.0),
        env.nominal,
        when,
        env.station,
    )
    env.current_bore = (initial.az, initial.el)
    policy = IntegratedDwellLineSearchPolicy(cfg, n_tau=5)

    vectorized = policy.candidate_scores(belief, env, when)
    brute_force = _brute_force_scores(policy, belief, env, when)

    assert np.allclose(vectorized, brute_force, rtol=0.0, atol=2.0e-12)


def test_later_dwell_cache_orientation_matches_absolute_epoch_scoring():
    """The flattened pass cache must map each H/Q slice to the right epoch."""
    cfg, belief, env, when = _case(temporal_step_s=1.25)
    env.pass_duration_s = 3.0 * env.dwell_time_s
    later = when + timedelta(seconds=2.0 * env.dwell_time_s)
    policy = IntegratedDwellLineSearchPolicy(
        cfg,
        n_tau=5,
        candidate_batch_size=2,
    )

    vectorized = policy.candidate_scores(belief, env, later)
    brute_force = _brute_force_scores(policy, belief, env, later)

    assert np.allclose(vectorized, brute_force, rtol=0.0, atol=2.0e-13)


def test_cache_reuses_weight_independent_geometry_and_rebuilds_for_particles():
    cfg, belief, env, when = _case(temporal_step_s=1.25)
    policy = IntegratedDwellLineSearchPolicy(cfg, n_tau=5)
    policy.candidate_scores(belief, env, when)
    initial_cache = policy._cache
    assert initial_cache is not None

    weights = np.arange(1.0, belief.n + 1.0)
    reweighted = ParticleBelief(
        particles=list(belief.particles),
        weights=weights / weights.sum(),
    )
    scores = policy.candidate_scores(reweighted, env, when)
    assert policy._cache is initial_cache
    assert np.allclose(
        scores,
        _brute_force_scores(policy, reweighted, env, when),
        rtol=0.0,
        atol=2.0e-13,
    )

    changed_particles = list(reweighted.particles)
    changed_particles[0] = replace(
        changed_particles[0],
        launch_slip_s=changed_particles[0].launch_slip_s + 2.0,
    )
    changed = ParticleBelief(
        particles=changed_particles,
        weights=reweighted.weights.copy(),
    )
    changed_scores = policy.candidate_scores(changed, env, when)
    assert policy._cache is not initial_cache
    assert np.allclose(
        changed_scores,
        _brute_force_scores(policy, changed, env, when),
        rtol=0.0,
        atol=2.0e-13,
    )


def test_compilation_matches_exact_policy_along_no_confirmation_branch():
    cfg, belief0, env, _ = _case(temporal_step_s=1.25)
    env.pass_duration_s = 3.0 * env.dwell_time_s
    factory = make_integrated_line_search_policy(
        n_tau=5,
        candidate_batch_size=2,
    )
    compiled = compile_no_confirmation_schedule(
        cfg,
        belief0,
        env,
        factory,
        np.random.default_rng(71),
    )

    belief = ParticleBelief(
        particles=list(belief0.particles),
        weights=belief0.weights.copy(),
    )
    policy = factory(cfg, np.random.default_rng(71))
    bore = env.current_bore
    for dwell in compiled.dwells:
        env.current_bore = bore
        assert policy(belief, env, dwell.start_time) == dwell.action
        belief = update_no_confirmation_belief(
            belief,
            dwell,
            env.nominal,
            env.station,
            cfg,
            np.random.default_rng(72),
        )
        if dwell.uses_rate_clamped_endpoint_fallback:
            bore = (dwell.pointing.az, dwell.pointing.el)
        else:
            endpoint = action_to_pointing(
                dwell.action,
                env.nominal,
                dwell.start_time + timedelta(seconds=env.dwell_time_s),
                env.station,
            )
            bore = (endpoint.az, endpoint.el)

    assert np.allclose(
        belief.weights,
        compiled.final_no_confirmation_belief.weights,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_factory_is_truth_independent_and_returns_action_in_envelope():
    cfg, belief, env, when = _case(temporal_step_s=1.0)

    class PoisonTruth:
        def __getattribute__(self, name):
            raise AssertionError(f"integrated policy inspected truth attribute {name!r}")

    env.truth = PoisonTruth()
    act = make_integrated_line_search_policy(cfg, n_tau=7)(
        cfg,
        np.random.default_rng(9),
    )
    action = act(belief, env, when)

    assert isinstance(action, TrackAction)
    assert action.cross_track_deg == 0.0
    assert -cfg.pomcp.along_track_offset_max_s <= action.along_track_s
    assert action.along_track_s <= cfg.pomcp.along_track_offset_max_s


def test_no_confirmation_update_normalizes_extreme_survival_in_log_domain():
    """Tiny but distinguishable branch probabilities must not reset to uniform."""
    cfg, belief, _, _ = _case(temporal_step_s=5.0, n_particles=2)
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            link_budget_enabled=False,
            false_alarm_rate=0.0,
            confirmation_true_accept_rate=1.0,
        ),
    )
    n_subintervals = 400
    pointings = np.tile(np.array([[0.0, 0.5]]), (n_subintervals, 1))
    geometry = np.empty((2, n_subintervals, 3), dtype=float)
    geometry[0, :, :] = (0.0, 0.5, 1.0e6)
    geometry[1, :, :] = (np.deg2rad(0.1), 0.5, 1.0e6)

    posterior = _integrated_no_confirmation_update(
        belief,
        pointings,
        geometry,
        cfg,
        subinterval_s=5.0,
        rng=np.random.default_rng(23),
    )

    assert np.isclose(np.sum(posterior.weights), 1.0)
    assert posterior.weights[1] > 1.0 - 1.0e-12
    assert posterior.weights[0] < 1.0e-12
