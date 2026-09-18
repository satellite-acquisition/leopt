"""Scientific-contract tests for binary-terminal policy compilation."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np

import antenna_pomdp.eval.compiled_schedule as compiled_schedule_module
from antenna_pomdp.baselines.line_search import make_line_search_policy
from antenna_pomdp.config import default_config
from antenna_pomdp.eval.compiled_schedule import (
    CompiledDwell,
    CompiledSchedule,
    compile_no_confirmation_schedule,
    evaluate_schedule,
    truth_event_hazards,
    truth_confirmation_hazards,
    update_no_confirmation_belief,
)
from antenna_pomdp.models.observation import (
    Pointing,
    false_confirmation_probability,
    true_acquisition_probability,
)
from antenna_pomdp.models.particle_filter import ParticleBelief, make_belief
from antenna_pomdp.orbit.geometry import Station, angular_separation, find_next_pass
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle, sample_leop_particles, sample_leop_truth
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
    slew_limited_pointing,
)


def _case():
    base = default_config()
    cfg = replace(
        base,
        antenna=replace(base.antenna, dwell_time_s=5.0, fwhm_deg=2.0),
        filter=replace(base.filter, n_particles=80, ess_resample_threshold=0.0),
        station=replace(base.station, pass_duration_s=25.0),
    )
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    truth = sample_leop_truth(cfg.orbit, cfg.leop, np.random.default_rng(10))
    belief = make_belief(
        sample_leop_particles(
            cfg.orbit,
            cfg.leop,
            n=cfg.filter.n_particles,
            rng=np.random.default_rng(11),
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


def test_compiled_actions_equal_online_actions_on_all_miss_history():
    cfg, station, _, belief0, env = _case()
    factory = make_line_search_policy(cfg, n_tau=17)
    compiled = compile_no_confirmation_schedule(
        cfg,
        belief0,
        env,
        factory,
        np.random.default_rng(22),
    )

    belief = ParticleBelief(
        particles=list(belief0.particles),
        weights=belief0.weights.copy(),
    )
    online_rng = np.random.default_rng(22)
    policy = factory(cfg, online_rng)
    bore = env.current_bore
    online_actions = []
    for k in range(env.horizon):
        when = env.pass_start + timedelta(seconds=k * env.dwell_time_s)
        env.current_bore = bore
        action = policy(belief, env, when)
        commanded = action_to_pointing(action, env.nominal, when, station)
        if bore is None:
            pointing = commanded
            rate_clamped = False
        else:
            separation = angular_separation(
                bore[0],
                bore[1],
                commanded.az,
                commanded.el,
            )
            rate_clamped = separation > env.max_slew_rad() + 1e-12
            pointing = slew_limited_pointing(
                bore[0],
                bore[1],
                commanded,
                env.max_slew_rad(),
            )
        dwell = CompiledDwell(
            start_time=when,
            action=action,
            pointing=pointing,
            uses_rate_clamped_endpoint_fallback=rate_clamped,
        )
        online_actions.append(action)
        belief = update_no_confirmation_belief(
            belief,
            dwell,
            env.nominal,
            station,
            cfg,
            online_rng,
        )
        if rate_clamped:
            bore = (pointing.az, pointing.el)
        else:
            end = action_to_pointing(
                action,
                env.nominal,
                when + timedelta(seconds=env.dwell_time_s),
                station,
            )
            bore = (end.az, end.el)

    assert online_actions == [d.action for d in compiled.dwells]
    assert np.allclose(
        belief.weights,
        compiled.final_no_confirmation_belief.weights,
        atol=2e-13,
    )


def test_exact_schedule_probability_matches_hazard_product():
    cfg, station, truth, belief, env = _case()
    schedule = compile_no_confirmation_schedule(
        cfg,
        belief,
        env,
        make_line_search_policy(cfg, n_tau=17),
        np.random.default_rng(33),
    )
    hazards = truth_confirmation_hazards(schedule, truth, station, cfg)
    result = evaluate_schedule(schedule, truth, station, cfg)
    assert np.isclose(result.p_failure, np.prod(1.0 - hazards))
    assert np.isclose(result.p_acquire + result.p_failure, 1.0)
    assert np.isclose(result.first_confirmation_pmf.sum(), result.p_acquire)
    assert result.p_false_confirmation == 0.0
    assert 0.0 <= result.restricted_mean_time_s <= env.pass_duration_s + 1e-9


def test_schedule_separates_correct_acquisition_from_false_confirmation():
    cfg, station, truth, belief, env = _case()
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            false_alarm_rate=0.2,
            confirmation_true_accept_rate=0.8,
            confirmation_false_accept_rate=0.1,
        ),
    )
    env.antenna_cfg = cfg.antenna
    schedule = compile_no_confirmation_schedule(
        cfg,
        belief,
        env,
        make_line_search_policy(cfg, n_tau=17),
        np.random.default_rng(44),
    )
    true_hazards, false_hazards = truth_event_hazards(schedule, truth, station, cfg)
    result = evaluate_schedule(schedule, truth, station, cfg)
    terminal = true_hazards + false_hazards
    survival = np.concatenate(([1.0], np.cumprod(1.0 - terminal[:-1])))

    assert np.all(false_hazards > 0.0)
    assert np.isclose(result.p_acquire, np.sum(survival * true_hazards))
    assert np.isclose(result.p_false_confirmation, np.sum(survival * false_hazards))
    assert np.isclose(
        result.p_acquire + result.p_false_confirmation + result.p_failure,
        1.0,
    )
    assert np.allclose(
        result.first_confirmation_pmf,
        result.first_true_acquisition_pmf + result.first_false_confirmation_pmf,
    )


def test_compilation_preserves_nonnull_initial_boresight():
    """Finite-slew compilation starts from the caller's actual mount state."""
    cfg, _, _, belief, env = _case()
    cfg = replace(cfg, antenna=replace(cfg.antenna, max_slew_rate_deg_s=0.1))
    env.antenna_cfg = cfg.antenna
    env.current_bore = (0.25, 0.35)

    def factory(_cfg, _rng):
        def policy(_belief, _env, _when):
            return TrackAction(along_track_s=120.0, cross_track_deg=0.0)

        return policy

    schedule = compile_no_confirmation_schedule(
        cfg,
        belief,
        env,
        factory,
        np.random.default_rng(31),
        max_steps=1,
        update_belief=False,
    )
    commanded = action_to_pointing(
        TrackAction(along_track_s=120.0, cross_track_deg=0.0),
        env.nominal,
        env.pass_start,
        env.station,
    )
    expected = slew_limited_pointing(
        env.current_bore[0],
        env.current_bore[1],
        commanded,
        env.max_slew_rad(),
    )

    assert np.isclose(schedule.dwells[0].pointing.az, expected.az)
    assert np.isclose(schedule.dwells[0].pointing.el, expected.el)
    assert schedule.dwells[0].uses_rate_clamped_endpoint_fallback
    assert schedule.has_rate_clamped_endpoint_fallback
    assert env.current_bore == (0.25, 0.35)


def test_constant_geometry_recovers_reference_dwell_probability(monkeypatch):
    """Subinterval integration exactly recovers a constant reference exposure."""
    cfg, station, truth, belief, env = _case()
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            link_budget_enabled=False,
            false_alarm_rate=0.2,
            confirmation_true_accept_rate=0.8,
            confirmation_false_accept_rate=0.1,
            temporal_quadrature_step_s=0.25,
        ),
    )
    pointing = Pointing(az=0.7, el=0.6)
    target_az = pointing.az + np.deg2rad(0.4)
    target_el = pointing.el
    target_range_m = 900e3

    def constant_geometry(_truth, epochs, _station):
        return np.tile(
            np.array([target_az, target_el, target_range_m]),
            (len(epochs), 1),
        )

    monkeypatch.setattr(
        compiled_schedule_module,
        "_particle_azel_range_many",
        constant_geometry,
    )
    schedule = CompiledSchedule(
        dwells=(
            CompiledDwell(
                start_time=env.pass_start,
                action=TrackAction(0.0, 0.0),
                pointing=pointing,
            ),
        ),
        final_no_confirmation_belief=belief,
        nominal=None,
    )
    true_hazard, false_hazard = truth_event_hazards(schedule, truth, station, cfg)
    separation = compiled_schedule_module._angular_separation_many(
        np.array([[pointing.az, pointing.el]]),
        np.array([[target_az, target_el]]),
    )[0]
    expected_true = true_acquisition_probability(
        separation,
        cfg.antenna,
        target_el,
        target_range_m,
    )
    expected_false = false_confirmation_probability(
        separation,
        cfg.antenna,
        target_el,
        target_range_m,
    )
    result = evaluate_schedule(schedule, truth, station, cfg)

    assert np.isclose(true_hazard[0], expected_true, atol=2e-12)
    assert np.isclose(false_hazard[0], expected_false, atol=2e-12)
    assert np.isclose(
        result.p_acquire + result.p_false_confirmation + result.p_failure,
        1.0,
        atol=1e-14,
    )
    # Events are timed on the subinterval grid, not charged at the old
    # end-of-dwell timestamp.
    assert 0.0 < result.conditional_mean_time_s < cfg.antenna.dwell_time_s


def test_temporal_quadrature_refinement_is_stable():
    cfg, station, truth, belief, env = _case()
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            fwhm_deg=0.5,
            temporal_quadrature_step_s=1.0,
            link_budget_enabled=False,
        ),
    )
    env.antenna_cfg = cfg.antenna

    def factory(_cfg, _rng):
        return lambda _belief, _env, _when: TrackAction(-62.0, 0.0)

    schedule = compile_no_confirmation_schedule(
        cfg,
        belief,
        env,
        factory,
        np.random.default_rng(52),
        max_steps=1,
        update_belief=False,
    )
    coarse = evaluate_schedule(schedule, truth, station, cfg)
    medium_cfg = replace(
        cfg,
        antenna=replace(cfg.antenna, temporal_quadrature_step_s=0.25),
    )
    fine_cfg = replace(
        cfg,
        antenna=replace(cfg.antenna, temporal_quadrature_step_s=0.125),
    )
    medium = evaluate_schedule(schedule, truth, station, medium_cfg)
    fine = evaluate_schedule(schedule, truth, station, fine_cfg)

    assert 0.05 < fine.p_acquire < 0.5
    assert abs(medium.p_acquire - fine.p_acquire) < 1e-5
    assert abs(medium.restricted_mean_time_s - fine.restricted_mean_time_s) < 1e-3
    assert abs(medium.p_acquire - fine.p_acquire) <= abs(coarse.p_acquire - fine.p_acquire) + 1e-12


def test_compilation_uses_evaluation_likelihood_and_never_truth():
    cfg, station, _, belief, env = _case()
    cfg = replace(
        cfg,
        antenna=replace(cfg.antenna, temporal_quadrature_step_s=0.5),
    )
    env.antenna_cfg = cfg.antenna

    class PoisonTruth:
        def __getattribute__(self, name):
            raise AssertionError(f"compilation inspected truth attribute {name!r}")

    env.truth = PoisonTruth()

    def factory(_cfg, _rng):
        return lambda _belief, _env, _when: TrackAction(-20.0, 0.25)

    schedule = compile_no_confirmation_schedule(
        cfg,
        belief,
        env,
        factory,
        np.random.default_rng(61),
        max_steps=1,
    )
    particle_failures = np.asarray(
        [
            evaluate_schedule(schedule, particle, station, cfg).p_failure
            for particle in belief.particles
        ]
    )
    expected_weights = belief.weights * particle_failures
    expected_weights /= expected_weights.sum()

    assert np.allclose(
        schedule.final_no_confirmation_belief.weights,
        expected_weights,
        atol=2e-13,
    )


def test_unclamped_dwell_follows_track_coordinates_not_fixed_azel():
    cfg, station, _, belief, env = _case()
    cfg = replace(
        cfg,
        antenna=replace(
            cfg.antenna,
            fwhm_deg=0.05,
            temporal_quadrature_step_s=0.25,
            max_slew_rate_deg_s=1e9,
        ),
    )
    env.antenna_cfg = cfg.antenna
    truth = OrbitParticle(propagator=env.nominal, launch_slip_s=0.0)

    def factory(_cfg, _rng):
        return lambda _belief, _env, _when: TrackAction(0.0, 0.0)

    moving_schedule = compile_no_confirmation_schedule(
        cfg,
        belief,
        env,
        factory,
        np.random.default_rng(71),
        max_steps=1,
        update_belief=False,
    )
    fixed_azel_schedule = replace(moving_schedule, nominal=None)
    moving = evaluate_schedule(moving_schedule, truth, station, cfg)
    fixed = evaluate_schedule(fixed_azel_schedule, truth, station, cfg)

    assert not moving_schedule.has_rate_clamped_endpoint_fallback
    assert moving.p_acquire > 0.9
    assert moving.p_acquire > fixed.p_acquire + 0.1
