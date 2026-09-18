"""Compile binary-terminal acquisition policies into open-loop schedules.

For the observation alphabet used by the acquisition model, a confirmed
acquisition terminates the search. Consequently every continuing history is
the same history: a sequence of ``no confirmation`` observations. Any
deterministic online policy can therefore be evaluated recursively along that
single branch before the pass starts. The resulting command list produces
exactly the same actions as the online policy up to acquisition.

This is a structural equivalence, not a claim that observations are useless in
general. It ceases to hold when the receiver supplies two or more distinct
nonterminal outcomes (for example calibrated power, Doppler residual, or an
unresolved trigger awaiting confirmation), or when unmodeled state becomes
known during the pass.

Randomized policies are covered conditionally on a pre-sampled policy/filter
random seed. This module keeps that random stream separate from truth and
receiver sampling so paired experiments remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import jday

from antenna_pomdp.config import Config
from antenna_pomdp.eval.leop_runner import PolicyFactory
from antenna_pomdp.models.observation import (
    Pointing,
    reference_event_probabilities,
    scale_competing_reference_probabilities,
)
from antenna_pomdp.models.particle_filter import ParticleBelief
from antenna_pomdp.orbit.geometry import Station, angular_separation
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
    slew_limited_pointing,
)


@dataclass(frozen=True)
class CompiledDwell:
    """One precomputed track-relative command.

    For an unclamped dwell, ``pointing`` is the command at ``start_time`` only;
    evaluation keeps ``action`` fixed and recomputes the moving boresight at
    every quadrature epoch. For a rate-clamped dwell, ``pointing`` is the
    slew-limited endpoint used by the explicitly labeled discrete fallback.
    """

    start_time: datetime
    action: TrackAction
    pointing: Pointing
    uses_rate_clamped_endpoint_fallback: bool = False


@dataclass(frozen=True)
class CompiledSchedule:
    """A policy evaluated along its unique nonterminal observation branch."""

    dwells: tuple[CompiledDwell, ...]
    final_no_confirmation_belief: ParticleBelief
    nominal: SGP4Propagator | None = None

    @property
    def has_rate_clamped_endpoint_fallback(self) -> bool:
        """Whether any dwell uses the discrete mechanical approximation."""
        return any(d.uses_rate_clamped_endpoint_fallback for d in self.dwells)


@dataclass(frozen=True)
class ScheduleEvaluation:
    """Deterministic competing-risk quadrature for one truth realization."""

    p_acquire: float
    p_false_confirmation: float
    p_failure: float
    first_confirmation_pmf: np.ndarray
    first_true_acquisition_pmf: np.ndarray
    first_false_confirmation_pmf: np.ndarray
    conditional_mean_time_s: float
    restricted_mean_time_s: float


@dataclass(frozen=True)
class _SubintervalRisks:
    """Conditional cause risks and event times on the quadrature grid."""

    p_true: np.ndarray
    p_false: np.ndarray
    event_time_s: np.ndarray
    n_per_dwell: int


def _quadrature(cfg: Config) -> tuple[int, float, np.ndarray]:
    """Return a uniform midpoint rule whose intervals do not exceed the cap."""
    dwell_s = float(cfg.antenna.dwell_time_s)
    maximum_step_s = float(cfg.antenna.temporal_quadrature_step_s)
    if dwell_s <= 0.0:
        raise ValueError("antenna.dwell_time_s must be positive")
    if maximum_step_s <= 0.0:
        raise ValueError("antenna.temporal_quadrature_step_s must be positive")
    n = max(1, int(np.ceil(dwell_s / maximum_step_s)))
    dt_s = dwell_s / n
    offsets_s = (np.arange(n, dtype=float) + 0.5) * dt_s
    return n, dt_s, offsets_s


def _as_utc_timestamp(when: datetime) -> float:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


def _propagate_positions_many(
    propagator: SGP4Propagator,
    epochs: tuple[datetime, ...] | list[datetime],
) -> np.ndarray:
    """Vectorized SGP4 positions, with a public-API fallback for test doubles."""
    if not epochs:
        return np.empty((0, 3), dtype=float)
    satrec = getattr(propagator, "_sat", None)
    sgp4_array = getattr(satrec, "sgp4_array", None)
    if sgp4_array is None:
        return np.stack([propagator.propagate(t).r_eci_m for t in epochs])

    normalized = [
        t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)
        for t in epochs
    ]
    jd, fraction = jday(
        np.asarray([t.year for t in normalized]),
        np.asarray([t.month for t in normalized]),
        np.asarray([t.day for t in normalized]),
        np.asarray([t.hour for t in normalized]),
        np.asarray([t.minute for t in normalized]),
        np.asarray([t.second + t.microsecond * 1e-6 for t in normalized]),
    )
    errors, positions_km, _ = sgp4_array(jd, fraction)
    if np.any(errors != 0):
        first = int(np.flatnonzero(errors != 0)[0])
        raise RuntimeError(
            f"SGP4 propagation error code {int(errors[first])} at {epochs[first].isoformat()}"
        )
    return np.asarray(positions_km, dtype=float) * 1e3


def _eci_to_azel_many(
    positions_eci_m: np.ndarray,
    observer_epochs: tuple[datetime, ...] | list[datetime],
    station: Station,
) -> np.ndarray:
    """Vectorized counterpart of ``eci_to_azel`` for one station."""
    if len(observer_epochs) == 0:
        return np.empty((0, 3), dtype=float)
    julian = (
        np.asarray([_as_utc_timestamp(t) for t in observer_epochs], dtype=float) / 86400.0
        + 2440587.5
    )
    centuries = (julian - 2451545.0) / 36525.0
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * centuries
        + 0.093104 * centuries**2
        - 6.2e-6 * centuries**3
    )
    theta = np.mod(gmst_s, 86400.0) * (2.0 * np.pi / 86400.0)
    c = np.cos(theta)
    s = np.sin(theta)
    ecef = np.column_stack(
        (
            c * positions_eci_m[:, 0] + s * positions_eci_m[:, 1],
            -s * positions_eci_m[:, 0] + c * positions_eci_m[:, 1],
            positions_eci_m[:, 2],
        )
    )
    rho = ecef - station.ecef_m
    enu = rho @ station.rot_ecef_to_enu.T
    ranges = np.linalg.norm(enu, axis=1)
    elevations = np.arcsin(np.clip(enu[:, 2] / ranges, -1.0, 1.0))
    azimuths = np.mod(np.arctan2(enu[:, 0], enu[:, 1]), 2.0 * np.pi)
    return np.column_stack((azimuths, elevations, ranges))


def _particle_azel_range_many(
    particle: OrbitParticle,
    observer_epochs: tuple[datetime, ...] | list[datetime],
    station: Station,
) -> np.ndarray:
    effective_epochs = [
        when - timedelta(seconds=particle.launch_slip_s) for when in observer_epochs
    ]
    positions = _propagate_positions_many(particle.propagator, effective_epochs)
    return _eci_to_azel_many(positions, observer_epochs, station)


def _track_pointings_many(
    action: TrackAction,
    nominal: SGP4Propagator,
    observer_epochs: tuple[datetime, ...] | list[datetime],
    station: Station,
) -> np.ndarray:
    """Vectorized ``action_to_pointing`` at several observer epochs."""
    target_epochs = [when + timedelta(seconds=action.along_track_s) for when in observer_epochs]
    positions0 = _propagate_positions_many(nominal, target_epochs)
    azel0 = _eci_to_azel_many(positions0, observer_epochs, station)
    positions1 = _propagate_positions_many(
        nominal,
        [when + timedelta(seconds=1.0) for when in target_epochs],
    )
    azel1 = _eci_to_azel_many(positions1, observer_epochs, station)

    az0 = azel0[:, 0]
    el0 = azel0[:, 1]
    daz = np.mod(azel1[:, 0] - az0 + np.pi, 2.0 * np.pi) - np.pi
    delta_el = azel1[:, 1] - el0
    tangent = np.column_stack((daz * np.cos(el0), delta_el))
    norms = np.linalg.norm(tangent, axis=1)
    safe = norms >= 1e-9
    tangent[safe] /= norms[safe, None]
    perpendicular = np.empty_like(tangent)
    perpendicular[safe] = np.column_stack((-tangent[safe, 1], tangent[safe, 0]))
    perpendicular[~safe] = np.array([1.0, 0.0])

    offset_rad = np.deg2rad(action.cross_track_deg)
    az = az0 + offset_rad * perpendicular[:, 0] / np.maximum(1e-6, np.cos(el0))
    el = el0 + offset_rad * perpendicular[:, 1]
    return np.column_stack(
        (
            np.mod(az, 2.0 * np.pi),
            np.clip(el, -np.pi / 2.0 + 1e-3, np.pi / 2.0 - 1e-3),
        )
    )


def _dwell_pointings(
    dwell: CompiledDwell,
    nominal: SGP4Propagator | None,
    midpoint_epochs: tuple[datetime, ...] | list[datetime],
    station: Station,
) -> np.ndarray:
    if dwell.uses_rate_clamped_endpoint_fallback or nominal is None:
        return np.tile(np.array([dwell.pointing.az, dwell.pointing.el]), (len(midpoint_epochs), 1))
    return _track_pointings_many(dwell.action, nominal, midpoint_epochs, station)


def _angular_separation_many(pointings: np.ndarray, targets: np.ndarray) -> np.ndarray:
    cos_separation = np.sin(pointings[..., 1]) * np.sin(targets[..., 1]) + np.cos(
        pointings[..., 1]
    ) * np.cos(targets[..., 1]) * np.cos(pointings[..., 0] - targets[..., 0])
    return np.arccos(np.clip(cos_separation, -1.0, 1.0))


def _resample_belief(
    belief: ParticleBelief,
    cfg: Config,
    rng: np.random.Generator,
) -> ParticleBelief:
    """Systematic resampling, matching the package particle-filter contract."""
    n = belief.n
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(belief.weights)
    cumulative[-1] = 1.0
    indices = np.clip(np.searchsorted(cumulative, positions), 0, n - 1)
    particles: list[OrbitParticle] = []
    for index in indices:
        source = belief.particles[int(index)]
        jitter = float(rng.normal(0.0, cfg.filter.process_noise_along_track_s))
        particles.append(
            OrbitParticle(
                propagator=source.propagator,
                launch_slip_s=source.launch_slip_s + jitter,
                weight=1.0,
            )
        )
    return ParticleBelief(particles=particles, weights=np.full(n, 1.0 / n))


def _integrated_no_confirmation_update(
    belief: ParticleBelief,
    pointings: np.ndarray,
    particle_geometry: np.ndarray,
    cfg: Config,
    subinterval_s: float,
    rng: np.random.Generator,
) -> ParticleBelief:
    """Apply one dwell's integrated no-terminal-event likelihood per particle."""
    dtheta = _angular_separation_many(pointings[None, :, :], particle_geometry[:, :, :2])
    sub_cfg = replace(cfg.antenna, dwell_time_s=subinterval_s)
    p_true_ref, p_false_ref = reference_event_probabilities(
        dtheta,
        sub_cfg,
        particle_geometry[:, :, 1],
        particle_geometry[:, :, 2],
    )
    p_true, p_false = scale_competing_reference_probabilities(
        p_true_ref,
        p_false_ref,
        sub_cfg,
    )
    terminal = np.clip(np.asarray(p_true) + np.asarray(p_false), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        log_survival = np.sum(np.log1p(-terminal), axis=1)

    # Normalize in the log domain.  Directly forming
    # ``weights * exp(log_survival)`` can underflow every particle during a
    # long, high-detection all-miss history even when their relative
    # likelihoods remain well defined.  A uniform reset in that case would
    # change the compiled policy solely because the branch is improbable.
    prior_weights = np.asarray(belief.weights, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mass = np.log(prior_weights) + log_survival
    finite = np.isfinite(log_mass)
    if not np.any(finite):
        # The conditioned branch is impossible under every represented
        # particle (or the supplied prior has no finite mass), so no posterior
        # is mathematically defined. Use the deterministic fallback.
        new_weights = np.full(belief.n, 1.0 / belief.n)
    else:
        maximum = float(np.max(log_mass[finite]))
        new_weights = np.zeros(belief.n, dtype=float)
        new_weights[finite] = np.exp(log_mass[finite] - maximum)
        new_weights /= float(np.sum(new_weights))
    candidate = ParticleBelief(particles=list(belief.particles), weights=new_weights)
    if candidate.effective_sample_size() < cfg.filter.ess_resample_threshold * candidate.n:
        return _resample_belief(candidate, cfg, rng)
    return candidate


def update_no_confirmation_belief(
    belief: ParticleBelief,
    dwell: CompiledDwell,
    nominal: SGP4Propagator,
    station: Station,
    cfg: Config,
    rng: np.random.Generator,
) -> ParticleBelief:
    """Apply the temporal model's all-no-confirmation update for one dwell.

    This truth-free helper is the online counterpart of the update used during
    compilation. Exposing the shared likelihood contract makes the structural
    compilation equivalence testable without reverting to a legacy
    start-of-dwell observation sample.
    """
    _, subinterval_s, offsets_s = _quadrature(cfg)
    midpoint_epochs = tuple(
        dwell.start_time + timedelta(seconds=float(offset_s)) for offset_s in offsets_s
    )
    pointings = _dwell_pointings(dwell, nominal, midpoint_epochs, station)
    particle_geometry = np.stack(
        [
            _particle_azel_range_many(particle, midpoint_epochs, station)
            for particle in belief.particles
        ]
    )
    return _integrated_no_confirmation_update(
        belief,
        pointings,
        particle_geometry,
        cfg,
        subinterval_s,
        rng,
    )


@dataclass(frozen=True)
class _IntegratedLineSearchCache:
    """Pass-wide geometry shared by every exact integrated-dwell decision."""

    key: tuple[object, ...]
    particle_signature: tuple[tuple[int, float], ...]
    track_pointings: np.ndarray
    dwell_start_pointings: np.ndarray
    particle_geometry: np.ndarray
    n_quadrature: int
    subinterval_s: float


class IntegratedDwellLineSearchPolicy:
    """Detection-greedy scalar search scored with the evaluator's dwell model.

    Unlike the lightweight midpoint scorer in
    :func:`antenna_pomdp.baselines.line_search.make_line_search_policy`, this
    policy integrates every candidate over the same moving time-of-validity
    boresight, cause-specific risks, and temporal midpoint quadrature used by
    :func:`evaluate_schedule`.  Candidate track pointings and particle geometry
    for the complete pass are cached on the first call.  Subsequent decisions
    therefore recompute only the candidate-by-particle event-risk tensor for the
    current dwell as posterior weights change.

    A finite-slew candidate is scored with the evaluator's explicitly labelled
    fixed-endpoint fallback.  The primary experiment's effectively unbounded
    slew rate never enters that fallback.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        n_tau: int | None = None,
        candidate_batch_size: int = 8,
    ) -> None:
        self._cfg = cfg
        resolved_n_tau = (
            max(1, int(cfg.pomcp.n_along_track_offsets)) if n_tau is None else max(1, int(n_tau))
        )
        if candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        self._candidate_batch_size = int(candidate_batch_size)
        self.taus_s = np.linspace(
            -float(cfg.pomcp.along_track_offset_max_s),
            float(cfg.pomcp.along_track_offset_max_s),
            resolved_n_tau,
        )
        self._cache: _IntegratedLineSearchCache | None = None

    @staticmethod
    def _particle_signature(
        belief: ParticleBelief,
    ) -> tuple[tuple[int, float], ...]:
        return tuple(
            (id(particle.propagator), float(particle.launch_slip_s))
            for particle in belief.particles
        )

    def _cache_key(self, env: PointingEnv) -> tuple[object, ...]:
        return (
            id(env.nominal),
            id(env.station),
            env.pass_start,
            int(env.horizon),
            float(env.dwell_time_s),
            tuple(float(value) for value in self.taus_s),
        )

    def _build_cache(
        self,
        belief: ParticleBelief,
        env: PointingEnv,
    ) -> _IntegratedLineSearchCache:
        n_quadrature, subinterval_s, offsets_s = _quadrature(self._cfg)
        midpoint_epochs = tuple(
            env.pass_start + timedelta(seconds=step * env.dwell_time_s + float(offset_s))
            for step in range(env.horizon)
            for offset_s in offsets_s
        )
        dwell_start_epochs = tuple(
            env.pass_start + timedelta(seconds=step * env.dwell_time_s)
            for step in range(env.horizon)
        )

        # (H, K, Q, 2): each action remains fixed in track coordinates while
        # its az/el command follows the moving nominal track through the dwell.
        track_pointings = np.stack(
            [
                _track_pointings_many(
                    TrackAction(float(tau_s), 0.0),
                    env.nominal,
                    midpoint_epochs,
                    env.station,
                )
                for tau_s in self.taus_s
            ],
            axis=0,
        )
        track_pointings = np.transpose(
            track_pointings.reshape(
                self.taus_s.size,
                env.horizon,
                n_quadrature,
                2,
            ),
            (1, 0, 2, 3),
        )

        # (H, K, 2): needed only to determine the same start-of-dwell
        # reachability flag used by compile_no_confirmation_schedule.
        dwell_start_pointings = np.stack(
            [
                _track_pointings_many(
                    TrackAction(float(tau_s), 0.0),
                    env.nominal,
                    dwell_start_epochs,
                    env.station,
                )
                for tau_s in self.taus_s
            ],
            axis=0,
        ).transpose(1, 0, 2)

        # (N, H, Q, 3): truth-free geometry for the complete empirical belief.
        particle_geometry = np.stack(
            [
                _particle_azel_range_many(
                    particle,
                    midpoint_epochs,
                    env.station,
                )
                for particle in belief.particles
            ],
            axis=0,
        ).reshape(belief.n, env.horizon, n_quadrature, 3)

        return _IntegratedLineSearchCache(
            key=self._cache_key(env),
            particle_signature=self._particle_signature(belief),
            track_pointings=track_pointings,
            dwell_start_pointings=dwell_start_pointings,
            particle_geometry=particle_geometry,
            n_quadrature=n_quadrature,
            subinterval_s=subinterval_s,
        )

    def _cache_for(
        self,
        belief: ParticleBelief,
        env: PointingEnv,
    ) -> _IntegratedLineSearchCache:
        key = self._cache_key(env)
        signature = self._particle_signature(belief)
        if (
            self._cache is None
            or self._cache.key != key
            or self._cache.particle_signature != signature
        ):
            self._cache = self._build_cache(belief, env)
        return self._cache

    @staticmethod
    def _step_index(env: PointingEnv, when: datetime) -> int:
        elapsed_s = (when - env.pass_start).total_seconds()
        raw_index = elapsed_s / float(env.dwell_time_s)
        step_index = int(round(raw_index))
        if not np.isclose(raw_index, step_index, rtol=0.0, atol=1.0e-9):
            raise ValueError("integrated line search requires a dwell-boundary decision epoch")
        if step_index < 0 or step_index >= env.horizon:
            raise ValueError("integrated line-search decision epoch is outside the pass")
        return step_index

    def candidate_scores(
        self,
        belief: ParticleBelief,
        env: PointingEnv,
        when: datetime,
    ) -> np.ndarray:
        """Expected correct-acquisition incidence for every scalar candidate."""
        cache = self._cache_for(belief, env)
        step_index = self._step_index(env, when)
        pointings = cache.track_pointings[step_index]

        bore = getattr(env, "current_bore", None)
        if bore is not None:
            start_pointings = cache.dwell_start_pointings[step_index]
            bore_array = np.broadcast_to(
                np.asarray(bore, dtype=float),
                start_pointings.shape,
            )
            separation = _angular_separation_many(start_pointings, bore_array)
            clamped = separation > env.max_slew_rad() + 1.0e-12
            if np.any(clamped):
                pointings = pointings.copy()
                for candidate_index in np.flatnonzero(clamped):
                    requested = Pointing(
                        float(start_pointings[candidate_index, 0]),
                        float(start_pointings[candidate_index, 1]),
                    )
                    endpoint = slew_limited_pointing(
                        float(bore[0]),
                        float(bore[1]),
                        requested,
                        env.max_slew_rad(),
                    )
                    pointings[candidate_index, :, 0] = endpoint.az
                    pointings[candidate_index, :, 1] = endpoint.el

        geometry = cache.particle_geometry[:, step_index]
        sub_cfg = replace(
            self._cfg.antenna,
            dwell_time_s=cache.subinterval_s,
        )
        weights = np.asarray(belief.weights, dtype=float)
        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            weights = np.full(belief.n, 1.0 / belief.n)
        else:
            weights = weights / total

        # A primary-size tensor has shape (193, 1000, 20).  Several risk
        # arrays coexist during evaluation, so materializing every candidate
        # at once creates a needlessly large transient allocation in each
        # experiment worker.  Candidate batches are independent and preserve
        # exactly the same per-candidate objective.
        scores = np.empty(self.taus_s.size, dtype=float)
        for start in range(0, self.taus_s.size, self._candidate_batch_size):
            stop = min(start + self._candidate_batch_size, self.taus_s.size)
            candidate_pointings = pointings[start:stop]
            dtheta = _angular_separation_many(
                candidate_pointings[:, None, :, :],
                geometry[None, :, :, :2],
            )
            p_true_ref, p_false_ref = reference_event_probabilities(
                dtheta,
                sub_cfg,
                geometry[None, :, :, 1],
                geometry[None, :, :, 2],
            )
            p_true, p_false = scale_competing_reference_probabilities(
                p_true_ref,
                p_false_ref,
                sub_cfg,
            )
            p_true = np.asarray(p_true, dtype=float)
            terminal = np.clip(p_true + np.asarray(p_false, dtype=float), 0.0, 1.0)
            survival_before = np.concatenate(
                (
                    np.ones(terminal.shape[:-1] + (1,), dtype=float),
                    np.cumprod(1.0 - terminal[..., :-1], axis=-1),
                ),
                axis=-1,
            )
            correct_incidence = np.sum(survival_before * p_true, axis=-1)
            scores[start:stop] = correct_incidence @ weights
        return scores

    def act(
        self,
        belief: ParticleBelief,
        env: PointingEnv,
        when: datetime,
    ) -> TrackAction:
        scores = self.candidate_scores(belief, env, when)
        return TrackAction(float(self.taus_s[int(np.argmax(scores))]), 0.0)


def make_integrated_line_search_policy(
    cfg: Config | None = None,
    *,
    n_tau: int | None = None,
    candidate_batch_size: int = 8,
) -> PolicyFactory:
    """Factory for exact integrated-dwell posterior detection-greedy search.

    ``cfg`` is accepted for API symmetry with the midpoint factory; the config
    supplied through the :class:`PolicyFactory` contract remains authoritative.
    """

    del cfg

    def factory(_cfg: Config, _rng: np.random.Generator):
        policy = IntegratedDwellLineSearchPolicy(
            _cfg,
            n_tau=n_tau,
            candidate_batch_size=candidate_batch_size,
        )
        return policy.act

    return factory


def compile_no_confirmation_schedule(
    cfg: Config,
    initial_belief: ParticleBelief,
    env: PointingEnv,
    policy_factory: PolicyFactory,
    rng: np.random.Generator,
    *,
    max_steps: int | None = None,
    update_belief: bool = True,
) -> CompiledSchedule:
    """Compile a policy by recursively applying the no-confirmation update.

    ``env.truth`` is never inspected. The environment is shallow-copied so the
    caller's live boresight is not mutated. An unclamped action remains fixed
    in track coordinates while its az/el command follows the nominal track
    throughout the dwell. A rate-clamped dwell retains the legacy
    discrete-endpoint approximation and advertises that fact on both the dwell
    and schedule. Set ``update_belief=False`` only for policies known to ignore
    all within-pass belief updates.

    The no-confirmation update is the product of the same quadrature
    subinterval survival likelihoods used by schedule evaluation; it is never
    a start-of-dwell sample and never depends on ``env.truth``.
    """
    # Preserve the caller-supplied initial mount state. Compilation operates on
    # a shallow copy so it cannot mutate the live environment, but a
    # finite-slew schedule must start from the same boresight as the online
    # policy.
    local_env = replace(env, current_bore=env.current_bore)
    policy = policy_factory(cfg, rng)
    belief = ParticleBelief(
        particles=list(initial_belief.particles),
        weights=np.asarray(initial_belief.weights, dtype=float).copy(),
    )
    n_steps = local_env.horizon if max_steps is None else min(local_env.horizon, max_steps)
    bore = env.current_bore
    dwells: list[CompiledDwell] = []
    n_quadrature, subinterval_s, offsets_s = _quadrature(cfg)

    # Precompute every initial particle's time-varying geometry over the pass.
    # In the released experiments resampling is disabled, reducing each
    # particle to one vectorized SGP4 call per compiled schedule. A cache miss
    # after optional jittered resampling is filled lazily without changing the
    # numerical contract.
    all_midpoints = tuple(
        local_env.pass_start + timedelta(seconds=k * local_env.dwell_time_s + float(offset_s))
        for k in range(n_steps)
        for offset_s in offsets_s
    )
    geometry_cache: dict[tuple[int, float], np.ndarray] = {}

    def geometry_for(particle: OrbitParticle) -> np.ndarray:
        key = (id(particle.propagator), float(particle.launch_slip_s))
        cached = geometry_cache.get(key)
        if cached is None:
            cached = _particle_azel_range_many(particle, all_midpoints, local_env.station)
            geometry_cache[key] = cached
        return cached

    for k in range(n_steps):
        when = local_env.pass_start + timedelta(seconds=k * local_env.dwell_time_s)
        local_env.current_bore = bore
        action = policy(belief, local_env, when)
        commanded = action_to_pointing(action, local_env.nominal, when, local_env.station)
        if bore is None:
            pointing = commanded
            rate_clamped = False
        else:
            separation = angular_separation(bore[0], bore[1], commanded.az, commanded.el)
            rate_clamped = separation > local_env.max_slew_rad() + 1e-12
            pointing = slew_limited_pointing(
                bore[0],
                bore[1],
                commanded,
                local_env.max_slew_rad(),
            )
        dwell = CompiledDwell(
            when,
            action,
            pointing,
            uses_rate_clamped_endpoint_fallback=rate_clamped,
        )
        dwells.append(dwell)
        if update_belief:
            start = k * n_quadrature
            stop = start + n_quadrature
            midpoint_epochs = all_midpoints[start:stop]
            pointings = _dwell_pointings(
                dwell,
                local_env.nominal,
                midpoint_epochs,
                local_env.station,
            )
            particle_geometry = np.stack(
                [geometry_for(particle)[start:stop] for particle in belief.particles]
            )
            belief = _integrated_no_confirmation_update(
                belief,
                pointings,
                particle_geometry,
                cfg,
                subinterval_s,
                rng=rng,
            )
        if rate_clamped:
            # Explicit fallback: the slew-limited endpoint stands in for the
            # continuous mechanical path over this dwell.
            bore = (pointing.az, pointing.el)
        else:
            end_time = when + timedelta(seconds=local_env.dwell_time_s)
            end_pointing = action_to_pointing(
                action,
                local_env.nominal,
                end_time,
                local_env.station,
            )
            bore = (end_pointing.az, end_pointing.el)

    return CompiledSchedule(tuple(dwells), belief, nominal=local_env.nominal)


def truth_confirmation_hazards(
    schedule: CompiledSchedule,
    truth: OrbitParticle,
    station: Station,
    cfg: Config,
) -> np.ndarray:
    """Per-dwell terminal-confirmation probabilities for one truth orbit.

    The compatibility helper returns the sum of correct-acquisition and
    false-confirmation hazards. Use :func:`truth_event_hazards` when the two
    outcomes must remain separate.
    """
    true_hazards, false_hazards = truth_event_hazards(schedule, truth, station, cfg)
    return np.clip(true_hazards + false_hazards, 0.0, 1.0)


def truth_event_hazards(
    schedule: CompiledSchedule,
    truth: OrbitParticle,
    station: Station,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dwell-aggregated ``(correct, false)`` conditional incidences."""
    risks = _subinterval_risks(schedule, truth, station, cfg)
    if risks.p_true.size == 0:
        return np.empty(0), np.empty(0)
    true_by_subinterval = risks.p_true.reshape(-1, risks.n_per_dwell)
    false_by_subinterval = risks.p_false.reshape(-1, risks.n_per_dwell)
    terminal = np.clip(true_by_subinterval + false_by_subinterval, 0.0, 1.0)
    survival_before = np.concatenate(
        (
            np.ones((terminal.shape[0], 1)),
            np.cumprod(1.0 - terminal[:, :-1], axis=1),
        ),
        axis=1,
    )
    return (
        np.sum(survival_before * true_by_subinterval, axis=1),
        np.sum(survival_before * false_by_subinterval, axis=1),
    )


def _subinterval_risks(
    schedule: CompiledSchedule,
    truth: OrbitParticle,
    station: Station,
    cfg: Config,
) -> _SubintervalRisks:
    """Evaluate cause-specific conditional risks at every midpoint node."""
    n_quadrature, subinterval_s, offsets_s = _quadrature(cfg)
    if len(schedule.dwells) == 0:
        empty = np.empty(0)
        return _SubintervalRisks(empty, empty.copy(), empty.copy(), n_quadrature)

    midpoint_epochs = tuple(
        dwell.start_time + timedelta(seconds=float(offset_s))
        for dwell in schedule.dwells
        for offset_s in offsets_s
    )
    geometry = _particle_azel_range_many(truth, midpoint_epochs, station)
    pointings = np.concatenate(
        [
            _dwell_pointings(
                dwell,
                schedule.nominal,
                midpoint_epochs[k * n_quadrature : (k + 1) * n_quadrature],
                station,
            )
            for k, dwell in enumerate(schedule.dwells)
        ],
        axis=0,
    )
    dtheta = _angular_separation_many(pointings, geometry[:, :2])
    sub_cfg = replace(cfg.antenna, dwell_time_s=subinterval_s)
    p_true_ref, p_false_ref = reference_event_probabilities(
        dtheta,
        sub_cfg,
        geometry[:, 1],
        geometry[:, 2],
    )
    p_true, p_false = scale_competing_reference_probabilities(
        p_true_ref,
        p_false_ref,
        sub_cfg,
    )
    first_start = schedule.dwells[0].start_time
    event_time_s = np.asarray(
        [(when - first_start).total_seconds() for when in midpoint_epochs],
        dtype=float,
    )
    return _SubintervalRisks(
        np.clip(np.asarray(p_true, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(p_false, dtype=float), 0.0, 1.0),
        event_time_s,
        n_quadrature,
    )


def evaluate_schedule(
    schedule: CompiledSchedule,
    truth: OrbitParticle,
    station: Station,
    cfg: Config,
) -> ScheduleEvaluation:
    """Evaluate a compiled schedule without receiver Monte Carlo noise.

    Conditional on the truth and midpoint-piecewise-constant geometry,
    independent competing hazards give exact first-event incidence within the
    quadrature model. The restricted mean time is aligned to correct
    acquisition: both false confirmations and no-confirmation outcomes are
    charged to the full schedule duration.
    """
    risks = _subinterval_risks(schedule, truth, station, cfg)
    terminal_risks = np.clip(risks.p_true + risks.p_false, 0.0, 1.0)
    if terminal_risks.size == 0:
        empty = np.empty(0)
        return ScheduleEvaluation(
            p_acquire=0.0,
            p_false_confirmation=0.0,
            p_failure=1.0,
            first_confirmation_pmf=empty,
            first_true_acquisition_pmf=empty.copy(),
            first_false_confirmation_pmf=empty.copy(),
            conditional_mean_time_s=float("nan"),
            restricted_mean_time_s=0.0,
        )

    survival_before = np.concatenate(([1.0], np.cumprod(1.0 - terminal_risks[:-1])))
    true_subinterval_pmf = survival_before * risks.p_true
    false_subinterval_pmf = survival_before * risks.p_false
    with np.errstate(divide="ignore"):
        log_failure = float(np.sum(np.log1p(-terminal_risks)))
    p_failure = float(np.exp(log_failure))
    p_acquire = float(np.sum(true_subinterval_pmf))
    p_false_confirmation = float(np.sum(false_subinterval_pmf))
    # The hazard identity makes these three outcomes sum to one analytically.
    # Normalize the sub-ulp floating-point residual so released probabilities
    # remain in [0, 1] and their conservation invariant is exact.
    normalizer = p_acquire + p_false_confirmation + p_failure
    if normalizer <= 0.0 or not np.isfinite(normalizer):
        raise FloatingPointError("invalid competing-risk probability normalization")
    true_subinterval_pmf = true_subinterval_pmf / normalizer
    false_subinterval_pmf = false_subinterval_pmf / normalizer
    p_failure = p_failure / normalizer
    p_acquire = float(np.sum(true_subinterval_pmf))
    p_false_confirmation = float(np.sum(false_subinterval_pmf))
    true_pmf = np.sum(
        true_subinterval_pmf.reshape(-1, risks.n_per_dwell),
        axis=1,
    )
    false_pmf = np.sum(
        false_subinterval_pmf.reshape(-1, risks.n_per_dwell),
        axis=1,
    )
    terminal_pmf = true_pmf + false_pmf
    conditional_mean = (
        float(np.dot(risks.event_time_s, true_subinterval_pmf) / p_acquire)
        if p_acquire > 0.0
        else float("nan")
    )
    horizon_s = len(schedule.dwells) * float(cfg.antenna.dwell_time_s)
    restricted_mean = float(
        np.dot(risks.event_time_s, true_subinterval_pmf)
        + horizon_s * (p_false_confirmation + p_failure)
    )
    return ScheduleEvaluation(
        p_acquire=p_acquire,
        p_false_confirmation=p_false_confirmation,
        p_failure=p_failure,
        first_confirmation_pmf=terminal_pmf,
        first_true_acquisition_pmf=true_pmf,
        first_false_confirmation_pmf=false_pmf,
        conditional_mean_time_s=conditional_mean,
        restricted_mean_time_s=restricted_mean,
    )
