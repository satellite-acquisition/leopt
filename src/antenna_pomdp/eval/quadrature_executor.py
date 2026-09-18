"""Cause-valued online execution under the compiled evaluator's time model.

This module is deliberately separate from :class:`~antenna_pomdp.pomdp.environment.PointingEnv`
and :mod:`antenna_pomdp.eval.leop_runner`. Those legacy interfaces take one
Bernoulli sample at the *start* of a dwell and expose it as ``detected: bool``;
that Boolean cannot distinguish a correct spacecraft acquisition from an
accepted false trigger.

The executor below instead uses the primary schedule evaluator's contract:

* an unclamped track-relative command moves with the nominal trajectory;
* each dwell is integrated by midpoint subintervals no longer than
  ``temporal_quadrature_step_s``;
* correct acquisition and false confirmation are mutually exclusive,
  cause-specific first events; and
* the continuing ``NONE`` branch applies the exact same integrated
  no-confirmation likelihood as policy compilation.

The analytic evaluator remains preferable for Monte Carlo studies because it
integrates receiver randomness exactly. This sampled executor exists for
online-contract tests, stochastic simulation, and adapters that need one
realized cause rather than an expectation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from antenna_pomdp.config import Config
from antenna_pomdp.eval.compiled_schedule import (
    CompiledDwell,
    CompiledSchedule,
    evaluate_schedule,
    update_no_confirmation_belief,
)
from antenna_pomdp.eval.leop_runner import PolicyFactory
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


class QuadratureOutcome(Enum):
    """Mutually exclusive outcome of one integrated dwell."""

    NONE = "no_confirmation"
    CORRECT = "correct_acquisition"
    FALSE = "false_confirmation"


@dataclass(frozen=True)
class DwellCauseProbabilities:
    """First-event cause probabilities conditional on entering one dwell."""

    p_correct: float
    p_false: float
    p_none: float

    def __post_init__(self) -> None:
        values = np.asarray((self.p_correct, self.p_false, self.p_none), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("dwell cause probabilities must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("dwell cause probabilities must lie in [0, 1]")
        if not np.isclose(float(np.sum(values)), 1.0, rtol=0.0, atol=5e-13):
            raise ValueError("dwell cause probabilities must sum to one")

    def as_array(self) -> np.ndarray:
        """Return probabilities ordered as ``CORRECT, FALSE, NONE``."""
        return np.asarray((self.p_correct, self.p_false, self.p_none), dtype=float)


@dataclass(frozen=True)
class QuadratureExecutionState:
    """Mutable-executor state represented as an immutable snapshot."""

    when: datetime
    step_idx: int
    done: bool
    last_outcome: QuadratureOutcome | None
    bore_az: float | None
    bore_el: float | None

    @property
    def acquired(self) -> bool:
        """Whether the terminal cause was a correct spacecraft acquisition."""
        return self.done and self.last_outcome is QuadratureOutcome.CORRECT

    @property
    def false_confirmed(self) -> bool:
        """Whether the terminal cause was an accepted false trigger."""
        return self.done and self.last_outcome is QuadratureOutcome.FALSE


@dataclass(frozen=True)
class QuadratureStepResult:
    """One commanded dwell, its conditional cause law, and sampled outcome."""

    dwell: CompiledDwell
    probabilities: DwellCauseProbabilities
    outcome: QuadratureOutcome
    next_state: QuadratureExecutionState


@dataclass(frozen=True)
class QuadratureRunResult:
    """Completed sampled pass plus its latest continuing-branch belief.

    When a terminal cause occurs, no terminal-conditioned posterior is needed;
    ``final_belief`` is the belief that entered that final dwell.
    """

    state: QuadratureExecutionState
    steps: tuple[QuadratureStepResult, ...]
    final_belief: ParticleBelief


def dwell_cause_probabilities(
    dwell: CompiledDwell,
    truth: OrbitParticle,
    nominal: SGP4Propagator | None,
    station: Station,
    cfg: Config,
) -> DwellCauseProbabilities:
    """Evaluate the cause law for one dwell without receiver sampling.

    A one-dwell :class:`CompiledSchedule` delegates all geometry, temporal
    quadrature, cause-risk rescaling, and rate-clamped fallback behavior to
    :func:`evaluate_schedule`. The single-particle belief merely satisfies the
    schedule record's provenance field; evaluator probabilities do not inspect
    it.
    """
    placeholder_belief = ParticleBelief(
        particles=[truth],
        weights=np.ones(1, dtype=float),
    )
    evaluation = evaluate_schedule(
        CompiledSchedule(
            dwells=(dwell,),
            final_no_confirmation_belief=placeholder_belief,
            nominal=nominal,
        ),
        truth,
        station,
        cfg,
    )
    probabilities = np.asarray(
        (
            evaluation.p_acquire,
            evaluation.p_false_confirmation,
            evaluation.p_failure,
        ),
        dtype=float,
    )
    # ``evaluate_schedule`` already normalizes the sub-ulp hazard residual.
    # Clip defensively at this API boundary, then normalize once more so a
    # categorical draw has an exact unit total even under future backends.
    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = float(np.sum(probabilities))
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("invalid dwell cause probability normalization")
    probabilities /= total
    return DwellCauseProbabilities(
        p_correct=float(probabilities[0]),
        p_false=float(probabilities[1]),
        p_none=float(probabilities[2]),
    )


def sample_dwell_outcome(
    probabilities: DwellCauseProbabilities,
    rng: np.random.Generator,
) -> QuadratureOutcome:
    """Sample one mutually exclusive cause from a dwell's first-event law."""
    draw = float(rng.random())
    if draw < probabilities.p_correct:
        return QuadratureOutcome.CORRECT
    if draw < probabilities.p_correct + probabilities.p_false:
        return QuadratureOutcome.FALSE
    return QuadratureOutcome.NONE


class QuadraturePolicyExecutor:
    """Execute a binary-terminal policy under cause-specific quadrature.

    ``policy_rng`` drives policy construction and optional particle-filter
    resampling. ``receiver_rng`` drives only the cause draw. Keeping them
    separate makes paired experiments reproducible and lets forced
    no-confirmation replay match :func:`compile_no_confirmation_schedule`
    exactly.
    """

    def __init__(
        self,
        cfg: Config,
        initial_belief: ParticleBelief,
        env: PointingEnv,
        policy_factory: PolicyFactory,
        policy_rng: np.random.Generator,
        receiver_rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self._env = replace(env, current_bore=env.current_bore)
        self._policy_rng = policy_rng
        self._receiver_rng = receiver_rng
        self._policy = policy_factory(cfg, policy_rng)
        self._belief = ParticleBelief(
            particles=list(initial_belief.particles),
            weights=np.asarray(initial_belief.weights, dtype=float).copy(),
        )
        bore = env.current_bore
        self._state = QuadratureExecutionState(
            when=env.pass_start,
            step_idx=0,
            done=False,
            last_outcome=None,
            bore_az=None if bore is None else float(bore[0]),
            bore_el=None if bore is None else float(bore[1]),
        )
        self._steps: list[QuadratureStepResult] = []

    @property
    def belief(self) -> ParticleBelief:
        """Current continuing-branch belief (returned as a defensive copy)."""
        return ParticleBelief(
            particles=list(self._belief.particles),
            weights=np.asarray(self._belief.weights, dtype=float).copy(),
        )

    @property
    def state(self) -> QuadratureExecutionState:
        return self._state

    @property
    def steps(self) -> tuple[QuadratureStepResult, ...]:
        return tuple(self._steps)

    def step(
        self,
        *,
        forced_outcome: QuadratureOutcome | None = None,
    ) -> QuadratureStepResult:
        """Execute one dwell.

        ``forced_outcome`` is a deterministic replay/test hook. It consumes no
        receiver random number. In particular, forcing ``NONE`` exposes the
        unique continuing history used by policy compilation.
        """
        if self._state.done:
            raise RuntimeError("cannot step a completed quadrature execution")
        if forced_outcome is not None and not isinstance(forced_outcome, QuadratureOutcome):
            raise TypeError("forced_outcome must be a QuadratureOutcome or None")

        bore = self._current_bore()
        self._env.current_bore = bore
        action = self._policy(self._belief, self._env, self._state.when)
        dwell = self._make_dwell(action, bore)
        probabilities = dwell_cause_probabilities(
            dwell,
            self._env.truth,
            self._env.nominal,
            self._env.station,
            self.cfg,
        )
        outcome = (
            sample_dwell_outcome(probabilities, self._receiver_rng)
            if forced_outcome is None
            else forced_outcome
        )

        if outcome is QuadratureOutcome.NONE:
            self._belief = update_no_confirmation_belief(
                self._belief,
                dwell,
                self._env.nominal,
                self._env.station,
                self.cfg,
                self._policy_rng,
            )

        next_bore = self._next_bore(dwell)
        next_step_idx = self._state.step_idx + 1
        self._state = QuadratureExecutionState(
            when=self._state.when + timedelta(seconds=self._env.dwell_time_s),
            step_idx=next_step_idx,
            done=outcome is not QuadratureOutcome.NONE or next_step_idx >= self._env.horizon,
            last_outcome=outcome,
            bore_az=next_bore[0],
            bore_el=next_bore[1],
        )
        self._env.current_bore = next_bore
        result = QuadratureStepResult(
            dwell=dwell,
            probabilities=probabilities,
            outcome=outcome,
            next_state=self._state,
        )
        self._steps.append(result)
        return result

    def run(self, *, force_no_confirmation: bool = False) -> QuadratureRunResult:
        """Run until a first event or horizon exhaustion."""
        while not self._state.done:
            self.step(forced_outcome=(QuadratureOutcome.NONE if force_no_confirmation else None))
        return QuadratureRunResult(
            state=self._state,
            steps=tuple(self._steps),
            final_belief=self.belief,
        )

    def compiled_no_confirmation_prefix(self) -> CompiledSchedule:
        """Return the executed prefix using the compiler's schedule record.

        This is defined only when every executed dwell observed ``NONE``;
        terminal branches do not have a no-confirmation posterior for their
        final dwell.
        """
        if any(step.outcome is not QuadratureOutcome.NONE for step in self._steps):
            raise RuntimeError("terminal execution is not a no-confirmation prefix")
        return CompiledSchedule(
            dwells=tuple(step.dwell for step in self._steps),
            final_no_confirmation_belief=self.belief,
            nominal=self._env.nominal,
        )

    def _current_bore(self) -> tuple[float, float] | None:
        if self._state.bore_az is None or self._state.bore_el is None:
            return None
        return self._state.bore_az, self._state.bore_el

    def _make_dwell(
        self,
        action: TrackAction,
        bore: tuple[float, float] | None,
    ) -> CompiledDwell:
        commanded = action_to_pointing(
            action,
            self._env.nominal,
            self._state.when,
            self._env.station,
        )
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
            rate_clamped = separation > self._env.max_slew_rad() + 1e-12
            pointing = slew_limited_pointing(
                bore[0],
                bore[1],
                commanded,
                self._env.max_slew_rad(),
            )
        return CompiledDwell(
            start_time=self._state.when,
            action=action,
            pointing=pointing,
            uses_rate_clamped_endpoint_fallback=rate_clamped,
        )

    def _next_bore(self, dwell: CompiledDwell) -> tuple[float, float]:
        if dwell.uses_rate_clamped_endpoint_fallback:
            return dwell.pointing.az, dwell.pointing.el
        endpoint = action_to_pointing(
            dwell.action,
            self._env.nominal,
            dwell.start_time + timedelta(seconds=self._env.dwell_time_s),
            self._env.station,
        )
        return endpoint.az, endpoint.el
