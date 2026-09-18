"""Evaluate acquisition policies across successive passes at one station.

An episode samples a truth orbit and a particle belief, then applies the
policy and observation updates through each nominal pass. Detection ends the
episode. Between passes, process noise is added with propagate_belief_to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np

from antenna_pomdp.baselines.scanner import (
    CenteredOneSidedTrackSweep,
    HierarchicalOneSidedTrackSweep,
    OneSidedTrackSweep,
    ProgressiveOneSidedTrackSweep,
    RandomScan,
    TrackLineSweep,
)
from antenna_pomdp.config import Config
from antenna_pomdp.eval.metrics import belief_entropy
from antenna_pomdp.models.particle_filter import (
    ParticleBelief,
    make_belief,
    propagate_belief_to,
    update,
)
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.leop import LeopPassWindow, enumerate_leop_passes
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import (
    OrbitParticle,
    sample_leop_particles,
    sample_leop_truth,
)
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
)
from antenna_pomdp.pomdp.pomcp import PomcpSolver


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


# A policy here is any callable: (belief, env, when, rng) -> TrackAction.
# Episode-level reset is the constructor's responsibility.
PolicyAct = Callable[[ParticleBelief, PointingEnv, datetime], TrackAction]
PolicyFactory = Callable[[Config, np.random.Generator], PolicyAct]


def make_pomcp_policy(cfg: Config) -> PolicyFactory:
    def factory(_cfg: Config, rng: np.random.Generator) -> PolicyAct:
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)
        solver = PomcpSolver(
            pomcp_cfg=_cfg.pomcp,
            antenna_cfg=_cfg.antenna,
            station=station,
            nominal=nominal,
        )

        def act(belief, env, when):
            idx, _ = solver.plan(belief, when, env, rng)
            return solver.actions[idx]

        return act

    return factory


def make_sweep_policy() -> PolicyFactory:
    def factory(_cfg: Config, _rng: np.random.Generator) -> PolicyAct:
        sweep = TrackLineSweep(_cfg.pomcp)

        def act(belief, env, when):
            return sweep.act()

        return act

    return factory


def make_one_sided_sweep_policy() -> PolicyFactory:
    """Prior-support sweep for a nonnegative launch-delay distribution."""

    def factory(_cfg: Config, _rng: np.random.Generator) -> PolicyAct:
        sweep = OneSidedTrackSweep(_cfg.pomcp)

        def act(_belief, _env, _when):
            return sweep.act()

        return act

    return factory


def make_centered_one_sided_sweep_policy() -> PolicyFactory:
    """Mid-support-first sweep for a uniform nonnegative launch-delay prior."""

    def factory(_cfg: Config, _rng: np.random.Generator) -> PolicyAct:
        sweep = CenteredOneSidedTrackSweep(_cfg.pomcp)

        def act(_belief, _env, _when):
            return sweep.act()

        return act

    return factory


def make_progressive_one_sided_sweep_policy() -> PolicyFactory:
    """Two-resolution one-sided schedule sized to the available dwell budget."""

    def factory(_cfg: Config, _rng: np.random.Generator) -> PolicyAct:
        sweep: ProgressiveOneSidedTrackSweep | None = None

        def act(_belief, env, _when):
            nonlocal sweep
            if sweep is None:
                sweep = ProgressiveOneSidedTrackSweep(
                    _cfg.pomcp,
                    n_steps=env.horizon,
                )
            return sweep.act()

        return act

    return factory


def make_hierarchical_one_sided_sweep_policy() -> PolicyFactory:
    """Three-stage horizon-aware one-sided refinement schedule."""

    def factory(_cfg: Config, _rng: np.random.Generator) -> PolicyAct:
        sweep: HierarchicalOneSidedTrackSweep | None = None

        def act(_belief, env, _when):
            nonlocal sweep
            if sweep is None:
                sweep = HierarchicalOneSidedTrackSweep(
                    _cfg.pomcp,
                    n_steps=env.horizon,
                )
            return sweep.act()

        return act

    return factory


def make_random_policy() -> PolicyFactory:
    def factory(_cfg: Config, rng: np.random.Generator) -> PolicyAct:
        scan = RandomScan(_cfg.pomcp, rng)

        def act(belief, env, when):
            return scan.act()

        return act

    return factory


def make_nominal_openloop_policy() -> PolicyFactory:
    """Open-loop baseline: always point at the nominal TLE track, never adapt.

    This is the strictest strawman — no observations, no belief, no sweep.
    Any closed-loop policy should beat this whenever the insertion error is
    larger than one beam-width.
    """
    _zero = TrackAction(along_track_s=0.0, cross_track_deg=0.0)

    def factory(_cfg: Config, _rng: np.random.Generator) -> PolicyAct:
        def act(belief, env, when):
            return _zero

        return act

    return factory


# ---------------------------------------------------------------------------
# Per-pass / per-episode records
# ---------------------------------------------------------------------------


@dataclass
class PassRecord:
    pass_number: int
    t_rise: datetime
    t_set: datetime
    max_elevation_deg: float
    n_steps: int
    detected: bool
    time_within_pass_s: float | None  # seconds after t_rise
    belief_entropy_at_start: float


@dataclass
class LeopEpisodeResult:
    """Outcome of one full LEOP episode (one ground-truth realisation)."""

    acquired: bool
    acquired_pass_number: int | None
    time_since_separation_s: float | None  # 0 = leop_start
    time_within_pass_s: float | None
    n_passes_attempted: int
    n_passes_available: int
    final_belief_entropy: float
    passes: list[PassRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-pass loop
# ---------------------------------------------------------------------------


def _run_one_pass(
    cfg: Config,
    nominal: SGP4Propagator,
    station: Station,
    truth: OrbitParticle,
    belief: ParticleBelief,
    pass_window: LeopPassWindow,
    policy_act: PolicyAct,
    rng: np.random.Generator,
) -> tuple[ParticleBelief, PassRecord]:
    """Run `policy_act` from pass rise to pass set. Returns updated belief and a record."""
    pass_duration = pass_window.duration_s
    env = PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=pass_window.t_rise,
        pass_duration_s=pass_duration,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    state = env.initial_state()
    entropy_at_start = belief_entropy(belief.weights)
    time_within_pass_s: float | None = None
    while not state.done:
        # Publish the live boresight so slew-aware controllers can plan reachable
        # pointings; the belief is updated with the *realised* (slew-limited)
        # pointing the antenna actually achieved this dwell.
        env.current_bore = None if state.bore_az is None else (state.bore_az, state.bore_el)
        action = policy_act(belief, env, state.when)
        next_state, detected, _, pointing = env.step(state, action, rng)
        belief = update(
            belief,
            pointing,
            detected,
            state.when,
            station,
            cfg.antenna,
            cfg.filter,
            rng,
        )
        if detected and time_within_pass_s is None:
            # Acquisition happened during the dwell that started at state.when.
            time_within_pass_s = (
                state.when - pass_window.t_rise
            ).total_seconds() + cfg.antenna.dwell_time_s
        state = next_state

    rec = PassRecord(
        pass_number=pass_window.pass_number,
        t_rise=pass_window.t_rise,
        t_set=pass_window.t_set,
        max_elevation_deg=pass_window.max_elevation_deg,
        n_steps=state.step_idx,
        detected=state.acquired,
        time_within_pass_s=time_within_pass_s,
        belief_entropy_at_start=entropy_at_start,
    )
    return belief, rec


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def run_leop_episode(
    cfg: Config,
    policy_factory: PolicyFactory,
    rng: np.random.Generator,
    *,
    leop_start: datetime | None = None,
) -> LeopEpisodeResult:
    """Run one multi-pass LEOP episode end-to-end."""
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    if leop_start is None:
        leop_start = nominal.epoch

    truth = sample_leop_truth(cfg.orbit, cfg.leop, rng)
    particles = sample_leop_particles(cfg.orbit, cfg.leop, n=cfg.filter.n_particles, rng=rng)
    belief = make_belief(particles)

    passes = enumerate_leop_passes(
        nominal,
        station,
        leop_start,
        min_elevation_deg=cfg.station.min_elevation_deg,
        leop_window_s=cfg.leop.leop_window_s,
    )
    policy_act = policy_factory(cfg, rng)

    records: list[PassRecord] = []
    prev_close: datetime | None = None
    for pw in passes:
        # Propagate belief from end of previous pass (or LEOP start) to this rise.
        ref = prev_close if prev_close is not None else leop_start
        if pw.t_rise > ref:
            belief = propagate_belief_to(
                belief,
                ref,
                pw.t_rise,
                drift_km_per_min=cfg.leop.inter_pass_drift_km_per_min,
                orbital_speed_km_s=cfg.leop.orbital_speed_km_s,
                rng=rng,
            )
        belief, rec = _run_one_pass(cfg, nominal, station, truth, belief, pw, policy_act, rng)
        records.append(rec)
        prev_close = pw.t_set
        if rec.detected:
            tsa = (rec.t_rise - leop_start).total_seconds() + (rec.time_within_pass_s or 0.0)
            return LeopEpisodeResult(
                acquired=True,
                acquired_pass_number=rec.pass_number,
                time_since_separation_s=tsa,
                time_within_pass_s=rec.time_within_pass_s,
                n_passes_attempted=len(records),
                n_passes_available=len(passes),
                final_belief_entropy=belief_entropy(belief.weights),
                passes=records,
            )

    return LeopEpisodeResult(
        acquired=False,
        acquired_pass_number=None,
        time_since_separation_s=None,
        time_within_pass_s=None,
        n_passes_attempted=len(records),
        n_passes_available=len(passes),
        final_belief_entropy=belief_entropy(belief.weights),
        passes=records,
    )


def run_leop_monte_carlo(
    cfg: Config,
    policy_factory: PolicyFactory,
    n_episodes: int,
    seed: int,
) -> list[LeopEpisodeResult]:
    """Convenience helper; not used by the parallel experiment driver."""
    master = np.random.default_rng(seed)
    out: list[LeopEpisodeResult] = []
    for _ in range(n_episodes):
        trial_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        out.append(run_leop_episode(cfg, policy_factory, trial_rng))
    return out


if __name__ == "__main__":
    from dataclasses import replace

    from antenna_pomdp.config import default_config

    base = default_config()
    # Tiny config for the sanity check.
    cfg = replace(
        base,
        antenna=replace(base.antenna, fwhm_deg=10.0),
        filter=replace(base.filter, n_particles=64),
        pomcp=replace(base.pomcp, n_rollouts=20, max_depth=3),
        leop=replace(base.leop, leop_window_s=6 * 3600.0),
    )
    rng = np.random.default_rng(0)
    ep = run_leop_episode(cfg, make_sweep_policy(), rng)
    print(
        f"[leop_runner] sweep: acquired={ep.acquired}, "
        f"pass={ep.acquired_pass_number}, n_passes_avail={ep.n_passes_available}, "
        f"records={len(ep.passes)}"
    )
    assert ep.n_passes_available >= 0
    print("[leop_runner] PASS")
