"""Monte Carlo runner that pits a policy against the POMDP environment.

Three policies are supported via a uniform interface (callable returning
`TrackAction` given the current belief / step):
    - `pomcp_policy`    : re-plans every step using a fresh POMCP tree.
    - `sweep_policy`    : open-loop track-line sweep.
    - `random_policy`   : uniform random over the action grid.

The runner is deliberately stateless: it takes a `Config`, an RNG, and a
policy-constructor; it returns a list of `TrialResult`s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from antenna_pomdp.baselines.scanner import RandomScan, TrackLineSweep
from antenna_pomdp.config import Config, default_config
from antenna_pomdp.eval.metrics import TrialResult, belief_entropy, summarize
from antenna_pomdp.models.particle_filter import (
    ParticleBelief,
    make_belief,
    update,
)
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import sample_particles, sample_truth
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
)
from antenna_pomdp.pomdp.pomcp import PomcpSolver


PolicyFactory = Callable[[Config, np.random.Generator], "Policy"]


class Policy:
    """Minimal duck-typed protocol — anything with `act(belief, env, when)` works."""

    def act(
        self,
        belief: ParticleBelief,
        env: PointingEnv,
        when,
    ) -> TrackAction:  # pragma: no cover - interface only
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Policy adapters
# ---------------------------------------------------------------------------


@dataclass
class PomcpPolicy:
    solver: PomcpSolver
    rng: np.random.Generator

    def act(self, belief, env, when):
        idx, _ = self.solver.plan(belief, when, env, self.rng)
        return self.solver.actions[idx]


@dataclass
class SweepPolicy:
    sweep: TrackLineSweep

    def act(self, belief, env, when):
        return self.sweep.act()


@dataclass
class RandomPolicy:
    scan: RandomScan

    def act(self, belief, env, when):
        return self.scan.act()


def make_pomcp_factory(cfg: Config) -> PolicyFactory:
    def factory(_cfg: Config, rng: np.random.Generator) -> PomcpPolicy:
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)
        solver = PomcpSolver(
            pomcp_cfg=_cfg.pomcp,
            antenna_cfg=_cfg.antenna,
            station=station,
            nominal=nominal,
        )
        return PomcpPolicy(solver=solver, rng=rng)

    return factory


def make_sweep_factory() -> PolicyFactory:
    def factory(_cfg: Config, _rng: np.random.Generator) -> SweepPolicy:
        return SweepPolicy(TrackLineSweep(_cfg.pomcp))

    return factory


def make_random_factory() -> PolicyFactory:
    def factory(_cfg: Config, rng: np.random.Generator) -> RandomPolicy:
        return RandomPolicy(RandomScan(_cfg.pomcp, rng))

    return factory


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_trial(
    cfg: Config,
    policy_factory: PolicyFactory,
    rng: np.random.Generator,
) -> TrialResult:
    """One Monte Carlo trial: sample truth + prior, run the pass, score outcome."""
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    truth = sample_truth(cfg.orbit, rng)
    particles = sample_particles(cfg.orbit, n=cfg.filter.n_particles, rng=rng)
    belief = make_belief(particles)

    pass_start = nominal.epoch
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

    policy = policy_factory(cfg, rng)
    state = env.initial_state()
    time_to_acquire: float | None = None

    while not state.done:
        action = policy.act(belief, env, state.when)
        pointing = action_to_pointing(action, nominal, state.when, station)
        new_state, detected, _, _ = env.step(state, action, rng)
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
        if detected and time_to_acquire is None:
            time_to_acquire = (state.step_idx + 1) * cfg.antenna.dwell_time_s
        state = new_state

    return TrialResult(
        acquired=state.acquired,
        time_to_acquire_s=time_to_acquire,
        final_belief_entropy=belief_entropy(belief.weights),
        n_steps=state.step_idx,
    )


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------


def run_monte_carlo(
    cfg: Config,
    policy_factory: PolicyFactory,
    n_trials: int | None = None,
    seed: int | None = None,
) -> list[TrialResult]:
    """Run `n_trials` independent trials. Defaults pulled from `cfg.eval`."""
    n_trials = n_trials if n_trials is not None else cfg.eval.n_trials
    seed = seed if seed is not None else cfg.eval.seed
    master = np.random.default_rng(seed)
    out: list[TrialResult] = []
    for _ in range(n_trials):
        trial_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        out.append(run_trial(cfg, policy_factory, trial_rng))
    return out


if __name__ == "__main__":
    cfg = default_config()
    # Use a fast config for the sanity check.
    fast_cfg = Config(
        orbit=cfg.orbit,
        station=cfg.station,
        antenna=cfg.antenna,
        filter=type(cfg.filter)(**{**cfg.filter.__dict__, "n_particles": 100}),
        pomcp=type(cfg.pomcp)(**{**cfg.pomcp.__dict__, "n_rollouts": 30, "max_depth": 4}),
        eval=type(cfg.eval)(n_trials=3, seed=0),
    )

    sweep_results = run_monte_carlo(fast_cfg, make_sweep_factory())
    random_results = run_monte_carlo(fast_cfg, make_random_factory())
    pomcp_results = run_monte_carlo(fast_cfg, make_pomcp_factory(fast_cfg))

    print("[runner] sweep :", summarize(sweep_results))
    print("[runner] random:", summarize(random_results))
    print("[runner] pomcp :", summarize(pomcp_results))
    print("[runner] PASS")
