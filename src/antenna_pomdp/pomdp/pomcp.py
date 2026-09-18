"""POMCP search with UCB1 action selection (Silver and Veness, 2010).

Each simulation samples a root particle, descends the tree, expands a leaf,
and backs up the discounted rollout return. Nodes store particle samples and
per-action visit counts, values, and binary-observation children.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from antenna_pomdp.config import AntennaConfig, PomcpConfig
from antenna_pomdp.models.particle_filter import ParticleBelief
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
    build_action_grid,
    simulate_observation_for_particle,
)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """One belief node in the POMCP tree."""

    depth: int
    visits: int = 0
    # action statistics
    n_a: dict[int, int] = field(default_factory=dict)
    q_a: dict[int, float] = field(default_factory=dict)
    # children indexed by (action_idx, obs_int)
    children: dict[tuple[int, int], "_Node"] = field(default_factory=dict)
    # particles attached during search (state samples that reached this node)
    particles: list[OrbitParticle] = field(default_factory=list)

    def is_leaf(self, n_actions: int) -> bool:
        return len(self.n_a) < n_actions


def _ucb_action(node: _Node, actions: list[TrackAction], c: float) -> int:
    """Return action index maximising UCB1 = Q + c * sqrt(log N / n)."""
    log_total = math.log(max(1, node.visits))
    best_idx = -1
    best_val = -math.inf
    for i in range(len(actions)):
        n = node.n_a.get(i, 0)
        if n == 0:
            return i  # always try unexplored first
        q = node.q_a[i]
        val = q + c * math.sqrt(log_total / n)
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass
class PomcpSolver:
    """Single-call POMCP planner. Holds tree root + diagnostics."""

    pomcp_cfg: PomcpConfig
    antenna_cfg: AntennaConfig
    station: Station
    nominal: SGP4Propagator

    def __post_init__(self) -> None:
        self.actions: list[TrackAction] = build_action_grid(self.pomcp_cfg)
        self.action_visits: np.ndarray = np.zeros(len(self.actions), dtype=int)
        # Root-node action visit counts from the most recent `plan()` call,
        # normalised so they sum to 1 across the full action grid (0 for any
        # action never tried). Filled in by `plan()`; empty before the first.
        self._last_root_visits: dict[int, float] = {i: 0.0 for i in range(len(self.actions))}

    @property
    def root_visit_counts(self) -> dict[int, float]:
        """Per-action visit distribution from the most recent `plan()` call.

        Returns a dict mapping every action index in the grid to its share of
        the root-node UCB visits (sums to 1.0). Actions that were never tried
        at the root are present with value 0.0. Empty (all zeros) before the
        first call to `plan()`.
        """
        return dict(self._last_root_visits)

    def plan(
        self,
        belief: ParticleBelief,
        current_time: datetime,
        env: PointingEnv,
        rng: np.random.Generator,
    ) -> tuple[int, _Node]:
        """Run rollouts, return (best_action_index, root_node).

        The returned root is useful for visualisation (per-action visit counts).
        `env` provides the dwell time / horizon; it must be the same env that
        will receive the chosen action.
        """
        root = _Node(depth=0)
        # Seed root particles from the current belief by weighted draws so that
        # rollouts represent the posterior, not the uniform prior.
        n_seed = min(self.pomcp_cfg.n_rollouts, max(50, belief.n))
        idx = rng.choice(belief.n, size=n_seed, replace=True, p=belief.weights)
        root.particles = [belief.particles[i] for i in idx]

        for _ in range(self.pomcp_cfg.n_rollouts):
            sampled = root.particles[rng.integers(0, len(root.particles))]
            self._simulate(root, sampled, current_time, env, rng)

        # Record the root-node visit distribution before returning, so that
        # callers can inspect it via the `root_visit_counts` property even if
        # they don't keep the returned `root` object.
        total_visits = float(sum(root.n_a.values()))
        if total_visits > 0.0:
            self._last_root_visits = {
                i: float(root.n_a.get(i, 0)) / total_visits for i in range(len(self.actions))
            }
        else:
            self._last_root_visits = {i: 0.0 for i in range(len(self.actions))}

        # Best action = highest Q. Fall back to most-visited if Q is empty.
        if not root.q_a:
            return 0, root
        best = max(root.q_a.items(), key=lambda kv: kv[1])[0]
        for i, n in root.n_a.items():
            self.action_visits[i] += n
        return best, root

    # ------------------------------------------------------------------
    def _simulate(
        self,
        node: _Node,
        particle: OrbitParticle,
        when: datetime,
        env: PointingEnv,
        rng: np.random.Generator,
    ) -> float:
        if node.depth >= self.pomcp_cfg.max_depth:
            return 0.0

        # Expand if leaf.
        if node.is_leaf(len(self.actions)):
            a_idx = len(node.n_a)  # next unexplored action
            r, next_time, child, det = self._step_in_tree(
                node, particle, when, a_idx, env, rng, create_child=True
            )
            future = self._rollout(particle, next_time, env, rng, depth=node.depth + 1)
            ret = r + self.pomcp_cfg.gamma * (0.0 if det else future)
            self._update(node, a_idx, ret)
            return ret

        a_idx = _ucb_action(node, self.actions, self.pomcp_cfg.ucb_c)
        r, next_time, child, det = self._step_in_tree(
            node, particle, when, a_idx, env, rng, create_child=True
        )
        if det:
            future = 0.0  # acquisition ends the episode
        else:
            future = self._simulate(child, particle, next_time, env, rng)
        ret = r + self.pomcp_cfg.gamma * future
        self._update(node, a_idx, ret)
        return ret

    # ------------------------------------------------------------------
    def _step_in_tree(
        self,
        node: _Node,
        particle: OrbitParticle,
        when: datetime,
        a_idx: int,
        env: PointingEnv,
        rng: np.random.Generator,
        create_child: bool,
    ) -> tuple[float, datetime, _Node, bool]:
        from datetime import timedelta

        action = self.actions[a_idx]
        pointing = action_to_pointing(action, self.nominal, when, self.station)
        detected = simulate_observation_for_particle(
            particle, when, pointing, self.station, self.antenna_cfg, rng
        )
        reward = self.pomcp_cfg.reward_detect if detected else self.pomcp_cfg.reward_step
        next_time = when + timedelta(seconds=env.dwell_time_s)
        key = (a_idx, int(detected))
        child = node.children.get(key)
        if child is None and create_child:
            child = _Node(depth=node.depth + 1)
            node.children[key] = child
        return reward, next_time, child or _Node(depth=node.depth + 1), detected

    # ------------------------------------------------------------------
    def _rollout(
        self,
        particle: OrbitParticle,
        when: datetime,
        env: PointingEnv,
        rng: np.random.Generator,
        depth: int,
    ) -> float:
        """Random-policy rollout. Returns discounted return from `when`."""
        from datetime import timedelta

        ret = 0.0
        discount = 1.0
        t = when
        d = depth
        while d < self.pomcp_cfg.max_depth:
            a = self.actions[rng.integers(0, len(self.actions))]
            pointing = action_to_pointing(a, self.nominal, t, self.station)
            detected = simulate_observation_for_particle(
                particle, t, pointing, self.station, self.antenna_cfg, rng
            )
            r = self.pomcp_cfg.reward_detect if detected else self.pomcp_cfg.reward_step
            ret += discount * r
            if detected:
                break
            discount *= self.pomcp_cfg.gamma
            t = t + timedelta(seconds=env.dwell_time_s)
            d += 1
        return ret

    # ------------------------------------------------------------------
    def _update(self, node: _Node, a_idx: int, ret: float) -> None:
        node.visits += 1
        n = node.n_a.get(a_idx, 0) + 1
        q = node.q_a.get(a_idx, 0.0)
        node.n_a[a_idx] = n
        node.q_a[a_idx] = q + (ret - q) / n


if __name__ == "__main__":
    from datetime import timedelta

    from antenna_pomdp.config import default_config
    from antenna_pomdp.models.particle_filter import make_belief
    from antenna_pomdp.orbit.sampler import sample_particles, sample_truth

    cfg = default_config()
    rng = np.random.default_rng(0)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_particles(cfg.orbit, n=64, rng=rng)
    belief = make_belief(particles)
    truth = sample_truth(cfg.orbit, rng)

    env = PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=nominal.epoch + timedelta(minutes=5),
        pass_duration_s=60.0,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )

    # Use a small number of rollouts so the sanity check stays fast.
    small_cfg = type(cfg.pomcp)(**{**cfg.pomcp.__dict__, "n_rollouts": 50, "max_depth": 5})
    solver = PomcpSolver(
        pomcp_cfg=small_cfg,
        antenna_cfg=cfg.antenna,
        station=station,
        nominal=nominal,
    )
    best, root = solver.plan(belief, env.pass_start, env, rng)
    print(f"[pomcp] root visits={root.visits}, |Q|={len(root.q_a)}, best action idx={best}")
    assert root.visits == small_cfg.n_rollouts
    print("[pomcp] PASS")
