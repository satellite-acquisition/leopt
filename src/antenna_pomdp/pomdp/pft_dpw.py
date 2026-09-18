"""Particle Filter Tree search with continuous-action progressive widening.

Based on Sunberg and Kochenderfer, "Online algorithms for POMDPs with
continuous state, action, and observation spaces," ICAPS 2018.

Each node stores a weighted particle belief and samples continuous TrackAction
candidates, bounded by |A(h)| <= k_a * N(h)^alpha_a. Binary observations key the
child beliefs directly; k_obs and alpha_obs are unused in this specialization.
The solver uses the same plan interface as PomcpSolver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from antenna_pomdp.config import AntennaConfig, Config, SolverConfig
from antenna_pomdp.models.observation import (
    delta_theta_from_azel,
    detection_probability,
    observation_likelihood,
)
from antenna_pomdp.models.particle_filter import (
    ParticleBelief,
    belief_azel_range,
)
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    Pointing,
    TrackAction,
    action_to_pointing,
    simulate_observation_for_particle,
    slew_limited_pointing,
)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class _BeliefNode:
    """One belief node in the PFT-DPW tree.

    Holds an explicit particle belief plus per-action statistics. Action
    children are keyed by the index of the action in `actions`; each action then
    has up to two observation children (detect / no-detect).
    """

    depth: int
    belief: ParticleBelief
    when: datetime
    visits: int = 0
    actions: list[TrackAction] = field(default_factory=list)
    n_a: dict[int, int] = field(default_factory=dict)
    q_a: dict[int, float] = field(default_factory=dict)
    # children keyed by (action_idx, obs_bit) -> child belief node
    children: dict[tuple[int, int], "_BeliefNode"] = field(default_factory=dict)
    # realised boresight (az, el) reached at this node; None => unconstrained
    bore: tuple[float, float] | None = None
    # cached belief-informed quantities (computed lazily, once per node)
    azelr: np.ndarray | None = None  # (m, 3) az/el/range of the node belief at `when`
    greedy: TrackAction | None = None  # myopic max-E[P_D] action over this belief
    greedy_pd: float = 0.0  # its expected one-step detection probability


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass
class PftDpwSolver:
    """Single-call PFT-DPW planner. Mirrors `PomcpSolver`'s interface."""

    solver_cfg: SolverConfig
    antenna_cfg: AntennaConfig
    station: Station
    nominal: SGP4Propagator

    def plan(
        self,
        belief: ParticleBelief,
        when: datetime,
        env: PointingEnv,
        rng: np.random.Generator,
    ) -> TrackAction:
        """Run `n_iterations` belief-MCTS simulations; return the best action.

        The root keeps the *full* belief (child nodes subsample to
        `n_belief_particles`) so the myopic greedy proposal and the final
        recommendation are computed at full belief resolution -- essential at
        narrow beams. The recommendation is *anchored* at the myopic greedy
        action (always root action 0): the tree's best-Q action is returned only
        when its estimated lookahead value exceeds the greedy action's by the
        configured guard margin; otherwise the greedy action is used. This is a
        variance-control heuristic, not a performance guarantee. It permits a
        different action when the implemented reward has sequential structure
        (e.g., slew costs).
        """
        # Slew budget per dwell (rad); huge default => unconstrained re-point.
        self._max_move = float(np.deg2rad(self.antenna_cfg.max_slew_rate_deg_s) * env.dwell_time_s)
        root = _BeliefNode(depth=0, belief=belief, when=when, bore=env.current_bore)
        self._ensure_greedy(root)  # greedy = root action 0, computed on full belief

        # With unconstrained re-pointing, return the myopic anchor.
        # This shortcut does not guarantee finite-horizon optimality.
        if self._max_move >= math.pi:
            return root.greedy

        for _ in range(self.solver_cfg.n_iterations):
            self._simulate(root, env, rng)

        if not root.q_a:
            return root.greedy if root.greedy is not None else TrackAction(0.0, 0.0)

        best_idx = max(root.q_a.items(), key=lambda kv: kv[1])[0]
        greedy_q = root.q_a.get(0, -math.inf)
        # Deviate from the myopic anchor only when the alternative is BOTH
        # well-visited (so its Q is trustworthy, not a low-sample fluke) AND
        # confidently better. Otherwise return the anchor. This is what keeps the
        # planner from chasing Q noise in the large unconstrained action space.
        if best_idx != 0:
            n_best = root.n_a.get(best_idx, 0)
            margin = root.q_a[best_idx] - greedy_q
            if (
                n_best < self.solver_cfg.deviate_min_visits
                or margin < self.solver_cfg.deviate_margin
            ):
                return root.actions[0]
        return root.actions[best_idx]

    # ------------------------------------------------------------------
    # Belief utilities
    # ------------------------------------------------------------------
    def _subsample(self, belief: ParticleBelief, rng: np.random.Generator) -> ParticleBelief:
        """Draw `n_belief_particles` weighted particles to form a node belief."""
        m = self.solver_cfg.n_belief_particles
        idx = rng.choice(belief.n, size=m, replace=True, p=belief.weights)
        parts = [belief.particles[i] for i in idx]
        return ParticleBelief(particles=parts, weights=np.full(m, 1.0 / m))

    def _sample_action(self, rng: np.random.Generator) -> TrackAction:
        """Uniform continuous action in the configured (Δτ, Δθ) box."""
        tau = float(
            rng.uniform(
                -self.solver_cfg.along_track_offset_max_s,
                self.solver_cfg.along_track_offset_max_s,
            )
        )
        theta = float(
            rng.uniform(
                -self.solver_cfg.cross_track_offset_max_deg,
                self.solver_cfg.cross_track_offset_max_deg,
            )
        )
        return TrackAction(along_track_s=tau, cross_track_deg=theta)

    # ------------------------------------------------------------------
    # Belief-informed search: greedy proposals + heuristic leaf value
    # ------------------------------------------------------------------
    def _node_azelr(self, node: _BeliefNode) -> np.ndarray:
        """Lazily cache the (m,3) az/el/range of the node belief at node.when."""
        if node.azelr is None:
            node.azelr = belief_azel_range(node.belief, node.when, self.station)
        return node.azelr

    def _realized_pointing(
        self, action: TrackAction, when: datetime, bore: tuple[float, float] | None
    ) -> Pointing:
        """Commanded pointing clamped to the slew-reachable set from `bore`."""
        p = action_to_pointing(action, self.nominal, when, self.station)
        if bore is None or self._max_move >= math.pi:
            return p
        return slew_limited_pointing(bore[0], bore[1], p, self._max_move)

    @staticmethod
    def _entropy(w: np.ndarray) -> float:
        w = w[w > 0.0]
        return float(-np.sum(w * np.log(w))) if w.size else 0.0

    def _scores(
        self,
        action: TrackAction,
        azelr: np.ndarray,
        w: np.ndarray,
        when: datetime,
        bore: tuple[float, float] | None,
    ) -> tuple[float, float]:
        """Return (E[P_D], expected one-step information gain) for `action`.

        The information gain is the Bayesian entropy reduction of the belief,
        ``H(w) - [p_det H(w|detect) + (1-p_det) H(w|no-detect)]`` -- the same
        objective as the strong myopic controller.
        """
        pointing = self._realized_pointing(action, when, bore)
        dthetas = np.array(
            [delta_theta_from_azel(pointing, azelr[i, 0], azelr[i, 1]) for i in range(len(azelr))]
        )
        pd_i = np.asarray(
            detection_probability(dthetas, self.antenna_cfg, azelr[:, 1], azelr[:, 2]), dtype=float
        )
        p_det = float(np.dot(w, pd_i))
        wd = w * pd_i
        sd = float(wd.sum())
        wn = w * (1.0 - pd_i)
        sn = float(wn.sum())
        h_post = (p_det * self._entropy(wd / sd) if sd > 0 else 0.0) + (
            (1.0 - p_det) * self._entropy(wn / sn) if sn > 0 else 0.0
        )
        return p_det, self._entropy(w) - h_post

    def _best_over(
        self,
        azelr: np.ndarray,
        w: np.ndarray,
        when: datetime,
        bore: tuple[float, float] | None,
        taus: np.ndarray,
        thetas: np.ndarray,
    ) -> tuple[TrackAction, float, float]:
        """Maximise info-gain over a τ×θ grid; also track the best E[P_D] seen.

        Returns (info-gain-optimal action, its info-gain, max E[P_D] over grid).
        """
        best_a, best_ig, max_pd = TrackAction(0.0, 0.0), -math.inf, 0.0
        for t in taus:
            for c in thetas:
                a = TrackAction(float(t), float(c))
                pd, ig = self._scores(a, azelr, w, when, bore)
                max_pd = max(max_pd, pd)
                if ig > best_ig:
                    best_ig, best_a = ig, a
        return best_a, best_ig, max_pd

    def _ensure_greedy(self, node: _BeliefNode) -> None:
        """Compute and cache the node's myopic anchor (max one-step info-gain).

        The anchor objective is *information gain*, matching the strong myopic
        controller -- max-E[P_D] gets stuck re-pointing the densest mode at narrow
        beams while info-gain sweeps efficiently. A coarse pass plus a local
        refine localises the anchor inside a narrow (e.g., 1°) beam. `greedy_pd`
        (used by the leaf value) is the belief's best achievable detection
        probability, the natural value-of-belief signal.
        """
        if node.greedy is not None:
            return
        azelr = self._node_azelr(node)
        w = np.asarray(node.belief.weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        tmax = self.solver_cfg.along_track_offset_max_s
        cmax = self.solver_cfg.cross_track_offset_max_deg

        n_t = 21
        coarse_taus = np.linspace(-tmax, tmax, n_t)
        coarse_thetas = np.linspace(-cmax, cmax, 5)
        best_a, _, max_pd = self._best_over(
            azelr, w, node.when, node.bore, coarse_taus, coarse_thetas
        )

        # Local refine around the coarse info-gain winner (≈1 s τ resolution).
        dt = 2.0 * tmax / (n_t - 1)
        dc = 0.5 * cmax
        fine_taus = np.clip(
            np.linspace(best_a.along_track_s - dt, best_a.along_track_s + dt, 15), -tmax, tmax
        )
        fine_thetas = np.clip(
            np.linspace(best_a.cross_track_deg - dc, best_a.cross_track_deg + dc, 7), -cmax, cmax
        )
        ref_a, _, ref_max_pd = self._best_over(
            azelr, w, node.when, node.bore, fine_taus, fine_thetas
        )
        node.greedy = ref_a
        node.greedy_pd = float(max(0.0, max_pd, ref_max_pd))

    def _propose_action(self, node: _BeliefNode, rng: np.random.Generator) -> TrackAction:
        """Belief-informed proposal for progressive widening.

        The first action is the myopic greedy action (so the tree is anchored at
        the one-step-optimal decision and can only improve on it); subsequent
        proposals jitter around the greedy action to explore its neighbourhood,
        with an occasional uniform draw for global exploration.
        """
        self._ensure_greedy(node)
        k = len(node.actions)
        if k == 0:
            return node.greedy  # type: ignore[return-value]
        if rng.random() < 0.2:
            return self._sample_action(rng)  # global exploration
        # Local jitter around the greedy action.
        tmax = self.solver_cfg.along_track_offset_max_s
        cmax = self.solver_cfg.cross_track_offset_max_deg
        tau = float(np.clip(node.greedy.along_track_s + rng.normal(0.0, 0.2 * tmax), -tmax, tmax))
        theta = float(
            np.clip(node.greedy.cross_track_deg + rng.normal(0.0, 0.3 * cmax), -cmax, cmax)
        )
        return TrackAction(tau, theta)

    def _ucb_select(self, node: _BeliefNode) -> int:
        """UCB1 over the node's tried actions: Q + c*sqrt(log N / n_a)."""
        log_total = math.log(max(1, node.visits))
        best_idx = -1
        best_val = -math.inf
        for i in range(len(node.actions)):
            n = node.n_a.get(i, 0)
            if n == 0:
                return i
            val = node.q_a[i] + self.solver_cfg.ucb_c * math.sqrt(log_total / n)
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # Simulation (one tree descent + backup)
    # ------------------------------------------------------------------
    def _simulate(
        self,
        node: _BeliefNode,
        env: PointingEnv,
        rng: np.random.Generator,
    ) -> float:
        if node.depth >= self.solver_cfg.max_depth:
            return 0.0

        # --- Action progressive widening (belief-informed proposals) ---
        if len(node.actions) < self.solver_cfg.k_action * (
            max(1, node.visits) ** self.solver_cfg.alpha_action
        ):
            node.actions.append(self._propose_action(node, rng))
            a_idx = len(node.actions) - 1
        else:
            a_idx = self._ucb_select(node)

        action = node.actions[a_idx]

        # --- Sample an observation from a node-belief state particle ---
        # The *realised* pointing respects the slew limit from the node boresight,
        # so the tree plans physically reachable trajectories.
        pointing = self._realized_pointing(action, node.when, node.bore)
        gen = node.belief.particles[rng.integers(0, node.belief.n)]
        detected = simulate_observation_for_particle(
            gen, node.when, pointing, self.station, self.antenna_cfg, rng
        )
        reward = self.solver_cfg.reward_detect if detected else self.solver_cfg.reward_step

        if detected:
            # Acquisition is terminal: no future, no child belief to expand.
            ret = reward
            self._update(node, a_idx, ret)
            return ret

        # --- Observation widening collapses to the single no-detect bit ---
        key = (a_idx, 0)
        child = node.children.get(key)
        if child is None:
            child = _BeliefNode(
                depth=node.depth + 1,
                belief=self._pf_step(node.belief, pointing, detected, node.when, rng),
                when=node.when + timedelta(seconds=env.dwell_time_s),
                bore=(pointing.az, pointing.el),
            )
            node.children[key] = child
            future = self._leaf_value(child, env)
        else:
            future = self._simulate(child, env, rng)

        ret = reward + self.solver_cfg.gamma * future
        self._update(node, a_idx, ret)
        return ret

    # ------------------------------------------------------------------
    def _pf_step(
        self,
        belief: ParticleBelief,
        pointing,
        detected: bool,
        when: datetime,
        rng: np.random.Generator,
    ) -> ParticleBelief:
        """Reweight node particles by the observation, then resample to size m."""
        azelr = belief_azel_range(belief, when, self.station)
        likelihoods = np.array(
            [
                float(
                    observation_likelihood(
                        detected,
                        delta_theta_from_azel(pointing, azelr[i, 0], azelr[i, 1]),
                        self.antenna_cfg,
                        azelr[i, 1],
                        azelr[i, 2],
                    )
                )
                for i in range(belief.n)
            ]
        )
        w = belief.weights * likelihoods
        total = float(w.sum())
        if total <= 0.0:
            w = np.full(belief.n, 1.0 / belief.n)
        else:
            w = w / total

        # Resample (with replacement) back to n_belief_particles.
        m = self.solver_cfg.n_belief_particles
        idx = rng.choice(belief.n, size=m, replace=True, p=w)
        parts = [belief.particles[i] for i in idx]
        return ParticleBelief(particles=parts, weights=np.full(m, 1.0 / m))

    # ------------------------------------------------------------------
    def _leaf_value(self, node: _BeliefNode, env: PointingEnv) -> float:
        """Heuristic value of a leaf belief: discounted detection under greedy.

        Assume the myopic greedy action is repeated. If its belief-averaged
        detection probability is ``p`` and the remaining horizon is ``H`` dwells,
        the discounted expected return of "keep trying greedy" is the geometric
        series with per-step success ``p`` and miss-discount ``gamma(1-p)``:

            V = p R_d sum_{k=0}^{H-1} (gamma (1-p))^k.

        """
        self._ensure_greedy(node)
        p = node.greedy_pd
        H = max(0, self.solver_cfg.max_depth - node.depth)
        if H == 0:
            return 0.0
        r_d = self.solver_cfg.reward_detect
        decay = self.solver_cfg.gamma * (1.0 - p)
        if decay >= 1.0 - 1e-9:
            return p * r_d * H
        return p * r_d * (1.0 - decay**H) / (1.0 - decay)

    # ------------------------------------------------------------------
    def _update(self, node: _BeliefNode, a_idx: int, ret: float) -> None:
        node.visits += 1
        n = node.n_a.get(a_idx, 0) + 1
        q = node.q_a.get(a_idx, 0.0)
        node.n_a[a_idx] = n
        node.q_a[a_idx] = q + (ret - q) / n


# ---------------------------------------------------------------------------
# LEOP policy factory (mirrors make_pomcp_policy)
# ---------------------------------------------------------------------------


def make_pft_policy(cfg: Config):
    """Return a PolicyFactory wrapping PFT-DPW for the LEOP harness."""

    def factory(_cfg: Config, rng: np.random.Generator):
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)
        solver = PftDpwSolver(
            solver_cfg=_cfg.solver,
            antenna_cfg=_cfg.antenna,
            station=station,
            nominal=nominal,
        )

        def act(belief, env, when):
            return solver.plan(belief, when, env, rng)

        return act

    return factory


if __name__ == "__main__":
    from dataclasses import replace

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

    small = replace(cfg.solver, n_iterations=50, max_depth=5, n_belief_particles=20)
    solver = PftDpwSolver(
        solver_cfg=small,
        antenna_cfg=cfg.antenna,
        station=station,
        nominal=nominal,
    )
    action = solver.plan(belief, env.pass_start, env, rng)
    print(
        f"[pft_dpw] action: along_track={action.along_track_s:.2f}s, "
        f"cross_track={action.cross_track_deg:.2f}deg"
    )
    assert isinstance(action, TrackAction)
    assert abs(action.along_track_s) <= small.along_track_offset_max_s + 1e-9
    assert abs(action.cross_track_deg) <= small.cross_track_offset_max_deg + 1e-9
    print("[pft_dpw] PASS")
