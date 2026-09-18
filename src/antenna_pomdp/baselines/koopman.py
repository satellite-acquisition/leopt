"""Koopman-inspired non-adaptive search-effort allocation.

Given the prior belief at pass start and a surrogate per-cell exponential
detection law

    P_detect(effort) = 1 - exp(-lambda * effort),

it allocates a fixed total dwell budget across the reachable along-track look-
angle cells so as to maximise the *prior* probability of detection. The optimal
allocation is the classical water-filling / Lagrangian solution (Koopman 1956;
Stone 1989, optimal search of a stationary target with an exponential detection
function):

    effort_j = max(0, (1/lambda) * ln(lambda * p_j / nu)),

with the threshold `nu` chosen so that sum_j effort_j = total_budget. Cells
whose prior mass `p_j` falls below `nu / lambda` receive zero effort.

The policy never conditions on within-pass observations: it computes the
surrogate-optimal allocation once at the start of each pass and then steps
blindly through it. It is *not* an upper bound for the implemented acquisition
model, whose beams overlap, link probability varies with geometry, actions are
ordered integer dwells, and mounts may be slew constrained.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from antenna_pomdp.config import Config, KoopmanConfig
from antenna_pomdp.models.particle_filter import ParticleBelief, belief_azel_range
from antenna_pomdp.orbit.geometry import Station, angular_separation
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
)


# ---------------------------------------------------------------------------
# Cell axis
# ---------------------------------------------------------------------------


def along_track_cell_offsets(koopman_cfg: KoopmanConfig, along_track_max_s: float) -> np.ndarray:
    """The `n_cells` along-track offsets (seconds) the budget is spread over.

    Symmetric grid on [-along_track_max_s, +along_track_max_s]; cross-track is
    fixed at 0 (the dominant LEOP uncertainty is along-track timing).
    """
    n = max(1, koopman_cfg.n_cells)
    return np.linspace(-along_track_max_s, along_track_max_s, n)


# ---------------------------------------------------------------------------
# Prior mass per cell (from the belief)
# ---------------------------------------------------------------------------


def cell_prior_mass(
    belief: ParticleBelief,
    cell_offsets: np.ndarray,
    nominal: SGP4Propagator,
    when: datetime,
    station: Station,
) -> np.ndarray:
    """Estimate prior probability mass `p_j` in each along-track cell.

    Each cell `j` corresponds to a nominal-track pointing offset by
    `cell_offsets[j]` seconds along-track (cross-track = 0). We project every
    particle's predicted az/el at `when` onto the nominal-track az/el curve and
    assign the particle (with its belief weight) to the nearest cell by
    angular separation in the topocentric (az/el) frame. The per-cell weights
    are normalised to sum to 1.

    This is a robust, geometry-aware proxy for the particle's along-track
    offset: it directly uses the look-angle the antenna would command, so cells
    that no particle's sky position aligns with correctly receive ~zero mass.
    """
    # Pointing (az, el) of each cell's nominal-track command at this time.
    cell_pointings = [
        action_to_pointing(TrackAction(float(dt), 0.0), nominal, when, station)
        for dt in cell_offsets
    ]
    cell_az = np.array([p.az for p in cell_pointings])
    cell_el = np.array([p.el for p in cell_pointings])

    azelr = belief_azel_range(belief, when, station)  # (N, 3)
    weights = np.asarray(belief.weights, dtype=float)
    wsum = float(weights.sum())
    weights = weights / wsum if wsum > 0.0 else np.full(belief.n, 1.0 / belief.n)

    n_cells = len(cell_offsets)
    mass = np.zeros(n_cells)
    for i in range(belief.n):
        p_az, p_el = float(azelr[i, 0]), float(azelr[i, 1])
        seps = np.array(
            [angular_separation(p_az, p_el, cell_az[j], cell_el[j]) for j in range(n_cells)]
        )
        j_star = int(np.argmin(seps))
        mass[j_star] += weights[i]

    total = float(mass.sum())
    if total <= 0.0:
        return np.full(n_cells, 1.0 / n_cells)
    return mass / total


# ---------------------------------------------------------------------------
# Water-filling allocation (Koopman / Stone)
# ---------------------------------------------------------------------------


def water_fill_effort(
    prior_mass: np.ndarray,
    total_budget: float,
    detection_rate_lambda: float,
) -> np.ndarray:
    """Optimal continuous effort allocation maximising prior P(detect).

    Solves  effort_j = max(0, (1/lambda) * ln(lambda * p_j / nu))  with `nu`
    set by bisection so that sum_j effort_j == total_budget. Returns a real-
    valued effort vector (same units as `total_budget`) summing to the budget.

    Cells with `lambda * p_j <= nu` get exactly zero effort (they are below the
    water line). With a uniform prior the allocation is uniform.
    """
    p = np.asarray(prior_mass, dtype=float)
    lam = float(detection_rate_lambda)
    budget = float(total_budget)
    if budget <= 0.0 or lam <= 0.0 or p.size == 0:
        return np.zeros_like(p)

    # Only cells with positive mass can ever receive effort.
    pos = p > 0.0
    if not np.any(pos):
        return np.zeros_like(p)

    lp = lam * p  # lambda * p_j; effort positive iff lp_j > nu

    def total_effort(nu: float) -> float:
        # effort_j = max(0, (1/lambda) * ln(lp_j / nu))
        with np.errstate(divide="ignore", invalid="ignore"):
            e = (1.0 / lam) * np.log(np.where(lp > nu, lp / nu, 1.0))
        e = np.where(lp > nu, e, 0.0)
        return float(e.sum())

    # Bisect nu in (0, max(lp)]. At nu -> 0+, total_effort -> +inf (>= budget);
    # at nu = max(lp), total_effort = 0 (<= budget). total_effort is monotone
    # decreasing in nu, so a unique root exists.
    nu_hi = float(lp.max())
    nu_lo = nu_hi * 1e-12
    # Ensure the low end over-shoots the budget.
    for _ in range(200):
        if total_effort(nu_lo) >= budget:
            break
        nu_lo *= 0.5
    for _ in range(200):
        mid = 0.5 * (nu_lo + nu_hi)
        if total_effort(mid) > budget:
            nu_lo = mid
        else:
            nu_hi = mid
    nu = 0.5 * (nu_lo + nu_hi)

    with np.errstate(divide="ignore", invalid="ignore"):
        effort = (1.0 / lam) * np.log(np.where(lp > nu, lp / nu, 1.0))
    effort = np.where(lp > nu, effort, 0.0)
    effort = np.maximum(0.0, effort)

    # Rescale to hit the budget exactly (bisection leaves a tiny residual).
    s = float(effort.sum())
    if s > 0.0:
        effort = effort * (budget / s)
    return effort


def allocate_dwells(
    prior_mass: np.ndarray,
    n_dwells_budget: int,
    detection_rate_lambda: float,
) -> np.ndarray:
    """Round the continuous water-filling effort to integer dwells per cell.

    Uses the largest-remainder (Hamilton) method so the integer allocation
    sums exactly to `n_dwells_budget` and tracks the continuous optimum as
    closely as possible.
    """
    budget = int(n_dwells_budget)
    n_cells = len(prior_mass)
    if budget <= 0 or n_cells == 0:
        return np.zeros(n_cells, dtype=int)

    effort = water_fill_effort(prior_mass, float(budget), detection_rate_lambda)
    s = float(effort.sum())
    if s <= 0.0:
        # Degenerate: dump everything on the single most-likely cell.
        out = np.zeros(n_cells, dtype=int)
        out[int(np.argmax(prior_mass))] = budget
        return out

    exact = effort * (budget / s)
    floor = np.floor(exact).astype(int)
    remainder = int(budget - int(floor.sum()))
    if remainder > 0:
        frac = exact - floor
        # Hand out the leftover dwells to the largest fractional parts.
        order = np.argsort(-frac)
        for k in range(remainder):
            floor[order[k % n_cells]] += 1
    return floor


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def build_schedule(
    n_dwells_per_cell: np.ndarray,
    cell_offsets: np.ndarray,
    prior_mass: np.ndarray,
) -> list[TrackAction]:
    """Flatten the per-cell dwell counts into an ordered open-loop schedule.

    Cells are visited in order of DESCENDING prior mass (search the most-likely
    cells first to minimise expected time-to-acquire). Within a cell the same
    `TrackAction(Delta_tau, 0.0)` is repeated `n_dwells` times.
    """
    order = np.argsort(-np.asarray(prior_mass, dtype=float))
    schedule: list[TrackAction] = []
    for j in order:
        n_j = int(n_dwells_per_cell[j])
        if n_j <= 0:
            continue
        action = TrackAction(float(cell_offsets[j]), 0.0)
        schedule.extend([action] * n_j)
    return schedule


@dataclass
class KoopmanSchedule:
    """A precomputed open-loop dwell schedule for one pass."""

    actions: list[TrackAction]
    cell_offsets: np.ndarray
    prior_mass: np.ndarray
    n_dwells_per_cell: np.ndarray


def plan_pass(
    belief: ParticleBelief,
    env: PointingEnv,
    when: datetime,
    nominal: SGP4Propagator,
    station: Station,
    koopman_cfg: KoopmanConfig,
    along_track_max_s: float,
) -> KoopmanSchedule:
    """Compute the surrogate-optimal dwell allocation for the pass starting now."""
    cell_offsets = along_track_cell_offsets(koopman_cfg, along_track_max_s)
    prior_mass = cell_prior_mass(belief, cell_offsets, nominal, when, station)
    n_per_cell = allocate_dwells(prior_mass, env.horizon, koopman_cfg.detection_rate_lambda)
    actions = build_schedule(n_per_cell, cell_offsets, prior_mass)
    return KoopmanSchedule(
        actions=actions,
        cell_offsets=cell_offsets,
        prior_mass=prior_mass,
        n_dwells_per_cell=n_per_cell,
    )


# ---------------------------------------------------------------------------
# Stateful open-loop policy
# ---------------------------------------------------------------------------


class KoopmanPolicy:
    """Stateful driver of the Koopman-inspired open-loop schedule across passes.

    `act(belief, env, when)` is called once per dwell. At the start of every
    new pass (detected via `when == env.pass_start` or a change of env
    identity) the optimal schedule is recomputed from the *current* belief and
    then stepped through, ignoring observations.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
        self._station = Station.from_config(cfg.station)
        # Reach as far along-track as the POMCP action grid can, so the
        # comparison is fair (same physical reach).
        self._along_track_max_s = float(cfg.pomcp.along_track_offset_max_s)
        self._schedule: KoopmanSchedule | None = None
        self._idx = 0
        self._env_key: tuple | None = None

    def _is_new_pass(self, env: PointingEnv, when: datetime) -> bool:
        """A new pass = a different env object/start, or the start dwell itself."""
        key = (id(env), env.pass_start)
        return key != self._env_key or when == env.pass_start

    def act(self, belief: ParticleBelief, env: PointingEnv, when: datetime) -> TrackAction:
        if self._schedule is None or self._is_new_pass(env, when):
            self._schedule = plan_pass(
                belief,
                env,
                when,
                self._nominal,
                self._station,
                self._cfg.koopman,
                self._along_track_max_s,
            )
            self._idx = 0
            self._env_key = (id(env), env.pass_start)

        sched = self._schedule
        if sched is None or not sched.actions:
            return TrackAction(0.0, 0.0)
        # Step through the fixed schedule; clamp/wrap defensively if the pass
        # runs longer than the planned budget (rounding edge cases).
        action = sched.actions[self._idx % len(sched.actions)]
        self._idx += 1
        return action


def make_koopman_policy():
    """PolicyFactory for the Koopman open-loop controller.

    Matches `eval.leop_runner.PolicyFactory`: (Config, rng) -> act, where
    act(belief, env, when) -> TrackAction.
    """

    def factory(_cfg: Config, _rng: np.random.Generator):
        policy = KoopmanPolicy(_cfg)

        def act(belief: ParticleBelief, env: PointingEnv, when: datetime) -> TrackAction:
            return policy.act(belief, env, when)

        return act

    return factory


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from datetime import timedelta

    from antenna_pomdp.config import default_config
    from antenna_pomdp.models.particle_filter import make_belief
    from antenna_pomdp.orbit.sampler import sample_leop_particles, sample_leop_truth

    cfg = default_config()
    rng = np.random.default_rng(0)

    # --- water-filling unit properties ---
    p_uniform = np.full(5, 0.2)
    e = water_fill_effort(p_uniform, total_budget=10.0, detection_rate_lambda=1.0)
    assert abs(e.sum() - 10.0) < 1e-6, e.sum()
    assert np.allclose(e, e[0]), e  # uniform prior -> uniform effort

    p_skew = np.array([0.6, 0.3, 0.07, 0.02, 0.01])
    # Small budget relative to the prior spread -> low-mass cells are starved.
    e2 = water_fill_effort(p_skew, total_budget=2.0, detection_rate_lambda=2.0)
    assert abs(e2.sum() - 2.0) < 1e-6, e2.sum()
    # More effort on higher-prior cells; some low-prior cells starved to zero.
    assert e2[0] >= e2[1] >= e2[2], e2
    assert np.any(e2 == 0.0), e2

    # --- integer allocation sums to budget ---
    n_alloc = allocate_dwells(p_skew, n_dwells_budget=20, detection_rate_lambda=2.0)
    assert int(n_alloc.sum()) == 20, n_alloc
    assert n_alloc[0] >= n_alloc[-1], n_alloc

    # --- end-to-end planning on a real belief ---
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_leop_particles(cfg.orbit, cfg.leop, n=120, rng=rng)
    belief = make_belief(particles)
    truth = sample_leop_truth(cfg.orbit, cfg.leop, rng)
    when = nominal.epoch + timedelta(minutes=5)
    env = PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=when,
        pass_duration_s=cfg.station.pass_duration_s,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )
    sched = plan_pass(
        belief, env, when, nominal, station, cfg.koopman, cfg.pomcp.along_track_offset_max_s
    )
    assert int(sched.n_dwells_per_cell.sum()) == env.horizon, (
        sched.n_dwells_per_cell.sum(),
        env.horizon,
    )
    assert len(sched.actions) == env.horizon, (len(sched.actions), env.horizon)
    assert abs(float(sched.prior_mass.sum()) - 1.0) < 1e-6
    # All scheduled actions are within bounds.
    for a in sched.actions:
        assert abs(a.along_track_s) <= cfg.pomcp.along_track_offset_max_s + 1e-6
        assert a.cross_track_deg == 0.0

    # --- policy returns in-bounds actions across a simulated pass ---
    factory = make_koopman_policy()
    act = factory(cfg, rng)
    state = env.initial_state()
    n_steps = 0
    while not state.done and n_steps < env.horizon:
        a = act(belief, env, state.when)
        assert abs(a.along_track_s) <= cfg.pomcp.along_track_offset_max_s + 1e-6
        state, _, _, _ = env.step(state, a, rng)
        n_steps += 1

    print(
        f"[koopman] horizon={env.horizon}, schedule len={len(sched.actions)}, "
        f"cells used={int((sched.n_dwells_per_cell > 0).sum())}/{cfg.koopman.n_cells}, "
        f"top cell mass={float(sched.prior_mass.max()):.3f}"
    )
    print("[koopman] PASS")
