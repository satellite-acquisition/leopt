"""Detect sustained particle-filter divergence and latch a sweep fallback.

The watchdog monitors effective sample size, peak weight, entropy, and
observation likelihood during pre-acquisition search. A supported detection
ends monitoring because the resulting belief concentration is expected.

ESS and likelihood alarms must persist for ``divergence_consecutive`` steps.
An already-collapsed belief under a miss latches immediately. An operator must
re-arm the session to clear the latch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antenna_pomdp.eval.metrics import belief_entropy
from antenna_pomdp.models.particle_filter import ParticleBelief

# Watchdog modes.
CLOSED_LOOP = "closed_loop"
FALLBACK_OPEN_LOOP = "fallback_open_loop"


@dataclass(frozen=True)
class WatchdogConfig:
    """Divergence thresholds. Fractions are of the particle count N."""

    ess_floor_frac: float = 0.25  # ESS < N/4 is a divergence tick
    divergence_consecutive: int = 3  # ticks in a row before latching
    max_weight_alarm: float = 0.99  # one particle owning ≥99% mass -> latch now
    entropy_floor_nats: float = 0.05  # entropy below this = collapsed -> latch now
    likelihood_floor: float = 1e-3  # obs support below this = "unexplained"


@dataclass
class WatchdogState:
    """Mutable per-session watchdog state."""

    consecutive_divergence: int = 0
    acquired: bool = False  # a supported detection has arrived -> stand down
    latched_open_loop: bool = False
    last_reason: str = ""

    def rearm(self) -> None:
        """Operator re-arm: clear the latch and counters (deliberate action)."""
        self.consecutive_divergence = 0
        self.latched_open_loop = False
        self.last_reason = ""


def max_normalized_weight(belief: ParticleBelief) -> float:
    w = np.asarray(belief.weights, dtype=float)
    s = float(w.sum())
    if s <= 0.0:
        return 1.0
    return float(np.max(w) / s)


def watchdog_step(
    belief: ParticleBelief,
    state: WatchdogState,
    cfg: WatchdogConfig,
    detected: bool = False,
    likelihood_total: float | None = None,
) -> str:
    """Fold one post-update belief into the watchdog; return the current mode.

    `detected` is the observation's detect bit; `likelihood_total` is the prior
    belief's support for that outcome (a detect with support ≈ 0 is unexplained).
    A supported detection is acquisition — the watchdog stands down. Otherwise a
    persistent (or extreme) collapse under no-detects latches to the open-loop
    fallback, which holds until `state.rearm()`.
    """
    if state.latched_open_loop:
        return FALLBACK_OPEN_LOOP

    supported = likelihood_total is None or likelihood_total >= cfg.likelihood_floor
    if detected and supported:
        # Acquisition: the collapse is success. Stand the watchdog down.
        state.acquired = True
        state.consecutive_divergence = 0
        return CLOSED_LOOP
    if state.acquired:
        # Already acquired — the belief is legitimately peaked; do not second-guess.
        return CLOSED_LOOP

    ess = belief.effective_sample_size()
    max_w = max_normalized_weight(belief)
    ent = belief_entropy(belief.weights)
    n = max(1, belief.n)

    def latch(reason: str) -> str:
        state.latched_open_loop = True
        state.last_reason = reason
        return FALLBACK_OPEN_LOOP

    # An already-degenerate belief under a no-detect is pathological -> latch now.
    if max_w >= cfg.max_weight_alarm:
        return latch(f"weight degeneracy: one particle holds {max_w * 100:.1f}% of the belief")
    if ent < cfg.entropy_floor_nats:
        return latch(f"belief collapsed under no-detects (entropy {ent:.3f} nats)")

    # Persistent ESS collapse, or observations that no particle explains.
    diverging = (ess < cfg.ess_floor_frac * n) or (
        likelihood_total is not None and likelihood_total < cfg.likelihood_floor
    )
    if diverging:
        state.consecutive_divergence += 1
    else:
        state.consecutive_divergence = 0
    if state.consecutive_divergence >= cfg.divergence_consecutive:
        return latch(
            f"belief diverging for {state.consecutive_divergence} steps without support "
            f"(ESS≈{ess:.1f} of {n})"
        )
    return CLOSED_LOOP


if __name__ == "__main__":
    from antenna_pomdp.orbit.propagator import SGP4Propagator
    from antenna_pomdp.orbit.sampler import OrbitParticle

    l1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
    l2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    parts = [OrbitParticle(SGP4Propagator(l1, l2), float(s)) for s in range(50)]
    cfg = WatchdogConfig()

    # Healthy uniform belief under a no-detect -> stays closed-loop.
    healthy = ParticleBelief(particles=parts, weights=np.full(50, 1.0 / 50))
    st = WatchdogState()
    assert watchdog_step(healthy, st, cfg, detected=False, likelihood_total=0.9) == CLOSED_LOOP

    # Degenerate belief under a no-detect -> immediate latch.
    w = np.full(50, 1e-6)
    w[0] = 1.0
    degen = ParticleBelief(particles=parts, weights=w / w.sum())
    st2 = WatchdogState()
    assert (
        watchdog_step(degen, st2, cfg, detected=False, likelihood_total=0.9) == FALLBACK_OPEN_LOOP
    )
    assert "degeneracy" in st2.last_reason

    # SAME degenerate belief but as a SUPPORTED DETECT = acquisition -> no latch.
    st3 = WatchdogState()
    assert watchdog_step(degen, st3, cfg, detected=True, likelihood_total=0.8) == CLOSED_LOOP
    assert st3.acquired and not st3.latched_open_loop

    # Persistent ESS collapse under no-detects (mass on ~5 of 50) -> latch on step 3.
    w2 = np.full(50, 1e-6)
    w2[:5] = 1.0
    peaked = ParticleBelief(particles=parts, weights=w2 / w2.sum())
    st4 = WatchdogState()
    modes = [
        watchdog_step(peaked, st4, cfg, detected=False, likelihood_total=0.5) for _ in range(3)
    ]
    assert modes == [CLOSED_LOOP, CLOSED_LOOP, FALLBACK_OPEN_LOOP], modes
    st4.rearm()
    assert not st4.latched_open_loop

    print(f"[watchdog] degen reason: {st2.last_reason}")
    print("[watchdog] PASS")
