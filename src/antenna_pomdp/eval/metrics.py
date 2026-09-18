"""Per-trial and aggregate metrics for the acquisition experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    """Outcome of one Monte Carlo trial."""

    acquired: bool
    time_to_acquire_s: float | None  # None if never acquired
    final_belief_entropy: float
    n_steps: int


def acquisition_probability(results: list[TrialResult]) -> float:
    """Empirical P(acquire) over a list of trials."""
    if not results:
        return 0.0
    return float(np.mean([r.acquired for r in results]))


def mean_time_to_acquire(results: list[TrialResult]) -> float:
    """Mean time-to-acquire over the successful trials only."""
    times = [r.time_to_acquire_s for r in results if r.acquired and r.time_to_acquire_s is not None]
    if not times:
        return float("nan")
    return float(np.mean(times))


def belief_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of a normalized weight vector (in nats)."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0.0]
    if w.size == 0:
        return 0.0
    return float(-np.sum(w * np.log(w)))


def summarize(results: list[TrialResult]) -> dict[str, float]:
    """One-line dict of acquisition probability, mean time-to-acquire, etc."""
    return {
        "n_trials": len(results),
        "p_acquire": acquisition_probability(results),
        "mean_time_to_acquire_s": mean_time_to_acquire(results),
        "mean_final_entropy": float(
            np.mean([r.final_belief_entropy for r in results]) if results else 0.0
        ),
    }


if __name__ == "__main__":
    results = [
        TrialResult(True, 12.0, 1.0, 5),
        TrialResult(True, 30.0, 0.5, 7),
        TrialResult(False, None, 4.0, 100),
    ]
    s = summarize(results)
    assert abs(s["p_acquire"] - 2 / 3) < 1e-9
    assert abs(s["mean_time_to_acquire_s"] - 21.0) < 1e-9
    print(f"[metrics] {s}")
    print("[metrics] PASS")
