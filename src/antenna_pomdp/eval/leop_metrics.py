"""Aggregate metrics for multi-pass LEOP episodes."""

from __future__ import annotations

import numpy as np

from antenna_pomdp.eval.leop_runner import LeopEpisodeResult


def cumulative_acquisition_curve(
    episodes: list[LeopEpisodeResult],
    time_grid_s: np.ndarray,
) -> np.ndarray:
    """Empirical P(acquired by t) on the provided grid (seconds since separation)."""
    if not episodes:
        return np.zeros_like(time_grid_s, dtype=float)
    acq_times = np.array(
        [ep.time_since_separation_s if ep.acquired else np.inf for ep in episodes],
        dtype=float,
    )
    return np.array([float(np.mean(acq_times <= t)) for t in time_grid_s])


def pass_success_rate(
    episodes: list[LeopEpisodeResult],
    max_pass: int,
) -> np.ndarray:
    """For each pass k = 1..max_pass, the fraction of episodes that acquired during pass k.

    Episodes that never acquired contribute 0 to all pass counts; episodes that
    acquired on pass k contribute 1 to pass k and 0 to all others. The marginal
    sums to `acquisition_probability`.
    """
    if not episodes:
        return np.zeros(max_pass)
    counts = np.zeros(max_pass)
    for ep in episodes:
        k = ep.acquired_pass_number
        if k is not None and 1 <= k <= max_pass:
            counts[k - 1] += 1
    return counts / len(episodes)


def mean_time_to_acquire(episodes: list[LeopEpisodeResult]) -> float:
    """Mean time-since-separation to acquisition, conditional on acquiring."""
    times = [ep.time_since_separation_s for ep in episodes if ep.acquired]
    if not times:
        return float("nan")
    return float(np.mean(times))


def failure_rate(episodes: list[LeopEpisodeResult]) -> float:
    """Fraction of episodes that never acquired within the LEOP window."""
    if not episodes:
        return 0.0
    return float(np.mean([not ep.acquired for ep in episodes]))


def summarize(episodes: list[LeopEpisodeResult], max_pass: int = 5) -> dict:
    return {
        "n_episodes": len(episodes),
        "p_acquire": float(np.mean([ep.acquired for ep in episodes])) if episodes else 0.0,
        "mean_time_to_acquire_s": mean_time_to_acquire(episodes),
        "failure_rate": failure_rate(episodes),
        "per_pass_success": pass_success_rate(episodes, max_pass).tolist(),
    }


if __name__ == "__main__":
    # Hand-built mini episodes for unit coverage.
    from antenna_pomdp.eval.leop_runner import LeopEpisodeResult

    eps = [
        LeopEpisodeResult(True, 1, 600.0, 100.0, 1, 4, 1.0),
        LeopEpisodeResult(True, 2, 6000.0, 200.0, 2, 4, 2.0),
        LeopEpisodeResult(False, None, None, None, 4, 4, 5.0),
    ]
    grid = np.array([300.0, 1000.0, 7000.0, 24 * 3600.0])
    cdf = cumulative_acquisition_curve(eps, grid)
    rates = pass_success_rate(eps, max_pass=4)
    s = summarize(eps, max_pass=4)
    print(f"[leop_metrics] cdf  : {cdf}")
    print(f"[leop_metrics] rates: {rates}")
    print(f"[leop_metrics] summ : {s}")
    assert abs(cdf[0] - 0.0) < 1e-9
    assert abs(cdf[2] - 2.0 / 3.0) < 1e-9
    assert abs(rates[0] - 1 / 3) < 1e-9 and abs(rates[1] - 1 / 3) < 1e-9
    assert abs(s["failure_rate"] - 1 / 3) < 1e-9
    print("[leop_metrics] PASS")
