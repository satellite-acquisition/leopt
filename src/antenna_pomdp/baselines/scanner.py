"""Open-loop pointing policies over along-track support.

TrackLineSweep alternates around the nominal track. OneSidedTrackSweep covers
nonnegative launch-delay support with nonpositive time offsets; its centered
variant starts at the support midpoint. Progressive and hierarchical variants
spend a finite dwell budget on increasingly fine support grids.

HierarchicalIntervalTrackSweep accepts any finite offset interval.
HierarchicalQuantileTrackSweep uses quantiles of a fixed design prior.
RandomScan samples the discrete action grid uniformly. None of these policies
updates its ordering from receiver outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from antenna_pomdp.config import PomcpConfig
from antenna_pomdp.pomdp.environment import TrackAction, build_action_grid


def hierarchical_refinement_levels(
    n_steps: int,
    *,
    start: float = 0.0,
    stop: float = 1.0,
) -> tuple[np.ndarray, ...]:
    """Return deterministic hierarchical refinement levels for a fixed horizon.

    ``start`` and ``stop`` define an oriented one-dimensional interval. The
    first level contains ``max(2, floor(H/3))`` uniform anchors (with explicit
    one- and two-step edge cases), the second contains adjacent-anchor
    midpoints, and the third contains the two child midpoints of each anchor
    interval in ``start``-to-``stop`` order. The final populated level is
    truncated to make the concatenated sequence exactly ``H`` points.

    Very short horizons can exhaust those three levels. In that case, the
    function preserves the legacy rule: repeatedly split the first largest
    gap in sorted coordinate order, appending each new point as a later level.
    For ``H=90``, the returned level lengths are exactly ``(30, 29, 31)``.

    The function is pure: its output depends only on the supplied horizon and
    interval, with no random state, truth state, belief, or receiver outcome.
    For ``H > 1``, every returned coordinate is unique.
    """

    if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, bool):
        raise TypeError("n_steps must be an integer")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    start = float(start)
    stop = float(stop)
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError("start and stop must be finite")
    if n_steps > 1 and start == stop:
        raise ValueError("start and stop must differ when n_steps > 1")

    if n_steps == 1:
        return (np.asarray([start], dtype=float),)
    if n_steps == 2:
        return (np.asarray([start, stop], dtype=float),)

    n_anchors = max(2, n_steps // 3)
    anchors = np.linspace(start, stop, n_anchors)
    first_midpoints = 0.5 * (anchors[:-1] + anchors[1:])
    child_midpoints = np.empty(2 * (n_anchors - 1), dtype=float)
    child_midpoints[0::2] = 0.5 * (anchors[:-1] + first_midpoints)
    child_midpoints[1::2] = 0.5 * (first_midpoints + anchors[1:])

    base_levels = (anchors, first_midpoints, child_midpoints)
    levels: list[np.ndarray] = []
    remaining = n_steps
    for level in base_levels:
        if remaining == 0:
            break
        selected = level[:remaining].copy()
        if selected.size:
            levels.append(selected)
            remaining -= selected.size

    if remaining:
        # Preserve the original short-horizon continuation exactly. Sorting in
        # physical coordinate order is significant when equal-size gaps tie.
        candidates = np.concatenate(base_levels)
        sorted_nodes = np.sort(candidates)
        continuation = np.empty(remaining, dtype=float)
        for index in range(remaining):
            gaps = np.diff(sorted_nodes)
            gap_index = int(np.argmax(gaps))
            midpoint = 0.5 * (sorted_nodes[gap_index] + sorted_nodes[gap_index + 1])
            continuation[index] = midpoint
            sorted_nodes = np.sort(np.append(sorted_nodes, midpoint))
        levels.append(continuation)

    flat = np.concatenate(levels)
    if flat.size != n_steps:
        raise AssertionError("hierarchical refinement did not fill the horizon")
    if np.unique(flat).size != n_steps:
        raise AssertionError("hierarchical refinement produced duplicate coordinates")
    return tuple(levels)


def _hierarchical_refinement_sequence(
    n_steps: int,
    *,
    start: float,
    stop: float,
) -> np.ndarray:
    """Flatten :func:`hierarchical_refinement_levels` without changing order."""

    return np.concatenate(hierarchical_refinement_levels(n_steps, start=start, stop=stop))


def hierarchical_quantile_offsets(
    quantile_levels: Sequence[float] | np.ndarray,
    offset_quantiles_s: Sequence[float] | np.ndarray,
    n_steps: int,
    *,
    lower_s: float | None = None,
    upper_s: float | None = None,
    start_at_upper: bool = False,
    allow_clipped_duplicates: bool = False,
) -> np.ndarray:
    """Map a hierarchical schedule through a precomputed design-prior table.

    ``quantile_levels`` must be strictly increasing and lie in ``[0, 1]``.
    Its endpoints define the finite design interval, so callers may use, for
    example, ``[0.001, 0.999]`` instead of an unbounded distribution's formal
    endpoints. ``offset_quantiles_s`` must be strictly monotone but may
    increase or decrease; decreasing offsets naturally represent positive
    launch delay mapping farther behind the nominal track.

    The table is fixed before execution. This helper performs deterministic
    linear interpolation only and does not estimate quantiles from an
    evaluation sample. Set ``start_at_upper=True`` to traverse from the upper
    to the lower design quantile, matching a near-nominal-to-delayed causal
    search order when projected track offset increases with quantile.
    Optional ``lower_s`` and ``upper_s`` bounds apply a common action-envelope
    clip after interpolation. Clipping that collapses multiple design nodes
    to the same command is rejected unless
    ``allow_clipped_duplicates=True`` explicitly declares repeated boundary
    exposures as part of the policy.
    """

    probabilities = np.asarray(quantile_levels, dtype=float)
    offsets = np.asarray(offset_quantiles_s, dtype=float)
    if probabilities.ndim != 1 or offsets.ndim != 1:
        raise ValueError("quantile tables must be one-dimensional")
    if probabilities.size < 2 or probabilities.shape != offsets.shape:
        raise ValueError("quantile tables must have the same length of at least two")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(offsets)):
        raise ValueError("quantile tables must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("quantile levels must lie in [0, 1]")
    if np.any(np.diff(probabilities) <= 0.0):
        raise ValueError("quantile levels must be strictly increasing")
    offset_differences = np.diff(offsets)
    if not (np.all(offset_differences > 0.0) or np.all(offset_differences < 0.0)):
        raise ValueError("offset quantiles must be strictly monotone")

    clip_lower = -np.inf if lower_s is None else float(lower_s)
    clip_upper = np.inf if upper_s is None else float(upper_s)
    if not np.isfinite(clip_lower) and lower_s is not None:
        raise ValueError("lower_s must be finite when provided")
    if not np.isfinite(clip_upper) and upper_s is not None:
        raise ValueError("upper_s must be finite when provided")
    if clip_lower > clip_upper:
        raise ValueError("lower_s must not exceed upper_s")

    if not isinstance(start_at_upper, (bool, np.bool_)):
        raise TypeError("start_at_upper must be boolean")
    if not isinstance(allow_clipped_duplicates, (bool, np.bool_)):
        raise TypeError("allow_clipped_duplicates must be boolean")

    start_probability, stop_probability = (
        (float(probabilities[-1]), float(probabilities[0]))
        if start_at_upper
        else (float(probabilities[0]), float(probabilities[-1]))
    )
    design_quantiles = _hierarchical_refinement_sequence(
        n_steps,
        start=start_probability,
        stop=stop_probability,
    )
    mapped = np.interp(design_quantiles, probabilities, offsets)
    mapped = np.clip(mapped, clip_lower, clip_upper)
    if not allow_clipped_duplicates and np.unique(mapped).size != n_steps:
        raise ValueError("quantile mapping or envelope clip produced duplicate offsets")
    return mapped


@dataclass
class HierarchicalIntervalTrackSweep:
    """Deterministic hierarchy over an arbitrary finite along-track interval.

    By default, traversal starts at ``upper_s`` and proceeds toward ``lower_s``.
    This makes ``[-T, 0]`` reproduce the one-sided launch-delay schedule
    ``0 -> -T``. Set ``start_at_upper=False`` to reverse the orientation.
    """

    lower_s: float
    upper_s: float
    n_steps: int
    start_at_upper: bool = True
    cross_track_deg: float = 0.0

    def __post_init__(self) -> None:
        lower_s = float(self.lower_s)
        upper_s = float(self.upper_s)
        cross_track_deg = float(self.cross_track_deg)
        if not np.isfinite(lower_s) or not np.isfinite(upper_s):
            raise ValueError("interval bounds must be finite")
        if lower_s > upper_s:
            raise ValueError("lower_s must not exceed upper_s")
        if not np.isfinite(cross_track_deg):
            raise ValueError("cross_track_deg must be finite")

        start, stop = (upper_s, lower_s) if self.start_at_upper else (lower_s, upper_s)
        offsets = _hierarchical_refinement_sequence(
            self.n_steps,
            start=start,
            stop=stop,
        )
        self._sequence = [TrackAction(float(offset), cross_track_deg) for offset in offsets]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self) -> TrackAction:
        action = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return action


@dataclass
class HierarchicalQuantileTrackSweep:
    """Deterministic hierarchy mapped through fixed design-prior quantiles."""

    quantile_levels: Sequence[float] | np.ndarray
    offset_quantiles_s: Sequence[float] | np.ndarray
    n_steps: int
    cross_track_deg: float = 0.0
    lower_s: float | None = None
    upper_s: float | None = None
    start_at_upper: bool = False
    allow_clipped_duplicates: bool = False

    def __post_init__(self) -> None:
        cross_track_deg = float(self.cross_track_deg)
        if not np.isfinite(cross_track_deg):
            raise ValueError("cross_track_deg must be finite")
        offsets = hierarchical_quantile_offsets(
            self.quantile_levels,
            self.offset_quantiles_s,
            self.n_steps,
            lower_s=self.lower_s,
            upper_s=self.upper_s,
            start_at_upper=self.start_at_upper,
            allow_clipped_duplicates=self.allow_clipped_duplicates,
        )
        self._sequence = [TrackAction(float(offset), cross_track_deg) for offset in offsets]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self) -> TrackAction:
        action = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return action


@dataclass
class TrackLineSweep:
    """Center-out sweep over along-track time offsets.

    The sequence visits zero offset first, then progressively larger negative
    and positive along-track offsets, always at zero cross-track offset. This
    matches the documented one-axis LEOP uncertainty model and avoids the
    previous implementation's accidental traversal of the full 2-D action
    grid. Once exhausted, the sequence wraps.
    """

    pomcp_cfg: PomcpConfig

    def __post_init__(self) -> None:
        n = max(1, self.pomcp_cfg.n_along_track_offsets)
        offsets = np.linspace(
            -self.pomcp_cfg.along_track_offset_max_s,
            self.pomcp_cfg.along_track_offset_max_s,
            n,
        )
        # Stable center-out ordering: 0, -Δ, +Δ, -2Δ, +2Δ, ...
        offsets = sorted(offsets, key=lambda x: (abs(float(x)), float(x)))
        self._sequence = [TrackAction(float(tau), 0.0) for tau in offsets]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self) -> TrackAction:
        a = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return a


@dataclass
class OneSidedTrackSweep:
    """Center-out sweep over the launch-delay-supported half of the grid.

    The full line-search grid spans ``[-T, T]``. For an odd number of full-grid
    points, this baseline keeps the same pitch but visits only
    ``0, -Δ, -2Δ, ..., -T``. It is specific to a prior whose launch delay is
    nonnegative and therefore maps to negative track-time offsets.
    """

    pomcp_cfg: PomcpConfig

    def __post_init__(self) -> None:
        n_full = max(1, self.pomcp_cfg.n_along_track_offsets)
        t_max = float(self.pomcp_cfg.along_track_offset_max_s)
        if n_full == 1 or t_max == 0.0:
            offsets = np.array([0.0])
        else:
            full = np.linspace(-t_max, t_max, n_full)
            offsets = full[full <= 1e-12][::-1]
        self._sequence = [TrackAction(float(tau), 0.0) for tau in offsets]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self) -> TrackAction:
        action = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return action


@dataclass
class CenteredOneSidedTrackSweep:
    """Center-out ordering over the launch-delay-supported half-grid.

    For the paper's uniform nonnegative launch-delay prior, the supported
    track-time interval is ``[-T, 0]`` and its midpoint is ``-T/2``. This
    baseline visits the supported grid in increasing distance from that
    midpoint, preserving the full-grid pitch and wrapping only after every
    supported point has been visited.
    """

    pomcp_cfg: PomcpConfig

    def __post_init__(self) -> None:
        n_full = max(1, self.pomcp_cfg.n_along_track_offsets)
        t_max = float(self.pomcp_cfg.along_track_offset_max_s)
        if n_full == 1 or t_max == 0.0:
            offsets = [0.0]
        else:
            full = np.linspace(-t_max, t_max, n_full)
            supported = full[full <= 1e-12]
            midpoint = -0.5 * t_max
            offsets = sorted(
                (float(value) for value in supported),
                key=lambda value: (abs(value - midpoint), -value),
            )
        self._sequence = [TrackAction(float(tau), 0.0) for tau in offsets]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self) -> TrackAction:
        action = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return action


@dataclass
class ProgressiveOneSidedTrackSweep:
    """Two-resolution, horizon-matched traversal of one-sided support.

    With ``H`` requested dwells, the sequence first visits
    ``ceil(H/2)`` equally spaced support nodes from zero to ``-T``, then visits
    the midpoint of each adjacent pair. For an even ``H``, the one remaining
    dwell revisits the support center. The 90-dwell paper case is therefore
    45 coarse nodes, 44 interval midpoints, and one central revisit.

    The rule is deterministic, independent of truth and receiver outcomes, and
    uses no fitted performance result. It is a stronger coverage comparator
    than repeating the 49 supported nodes inherited from a symmetric
    97-point parent grid.
    """

    pomcp_cfg: PomcpConfig
    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be positive")
        t_max = float(self.pomcp_cfg.along_track_offset_max_s)
        if self.n_steps == 1 or t_max == 0.0:
            offsets = [0.0]
        elif self.n_steps == 2:
            offsets = [0.0, -t_max]
        else:
            n_coarse = (self.n_steps + 1) // 2
            coarse = np.linspace(0.0, -t_max, n_coarse)
            midpoints = 0.5 * (coarse[:-1] + coarse[1:])
            offsets = [*(float(value) for value in coarse)]
            offsets.extend(float(value) for value in midpoints)
            if len(offsets) < self.n_steps:
                offsets.append(-0.5 * t_max)
        if len(offsets) != self.n_steps:
            raise AssertionError("progressive sweep construction did not fill the horizon")
        self._sequence = [TrackAction(float(tau), 0.0) for tau in offsets]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def act(self) -> TrackAction:
        action = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return action


@dataclass
class HierarchicalOneSidedTrackSweep:
    """Three-stage, horizon-derived refinement of one-sided support.

    For ``H`` requested dwells, the rule chooses ``floor(H/3)`` uniform anchors
    from zero to ``-T``, then visits all adjacent anchor midpoints, then visits
    child-interval midpoints in the same zero-to-``-T`` order until the horizon
    is full. The 90-dwell paper schedule therefore contains 30 anchors, 29
    first-level midpoints, and the first 31 second-level midpoints. All 90
    commands are unique.

    The rule is deterministic and independent of truth, receiver outcomes,
    link scores, and particle-belief samples. It was developed during a
    comparator red-team pilot; final estimates must therefore use an untouched
    evaluation seed.
    """

    pomcp_cfg: PomcpConfig
    n_steps: int

    def __post_init__(self) -> None:
        t_max = float(self.pomcp_cfg.along_track_offset_max_s)
        interval = HierarchicalIntervalTrackSweep(
            lower_s=-t_max,
            upper_s=0.0,
            n_steps=self.n_steps,
        )
        self._sequence = interval._sequence
        self._idx = interval._idx

    def reset(self) -> None:
        self._idx = 0

    def act(self) -> TrackAction:
        action = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return action


@dataclass
class RandomScan:
    """Uniform random pick from the discrete action grid."""

    pomcp_cfg: PomcpConfig
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self._actions = build_action_grid(self.pomcp_cfg)

    def reset(self) -> None:
        pass

    def act(self) -> TrackAction:
        return self._actions[self.rng.integers(0, len(self._actions))]


if __name__ == "__main__":
    from antenna_pomdp.config import default_config

    cfg = default_config()
    sweep = TrackLineSweep(cfg.pomcp)
    one_sided = OneSidedTrackSweep(cfg.pomcp)
    centered_one_sided = CenteredOneSidedTrackSweep(cfg.pomcp)
    progressive_one_sided = ProgressiveOneSidedTrackSweep(cfg.pomcp, n_steps=90)
    hierarchical_one_sided = HierarchicalOneSidedTrackSweep(cfg.pomcp, n_steps=90)
    rnd = RandomScan(cfg.pomcp, np.random.default_rng(0))
    n = len(sweep._sequence)
    actions = [sweep.act() for _ in range(2 * n)]
    # First and (n+1)-th should be identical (wraparound).
    assert actions[0] == actions[n], (actions[0], actions[n])
    one_sided_actions = [one_sided.act() for _ in range(len(one_sided._sequence))]
    assert all(action.along_track_s <= 0.0 for action in one_sided_actions)
    assert one_sided_actions[0].along_track_s == 0.0
    assert centered_one_sided.act().along_track_s == -0.5 * cfg.pomcp.along_track_offset_max_s
    progressive_actions = [
        progressive_one_sided.act() for _ in range(len(progressive_one_sided._sequence))
    ]
    assert len(progressive_actions) == 90
    assert all(action.along_track_s <= 0.0 for action in progressive_actions)
    hierarchical_actions = [
        hierarchical_one_sided.act() for _ in range(len(hierarchical_one_sided._sequence))
    ]
    assert len(hierarchical_actions) == 90
    assert len({action.along_track_s for action in hierarchical_actions}) == 90
    rnd_actions = [rnd.act() for _ in range(50)]
    assert len(set(id(a) for a in rnd_actions)) > 1
    print(f"[baselines] sweep length = {n}, random scan distinct = {len(set(rnd_actions))}")
    print("[baselines] PASS")
