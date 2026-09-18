"""Focused contracts for deterministic open-loop scanner baselines."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from antenna_pomdp.baselines.scanner import (
    HierarchicalIntervalTrackSweep,
    HierarchicalOneSidedTrackSweep,
    HierarchicalQuantileTrackSweep,
    hierarchical_quantile_offsets,
    hierarchical_refinement_levels,
)
from antenna_pomdp.config import default_config


def _offsets(sweep) -> np.ndarray:
    return np.asarray([action.along_track_s for action in sweep._sequence])


@pytest.mark.parametrize("n_steps", [1, 2, *range(3, 24), 89, 90, 91, 193])
def test_refinement_levels_fill_horizon_with_unique_bounded_points(n_steps):
    levels = hierarchical_refinement_levels(
        n_steps,
        start=4.25,
        stop=-17.5,
    )
    points = np.concatenate(levels)

    assert points.size == n_steps
    assert np.unique(points).size == n_steps
    assert np.all((-17.5 <= points) & (points <= 4.25))
    assert points[0] == 4.25
    if n_steps > 1:
        assert levels[0][-1] == -17.5


def test_refinement_levels_preserve_frozen_90_step_order_exactly():
    levels = hierarchical_refinement_levels(90, start=0.0, stop=-360.0)
    anchors = np.linspace(0.0, -360.0, 30)
    first_midpoints = 0.5 * (anchors[:-1] + anchors[1:])
    child_midpoints = np.empty(58, dtype=float)
    child_midpoints[0::2] = 0.5 * (anchors[:-1] + first_midpoints)
    child_midpoints[1::2] = 0.5 * (first_midpoints + anchors[1:])

    assert tuple(level.size for level in levels) == (30, 29, 31)
    assert np.array_equal(levels[0], anchors)
    assert np.array_equal(levels[1], first_midpoints)
    assert np.array_equal(levels[2], child_midpoints[:31])


@pytest.mark.parametrize("n_steps", [0, -1])
def test_refinement_levels_reject_nonpositive_horizons(n_steps):
    with pytest.raises(ValueError, match="positive"):
        hierarchical_refinement_levels(n_steps)


@pytest.mark.parametrize("n_steps", [True, 1.5, "3"])
def test_refinement_levels_reject_noninteger_horizons(n_steps):
    with pytest.raises(TypeError, match="integer"):
        hierarchical_refinement_levels(n_steps)


def test_refinement_levels_handle_degenerate_and_nonfinite_intervals():
    (only_level,) = hierarchical_refinement_levels(1, start=-2.0, stop=-2.0)
    assert np.array_equal(only_level, [-2.0])

    with pytest.raises(ValueError, match="differ"):
        hierarchical_refinement_levels(2, start=-2.0, stop=-2.0)
    with pytest.raises(ValueError, match="finite"):
        hierarchical_refinement_levels(1, start=np.inf, stop=0.0)


@pytest.mark.parametrize("start_at_upper", [True, False])
def test_interval_sweep_respects_bounds_orientation_and_reset(start_at_upper):
    lower_s, upper_s, n_steps = -17.5, 4.25, 19
    sweep = HierarchicalIntervalTrackSweep(
        lower_s=lower_s,
        upper_s=upper_s,
        n_steps=n_steps,
        start_at_upper=start_at_upper,
        cross_track_deg=0.75,
    )
    start, stop = (upper_s, lower_s) if start_at_upper else (lower_s, upper_s)
    expected = np.concatenate(hierarchical_refinement_levels(n_steps, start=start, stop=stop))

    assert np.array_equal(_offsets(sweep), expected)
    assert np.unique(expected).size == n_steps
    assert np.all((lower_s <= expected) & (expected <= upper_s))
    assert all(action.cross_track_deg == 0.75 for action in sweep._sequence)

    first_pass = [sweep.act() for _ in range(n_steps)]
    assert sweep.act() == first_pass[0]
    sweep.reset()
    assert sweep.act() == first_pass[0]


def test_interval_sweep_validates_bounds_and_degenerate_support():
    with pytest.raises(ValueError, match="lower_s"):
        HierarchicalIntervalTrackSweep(lower_s=1.0, upper_s=-1.0, n_steps=3)
    with pytest.raises(ValueError, match="differ"):
        HierarchicalIntervalTrackSweep(lower_s=2.0, upper_s=2.0, n_steps=2)

    sweep = HierarchicalIntervalTrackSweep(
        lower_s=2.0,
        upper_s=2.0,
        n_steps=1,
    )
    assert sweep.act().along_track_s == 2.0


def test_one_sided_sweep_is_bitwise_compatible_with_frozen_90_step_rule():
    cfg = default_config()
    t_max = float(cfg.pomcp.along_track_offset_max_s)
    sweep = HierarchicalOneSidedTrackSweep(cfg.pomcp, n_steps=90)

    anchors = np.linspace(0.0, -t_max, 30)
    first_midpoints = 0.5 * (anchors[:-1] + anchors[1:])
    child_midpoints = np.empty(58, dtype=float)
    child_midpoints[0::2] = 0.5 * (anchors[:-1] + first_midpoints)
    child_midpoints[1::2] = 0.5 * (first_midpoints + anchors[1:])
    legacy = np.concatenate((anchors, first_midpoints, child_midpoints))[:90]

    assert np.array_equal(_offsets(sweep), legacy)
    assert all(action.cross_track_deg == 0.0 for action in sweep._sequence)


def test_quantile_hierarchy_uses_only_fixed_precomputed_design_table():
    quantile_levels = np.asarray([0.001, 0.10, 0.50, 0.90, 0.999])
    offset_quantiles_s = np.asarray([0.0, -4.0, -18.0, -55.0, -140.0])
    frozen_levels = quantile_levels.copy()
    frozen_offsets = offset_quantiles_s.copy()
    n_steps = 17

    design_quantiles = np.concatenate(
        hierarchical_refinement_levels(
            n_steps,
            start=float(frozen_levels[0]),
            stop=float(frozen_levels[-1]),
        )
    )
    expected = np.interp(design_quantiles, frozen_levels, frozen_offsets)
    mapped = hierarchical_quantile_offsets(
        quantile_levels,
        offset_quantiles_s,
        n_steps,
    )
    sweep = HierarchicalQuantileTrackSweep(
        quantile_levels,
        offset_quantiles_s,
        n_steps,
    )

    assert np.array_equal(mapped, expected)
    assert np.array_equal(_offsets(sweep), expected)
    assert np.unique(expected).size == n_steps
    assert np.all((frozen_offsets[-1] <= expected) & (expected <= frozen_offsets[0]))

    # Execution has no truth, observation, belief, or RNG input, and the
    # already-constructed sequence is insulated from caller-side table changes.
    assert list(inspect.signature(sweep.act).parameters) == []
    quantile_levels[:] = 0.5
    offset_quantiles_s[:] = 999.0
    assert np.array_equal(_offsets(sweep), expected)


def test_quantile_hierarchy_clips_single_tail_node_to_action_envelope():
    quantile_levels = np.asarray([0.001, 0.01, 0.50, 0.999])
    offset_quantiles_s = np.asarray([-125.4, -115.0, -50.0, 0.0])
    n_steps = 90

    unclipped = hierarchical_quantile_offsets(
        quantile_levels,
        offset_quantiles_s,
        n_steps,
    )
    clipped = hierarchical_quantile_offsets(
        quantile_levels,
        offset_quantiles_s,
        n_steps,
        lower_s=-120.0,
        upper_s=120.0,
    )
    sweep = HierarchicalQuantileTrackSweep(
        quantile_levels,
        offset_quantiles_s,
        n_steps,
        lower_s=-120.0,
        upper_s=120.0,
    )

    assert np.count_nonzero(unclipped < -120.0) == 1
    assert clipped[0] == -120.0
    assert np.array_equal(clipped[1:], unclipped[1:])
    assert np.array_equal(_offsets(sweep), clipped)
    assert np.unique(clipped).size == n_steps
    assert np.all((-120.0 <= clipped) & (clipped <= 120.0))


def test_quantile_hierarchy_can_traverse_upper_tail_first():
    quantile_levels = np.asarray([0.001, 0.25, 0.50, 0.75, 0.999])
    offset_quantiles_s = np.asarray([-125.0, -90.0, -60.0, -30.0, 6.0])
    n_steps = 17

    design_quantiles = np.concatenate(
        hierarchical_refinement_levels(
            n_steps,
            start=float(quantile_levels[-1]),
            stop=float(quantile_levels[0]),
        )
    )
    expected = np.interp(design_quantiles, quantile_levels, offset_quantiles_s)
    mapped = hierarchical_quantile_offsets(
        quantile_levels,
        offset_quantiles_s,
        n_steps,
        start_at_upper=True,
    )
    sweep = HierarchicalQuantileTrackSweep(
        quantile_levels,
        offset_quantiles_s,
        n_steps,
        start_at_upper=True,
    )

    assert np.array_equal(mapped, expected)
    assert np.array_equal(_offsets(sweep), expected)
    assert mapped[0] == offset_quantiles_s[-1]
    first_level_size = hierarchical_refinement_levels(n_steps)[0].size
    assert mapped[first_level_size - 1] == offset_quantiles_s[0]


def test_quantile_hierarchy_requires_explicit_permission_for_clipped_repeats():
    quantile_levels = np.asarray([0.001, 0.25, 0.50, 0.75, 0.999])
    offset_quantiles_s = np.asarray([-200.0, -160.0, -60.0, -30.0, 6.0])

    with pytest.raises(ValueError, match="duplicate"):
        hierarchical_quantile_offsets(
            quantile_levels,
            offset_quantiles_s,
            17,
            lower_s=-120.0,
            upper_s=120.0,
            start_at_upper=True,
        )

    mapped = hierarchical_quantile_offsets(
        quantile_levels,
        offset_quantiles_s,
        17,
        lower_s=-120.0,
        upper_s=120.0,
        start_at_upper=True,
        allow_clipped_duplicates=True,
    )
    sweep = HierarchicalQuantileTrackSweep(
        quantile_levels,
        offset_quantiles_s,
        17,
        lower_s=-120.0,
        upper_s=120.0,
        start_at_upper=True,
        allow_clipped_duplicates=True,
    )

    assert np.count_nonzero(mapped == -120.0) > 1
    assert np.array_equal(_offsets(sweep), mapped)
    assert np.all((-120.0 <= mapped) & (mapped <= 120.0))


def test_quantile_hierarchy_rejects_invalid_or_collapsing_envelope():
    with pytest.raises(ValueError, match="lower_s"):
        hierarchical_quantile_offsets(
            [0.0, 1.0],
            [-10.0, 10.0],
            5,
            lower_s=2.0,
            upper_s=-2.0,
        )
    with pytest.raises(ValueError, match="duplicate"):
        hierarchical_quantile_offsets(
            [0.0, 1.0],
            [-10.0, 10.0],
            5,
            lower_s=-1.0,
            upper_s=1.0,
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("start_at_upper", 1),
        ("allow_clipped_duplicates", "yes"),
    ],
)
def test_quantile_hierarchy_rejects_nonboolean_controls(keyword, value):
    with pytest.raises(TypeError, match="boolean"):
        hierarchical_quantile_offsets(
            [0.0, 1.0],
            [-10.0, 10.0],
            5,
            **{keyword: value},
        )


@pytest.mark.parametrize(
    ("quantile_levels", "offset_quantiles_s", "message"),
    [
        ([0.0], [0.0], "at least two"),
        ([0.0, 0.5, 0.5], [0.0, -1.0, -2.0], "strictly increasing"),
        ([-0.1, 1.0], [0.0, -1.0], r"\[0, 1\]"),
        ([0.0, 1.0], [0.0, 0.0], "strictly monotone"),
        ([0.0, 0.5, 1.0], [0.0, -2.0, -1.0], "strictly monotone"),
    ],
)
def test_quantile_hierarchy_rejects_invalid_design_tables(
    quantile_levels,
    offset_quantiles_s,
    message,
):
    with pytest.raises(ValueError, match=message):
        hierarchical_quantile_offsets(
            quantile_levels,
            offset_quantiles_s,
            n_steps=5,
        )
