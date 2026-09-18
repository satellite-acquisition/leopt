"""Unit tests for the rest-to-rest mount model."""

from __future__ import annotations

import numpy as np
import pytest

from antenna_pomdp.control.mount import (
    MountLimits,
    boresight_elevation,
    elevation_constraint_violation,
    great_circle_distance,
    maximum_slew_angle,
    minimum_slew_time,
    move_toward,
    transition_feasible,
    transition_rate_acceleration_violations,
)


def _limits() -> MountLimits:
    return MountLimits(
        max_rate_rad_s=np.deg2rad(2.0),
        max_accel_rad_s2=np.deg2rad(1.0),
        settle_s=0.5,
        dwell_s=1.0,
    )


def _direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ]
    )


def test_triangular_and_trapezoidal_slew_times_match_closed_form() -> None:
    limits = _limits()
    assert minimum_slew_time(np.deg2rad(1.0), limits) == pytest.approx(2.0)
    assert minimum_slew_time(np.deg2rad(8.0), limits) == pytest.approx(6.0)
    assert maximum_slew_angle(2.0, limits) == pytest.approx(np.deg2rad(1.0))
    assert maximum_slew_angle(6.0, limits) == pytest.approx(np.deg2rad(8.0))


@pytest.mark.parametrize("target", [np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])])
def test_settling_and_dwell_must_fit_even_without_motion(target: np.ndarray) -> None:
    limits = _limits()
    start = np.array([0.0, 1.0, 0.0])
    interval = 1.0

    assert limits.motion_time_available(interval) < 0.0
    assert not transition_feasible(start, target, limits, interval)
    assert transition_rate_acceleration_violations(start, target, limits, interval) == (
        float("inf"),
        float("inf"),
    )
    with pytest.raises(ValueError, match="settling plus dwell"):
        move_toward(start, target, limits, interval)


def test_stationary_dwell_fits_at_exact_settling_and_dwell_budget() -> None:
    limits = _limits()
    start = np.array([0.0, 1.0, 0.0])
    interval = limits.settle_s + limits.dwell_s

    assert limits.motion_time_available(interval) == 0.0
    assert transition_feasible(start, start, limits, interval)
    assert transition_rate_acceleration_violations(start, start, limits, interval) == (0.0, 0.0)
    np.testing.assert_allclose(move_toward(start, start, limits, interval), start)


def test_move_toward_respects_stage_budget() -> None:
    limits = _limits()
    first = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    realized = move_toward(first, target, limits, decision_interval_s=4.5)
    available = limits.motion_time_available(4.5)
    assert minimum_slew_time(great_circle_distance(first, realized), limits) <= available + 1e-12
    assert great_circle_distance(realized, target) > 0.0
    assert not transition_feasible(first, target, limits, decision_interval_s=4.5)


def test_elevation_limits_reject_endpoint_and_great_circle_excursions() -> None:
    limits = MountLimits(
        max_rate_rad_s=np.deg2rad(100.0),
        max_accel_rad_s2=np.deg2rad(100.0),
        settle_s=0.0,
        dwell_s=1.0,
        min_elevation_rad=np.deg2rad(10.0),
        max_elevation_rad=np.deg2rad(85.0),
    )
    start = _direction(0.0, 20.0)
    assert not transition_feasible(
        start,
        _direction(20.0, 5.0),
        limits,
        decision_interval_s=10.0,
    )

    first = _direction(0.0, 80.0)
    second = _direction(180.0, 80.0)
    assert elevation_constraint_violation(first, second, limits) == pytest.approx(np.deg2rad(5.0))
    assert not transition_feasible(first, second, limits, decision_interval_s=10.0)


def test_partial_slew_stays_inside_elevation_and_kinematic_envelopes() -> None:
    limits = MountLimits(
        max_rate_rad_s=np.deg2rad(2.0),
        max_accel_rad_s2=np.deg2rad(1.0),
        settle_s=0.5,
        dwell_s=1.0,
        min_elevation_rad=np.deg2rad(10.0),
        max_elevation_rad=np.deg2rad(85.0),
    )
    start = _direction(0.0, 20.0)
    requested = _direction(50.0, -15.0)
    realized = move_toward(start, requested, limits, decision_interval_s=4.5)

    elevation = boresight_elevation(realized)
    assert limits.min_elevation_rad <= elevation <= limits.max_elevation_rad
    assert transition_feasible(start, realized, limits, decision_interval_s=4.5)
    rate_violation, acceleration_violation = transition_rate_acceleration_violations(
        start,
        realized,
        limits,
        decision_interval_s=4.5,
    )
    assert rate_violation <= 1.0e-12
    assert acceleration_violation <= 1.0e-12


def test_replayed_kinematic_residuals_detect_an_overlong_transition() -> None:
    limits = _limits()
    first = _direction(0.0, 20.0)
    second = _direction(0.0, 30.0)
    rate_violation, acceleration_violation = transition_rate_acceleration_violations(
        first,
        second,
        limits,
        decision_interval_s=4.5,
    )
    assert rate_violation > 0.0
    assert acceleration_violation > 0.0
