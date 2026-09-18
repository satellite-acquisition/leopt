"""Rest-to-rest antenna-mount feasibility for settled acquisition dwells.

Slews follow symmetric acceleration profiles along great-circle arcs. The
model includes rate, acceleration, settling, and dwell limits, without a
vendor-specific servo model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MountLimits:
    """Mechanical limits for rest-to-rest pointing transitions.

    All angular quantities are radians.  ``decision_interval_s`` is the time
    from one settled dwell start to the next; the interval must accommodate a
    rest-to-rest slew, settling, and the following dwell.
    """

    max_rate_rad_s: float
    max_accel_rad_s2: float
    settle_s: float
    dwell_s: float
    min_elevation_rad: float = 0.0
    max_elevation_rad: float = np.pi / 2.0

    def __post_init__(self) -> None:
        if self.max_rate_rad_s <= 0.0:
            raise ValueError("max_rate_rad_s must be positive")
        if self.max_accel_rad_s2 <= 0.0:
            raise ValueError("max_accel_rad_s2 must be positive")
        if self.settle_s < 0.0 or self.dwell_s <= 0.0:
            raise ValueError("settle_s must be nonnegative and dwell_s positive")
        if not (-np.pi / 2.0 <= self.min_elevation_rad < self.max_elevation_rad <= np.pi / 2.0):
            raise ValueError("invalid elevation limits")

    def motion_time_available(self, decision_interval_s: float) -> float:
        """Motion budget in seconds; negative if settling and dwell do not fit."""
        return decision_interval_s - self.settle_s - self.dwell_s


def great_circle_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Angular distance between two three-dimensional unit vectors."""
    first_u = np.asarray(first, dtype=float)
    second_u = np.asarray(second, dtype=float)
    first_u = first_u / np.linalg.norm(first_u)
    second_u = second_u / np.linalg.norm(second_u)
    return float(np.arccos(np.clip(first_u @ second_u, -1.0, 1.0)))


def boresight_elevation(vector: np.ndarray) -> float:
    """Elevation of an east--north--up unit vector in radians."""
    unit = np.asarray(vector, dtype=float)
    unit = unit / np.linalg.norm(unit)
    return float(np.arcsin(np.clip(unit[2], -1.0, 1.0)))


def _arc_basis(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a deterministic great-circle basis and arc length."""
    first_u = np.asarray(first, dtype=float)
    second_u = np.asarray(second, dtype=float)
    first_u = first_u / np.linalg.norm(first_u)
    second_u = second_u / np.linalg.norm(second_u)
    cosine = float(np.clip(first_u @ second_u, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    tangent = second_u - cosine * first_u
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-12:
        # The shortest arc is ambiguous only for antipodal vectors.  Pick a
        # stable orthogonal direction; either choice has the same length.
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(first_u)))] = 1.0
        tangent = axis - float(axis @ first_u) * first_u
        tangent_norm = float(np.linalg.norm(tangent))
    tangent /= tangent_norm
    return first_u, tangent, angle


def _arc_elevation_extrema(
    first: np.ndarray,
    tangent: np.ndarray,
    distance_rad: float,
) -> tuple[float, float]:
    """Minimum and maximum elevation along a great-circle prefix."""
    distance = max(0.0, float(distance_rad))
    candidates = [0.0, distance]
    stationary = float(np.arctan2(tangent[2], first[2]))
    for shift in range(-2, 3):
        location = stationary + shift * np.pi
        if 0.0 < location < distance:
            candidates.append(location)
    vertical = [first[2] * np.cos(value) + tangent[2] * np.sin(value) for value in candidates]
    return (
        float(np.arcsin(np.clip(min(vertical), -1.0, 1.0))),
        float(np.arcsin(np.clip(max(vertical), -1.0, 1.0))),
    )


def elevation_constraint_violation(
    first: np.ndarray,
    second: np.ndarray,
    limits: MountLimits,
) -> float:
    """Maximum elevation-limit violation along the shortest slew, in radians."""
    first_u, tangent, angle = _arc_basis(first, second)
    minimum, maximum = _arc_elevation_extrema(first_u, tangent, angle)
    return max(
        0.0,
        limits.min_elevation_rad - minimum,
        maximum - limits.max_elevation_rad,
    )


def _clamp_target_elevation(
    target: np.ndarray,
    reference: np.ndarray,
    limits: MountLimits,
) -> np.ndarray:
    """Project a requested ENU direction onto the allowed elevation interval."""
    target_u = np.asarray(target, dtype=float)
    target_u = target_u / np.linalg.norm(target_u)
    elevation = boresight_elevation(target_u)
    constrained = float(np.clip(elevation, limits.min_elevation_rad, limits.max_elevation_rad))
    if abs(constrained - elevation) <= 1.0e-15:
        return target_u
    horizontal = target_u[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1.0e-12:
        horizontal = np.asarray(reference, dtype=float)[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1.0e-12:
        horizontal = np.array([1.0, 0.0])
        horizontal_norm = 1.0
    direction = horizontal / horizontal_norm
    return np.array(
        [
            np.cos(constrained) * direction[0],
            np.cos(constrained) * direction[1],
            np.sin(constrained),
        ]
    )


def minimum_slew_time(angle_rad: float, limits: MountLimits) -> float:
    """Minimum rest-to-rest slew time for a nonnegative angular distance.

    The optimal one-axis profile is triangular below ``v_max**2 / a_max`` and
    trapezoidal above it.  Great-circle distance is used as the scalar motion
    coordinate for the synthetic mount.
    """
    angle = float(angle_rad)
    if angle < 0.0:
        raise ValueError("angle_rad must be nonnegative")
    if angle == 0.0:
        return 0.0
    rate = limits.max_rate_rad_s
    accel = limits.max_accel_rad_s2
    triangular_limit = rate * rate / accel
    if angle <= triangular_limit:
        return 2.0 * np.sqrt(angle / accel)
    return angle / rate + rate / accel


def maximum_slew_angle(duration_s: float, limits: MountLimits) -> float:
    """Invert :func:`minimum_slew_time` for a rest-to-rest time budget."""
    duration = max(0.0, float(duration_s))
    rate = limits.max_rate_rad_s
    accel = limits.max_accel_rad_s2
    time_to_reach_rate_and_stop = 2.0 * rate / accel
    if duration <= time_to_reach_rate_and_stop:
        return accel * duration * duration / 4.0
    return rate * duration - rate * rate / accel


def transition_feasible(
    first: np.ndarray,
    second: np.ndarray,
    limits: MountLimits,
    decision_interval_s: float,
) -> bool:
    """Whether the target can be reached, settled, and dwelled in one stage."""
    available = limits.motion_time_available(decision_interval_s)
    if available < -1.0e-12:
        return False
    if elevation_constraint_violation(first, second, limits) > 1.0e-12:
        return False
    distance = great_circle_distance(first, second)
    return minimum_slew_time(distance, limits) <= available + 1.0e-12


def transition_rate_acceleration_violations(
    first: np.ndarray,
    second: np.ndarray,
    limits: MountLimits,
    decision_interval_s: float,
) -> tuple[float, float]:
    """Lower-bound rate and acceleration excess for one rest-to-rest slew.

    The returned values are in radians/s and radians/s^2.  A zero pair is
    equivalent to satisfying the scalar slew envelope within the available
    motion time.  The calculation is independent of the command generator, so
    replaying a saved schedule can detect an accidentally overlong transition.
    Both residuals are infinite if settling and dwell alone exceed the interval.
    """
    duration = limits.motion_time_available(decision_interval_s)
    if duration < -1.0e-12:
        return float("inf"), float("inf")
    angle = great_circle_distance(first, second)
    if angle <= 1.0e-15:
        return 0.0, 0.0
    if duration <= 0.0:
        return float("inf"), float("inf")

    minimum_acceleration = 4.0 * angle / (duration * duration)
    acceleration_violation = max(0.0, minimum_acceleration - limits.max_accel_rad_s2)
    if acceleration_violation > 0.0:
        minimum_peak_rate = 2.0 * angle / duration
    else:
        discriminant = max(
            0.0,
            duration * duration - 4.0 * angle / limits.max_accel_rad_s2,
        )
        minimum_peak_rate = 0.5 * limits.max_accel_rad_s2 * (duration - np.sqrt(discriminant))
    rate_violation = max(0.0, minimum_peak_rate - limits.max_rate_rad_s)
    return float(rate_violation), float(acceleration_violation)


def move_toward(
    first: np.ndarray,
    target: np.ndarray,
    limits: MountLimits,
    decision_interval_s: float,
) -> np.ndarray:
    """Reach ``target`` or the farthest settled point toward it.

    Spherical linear interpolation avoids azimuth-wrap and zenith artifacts.
    The returned unit vector is the realized boresight at the next dwell.
    Raises ValueError if the interval cannot accommodate settling and dwell.
    """
    available = limits.motion_time_available(decision_interval_s)
    if available < -1.0e-12:
        raise ValueError("decision interval is shorter than settling plus dwell time")
    first_u = np.asarray(first, dtype=float)
    first_u = first_u / np.linalg.norm(first_u)
    first_elevation = boresight_elevation(first_u)
    if not (
        limits.min_elevation_rad - 1.0e-12 <= first_elevation <= limits.max_elevation_rad + 1.0e-12
    ):
        raise ValueError("the starting boresight violates the elevation limits")

    target_u = _clamp_target_elevation(target, first_u, limits)
    first_u, tangent, angle = _arc_basis(first_u, target_u)
    if angle <= 1.0e-15:
        return target_u
    reachable = maximum_slew_angle(available, limits)
    travel = min(angle, reachable)

    minimum, maximum = _arc_elevation_extrema(first_u, tangent, travel)
    if minimum < limits.min_elevation_rad - 1.0e-12 or maximum > limits.max_elevation_rad + 1.0e-12:
        # Stop at the first elevation boundary on the requested great-circle
        # path.  Prefix feasibility is monotone, so bisection is deterministic.
        lower = 0.0
        upper = travel
        for _ in range(60):
            midpoint = 0.5 * (lower + upper)
            minimum, maximum = _arc_elevation_extrema(first_u, tangent, midpoint)
            if (
                minimum >= limits.min_elevation_rad - 1.0e-13
                and maximum <= limits.max_elevation_rad + 1.0e-13
            ):
                lower = midpoint
            else:
                upper = midpoint
        travel = lower

    realized = np.cos(travel) * first_u + np.sin(travel) * tangent
    realized /= np.linalg.norm(realized)
    # Remove sub-ulp boundary drift without changing the intended azimuth.
    return _clamp_target_elevation(realized, first_u, limits)
