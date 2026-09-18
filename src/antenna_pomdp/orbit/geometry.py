"""Observer geometry: TEME to ECEF to topocentric azimuth/elevation.

Earth rotation uses GMST, without precession, nutation, or polar motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from antenna_pomdp.config import StationConfig
from antenna_pomdp.orbit.propagator import SGP4Propagator

_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


# ---------------------------------------------------------------------------
# Time + Earth rotation
# ---------------------------------------------------------------------------


def _julian_date(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    unix = dt.timestamp()
    return unix / 86400.0 + 2440587.5


def gmst_rad(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU 1982, low-precision form)."""
    jd = _julian_date(dt)
    T = (jd - 2451545.0) / 36525.0
    # seconds; Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    gmst_rad_val = (gmst_s % 86400.0) * (2.0 * np.pi / 86400.0)
    return gmst_rad_val % (2.0 * np.pi)


def rot_eci_to_ecef(dt: datetime) -> np.ndarray:
    """3x3 rotation matrix from TEME-treated-as-ECI to ECEF (no polar motion)."""
    theta = gmst_rad(dt)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Station geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Station:
    """Geodetic + ECEF station representation."""

    latitude_rad: float
    longitude_rad: float
    altitude_m: float
    ecef_m: np.ndarray  # shape (3,)
    rot_ecef_to_enu: np.ndarray  # shape (3, 3)

    @classmethod
    def from_config(cls, cfg: StationConfig) -> "Station":
        lat = np.deg2rad(cfg.latitude_deg)
        lon = np.deg2rad(cfg.longitude_deg)
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
        x = (N + cfg.altitude_m) * np.cos(lat) * np.cos(lon)
        y = (N + cfg.altitude_m) * np.cos(lat) * np.sin(lon)
        z = (N * (1.0 - _WGS84_E2) + cfg.altitude_m) * np.sin(lat)
        # ECEF -> ENU rotation
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        R = np.array(
            [
                [-sin_lon, cos_lon, 0.0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ]
        )
        return cls(
            latitude_rad=lat,
            longitude_rad=lon,
            altitude_m=cfg.altitude_m,
            ecef_m=np.array([x, y, z]),
            rot_ecef_to_enu=R,
        )


# ---------------------------------------------------------------------------
# az / el
# ---------------------------------------------------------------------------


def eci_to_azel(
    r_eci_m: np.ndarray, when: datetime, station: Station
) -> tuple[float, float, float]:
    """Return (azimuth, elevation, range) in radians/radians/meters."""
    R = rot_eci_to_ecef(when)
    r_ecef = R @ r_eci_m
    rho_ecef = r_ecef - station.ecef_m
    enu = station.rot_ecef_to_enu @ rho_ecef
    e, n, u = enu
    rng = float(np.linalg.norm(enu))
    el = float(np.arcsin(u / rng))
    az = float(np.arctan2(e, n)) % (2.0 * np.pi)
    return az, el, rng


def azel_unit_vector(az: float, el: float) -> np.ndarray:
    """ENU unit vector for a given az/el (rad)."""
    return np.array(
        [
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el),
        ]
    )


def angular_separation(az1: float, el1: float, az2: float, el2: float) -> float:
    """Great-circle angular separation between two az/el directions (rad)."""
    v1 = azel_unit_vector(az1, el1)
    v2 = azel_unit_vector(az2, el2)
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Pass window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassWindow:
    """Acquired by `find_next_pass`: open / close times above min elevation."""

    open: datetime
    close: datetime
    peak: datetime
    peak_elevation_rad: float


def find_next_pass(
    propagator: SGP4Propagator,
    station: Station,
    start: datetime,
    *,
    min_elevation_rad: float,
    search_horizon_s: float = 6 * 3600.0,
    coarse_step_s: float = 30.0,
) -> PassWindow | None:
    """Find the next pass of `propagator` above `station`. None if none found.

    Linear scan with a coarse step, then a fine step to bracket open/close.
    Uses a fixed sampling step; not optimized for
    deep-space.
    """
    n = int(search_horizon_s / coarse_step_s) + 1
    above = False
    open_t: datetime | None = None
    peak_t: datetime | None = None
    peak_el = -np.pi
    for i in range(n):
        t = start + timedelta(seconds=i * coarse_step_s)
        s = propagator.propagate(t)
        _, el, _ = eci_to_azel(s.r_eci_m, t, station)
        if el > min_elevation_rad and not above:
            above = True
            open_t = t
            peak_t = t
            peak_el = el
        elif el > peak_el and above:
            peak_el = el
            peak_t = t
        elif el < min_elevation_rad and above:
            return PassWindow(open=open_t, close=t, peak=peak_t, peak_elevation_rad=peak_el)
    return None


if __name__ == "__main__":
    from antenna_pomdp.config import default_config

    cfg = default_config()
    station = Station.from_config(cfg.station)
    prop = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)

    # Geometry sanity: zenith vector at the station should map to el = 90 deg.
    R = rot_eci_to_ecef(prop.epoch)
    zenith_ecef = station.ecef_m / np.linalg.norm(station.ecef_m)
    zenith_eci = R.T @ (station.ecef_m + zenith_ecef * 1e6)  # 1000 km up
    _, el, _ = eci_to_azel(zenith_eci, prop.epoch, station)
    assert el > np.deg2rad(89.0), f"zenith elevation should be ~90 deg, got {np.rad2deg(el):.3f}"

    # Try to find a pass.
    window = find_next_pass(
        prop,
        station,
        prop.epoch,
        min_elevation_rad=np.deg2rad(cfg.station.min_elevation_deg),
    )
    if window is not None:
        dur = (window.close - window.open).total_seconds()
        print(
            f"[geometry] next pass: {window.open.isoformat()} "
            f"(peak el {np.rad2deg(window.peak_elevation_rad):.1f} deg, "
            f"dur {dur:.0f} s)"
        )
    else:
        print("[geometry] no pass within search horizon (nominal TLE is illustrative only)")
    print("[geometry] PASS")
