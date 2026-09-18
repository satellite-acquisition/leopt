"""Thin SGP4 wrapper used by the rest of the package.

We deliberately keep this module minimal: parse a TLE, propagate to a datetime,
return ECI (TEME) position and velocity in meters / meters-per-second. Higher
layers handle frames and az/el.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, jday


@dataclass(frozen=True)
class OrbitState:
    """ECI (TEME) position / velocity at a specific UTC epoch, SI units."""

    epoch: datetime
    r_eci_m: np.ndarray  # shape (3,)
    v_eci_m_s: np.ndarray  # shape (3,)


class SGP4Propagator:
    """Wraps a single `sgp4.api.Satrec` so callers don't touch JD arithmetic."""

    def __init__(self, tle_line1: str, tle_line2: str):
        self._sat = Satrec.twoline2rv(tle_line1, tle_line2)
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2

    @property
    def epoch(self) -> datetime:
        """TLE reference epoch as timezone-aware UTC datetime."""
        # `sgp4` exposes `jdsatepoch + jdsatepochF` as Julian Date.
        jd = self._sat.jdsatepoch + self._sat.jdsatepochF
        # JD 2440587.5 = 1970-01-01T00:00:00 UTC
        unix_seconds = (jd - 2440587.5) * 86400.0
        return datetime.fromtimestamp(unix_seconds, tz=timezone.utc)

    def propagate(self, when: datetime) -> OrbitState:
        """Propagate to UTC `when`. Raises if SGP4 reports a propagation error."""
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        jd, fr = jday(
            when.year,
            when.month,
            when.day,
            when.hour,
            when.minute,
            when.second + when.microsecond * 1e-6,
        )
        err, r_km, v_km_s = self._sat.sgp4(jd, fr)
        if err != 0:
            raise RuntimeError(f"SGP4 propagation error code {err} at {when.isoformat()}")
        return OrbitState(
            epoch=when,
            r_eci_m=np.asarray(r_km, dtype=float) * 1e3,
            v_eci_m_s=np.asarray(v_km_s, dtype=float) * 1e3,
        )

    def propagate_many(self, epochs: list[datetime]) -> list[OrbitState]:
        """Vectorless convenience helper. Returns one state per requested epoch."""
        return [self.propagate(e) for e in epochs]


def propagate_grid(
    tle_line1: str,
    tle_line2: str,
    start: datetime,
    duration_s: float,
    step_s: float,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Convenience: propagate to a uniform time grid; returns (epochs, R, V)."""
    prop = SGP4Propagator(tle_line1, tle_line2)
    n = int(round(duration_s / step_s)) + 1
    epochs = [start + timedelta(seconds=i * step_s) for i in range(n)]
    states = prop.propagate_many(epochs)
    R = np.stack([s.r_eci_m for s in states])
    V = np.stack([s.v_eci_m_s for s in states])
    return epochs, R, V


if __name__ == "__main__":
    from antenna_pomdp.config import default_config

    cfg = default_config()
    prop = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    s = prop.propagate(prop.epoch + timedelta(minutes=5))
    r = np.linalg.norm(s.r_eci_m)
    v = np.linalg.norm(s.v_eci_m_s)
    assert 6.5e6 < r < 8.0e6, f"unexpected radius {r:.0f} m"
    assert 6.5e3 < v < 8.5e3, f"unexpected speed {v:.0f} m/s"
    print(f"[propagator] |r| = {r / 1e3:.1f} km, |v| = {v / 1e3:.3f} km/s")
    print("[propagator] PASS")
