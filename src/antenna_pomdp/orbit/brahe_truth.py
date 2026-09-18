"""Deterministic Brahe SGP4 bridge for synthetic truth checks.

The maintained search engine propagates TLE particles with ``python-sgp4``
and converts their native TEME states to topocentric coordinates with a
lightweight GMST rotation.  Brahe exposes its SGP4 result through transformed
inertial and Earth-fixed frames, so comparing the two libraries' raw inertial
vectors would mix reference frames.  This module instead provides the common
observable needed by the acquisition problem: azimuth, elevation, and range
from a specified ground station.

Brahe's convenience initializers may refresh stale Earth-orientation and space-
weather files over the network.  Synthetic tests must be deterministic and
offline, so :func:`configure_static_brahe_environment` installs fixed providers
that never download data.  These constants are a synthetic convention, not an
operational-fidelity model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import brahe as bh
import numpy as np

from antenna_pomdp.config import StationConfig


# Public specifications let evidence manifests record the exact offline
# convention without duplicating values in an experiment driver.
STATIC_EOP_CONVENTION = {
    "provider": "StaticEOPProvider.from_zero",
    "values": "all Earth-orientation corrections zero",
}
STATIC_SPACE_WEATHER_VALUES = {
    "kp": 3.0,
    "ap": 15.0,
    "f107": 150.0,
    "f107a": 145.0,
    "s": 100,
}


def configure_static_brahe_environment() -> None:
    """Install deterministic, network-free global providers in Brahe.

    Brahe 1.3.2 requires a global EOP provider for its frame transforms.  A
    static all-zero EOP model is consistent with the research engine's
    low-precision ``UT1 approximately UTC`` geometry.  Fixed space-weather
    values make subsequent ``leo_default`` numerical propagation reproducible
    without calling ``initialize_eop`` or ``initialize_sw``.

    The function is intentionally idempotent: each call replaces the process-
    global providers with new providers carrying the same fixed values.
    """

    bh.set_global_eop_provider(bh.StaticEOPProvider.from_zero())
    bh.set_global_space_weather_provider(
        bh.StaticSpaceWeatherProvider.from_values(**STATIC_SPACE_WEATHER_VALUES)
    )


def _as_brahe_epoch(when: datetime) -> bh.Epoch:
    """Convert a Python datetime to a Brahe UTC epoch."""

    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    else:
        when = when.astimezone(timezone.utc)
    return bh.Epoch.from_unix_timestamp(when.timestamp())


@dataclass(frozen=True)
class TopocentricState:
    """Brahe-propagated line of sight in SI/radian units."""

    epoch: datetime
    az_rad: float
    el_rad: float
    range_m: float


class BraheSgp4Topocentric:
    """Propagate one TLE with Brahe SGP4 and return ground-relative geometry."""

    def __init__(
        self,
        tle_line1: str,
        tle_line2: str,
        station: StationConfig,
        *,
        configure_environment: bool = True,
    ) -> None:
        if configure_environment:
            configure_static_brahe_environment()
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.station = station
        self._propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2)
        # Brahe's PointLocation constructor is (longitude, latitude, altitude).
        self._location = bh.PointLocation(
            station.longitude_deg,
            station.latitude_deg,
            station.altitude_m,
        )

    def azel_range(self, when: datetime) -> TopocentricState:
        """Return azimuth, elevation, and range at ``when``.

        ``state_itrf`` performs Brahe's internally consistent TEME-to-fixed
        transformation.  The final ENZ-to-az/el conversion uses radians, while
        its third component remains range in metres.
        """

        epoch = _as_brahe_epoch(when)
        state_itrf = np.asarray(self._propagator.state_itrf(epoch), dtype=float)
        station_ecef = np.asarray(self._location.center_ecef(), dtype=float)
        relative_enz = bh.relative_position_ecef_to_enz(
            station_ecef,
            state_itrf[:3],
            bh.EllipsoidalConversionType.GEODETIC,
        )
        azel = np.asarray(
            bh.position_enz_to_azel(relative_enz, bh.AngleFormat.RADIANS),
            dtype=float,
        )
        return TopocentricState(
            epoch=(
                when.replace(tzinfo=timezone.utc)
                if when.tzinfo is None
                else when.astimezone(timezone.utc)
            ),
            az_rad=float(azel[0]) % (2.0 * np.pi),
            el_rad=float(azel[1]),
            range_m=float(azel[2]),
        )


def brahe_sgp4_azel_range(
    tle_line1: str,
    tle_line2: str,
    when: datetime,
    station: StationConfig,
) -> tuple[float, float, float]:
    """One-shot convenience wrapper returning ``(az, el, range)``."""

    state = BraheSgp4Topocentric(tle_line1, tle_line2, station).azel_range(when)
    return state.az_rad, state.el_rad, state.range_m
