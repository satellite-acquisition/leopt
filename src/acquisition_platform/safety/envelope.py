"""Classify pointing commands against antenna and transmit constraints.

Checks cover the mechanical elevation floor, operational elevation mask,
zenith keyhole, Sun exclusion cones, and RF no-transmit sectors. A command can
be rejected, allowed with a warning, or allowed with transmit inhibited.

Defaults represent a small S-band dish. Configure limits for the actual
antenna; these values do not establish hardware safety or certification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from antenna_pomdp.orbit.geometry import Station, _julian_date, angular_separation, eci_to_azel

# Verdict severity ladder (higher = worse). `ok` and `warn` still actuate and
# transmit; `keyhole` actuates (flagged) and transmits; `rf_inhibit` actuates but
# does not transmit; `reject` does neither.
_SEVERITY = {"ok": 0, "warn": 1, "keyhole": 2, "rf_inhibit": 3, "reject": 4}
_AU_M = 1.495978707e11


@dataclass(frozen=True)
class KeepOutBox:
    """An az/el sector keep-out. `transmit_only` => track is fine, no keying."""

    az_min_deg: float
    az_max_deg: float
    el_min_deg: float
    el_max_deg: float
    name: str = "keep-out"
    transmit_only: bool = True

    def contains(self, az_deg: float, el_deg: float) -> bool:
        if not (self.el_min_deg <= el_deg <= self.el_max_deg):
            return False
        # A full (or over-full) azimuth span is an all-azimuth ring; guard it
        # before the modulo, which would otherwise map 360 -> 0 and match only az=0.
        if (self.az_max_deg - self.az_min_deg) >= 360.0:
            return True
        a = az_deg % 360.0
        lo = self.az_min_deg % 360.0
        hi = self.az_max_deg % 360.0
        return lo <= a <= hi if lo <= hi else (a >= lo or a <= hi)  # wrap-safe


@dataclass(frozen=True)
class SafetyConfig:
    """Per-antenna pointing envelope. Degrees throughout. Defaults: 3-5 m S-band."""

    hard_el_lower_deg: float = 3.0  # mechanical floor — below this is REJECT
    min_elevation_deg: float = 5.0  # operational mask — below this is RF_INHIBIT
    keyhole_half_angle_deg: float = 3.0  # continuous-track cap at 90 - this
    sun_damage_half_angle_deg: float = 5.0  # HARD reject cone around the Sun
    sun_avoid_half_angle_deg: float = 20.0  # WARN (degraded / G/T loss) cone
    max_slew_rate_deg_s: float = 10.0  # documented for context / rate checks
    keep_out: tuple[KeepOutBox, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # The operational mask cannot sit below the mechanical floor.
        if self.min_elevation_deg < self.hard_el_lower_deg:
            object.__setattr__(self, "min_elevation_deg", self.hard_el_lower_deg)


@dataclass(frozen=True)
class EnvelopeVerdict:
    """The result of classifying one pointing against the envelope."""

    status: str  # ok | warn | keyhole | rf_inhibit | reject
    actuatable: bool  # is it safe to slew/point the pedestal here?
    transmit_ok: bool  # is it safe to key the transmitter here?
    severity: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "actuatable": self.actuatable,
            "transmit_ok": self.transmit_ok,
            "severity": self.severity,
            "reasons": list(self.reasons),
        }


def classify_pointing(
    az_deg: float,
    el_deg: float,
    cfg: SafetyConfig,
    sun_sep_deg: float | None = None,
) -> EnvelopeVerdict:
    """Classify a pointing (degrees) against the envelope. Pure / offline.

    `sun_sep_deg` is the angular separation between the boresight and the Sun; if
    None the sun checks are skipped (no ephemeris available).
    """
    reasons: list[str] = []
    status = "ok"

    def raise_to(s: str, why: str) -> None:
        nonlocal status
        reasons.append(why)
        if _SEVERITY[s] > _SEVERITY[status]:
            status = s

    if el_deg < cfg.hard_el_lower_deg:
        raise_to(
            "reject",
            f"elevation {el_deg:.1f}° below mechanical floor {cfg.hard_el_lower_deg:.1f}°",
        )
    elif el_deg < cfg.min_elevation_deg:
        raise_to(
            "rf_inhibit",
            f"elevation {el_deg:.1f}° below transmit mask {cfg.min_elevation_deg:.1f}°",
        )

    keyhole_cap = 90.0 - cfg.keyhole_half_angle_deg
    if el_deg > keyhole_cap:
        raise_to(
            "keyhole",
            f"elevation {el_deg:.1f}° inside the {cfg.keyhole_half_angle_deg:.1f}° zenith keyhole",
        )

    if sun_sep_deg is not None:
        if sun_sep_deg < cfg.sun_damage_half_angle_deg:
            raise_to(
                "reject",
                f"boresight {sun_sep_deg:.1f}° from Sun — inside "
                f"{cfg.sun_damage_half_angle_deg:.1f}° damage cone",
            )
        elif sun_sep_deg < cfg.sun_avoid_half_angle_deg:
            raise_to(
                "warn",
                f"boresight {sun_sep_deg:.1f}° from Sun — inside "
                f"{cfg.sun_avoid_half_angle_deg:.1f}° avoidance cone (degraded G/T)",
            )

    for box in cfg.keep_out:
        if box.contains(az_deg, el_deg):
            # transmit_only => track-ok/no-keying (rf_inhibit); a hard keep-out
            # (transmit_only=False) bars the pedestal from slewing in at all.
            if box.transmit_only:
                raise_to("rf_inhibit", f"inside transmit keep-out '{box.name}'")
            else:
                raise_to("reject", f"inside hard keep-out '{box.name}'")

    actuatable = status != "reject"
    transmit_ok = status not in ("reject", "rf_inhibit")
    return EnvelopeVerdict(
        status=status,
        actuatable=actuatable,
        transmit_ok=transmit_ok,
        severity=_SEVERITY[status],
        reasons=tuple(reasons) or ("within envelope",),
    )


def _sun_eci_m(when: datetime) -> np.ndarray:
    """Low-precision geocentric solar position in ECI (metres).

    Astronomical-Almanac low-precision formula (~0.01° accuracy) — far finer than
    the multi-degree keep-out cones. At 1 AU the station offset is negligible, so
    the topocentric direction equals this geocentric one for keep-out purposes.
    """
    n = _julian_date(when) - 2451545.0
    mean_lon = (280.460 + 0.9856474 * n) % 360.0
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
    lam = np.deg2rad(mean_lon + 1.915 * np.sin(g) + 0.020 * np.sin(2.0 * g))
    eps = np.deg2rad(23.439 - 4.0e-7 * n)
    u = np.array([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)])
    return _AU_M * u


def sun_azel(when: datetime, station: Station) -> tuple[float, float]:
    """Apparent Sun (az, el) in radians at a station (low precision)."""
    az, el, _ = eci_to_azel(_sun_eci_m(when), when, station)
    return az, el


def evaluate_pointing(
    az_rad: float,
    el_rad: float,
    when: datetime,
    station: Station,
    cfg: SafetyConfig,
) -> EnvelopeVerdict:
    """Classify a session pointing (radians) at a time/station, sun checks on."""
    s_az, s_el = sun_azel(when, station)
    sun_sep_deg = float(np.rad2deg(angular_separation(az_rad, el_rad, s_az, s_el)))
    return classify_pointing(
        float(np.rad2deg(az_rad)), float(np.rad2deg(el_rad)), cfg, sun_sep_deg=sun_sep_deg
    )


if __name__ == "__main__":
    from datetime import timezone

    from antenna_pomdp.config import StationConfig

    cfg = SafetyConfig()
    # Below the mechanical floor -> reject.
    v = classify_pointing(120.0, 2.0, cfg)
    assert v.status == "reject" and not v.actuatable, v
    # Between floor and mask -> rf_inhibit (track ok, no transmit).
    v = classify_pointing(120.0, 4.0, cfg)
    assert v.status == "rf_inhibit" and v.actuatable and not v.transmit_ok, v
    # Near zenith -> keyhole.
    v = classify_pointing(120.0, 88.5, cfg)
    assert v.status == "keyhole" and v.actuatable and v.transmit_ok, v
    # Straight at the Sun -> hard reject.
    v = classify_pointing(120.0, 45.0, cfg, sun_sep_deg=2.0)
    assert v.status == "reject" and not v.actuatable, v
    # Near the Sun -> warn.
    v = classify_pointing(120.0, 45.0, cfg, sun_sep_deg=12.0)
    assert v.status == "warn" and v.actuatable and v.transmit_ok, v
    # Keep-out box -> rf_inhibit.
    cfg2 = SafetyConfig(keep_out=(KeepOutBox(100.0, 140.0, 0.0, 30.0, name="GEO-arc"),))
    v = classify_pointing(120.0, 20.0, cfg2)
    assert v.status == "rf_inhibit" and "GEO-arc" in v.reasons[0], v
    # Clean pointing -> ok.
    v = classify_pointing(120.0, 45.0, cfg, sun_sep_deg=90.0)
    assert v.status == "ok" and v.actuatable and v.transmit_ok, v

    # Sun ephemeris sanity: the Sun should be up around local noon somewhere.
    st = Station.from_config(StationConfig(latitude_deg=0.0, longitude_deg=0.0))
    az, el = sun_azel(datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc), st)
    assert -np.pi / 2 <= el <= np.pi / 2
    print(f"[envelope] sun at equator noon-ish: el={np.rad2deg(el):.1f}°")
    print("[envelope] PASS")
