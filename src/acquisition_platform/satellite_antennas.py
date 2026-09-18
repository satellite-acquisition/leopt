"""Spacecraft antenna presets and a heuristic acquisition hold time.

The model uses ``clip((1 - coverage) * T_tumble, HOLD_FLOOR_S, HOLD_CAP_S)``
with a representative 120 s tumble period and a 30 s minimum hold. Coverage is
the approximate fraction of the sphere with usable gain. A low-elevation margin
extends the hold for weak passes. The station is assumed to track the predicted
trajectory throughout; the model does not guarantee a link.
"""

from __future__ import annotations

from dataclasses import dataclass

# Representative LEOP tumble period (s); a few deg/s post-separation tip-off.
T_TUMBLE_S = 120.0
HOLD_FLOOR_S = 30.0  # minimum hold to establish lock / acquire frames
HOLD_CAP_S = 60.0  # cap the nominal (high-elevation) hold
HOLD_MAX_S = 90.0  # absolute cap after the low-elevation margin


@dataclass(frozen=True)
class SatAntenna:
    """A spacecraft-side antenna configuration option."""

    id: str
    name: str
    description: str
    coverage: float  # approx fraction of 4*pi sphere with usable gain
    base_hold_s: float  # nominal hold (high elevation) to ensure a link


def _base_hold(coverage: float) -> float:
    raw = (1.0 - coverage) * T_TUMBLE_S
    return float(min(HOLD_CAP_S, max(HOLD_FLOOR_S, round(raw))))


# Ordered least -> most coverage (i.e. longest -> shortest hold). Default is the
# turnstile, the most common LEOP TT&C antenna.
_DEFS = [
    (
        "patch_single",
        "Single patch (directional)",
        "Body-mounted microstrip patch; roughly hemispherical with deep nulls "
        "when tumbling. Reliable only with attitude control.",
        0.30,
    ),
    (
        "monopole",
        "Deployable monopole / dipole",
        "Single tape-spring monopole or dipole (typically UHF); toroidal "
        "pattern with nulls along the antenna axis.",
        0.60,
    ),
    (
        "turnstile",
        "Turnstile / crossed dipole (near-omni)",
        "Four monopoles phased to a circularly polarized near-omni pattern; "
        "the standard LEOP TT&C antenna for tumbling spacecraft.",
        0.85,
    ),
    (
        "canted_turnstile",
        "Canted turnstile (CP, no blind spots)",
        "Canted turnstile (e.g. GomSpace ANT430 class); near-omni circular "
        "polarization engineered to avoid fade during tumble.",
        0.88,
    ),
    (
        "dual_omni",
        "Dual antennas (full-sphere)",
        "Two antennas on opposite faces, combined or switched for near full-sphere coverage.",
        0.92,
    ),
]

SAT_ANTENNAS: dict[str, SatAntenna] = {
    aid: SatAntenna(aid, name, desc, cov, _base_hold(cov)) for (aid, name, desc, cov) in _DEFS
}

DEFAULT_SAT_ANTENNA = "turnstile"


def list_sat_antennas() -> list[SatAntenna]:
    return list(SAT_ANTENNAS.values())


def get_sat_antenna(aid: str) -> SatAntenna:
    return SAT_ANTENNAS.get(aid, SAT_ANTENNAS[DEFAULT_SAT_ANTENNA])


def recommended_hold_s(ant: SatAntenna, elevation_deg: float) -> float:
    """Empirical hold (s) for this antenna at a given pass elevation.

    Applies a low-elevation margin (up to +25% below 20 deg) to the antenna's
    nominal hold and clamps to a sane operational range.
    """
    margin = 1.0 + 0.25 * max(0.0, min(1.0, (20.0 - elevation_deg) / 20.0))
    return float(min(HOLD_MAX_S, max(HOLD_FLOOR_S, ant.base_hold_s * margin)))


if __name__ == "__main__":
    for a in list_sat_antennas():
        hi = recommended_hold_s(a, 45.0)
        lo = recommended_hold_s(a, 5.0)
        print(
            f"  {a.id:18s} coverage={a.coverage:.2f}  hold {hi:.0f}s (hi-el) .. {lo:.0f}s (lo-el)"
        )
    assert get_sat_antenna("turnstile").id == "turnstile"
    assert get_sat_antenna("nope").id == DEFAULT_SAT_ANTENNA  # falls back
    assert recommended_hold_s(SAT_ANTENNAS["patch_single"], 45.0) >= 30.0
    print("[satellite_antennas] PASS")
