"""Export offline pointing tracks and propagated orbit ephemerides.

``to_csv`` and ``to_ccsds_pointing_kvn`` emit azimuth/elevation tracks.
``to_oem_kvn`` propagates a TLE into a CCSDS OEM for consumers that compute
their own pointing. Exports do not send commands to hardware.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from acquisition_platform.drivers.base import TrackSample

if TYPE_CHECKING:  # pragma: no cover - typing only
    from antenna_pomdp.orbit.propagator import SGP4Propagator

_CSV_HEADER = "time_utc,az_deg,el_deg,az_rate_deg_s,el_rate_deg_s"


def _iso(dt: datetime) -> str:
    """ISO-8601 UTC with a trailing Z (CCSDS-style epoch)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_csv(track: list[TrackSample]) -> str:
    """Serialize a program track as CSV with an ISO-8601 epoch column."""
    lines = [_CSV_HEADER]
    for s in track:
        lines.append(
            f"{_iso(s.t)},{s.az_deg:.6f},{s.el_deg:.6f},{s.az_rate_deg_s:.6f},{s.el_rate_deg_s:.6f}"
        )
    return "\n".join(lines) + "\n"


def to_ccsds_pointing_kvn(
    track: list[TrackSample],
    object_name: str = "SPACECRAFT",
    originator: str = "ACQUISITION_PLATFORM",
) -> str:
    """A CCSDS-flavored KVN pointing/program-track message.

    Mirrors the CCSDS APM/pointing convention: a COMMENT header, a META block
    (ORIGINATOR, OBJECT_NAME, TIME_SYSTEM=UTC), then `<epoch> <az_deg> <el_deg>`
    data rows in az/el degrees. This is a pragmatic, self-describing text load
    for an ACU that consumes a pointing track; it is NOT a full NDM document.
    """
    now = _iso(datetime.now(tz=timezone.utc))
    lines = [
        "COMMENT CCSDS-flavored antenna pointing / program-track message",
        "COMMENT Rows: EPOCH AZ_DEG EL_DEG (topocentric az/el, degrees)",
        f"CREATION_DATE = {now}",
        f"ORIGINATOR = {originator}",
        "META_START",
        f"OBJECT_NAME = {object_name}",
        "TIME_SYSTEM = UTC",
        "ANGLE_TYPE = AZEL",
        "META_STOP",
    ]
    for s in track:
        lines.append(f"{_iso(s.t)} {s.az_deg:.6f} {s.el_deg:.6f}")
    return "\n".join(lines) + "\n"


def to_oem_kvn(
    propagator: SGP4Propagator,
    times: list[datetime],
    object_name: str = "SPACECRAFT",
    object_id: str = "0000-000A",
) -> str:
    """A real CCSDS OEM (Orbit Ephemeris Message, 502.0-B) in KVN form.

    Propagates `propagator` to each requested time and emits CCSDS-unit rows
    `EPOCH X Y Z X_DOT Y_DOT Z_DOT` in km and km/s under a proper META block
    (CCSDS_OEM_VERS=2.0, REF_FRAME=TEME, CENTER_NAME=EARTH). This is the
    orbit-ephemeris form an ACU that does its own pointing consumes.
    """
    if not times:
        raise ValueError("to_oem_kvn requires at least one epoch")
    ordered = sorted(times)
    now = _iso(datetime.now(tz=timezone.utc))
    header = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE = {now}",
        "ORIGINATOR = ACQUISITION_PLATFORM",
        "META_START",
        f"OBJECT_NAME = {object_name}",
        f"OBJECT_ID = {object_id}",
        "CENTER_NAME = EARTH",
        "REF_FRAME = TEME",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {_iso(ordered[0])}",
        f"STOP_TIME = {_iso(ordered[-1])}",
        "META_STOP",
    ]
    rows: list[str] = []
    for t in ordered:
        st = propagator.propagate(t)
        r_km = st.r_eci_m / 1e3
        v_km_s = st.v_eci_m_s / 1e3
        rows.append(
            f"{_iso(t)} "
            f"{r_km[0]:.6f} {r_km[1]:.6f} {r_km[2]:.6f} "
            f"{v_km_s[0]:.9f} {v_km_s[1]:.9f} {v_km_s[2]:.9f}"
        )
    return "\n".join(header + rows) + "\n"


if __name__ == "__main__":
    from datetime import timedelta

    from antenna_pomdp.orbit.propagator import SGP4Propagator

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    track = [
        TrackSample(t0, 350.0, 10.0, 0.5, 1.0),
        TrackSample(t0 + timedelta(seconds=1), 351.0, 11.0, 0.5, 1.0),
    ]
    csv = to_csv(track)
    assert csv.startswith(_CSV_HEADER) and csv.count("\n") == 3
    kvn = to_ccsds_pointing_kvn(track, object_name="TESTSAT")
    assert "META_START" in kvn and "TIME_SYSTEM = UTC" in kvn
    assert "OBJECT_NAME = TESTSAT" in kvn

    l1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
    l2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    prop = SGP4Propagator(l1, l2)
    times = [prop.epoch + timedelta(seconds=60 * i) for i in range(3)]
    oem = to_oem_kvn(prop, times, object_name="TESTSAT", object_id="2024-001A")
    assert "CCSDS_OEM_VERS = 2.0" in oem and "REF_FRAME = TEME" in oem
    # Data rows: epoch token + 3 position + 3 velocity = 7 whitespace fields.
    data_line = oem.strip().splitlines()[-1]
    assert len(data_line.split()) == 7, data_line
    print("[program_track] PASS")
