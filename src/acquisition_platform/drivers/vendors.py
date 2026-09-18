"""Vendor adapters: one clean interface, one thin subclass per vendor dialect.

Each adapter is a `StreamDriver` that differs ONLY by its record encoder and
documented defaults. The exact on-the-wire framing every vendor uses is
proprietary and site-configurable; the encoders here are a documented,
vendor-APPROXIMATE mapping meant to be tweaked to a specific ACU's ICD without
touching the rest of the stack.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from acquisition_platform.drivers.base import TrackSample
from acquisition_platform.drivers.stream import StreamDriver, encode_csv


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class GenericRotatorDriver(StreamDriver):
    """Plain default: az/el(/rate) CSV lines. A safe lowest-common-denominator."""

    description = "Generic rotator (az/el/rate CSV stream)"

    def __init__(self, host: str = "127.0.0.1", port: int = 5005, **kwargs: object) -> None:
        super().__init__(host, port, encoder=encode_csv, **kwargs)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        return "generic_rotator"


def _encode_kongsberg(sample: TrackSample) -> bytes:
    """Kongsberg-style ACU record (approximate field mapping, JSON per line).

    Mapping: {"cmd":"track", "az", "el", "az_rate", "el_rate", "t":ISO-UTC}.
    """
    obj = {
        "cmd": "track",
        "az": round(sample.az_deg, 6),
        "el": round(sample.el_deg, 6),
        "az_rate": round(sample.az_rate_deg_s, 6),
        "el_rate": round(sample.el_rate_deg_s, 6),
        "t": _iso(sample.t),
    }
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


class KongsbergDriver(StreamDriver):
    """Kongsberg-style ACU: JSON `track` command records (approximate framing)."""

    description = "Kongsberg-style ACU (JSON track records)"

    def __init__(self, host: str = "127.0.0.1", port: int = 5010, **kwargs: object) -> None:
        super().__init__(host, port, encoder=_encode_kongsberg, **kwargs)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        return "kongsberg"


def _encode_comtech(sample: TrackSample) -> bytes:
    """ViaSat/Comtech-style delimited command line (approximate framing).

    `TRK,<t>,<az>,<el>,<az_rate>,<el_rate>` with an ISO-UTC epoch, degrees.
    """
    line = (
        f"TRK,{_iso(sample.t)},{sample.az_deg:.4f},{sample.el_deg:.4f},"
        f"{sample.az_rate_deg_s:.4f},{sample.el_rate_deg_s:.4f}\n"
    )
    return line.encode("utf-8")


class ComtechDriver(StreamDriver):
    """ViaSat/Comtech-style ACU: a `TRK,...` delimited command line per sample."""

    description = "ViaSat/Comtech-style ACU (delimited TRK command line)"

    def __init__(self, host: str = "127.0.0.1", port: int = 5020, **kwargs: object) -> None:
        super().__init__(host, port, encoder=_encode_comtech, **kwargs)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        return "comtech"


if __name__ == "__main__":
    from datetime import timedelta

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    track = [
        TrackSample(t0, 10.0, 20.0, 0.5, 0.1),
        TrackSample(t0 + timedelta(seconds=1), 10.5, 20.1, 0.5, 0.1),
    ]
    gen = GenericRotatorDriver(dry_run=True)
    kon = KongsbergDriver(dry_run=True)
    com = ComtechDriver(dry_run=True)
    for drv in (gen, kon, com):
        with drv:
            drv.send_track(track)
        assert len(drv.sent_bytes) == 2, drv.name

    assert gen.sent_bytes[0].startswith(b"2024-01-01")  # CSV epoch first
    kobj = json.loads(kon.sent_bytes[0].decode().strip())
    assert kobj["cmd"] == "track" and kobj["az"] == 10.0
    assert com.sent_bytes[0].startswith(b"TRK,")
    print("[vendors] PASS")
