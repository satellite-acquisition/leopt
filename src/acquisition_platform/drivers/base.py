"""Track samples, dwell interpolation, and the antenna driver interface.

``densify_track`` converts sparse planner dwells to a regularly sampled
azimuth/elevation track. Public driver angles are in degrees and times are
UTC-aware datetimes; the planner adapters convert from radians.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from acquisition_platform.planner.session import PassPlan


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackSample:
    """One point on a continuous program track (degrees, UTC).

    `az_rate_deg_s` / `el_rate_deg_s` are the local angular rates the pedestal
    should be turning at when it reaches this point (finite-difference estimate).
    """

    t: datetime
    az_deg: float
    el_deg: float
    az_rate_deg_s: float
    el_rate_deg_s: float


# A "dwell" is the transport-agnostic, normalized input this layer consumes:
#   {"when": datetime (UTC), "az_deg": float, "el_deg": float, "dwell_s": float}
# Both `dwells_from_passplan` and `dwells_from_passout` emit exactly this shape.


def dwells_from_passplan(passplan: PassPlan) -> list[dict]:
    """Normalize an internal `PassPlan` (radians) into dwell dicts (degrees).

    Imports the planner type lazily so this module has no hard dependency on it.
    """
    from math import degrees  # local import keeps the hot path tiny

    dwells: list[dict] = []
    for pt in passplan.pointings:
        dwells.append(
            {
                "when": pt.when,
                "az_deg": degrees(pt.az_rad),
                "el_deg": degrees(pt.el_rad),
                "dwell_s": float(pt.dwell_s),
            }
        )
    return dwells


def dwells_from_passout(passout_dict: dict) -> list[dict]:
    """Normalize a wire `PassOut` dict (degrees, ISO epochs) into dwell dicts.

    Accepts the JSON shape produced by `acquisition_platform.ingest.schemas`:
    a pass dict with a `pointings` list of `{t_s, az_deg, el_deg, dwell_s, ...}`.
    The absolute epoch of each dwell is `rise + t_s` seconds.
    """
    rise = _parse_epoch(passout_dict["rise"])
    dwells: list[dict] = []
    for pt in passout_dict.get("pointings", []):
        when = rise + timedelta(seconds=float(pt["t_s"]))
        dwells.append(
            {
                "when": when,
                "az_deg": float(pt["az_deg"]),
                "el_deg": float(pt["el_deg"]),
                "dwell_s": float(pt.get("dwell_s", 0.0)),
            }
        )
    return dwells


def _parse_epoch(value: str | datetime) -> datetime:
    """Coerce an ISO-8601 string or datetime into a tz-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Densification: sparse dwells -> continuous program track
# ---------------------------------------------------------------------------


def _az_shortest_delta(a0: float, a1: float) -> float:
    """Signed shortest angular path (deg) from a0 to a1 across the 0/360 wrap."""
    return (a1 - a0 + 180.0) % 360.0 - 180.0


def densify_track(dwells: list[dict], rate_hz: float = 1.0) -> list[TrackSample]:
    """Interpolate sparse dwells into a continuous az/el/rate program track.

    Between consecutive dwells we linearly interpolate az/el vs. time at
    `rate_hz` samples/second, taking the SHORTEST angular path in azimuth (so a
    track crossing 350 -> 010 deg sweeps +20 deg, not -340). Az/el rates are a
    forward finite difference (deg/s), with the final sample reusing the previous
    rate. A single dwell yields a single zero-rate sample.

    Requires dwells sorted by `when`; raises on an empty list or `rate_hz <= 0`.
    """
    if not dwells:
        raise ValueError("densify_track requires at least one dwell")
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")

    ordered = sorted(dwells, key=lambda d: d["when"])
    if len(ordered) == 1:
        d = ordered[0]
        return [TrackSample(_parse_epoch(d["when"]), d["az_deg"], d["el_deg"], 0.0, 0.0)]

    dt_s = 1.0 / rate_hz
    # 1) Build the (time, az, el) samples with azimuth unwrapped along the path.
    times: list[datetime] = []
    azs: list[float] = []
    els: list[float] = []
    t0 = _parse_epoch(ordered[0]["when"])
    for i in range(len(ordered) - 1):
        a = ordered[i]
        b = ordered[i + 1]
        ta = _parse_epoch(a["when"])
        tb = _parse_epoch(b["when"])
        span = (tb - ta).total_seconds()
        az_a = a["az_deg"]
        az_delta = _az_shortest_delta(az_a, b["az_deg"])
        el_a = a["el_deg"]
        el_delta = b["el_deg"] - el_a
        n_steps = max(1, int(math.floor(span * rate_hz + 1e-9)))
        for k in range(n_steps):
            frac = (k / n_steps) if span > 0 else 0.0
            offset = (ta - t0).total_seconds() + frac * span
            times.append(t0 + timedelta(seconds=offset))
            azs.append((az_a + frac * az_delta) % 360.0)
            els.append(el_a + frac * el_delta)
    # Always emit the final dwell exactly.
    last = ordered[-1]
    times.append(_parse_epoch(last["when"]))
    azs.append(last["az_deg"] % 360.0)
    els.append(last["el_deg"])

    # 2) Rates via forward finite difference on the (possibly wrapped) az series.
    samples: list[TrackSample] = []
    n = len(times)
    for i in range(n):
        if i < n - 1:
            step = (times[i + 1] - times[i]).total_seconds()
            if step > 0:
                az_rate = _az_shortest_delta(azs[i], azs[i + 1]) / step
                el_rate = (els[i + 1] - els[i]) / step
            else:
                az_rate = el_rate = 0.0
        else:  # last sample reuses the previous rate
            az_rate = samples[-1].az_rate_deg_s if samples else 0.0
            el_rate = samples[-1].el_rate_deg_s if samples else 0.0
        samples.append(TrackSample(times[i], azs[i], els[i], az_rate, el_rate))
    _ = dt_s  # documented cadence; sampling above already honours rate_hz
    return samples


# ---------------------------------------------------------------------------
# Driver base class
# ---------------------------------------------------------------------------


class AntennaDriver(abc.ABC):
    """Abstract "commands OUT" adapter to a single antenna control endpoint.

    Concrete drivers translate `TrackSample`s into a vendor wire dialect. When
    `dry_run` is True no hardware/socket is touched: every commanded sample is
    appended to `self.sent`, which makes the whole surface unit-testable.
    """

    #: Free-form capability tags, e.g. {"program_track", "get_position", "park"}.
    capabilities: frozenset[str] = frozenset()
    #: One of "file" | "tcp"; used by the registry metadata.
    transport: str = "tcp"
    #: Short human description for the registry listing.
    description: str = ""

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.sent: list[TrackSample] = []

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable driver name."""

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        """Open the connection to the endpoint. No-op in dry-run by default."""

    def close(self) -> None:
        """Release any connection resources. No-op by default."""

    def park(self) -> None:
        """Return the pedestal to a safe stow position. Override per vendor."""

    # -- commands ----------------------------------------------------------
    @abc.abstractmethod
    def command(self, sample: TrackSample) -> None:
        """Command a single track point. Must honour `self.dry_run`."""

    def send_track(self, track: list[TrackSample]) -> None:
        """Send a full program track. Default: command points in order."""
        for sample in track:
            self.command(sample)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> AntennaDriver:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# A tiny concrete driver used only for the smoke test / as a reference impl.
@dataclass
class _RecordingDriver(AntennaDriver):
    """Minimal dry-run driver: records every command. For tests/smoke only."""

    def __init__(self) -> None:
        super().__init__(dry_run=True)

    @property
    def name(self) -> str:
        return "recording"

    def command(self, sample: TrackSample) -> None:
        self.sent.append(sample)


# Keep dataclass field import referenced (used by subclasses/tests patterns).
_ = field


if __name__ == "__main__":
    _t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _dwells = [
        {"when": _t0, "az_deg": 350.0, "el_deg": 10.0, "dwell_s": 30.0},
        {"when": _t0 + timedelta(seconds=30), "az_deg": 10.0, "el_deg": 40.0, "dwell_s": 30.0},
        {"when": _t0 + timedelta(seconds=60), "az_deg": 30.0, "el_deg": 20.0, "dwell_s": 30.0},
    ]
    track = densify_track(_dwells, rate_hz=2.0)
    assert len(track) > 3, "expected a densified track"
    # Shortest-path wrap: 350 -> 010 should sweep POSITIVE (+20), never -340.
    assert track[0].az_rate_deg_s > 0.0, track[0]
    assert all(0.0 <= s.az_deg < 360.0 for s in track)

    drv = _RecordingDriver()
    with drv:
        drv.send_track(track)
    assert len(drv.sent) == len(track)
    drv.park()
    print("[base] PASS")
