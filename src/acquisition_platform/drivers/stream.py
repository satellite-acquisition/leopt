"""Stream azimuth, elevation, and rates as newline-delimited TCP records.

Encoders accept a ``TrackSample`` and return bytes; JSON and CSV encoders are
provided. Dry runs record data without opening a socket. ``pacing=True`` sends
samples at the track cadence; the default sends without waiting.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from datetime import datetime, timezone

from acquisition_platform.drivers.base import AntennaDriver, TrackSample

Encoder = Callable[[TrackSample], bytes]


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def encode_json(sample: TrackSample) -> bytes:
    """One compact JSON object per line: {t, az, el, az_rate, el_rate}."""
    obj = {
        "t": _iso(sample.t),
        "az": round(sample.az_deg, 6),
        "el": round(sample.el_deg, 6),
        "az_rate": round(sample.az_rate_deg_s, 6),
        "el_rate": round(sample.el_rate_deg_s, 6),
    }
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


def encode_csv(sample: TrackSample) -> bytes:
    """One CSV line: t,az,el,az_rate,el_rate (ISO epoch, degrees)."""
    line = (
        f"{_iso(sample.t)},{sample.az_deg:.6f},{sample.el_deg:.6f},"
        f"{sample.az_rate_deg_s:.6f},{sample.el_rate_deg_s:.6f}\n"
    )
    return line.encode("utf-8")


class StreamDriver(AntennaDriver):
    """Stream newline-delimited track records to a TCP endpoint."""

    capabilities = frozenset({"program_track", "stream"})
    transport = "tcp"
    description = "Generic newline-delimited az/el/rate TCP stream"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5005,
        *,
        encoder: Encoder | None = None,
        dry_run: bool = False,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.host = host
        self.port = port
        self.encoder: Encoder = encoder or encode_csv
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        #: Raw bytes of every record sent (populated in dry-run and live).
        self.sent_bytes: list[bytes] = []

    @property
    def name(self) -> str:
        return "stream"

    def connect(self) -> None:
        if self.dry_run:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def command(self, sample: TrackSample) -> None:
        """Encode and send one record. Records to `sent`/`sent_bytes` always."""
        payload = self.encoder(sample)
        self.sent.append(sample)
        self.sent_bytes.append(payload)
        if not self.dry_run and self._sock is not None:
            self._sock.sendall(payload)

    def send_track(self, track: list[TrackSample], *, pacing: bool = False) -> None:
        """Stream the track. If `pacing`, sleep to match inter-sample wall time.

        Tests pass `pacing=False` (default) so the stream runs instantly.
        """
        prev_t: datetime | None = None
        for sample in track:
            if pacing and prev_t is not None:
                dt = (sample.t - prev_t).total_seconds()
                if dt > 0:
                    time.sleep(dt)
            self.command(sample)
            prev_t = sample.t


if __name__ == "__main__":
    from datetime import timedelta

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    track = [
        TrackSample(t0, 10.0, 20.0, 0.5, 0.1),
        TrackSample(t0 + timedelta(seconds=1), 10.5, 20.1, 0.5, 0.1),
    ]
    # Dry-run with each built-in encoder.
    for enc in (encode_csv, encode_json):
        drv = StreamDriver(encoder=enc, dry_run=True)
        with drv:
            drv.send_track(track)
        assert len(drv.sent_bytes) == 2
        assert all(b.endswith(b"\n") for b in drv.sent_bytes)
    obj = json.loads(encode_json(track[0]).decode().strip())
    assert obj["az"] == 10.0 and "t" in obj

    # Real socket: stream to a loopback listener that collects the bytes.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    drv = StreamDriver(host=host, port=port, encoder=encode_json)
    drv.connect()
    conn, _ = srv.accept()
    drv.send_track(track, pacing=False)
    drv.close()
    received = b""
    conn.settimeout(1.0)
    while True:
        try:
            chunk = conn.recv(4096)
        except OSError:
            break
        if not chunk:
            break
        received += chunk
    conn.close()
    srv.close()
    assert received.count(b"\n") == 2, received
    print("[stream] PASS")
