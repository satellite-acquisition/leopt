"""Hamlib rotctld and SatNOGS rotator drivers.

The TCP protocol uses ``P <az> <el>`` to set position and ``p`` to read it.
Dry runs record commands in ``sent``. The loopback server exercises the socket
transport without hardware.
"""

from __future__ import annotations

import socket
import threading
from datetime import datetime, timezone

from acquisition_platform.drivers.base import AntennaDriver, TrackSample

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 4533


class RotctldDriver(AntennaDriver):
    """Speaks the Hamlib `rotctld` TCP text protocol over stdlib sockets."""

    capabilities = frozenset({"program_track", "get_position", "park"})
    transport = "tcp"
    description = "Hamlib rotctld rotator daemon (az/el text protocol)"

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        *,
        dry_run: bool = False,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None

    @property
    def name(self) -> str:
        return "rotctld"

    # -- lifecycle ---------------------------------------------------------
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

    # -- protocol ----------------------------------------------------------
    def _send_line(self, line: str) -> str:
        """Send one command line and return the daemon's reply text."""
        if self.dry_run or self._sock is None:
            return "RPRT 0"
        self._sock.sendall(line.encode("ascii"))
        return self._sock.recv(1024).decode("ascii", errors="replace")

    def command(self, sample: TrackSample) -> None:
        """Set position: `P <az> <el>`. Records in dry-run, else checks RPRT 0."""
        self.sent.append(sample)
        reply = self._send_line(f"P {sample.az_deg:.6f} {sample.el_deg:.6f}\n")
        if not self.dry_run and "RPRT 0" not in reply:
            raise RuntimeError(f"rotctld rejected set-position: {reply!r}")

    def park(self) -> None:
        """Stow to az=0 el=0 via a `P 0 0` set-position."""
        reply = self._send_line("P 0 0\n")
        if not self.dry_run and "RPRT 0" not in reply:
            raise RuntimeError(f"rotctld rejected park: {reply!r}")

    def get_position(self) -> tuple[float, float]:
        """Read current (az_deg, el_deg) via `p`. Returns (0, 0) in dry-run."""
        if self.dry_run or self._sock is None:
            return (0.0, 0.0)
        reply = self._send_line("p\n")
        parts = [p for p in reply.replace("\r", "").split("\n") if p.strip()]
        if len(parts) < 2:
            raise RuntimeError(f"rotctld get-position malformed: {reply!r}")
        return (float(parts[0]), float(parts[1]))


class SatnogsRotatorDriver(RotctldDriver):
    """SatNOGS rotator: rotctld-compatible wire protocol, optional client label.

    SatNOGS ground-station rotators are driven through rotctld, so the transport
    is identical; this subclass only carries an optional `client_id` label for
    provenance/logging and a distinct registry identity.
    """

    description = "SatNOGS rotator (rotctld-compatible)"

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        *,
        client_id: str | None = None,
        dry_run: bool = False,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__(host, port, dry_run=dry_run, timeout_s=timeout_s)
        self.client_id = client_id

    @property
    def name(self) -> str:
        return "satnogs"


class _LoopbackRotctld:
    """A minimal in-process rotctld server for tests/smoke (no hardware).

    Understands `P az el` (stores position, replies `RPRT 0`) and `p` (replies
    the stored az/el on two lines). Use as a context manager; `.port` gives the
    bound ephemeral port to point a `RotctldDriver` at.
    """

    def __init__(self, host: str = _DEFAULT_HOST) -> None:
        self.host = host
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.az = 0.0
        self.el = 0.0

    def __enter__(self) -> _LoopbackRotctld:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        try:
            self._srv.close()
        finally:
            if self._thread is not None:
                self._thread.join(timeout=1.0)

    def _serve(self) -> None:
        self._srv.settimeout(0.25)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                continue
            with conn:
                self._handle(conn)

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = ""
        while not self._stop.is_set():
            try:
                chunk = conn.recv(1024)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk.decode("ascii", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                conn.sendall(self._reply(line.strip()).encode("ascii"))

    def _reply(self, line: str) -> str:
        if not line:
            return "RPRT 0\n"
        tok = line.split()
        if tok[0] in ("P", "set_pos") and len(tok) >= 3:
            self.az, self.el = float(tok[1]), float(tok[2])
            return "RPRT 0\n"
        if tok[0] in ("p", "get_pos"):
            return f"{self.az:.6f}\n{self.el:.6f}\n"
        return "RPRT 0\n"


if __name__ == "__main__":
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    samples = [
        TrackSample(t0, 123.4, 45.6, 0.1, 0.2),
        TrackSample(t0, 130.0, 50.0, 0.1, 0.2),
    ]
    # Dry-run: no socket, commands recorded.
    dry = RotctldDriver(dry_run=True)
    with dry:
        dry.send_track(samples)
    assert len(dry.sent) == 2

    # Real socket round-trip against the in-process loopback server.
    with _LoopbackRotctld() as srv:
        drv = RotctldDriver(host=srv.host, port=srv.port)
        with drv:
            drv.command(samples[0])
            az, el = drv.get_position()
            drv.park()
        assert abs(az - 123.4) < 1e-3 and abs(el - 45.6) < 1e-3, (az, el)
        assert abs(srv.az) < 1e-9 and abs(srv.el) < 1e-9  # parked to 0,0

    sat = SatnogsRotatorDriver(client_id="gs-42", dry_run=True)
    assert sat.name == "satnogs" and sat.client_id == "gs-42"
    print("[rotctld] PASS")
