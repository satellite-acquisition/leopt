"""Read a spectrum analyzer's SCPI marker amplitude over TCP.

The source subtracts a reference noise floor to estimate SNR. Command strings
are configurable for different instruments. Simulated and loopback sources
support tests without an attached analyzer.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np

from acquisition_platform.observations.base import Observation, ObservationSource

GeometryFn = Callable[[datetime], "tuple[float, float, float]"]


class ScpiSpectrumAnalyzerSource(ObservationSource):
    """Peak-marker beacon detector over a SCPI/TCP spectrum analyzer."""

    name = "spectrum-scpi"
    capabilities = frozenset({"detect", "snr"})

    def __init__(
        self,
        host: str,
        port: int = 5025,
        *,
        marker_query: str = ":CALC:MARK1:MAX;:CALC:MARK1:Y?",
        noise_floor_dbm: float = -110.0,
        detect_snr_db: float = 8.0,
        timeout_s: float = 2.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.marker_query = marker_query
        self.noise_floor_dbm = noise_floor_dbm
        self.detect_snr_db = detect_snr_db
        self.timeout_s = timeout_s
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sock: socket.socket | None = None

    def open(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _query(self, command: str) -> str:
        assert self._sock is not None
        self._sock.sendall(command.encode("ascii") + b"\n")
        chunks: list[bytes] = []
        while b"\n" not in b"".join(chunks):
            data = self._sock.recv(256)
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks).decode("ascii", errors="replace").strip()

    def poll(self) -> Observation | None:
        if self._sock is None:
            self.open()
        reply = self._query(self.marker_query)
        try:
            marker_dbm = float(reply.split(",")[0].split()[0])
        except (ValueError, IndexError):
            return Observation(
                t=self._clock(),
                detected=False,
                locked=False,
                source=self.name,
                meta={"scpi_reply": reply},
            )
        snr = marker_dbm - self.noise_floor_dbm
        detected = snr >= self.detect_snr_db
        return Observation(
            t=self._clock(),
            detected=detected,
            locked=detected,
            snr_db=snr if detected else None,
            source=self.name,
            meta={"marker_dbm": marker_dbm},
        )


class SimulatedSpectrumAnalyzer(ObservationSource):
    """Synthesise a marker amplitude from pass geometry (no instrument)."""

    name = "spectrum-sim"
    capabilities = frozenset({"detect", "snr"})

    def __init__(
        self,
        geometry_fn: GeometryFn,
        antenna_cfg,
        snr_cfg=None,
        *,
        detect_snr_db: float = 8.0,
        seed: int = 0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        from antenna_pomdp.models.snr_observation import SnrConfig

        self.geometry_fn = geometry_fn
        self.antenna_cfg = antenna_cfg
        self.snr_cfg = snr_cfg or SnrConfig()
        self.detect_snr_db = detect_snr_db
        self._rng = np.random.default_rng(seed)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def poll(self) -> Observation | None:
        from antenna_pomdp.models.snr_observation import predicted_ebn0_db

        t = self._clock()
        dtheta, el, rng_m = self.geometry_fn(t)
        snr = float(predicted_ebn0_db(dtheta, self.antenna_cfg, el, rng_m, self.snr_cfg))
        snr += float(self._rng.normal(0.0, 1.0))
        detected = snr >= self.detect_snr_db
        return Observation(
            t=t,
            detected=detected,
            locked=detected,
            snr_db=snr if detected else None,
            source=self.name,
        )


class LoopbackScpiServer:
    """A tiny in-process SCPI server that answers the marker query with a level.

    Used by the smoke test / unit tests to drive `ScpiSpectrumAnalyzerSource`
    over a real socket without an instrument. Context-managed; binds an ephemeral
    port exposed as `.host` / `.port`.
    """

    def __init__(self, marker_dbm: float = -95.0) -> None:
        self.marker_dbm = marker_dbm
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.host, self.port = self._srv.getsockname()
        self._thread: threading.Thread | None = None

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        with conn:
            buf = b""
            while True:
                try:
                    data = conn.recv(256)
                except OSError:
                    return
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    _line, _, buf = buf.partition(b"\n")
                    conn.sendall(f"{self.marker_dbm:.2f}\n".encode())

    def __enter__(self) -> LoopbackScpiServer:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._srv.close()


if __name__ == "__main__":
    from antenna_pomdp.config import default_config

    cfg = default_config()
    clock = lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: E731

    # Real socket round-trip against the loopback SCPI server: strong marker -> detect.
    with LoopbackScpiServer(marker_dbm=-95.0) as srv:
        src = ScpiSpectrumAnalyzerSource(
            srv.host, srv.port, noise_floor_dbm=-110.0, detect_snr_db=8.0, clock=clock
        )
        with src:
            obs = src.poll()
    assert obs.detected and abs(obs.snr_db - 15.0) < 1e-6, obs  # -95 - (-110) = 15 dB

    # Weak marker -> no detect.
    with LoopbackScpiServer(marker_dbm=-118.0) as srv:
        src = ScpiSpectrumAnalyzerSource(srv.host, srv.port, noise_floor_dbm=-110.0, clock=clock)
        with src:
            weak = src.poll()
    assert not weak.detected, weak

    # Simulated instrument tracks geometry.
    on = SimulatedSpectrumAnalyzer(
        lambda t: (0.0, np.deg2rad(80.0), 6e5), cfg.antenna, seed=1, clock=clock
    )
    det = np.mean([on.poll().detected for _ in range(20)])
    assert det > 0.8, det

    print(
        f"[obs.spectrum] loopback SNR={obs.snr_db:.1f} dB (detect={obs.detected}); sim det={det:.2f}"
    )
    print("[obs.spectrum] PASS")
