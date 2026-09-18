"""Read carrier lock and Eb/N0 from a modem over SNMP or serial.

SNMP OIDs and serial queries are configurable. The hardware adapters require
``pysnmp`` or ``pyserial`` respectively. ``SimulatedModemSource`` produces
reports from the engine's geometry and observation model without hardware.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone

from acquisition_platform.observations.base import Observation, ObservationSource

# A geometry provider: given a time, return (delta_theta_rad, elevation_rad,
# range_m) of the true satellite relative to the commanded boresight.
GeometryFn = Callable[[datetime], "tuple[float, float, float]"]


class SnmpModemSource(ObservationSource):
    """Poll lock-state and Eb/N0 OIDs with PySNMP 7.1.6 or newer.

    ``poll`` blocks while querying the modem. In async code, call it with
    ``await asyncio.to_thread(source.poll)``.
    """

    name = "modem-snmp"
    capabilities = frozenset({"detect", "ebn0"})

    def __init__(
        self,
        host: str,
        lock_oid: str,
        ebn0_oid: str,
        *,
        community: str = "public",
        port: int = 161,
        ebn0_scale: float = 1.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.community = community
        self.lock_oid = lock_oid
        self.ebn0_oid = ebn0_oid
        self.ebn0_scale = ebn0_scale  # e.g. 0.1 if the OID reports tenths of dB
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._opened = False

    def open(self) -> None:
        try:
            from pysnmp.hlapi.v3arch.asyncio import get_cmd  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SnmpModemSource needs pysnmp>=7.1.6; install the adapters extra or "
                "use SimulatedModemSource for demos/tests."
            ) from exc
        self._opened = True

    async def _get_async(self, oid: str) -> str:
        from pysnmp.hlapi.v3arch.asyncio import (
            CommunityData,
            ContextData,
            ObjectIdentity,
            ObjectType,
            SnmpEngine,
            UdpTransportTarget,
            get_cmd,
        )

        engine = SnmpEngine()
        try:
            target = await UdpTransportTarget.create((self.host, self.port))
            err_ind, err_stat, _, var_binds = await get_cmd(
                engine,
                CommunityData(self.community),
                target,
                ContextData(),
                ObjectType(ObjectIdentity(oid)),
            )
        finally:
            engine.close_dispatcher()
        if err_ind or err_stat:
            raise RuntimeError(f"SNMP GET {oid} failed: {err_ind or err_stat}")
        if not var_binds:
            raise RuntimeError(f"SNMP GET {oid} returned no value")
        return str(var_binds[0][1])

    def _get(self, oid: str) -> str:
        return asyncio.run(self._get_async(oid))

    def poll(self) -> Observation | None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "SnmpModemSource.poll() blocks; use await asyncio.to_thread(source.poll) "
                "from an active event loop."
            )
        if not self._opened:
            self.open()
        lock_raw = self._get(self.lock_oid).strip().lower()
        locked = lock_raw in {"1", "true", "up", "locked", "lock"}
        ebn0 = float(self._get(self.ebn0_oid)) * self.ebn0_scale if locked else None
        return Observation(
            t=self._clock(),
            detected=locked,
            locked=locked,
            ebn0_db=ebn0,
            source=self.name,
            meta={"lock_raw": lock_raw},
        )


class SerialModemSource(ObservationSource):
    """Query a modem's Eb/N0 over an ASCII serial line (needs `pyserial`)."""

    name = "modem-serial"
    capabilities = frozenset({"detect", "ebn0"})

    def __init__(
        self,
        port: str,
        *,
        baud: int = 9600,
        lock_query: bytes = b"STAT:LOCK?\n",
        ebn0_query: bytes = b"MEAS:EBNO?\n",
        timeout_s: float = 1.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.lock_query = lock_query
        self.ebn0_query = ebn0_query
        self.timeout_s = timeout_s
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._ser = None

    def open(self) -> None:
        try:
            import serial  # noqa: F401
        except ImportError as exc:  # pragma: no cover - hardware path
            raise RuntimeError(
                "SerialModemSource needs the optional 'pyserial' package; install it "
                "or use SimulatedModemSource for demos/tests."
            ) from exc
        import serial

        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)

    def close(self) -> None:  # pragma: no cover - hardware path
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def _ask(self, query: bytes) -> str:  # pragma: no cover - hardware path
        self._ser.reset_input_buffer()
        self._ser.write(query)
        return self._ser.readline().decode("ascii", errors="replace").strip()

    def poll(self) -> Observation | None:  # pragma: no cover - hardware path
        if self._ser is None:
            self.open()
        lock_raw = self._ask(self.lock_query).lower()
        locked = lock_raw in {"1", "true", "up", "locked", "lock", "on"}
        ebn0 = None
        if locked:
            try:
                ebn0 = float(self._ask(self.ebn0_query).split()[0])
            except (ValueError, IndexError):
                ebn0 = None
        return Observation(
            t=self._clock(), detected=locked, locked=locked, ebn0_db=ebn0, source=self.name
        )


class SimulatedModemSource(ObservationSource):
    """A physics-based stand-in: synthesise lock + Eb/N0 from pass geometry.

    `geometry_fn(t) -> (delta_theta_rad, elevation_rad, range_m)` supplies the
    truth geometry (e.g. from a simulation or the console's Keplerian model). Each
    `poll()` draws a realistic report through the SNR observation model, so the
    simulated modem behaves like the real one and drives the same belief update.
    """

    name = "modem-sim"
    capabilities = frozenset({"detect", "ebn0"})

    def __init__(
        self,
        geometry_fn: GeometryFn,
        antenna_cfg,
        snr_cfg=None,
        *,
        seed: int = 0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        from antenna_pomdp.models.snr_observation import SnrConfig

        self.geometry_fn = geometry_fn
        self.antenna_cfg = antenna_cfg
        self.snr_cfg = snr_cfg or SnrConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        import numpy as np

        self._rng = np.random.default_rng(seed)

    def poll(self) -> Observation | None:
        from antenna_pomdp.models.snr_observation import sample_snr_observation

        t = self._clock()
        dtheta, el, rng_m = self.geometry_fn(t)
        snr_obs = sample_snr_observation(
            dtheta, self.antenna_cfg, el, rng_m, self.snr_cfg, self._rng
        )
        return Observation(
            t=t,
            detected=snr_obs.locked,
            locked=snr_obs.locked,
            ebn0_db=snr_obs.ebn0_db,
            source=self.name,
            meta={"delta_theta_deg": float(__import__("numpy").rad2deg(dtheta))},
        )


if __name__ == "__main__":
    import numpy as np

    from antenna_pomdp.config import default_config

    cfg = default_config()
    ticks = iter(range(1000))
    clock = lambda: datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # noqa: E731

    # On-boresight, near/high pass -> locked with a healthy Eb/N0.
    on_boresight: GeometryFn = lambda t: (0.0, np.deg2rad(80.0), 6.0e5)  # noqa: E731
    m = SimulatedModemSource(on_boresight, cfg.antenna, seed=1, clock=clock)
    reports = [m.poll() for _ in range(20)]
    lock_rate = np.mean([r.locked for r in reports])
    assert lock_rate > 0.9, lock_rate
    assert all(r.ebn0_db is not None for r in reports if r.locked)

    # Far off-boresight, low/far pass -> mostly no lock.
    off: GeometryFn = lambda t: (np.deg2rad(25.0), np.deg2rad(6.0), 2.0e6)  # noqa: E731
    m2 = SimulatedModemSource(off, cfg.antenna, seed=2, clock=clock)
    lock_rate_off = np.mean([m2.poll().locked for _ in range(20)])
    assert lock_rate_off < 0.2, lock_rate_off

    # The graded report maps to a locked SnrObservation the engine can use.
    good = next(r for r in reports if r.locked)
    assert good.to_snr().locked and good.to_snr().ebn0_db is not None

    # Hardware transports raise a clear error without their optional dep present.
    try:
        SnmpModemSource("10.0.0.5", "1.3.6.1.4.1.1.1", "1.3.6.1.4.1.1.2").open()
    except RuntimeError as e:
        assert "pysnmp" in str(e)

    print(f"[obs.modem] sim lock rate: on-axis={lock_rate:.2f}, off-axis={lock_rate_off:.2f}")
    print("[obs.modem] PASS")
