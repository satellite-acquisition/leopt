"""Tests for the observation sources, SNR-valued observe, and track export."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from types import ModuleType

import numpy as np
import pytest
from fastapi.testclient import TestClient

from acquisition_platform.observations.base import Observation
from acquisition_platform.observations.manual import ManualSource
from acquisition_platform.observations.modem import SimulatedModemSource, SnmpModemSource
from acquisition_platform.observations.registry import get_source, list_sources
from acquisition_platform.observations.sdr import SimulatedSdrSource, band_snr_db
from acquisition_platform.observations.spectrum_analyzer import (
    LoopbackScpiServer,
    ScpiSpectrumAnalyzerSource,
)
from acquisition_platform.service.app import app
from antenna_pomdp.config import default_config

_CLOCK = lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: E731
SSO = (
    "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
    "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
)


# ---------------------------------------------------------------------------
# Observation value object
# ---------------------------------------------------------------------------


def test_observation_snr_mapping():
    now = _CLOCK()
    graded = Observation(t=now, detected=True, locked=True, ebn0_db=8.0)
    assert graded.has_snr() and graded.to_snr().locked
    bare = Observation(t=now, detected=True)
    assert bare.to_snr() is None  # detected but ungraded -> use detect update
    nolock = Observation(t=now, detected=False, locked=False)
    assert nolock.to_snr().locked is False


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_manual_source_queue():
    src = ManualSource(clock=_CLOCK)
    src.report(True, ebn0_db=9.0)
    src.report(False)
    a = src.poll()
    assert a.detected and a.to_snr().locked
    assert src.poll().detected is False
    assert src.poll() is None


def test_simulated_modem_lock_tracks_geometry():
    cfg = default_config()
    on = SimulatedModemSource(
        lambda t: (0.0, np.deg2rad(80.0), 6e5), cfg.antenna, seed=1, clock=_CLOCK
    )
    off = SimulatedModemSource(
        lambda t: (np.deg2rad(25.0), np.deg2rad(6.0), 2e6), cfg.antenna, seed=2, clock=_CLOCK
    )
    assert np.mean([on.poll().locked for _ in range(20)]) > 0.9
    assert np.mean([off.poll().locked for _ in range(20)]) < 0.2


def test_snmp_modem_reports_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysnmp.hlapi.v3arch.asyncio", None)
    src = SnmpModemSource("10.0.0.5", "1.3.6.1.4.1.1.1", "1.3.6.1.4.1.1.2")
    with pytest.raises(RuntimeError, match="pysnmp>=7.1.6"):
        src.open()


@pytest.fixture
def snmp_api(monkeypatch):
    api = ModuleType("pysnmp.hlapi.v3arch.asyncio")
    api.responses = {
        "lock": (None, 0, 0, [("lock", " LOCKED ")]),
        "ebn0": (None, 0, 0, [("ebn0", "84")]),
    }
    api.requests = []
    api.engines = []
    api.transport_error = None

    class Engine:
        def __init__(self):
            self.closed = False
            api.engines.append(self)

        def close_dispatcher(self):
            self.closed = True

    class Target:
        @classmethod
        async def create(cls, address):
            if api.transport_error:
                raise api.transport_error
            return address

    async def get_cmd(engine, community, target, context, oid):
        api.requests.append((community, target, oid))
        return api.responses[oid]

    api.SnmpEngine = Engine
    api.UdpTransportTarget = Target
    api.CommunityData = str
    api.ContextData = lambda: None
    api.ObjectIdentity = str
    api.ObjectType = str
    api.get_cmd = get_cmd
    monkeypatch.setitem(sys.modules, api.__name__, api)
    return api


def test_snmp_modem_reads_lock_and_scaled_ebn0(snmp_api):
    source = SnmpModemSource(
        "modem.example", "lock", "ebn0", port=1161, community="station",
        ebn0_scale=0.1, clock=_CLOCK,
    )
    obs = source.poll()
    assert obs.detected and obs.locked
    assert obs.ebn0_db == pytest.approx(8.4)
    assert obs.t == _CLOCK() and obs.meta == {"lock_raw": "locked"}
    assert snmp_api.requests == [
        ("station", ("modem.example", 1161), "lock"),
        ("station", ("modem.example", 1161), "ebn0"),
    ]
    assert len(snmp_api.engines) == 2
    assert all(engine.closed for engine in snmp_api.engines)


def test_snmp_modem_skips_ebn0_without_lock(snmp_api):
    snmp_api.responses["lock"] = (None, 0, 0, [("lock", "0")])
    obs = SnmpModemSource("modem.example", "lock", "ebn0").poll()
    assert obs.locked is False and obs.detected is False
    assert obs.ebn0_db is None
    assert len(snmp_api.requests) == 1
    assert snmp_api.engines[0].closed


@pytest.mark.parametrize(
    "response, message",
    [
        (("request timed out", 0, 0, []), "request timed out"),
        ((None, "authorizationError", 1, []), "authorizationError"),
        ((None, 0, 0, []), "returned no value"),
    ],
)
def test_snmp_modem_propagates_response_errors(snmp_api, response, message):
    snmp_api.responses["lock"] = response
    with pytest.raises(RuntimeError, match=message):
        SnmpModemSource("modem.example", "lock", "ebn0").poll()
    assert snmp_api.engines[0].closed


def test_snmp_modem_closes_engine_after_transport_error(snmp_api):
    snmp_api.transport_error = OSError("host lookup failed")
    with pytest.raises(OSError, match="host lookup failed"):
        SnmpModemSource("modem.example", "lock", "ebn0").poll()
    assert snmp_api.engines[0].closed


def test_snmp_modem_async_caller_uses_worker_thread(snmp_api):
    source = SnmpModemSource("modem.example", "lock", "ebn0")

    async def run():
        with pytest.raises(RuntimeError, match="asyncio.to_thread"):
            source.poll()
        assert not snmp_api.requests
        return await asyncio.to_thread(source.poll)

    assert asyncio.run(run()).locked
    assert all(engine.closed for engine in snmp_api.engines)


def test_band_snr_db_recovers_tone():
    freqs = np.linspace(-1e6, 1e6, 4096)
    spec = np.zeros_like(freqs)
    spec[np.argmin(np.abs(freqs - 1e4))] = 30.0
    assert 25.0 < band_snr_db(spec, freqs, 1e4, 2e4) <= 30.0


def test_simulated_sdr_detects_strong_geometry():
    cfg = default_config()
    on = SimulatedSdrSource(
        lambda t: (0.0, np.deg2rad(80.0), 6e5), cfg.antenna, seed=1, clock=_CLOCK
    )
    assert np.mean([on.poll().detected for _ in range(30)]) > 0.9


def test_scpi_spectrum_over_loopback_socket():
    with LoopbackScpiServer(marker_dbm=-95.0) as srv:
        src = ScpiSpectrumAnalyzerSource(
            srv.host, srv.port, noise_floor_dbm=-110.0, detect_snr_db=8.0, clock=_CLOCK
        )
        with src:
            obs = src.poll()
    assert obs.detected and abs(obs.snr_db - 15.0) < 1e-6


def test_registry_lists_and_builds():
    ids = {s["id"] for s in list_sources()}
    assert {"manual", "modem_sim", "sdr_sim", "spectrum_sim"} <= ids
    assert get_source("manual").name == "manual"
    with pytest.raises(KeyError):
        get_source("nope")


# ---------------------------------------------------------------------------
# Service: SNR observe + observation sources + track export
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _plan(client, network="polar_sso"):
    body = {
        "ingest": {"text": SSO},
        "network": network,
        "antenna": {"fwhm_deg": 10.0, "dwell_s": 10.0},
        "horizon_h": 10.0,
        "n_particles": 80,
        "rollouts": 10,
        "policy": "sweep",
    }
    res = client.post("/api/plan", json=body).json()
    return res


def test_api_observation_sources(client):
    r = client.get("/api/observation-sources")
    assert r.status_code == 200
    assert any(s["id"] == "modem_sim" for s in r.json())


def test_api_snr_observe(client):
    res = _plan(client)
    sid = res["session_id"]
    first = next(p for p in res["passes"] if p["pointings"])
    r = client.post(
        "/api/observe",
        json={
            "session_id": sid,
            "pass_idx": first["idx"],
            "dwell_idx": 0,
            "detected": True,
            "locked": True,
            "ebn0_db": 9.0,
            "source": "modem_sim",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["session_id"] == sid


def test_api_drivers_and_track_export(client):
    assert any(d["id"] == "rotctld" for d in client.get("/api/drivers").json())
    res = _plan(client)
    sid = res["session_id"]
    first = next(p for p in res["passes"] if p["pointings"])
    for fmt, needle in [("csv", "az_deg"), ("ccsds_pointing", "CCSDS"), ("oem", "CCSDS_OEM_VERS")]:
        r = client.post(
            "/api/export/track",
            json={"session_id": sid, "pass_idx": first["idx"], "format": fmt, "rate_hz": 0.5},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_samples"] > 0 and needle in body["content"]
