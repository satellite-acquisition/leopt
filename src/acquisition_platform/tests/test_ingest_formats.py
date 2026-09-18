"""Tests for the multi-format orbit ingest + catalog + track-export endpoints."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from acquisition_platform.ingest.catalog import CatalogPoller, extract_tle_pair
from acquisition_platform.ingest.detect import detect_format, parse_any
from acquisition_platform.ingest.json_ingest import SAMPLE_OMM_JSON, omm_to_tle, parse_json
from acquisition_platform.ingest.ndm import SAMPLE_OEM_XML, SAMPLE_OPM_XML, parse_ndm_xml
from acquisition_platform.ingest.oem import SAMPLE_OEM, Ephemeris, parse_oem
from acquisition_platform.ingest.opm import OpmState, SAMPLE_OPM
from acquisition_platform.ingest.tle import SAMPLE_TLE
from acquisition_platform.service.app import app

SSO = (
    "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
    "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_oem_parse_and_interpolation():
    eph = parse_oem(SAMPLE_OEM)
    assert eph.n == 6
    r_node, _ = eph.at(eph.epochs[2])
    assert np.allclose(r_node, eph.r_eci_m[2], atol=1e-3)  # Lagrange exact on nodes
    opm = eph.to_opm()
    assert isinstance(opm, OpmState) and opm.epoch == eph.start


def test_ndm_xml_opm_and_oem():
    opm = parse_ndm_xml(SAMPLE_OPM_XML)
    assert isinstance(opm, OpmState) and opm.frame == "RTN"
    assert abs(np.sqrt(opm.cov_6x6[0, 0]) - 3.0) < 1e-6
    eph = parse_ndm_xml(SAMPLE_OEM_XML)
    assert isinstance(eph, Ephemeris) and eph.n == 3


def test_omm_json_to_tle_roundtrip():
    _, l1, l2 = parse_json(SAMPLE_OMM_JSON)
    assert l1.startswith("1 ") and l2.startswith("2 ")
    from antenna_pomdp.orbit.propagator import SGP4Propagator
    from antenna_pomdp.orbit.sampler import _parse_tle_fields

    prop = SGP4Propagator(l1, l2)  # synthesised lines must parse under SGP4
    assert 6.5e6 < np.linalg.norm(prop.propagate(prop.epoch).r_eci_m) < 8.0e6
    f = _parse_tle_fields(l1, l2)
    assert abs(f["inc_deg"] - 51.64) < 1e-3


def test_omm_passthrough_existing_lines():
    l1, l2 = omm_to_tle({"TLE_LINE1": "1 ...", "TLE_LINE2": "2 ..."})
    assert l1 == "1 ..." and l2 == "2 ..."


@pytest.mark.parametrize(
    "payload,expected",
    [
        (SAMPLE_TLE, "tle"),
        (SAMPLE_OPM, "opm"),
        (SAMPLE_OEM, "oem"),
        (SAMPLE_OPM_XML, "ndm"),
        (SAMPLE_OMM_JSON, "json"),
    ],
)
def test_autodetect(payload, expected):
    assert detect_format(payload) == expected
    out = parse_any(payload)
    assert isinstance(out.opm, OpmState)
    assert np.linalg.norm(out.opm.r_eci_m) > 6.0e6


def test_oem_via_parse_any_keeps_ephemeris():
    out = parse_any(SAMPLE_OEM)
    assert out.ephemeris is not None and out.line1 is None


# ---------------------------------------------------------------------------
# Catalog poller (offline)
# ---------------------------------------------------------------------------


def test_extract_tle_pair():
    assert extract_tle_pair(SSO) is not None
    assert extract_tle_pair("No GP data found") is None


def test_catalog_poller_finds_on_third_attempt():
    state = {"n": 0}

    def fetch():
        state["n"] += 1
        return extract_tle_pair(SSO) if state["n"] >= 3 else None

    ticks = {"t": 0.0}
    res = CatalogPoller(fetch=fetch, interval_s=5.0, max_attempts=10).wait(
        sleep=lambda s: ticks.__setitem__("t", ticks["t"] + s), clock=lambda: ticks["t"]
    )
    assert res.found and res.attempts == 3


def test_catalog_poller_swallows_network_errors():
    def boom():
        raise OSError("refused")

    assert CatalogPoller(fetch=boom).poll_once() is None


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.mark.parametrize("payload", [SAMPLE_OEM, SAMPLE_OPM_XML, SAMPLE_OMM_JSON])
def test_api_ingest_autodetects(client, payload):
    r = client.post("/api/ingest", json={"text": payload})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_format"] in {"oem", "ndm", "json"}
    assert len(body["r_eci_m"]) == 3


def test_api_ingest_oem_reports_ephemeris(client):
    body = client.post("/api/ingest", json={"text": SAMPLE_OEM}).json()
    assert body["has_ephemeris"] and body["ephemeris_span_s"] > 0


def test_api_plan_from_oem(client):
    body = {
        "ingest": {"text": SAMPLE_OEM},
        "network": "single_university",
        "antenna": {"fwhm_deg": 10.0, "dwell_s": 10.0},
        "horizon_h": 8.0,
        "n_particles": 48,
        "rollouts": 10,
        "policy": "sweep",
    }
    r = client.post("/api/plan", json=body)
    assert r.status_code == 200, r.text  # parses + seeds even with no TLE lines


def test_api_catalog_requires_identifier(client):
    r = client.post("/api/catalog/tle", json={"source": "celestrak"})
    assert r.status_code == 400


def test_api_catalog_spacetrack_needs_credentials(client):
    r = client.post("/api/catalog/tle", json={"source": "spacetrack", "catnr": 25544})
    assert r.status_code == 400
