"""End-to-end tests for the acquisition_platform package.

Kept fast (tiny particle / rollout counts) so the suite runs in a few seconds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from acquisition_platform.estimator.ukf import SgpUkf, cov_in_eci
from acquisition_platform.ingest.opm import SAMPLE_OPM, parse_opm
from acquisition_platform.ingest.tle import SAMPLE_TLE, parse_tle
from acquisition_platform.planner.session import PlanSession
from acquisition_platform.service.app import app
from antenna_pomdp.config import default_config, network_single_university

SSO = (
    "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
    "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
)


def _tiny_cfg(l1: str, l2: str):
    base = default_config()
    return replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=l1, nominal_tle_line2=l2),
        antenna=replace(base.antenna, fwhm_deg=10.0, dwell_time_s=10.0),
        filter=replace(base.filter, n_particles=64),
        pomcp=replace(base.pomcp, n_rollouts=20, max_depth=3),
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_opm_parse_roundtrip():
    st = parse_opm(SAMPLE_OPM)
    assert st.frame == "RTN"
    assert st.cov_6x6.shape == (6, 6)
    assert np.allclose(st.cov_6x6, st.cov_6x6.T)
    # radial 1-sigma = sqrt(9e-6 km^2) = 3 m
    assert abs(np.sqrt(st.cov_6x6[0, 0]) - 3.0) < 1e-6


def test_opm_missing_covariance_falls_back():
    no_cov = "\n".join(
        line for line in SAMPLE_OPM.splitlines() if not line.startswith(("C", "COV_REF"))
    )
    st = parse_opm(no_cov)
    assert st.frame == "ECI"
    assert len(st.warnings) == 1
    assert st.cov_6x6[0, 0] > 0.0


def test_tle_parse():
    opm, l1, l2 = parse_tle(SAMPLE_TLE)
    r = np.linalg.norm(opm.r_eci_m) / 1e3
    assert 6.5e3 < r < 8.0e3
    assert l1.startswith("1 ") and l2.startswith("2 ")


# ---------------------------------------------------------------------------
# UKF
# ---------------------------------------------------------------------------


def test_ukf_seed_produces_spread():
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, l1, l2 = parse_tle(SSO, cov_6x6=cov, frame="RTN")
    cfg = _tiny_cfg(l1, l2)
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=200, rng=np.random.default_rng(0))
    assert belief.n == 200
    slips = np.array([p.launch_slip_s for p in belief.particles])
    assert float(np.std(slips)) > 0.0


def test_ukf_rtn_to_eci_preserves_position_trace():
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, _, _ = parse_tle(SSO, cov_6x6=cov, frame="RTN")
    eci = cov_in_eci(opm)
    assert abs(np.trace(eci[:3, :3]) - np.trace(cov[:3, :3])) < 1.0


# ---------------------------------------------------------------------------
# PlanSession
# ---------------------------------------------------------------------------


def _build_session(policy="pomcp"):
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, l1, l2 = parse_tle(SSO, cov_6x6=cov, frame="RTN")
    cfg = _tiny_cfg(l1, l2)
    rng = np.random.default_rng(0)
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)
    return PlanSession(
        config=cfg,
        network=network_single_university(),
        belief=belief,
        policy=policy,
        leop_start=opm.epoch,
        ukf=ukf,
        rng=rng,
    )


def test_plan_network_returns_passes_with_pointings():
    sess = _build_session("sweep")
    plans = sess.plan_network(horizon_h=10.0)
    assert len(plans) >= 1
    assert any(len(p.pointings) > 0 for p in plans)
    for p in plans:
        for pt in p.pointings:
            assert 0.0 <= pt.p_detect <= 1.0


def test_observe_changes_belief_entropy():
    sess = _build_session("sweep")
    plans = sess.plan_network(horizon_h=10.0)
    first = next(p for p in plans if p.pointings)
    ent0 = sess.belief_summary()["entropy"]
    sess.observe(first.idx, 0, detected=False)
    ent1 = sess.belief_summary()["entropy"]
    # A measurement update reweights particles -> entropy moves.
    assert ent1 != ent0


def test_belief_summary_grid_shape():
    sess = _build_session("sweep")
    sess.plan_network(horizon_h=10.0)
    g = sess.belief_summary(n_along=25, n_cross=21)["grid"]
    assert len(g["along_s"]) == 25
    assert len(g["cross_deg"]) == 21
    assert len(g["weight"]) == 25 and len(g["weight"][0]) == 21


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_api_networks(client):
    r = client.get("/api/networks")
    assert r.status_code == 200
    nets = r.json()
    assert len(nets) == 4
    ids = {n["id"] for n in nets}
    assert "single_university" in ids
    for n in nets:
        assert n["n_stations"] == len(n["stations"])


def test_api_plan_and_observe(client):
    body = {
        "ingest": {"format": "tle", "text": SSO},
        "network": "single_university",
        "antenna": {"fwhm_deg": 10.0, "dwell_s": 10.0, "pd_onaxis": 0.95, "pfa": 0.01},
        "horizon_h": 10.0,
        "n_particles": 64,
        "rollouts": 20,
        "policy": "sweep",
    }
    r = client.post("/api/plan", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["passes"], res
    sid = res["session_id"]
    first = next(p for p in res["passes"] if p["pointings"])

    obs = client.post(
        "/api/observe",
        json={
            "session_id": sid,
            "pass_idx": first["idx"],
            "dwell_idx": 0,
            "detected": False,
        },
    )
    assert obs.status_code == 200, obs.text
    assert obs.json()["session_id"] == sid

    bel = client.get(f"/api/belief/{sid}")
    assert bel.status_code == 200
    assert "entropy" in bel.json()
