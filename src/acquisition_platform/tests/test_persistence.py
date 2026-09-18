"""Tests for durable session persistence + the hash-chained audit log."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from acquisition_platform.estimator.ukf import SgpUkf
from acquisition_platform.ingest.tle import parse_tle
from acquisition_platform.planner.controller_select import GroundAntennaSpec
from acquisition_platform.planner.session import PlanSession
from acquisition_platform.safety import WatchdogConfig
from acquisition_platform.service.store import SessionStore
from antenna_pomdp.config import default_config, network_single_university

SSO = (
    "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
    "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
)


def _session(policy="infogreedy", ground=None, watchdog_cfg=None, authority="L1_SHADOW"):
    base = default_config()
    l1, l2 = SSO.splitlines()
    cfg = replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=l1, nominal_tle_line2=l2),
        antenna=replace(base.antenna, fwhm_deg=10.0, dwell_time_s=10.0),
        filter=replace(base.filter, n_particles=48),
    )
    rng = np.random.default_rng(0)
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, _, _ = parse_tle(SSO, cov_6x6=cov, frame="RTN")
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)
    kw = {}
    if watchdog_cfg is not None:
        kw["watchdog_cfg"] = watchdog_cfg
    return PlanSession(
        config=cfg,
        network=network_single_university(),
        belief=belief,
        policy=policy,
        ground_antenna=ground,
        authority_level=authority,
        leop_start=opm.epoch,
        ukf=ukf,
        rng=rng,
        **kw,
    )


# ---------------------------------------------------------------------------
# Session round-trip
# ---------------------------------------------------------------------------


def test_session_roundtrip_belief_fidelity():
    sess = _session()
    plans = sess.plan_network(horizon_h=8.0)
    first = next(p for p in plans if p.pointings)
    sess.observe(first.idx, 0, detected=False)

    store = SessionStore(":memory:")
    store.save_session("s1", sess)
    r = store.load_session("s1")
    assert r is not None
    assert np.allclose(r.belief.weights, sess.belief.weights)
    assert [p.launch_slip_s for p in r.belief.particles] == [
        p.launch_slip_s for p in sess.belief.particles
    ]
    assert r.leop_start == sess.leop_start
    assert r.authority_level == sess.authority_level
    assert r.policy == sess.policy


def test_reload_is_deterministic():
    sess = _session()
    sess.plan_network(horizon_h=8.0)
    store = SessionStore(":memory:")
    store.save_session("s1", sess)

    def fp(s):
        return [(round(pt.az_rad, 9), round(pt.el_rad, 9)) for p in s._plans for pt in p.pointings]

    assert fp(store.load_session("s1")) == fp(store.load_session("s1"))


def test_roundtrip_preserves_watchdog_fallback_state():
    # Trip the watchdog, then persist+restore; the fallback + rationale survive.
    sess = _session(watchdog_cfg=WatchdogConfig(ess_floor_frac=0.999, divergence_consecutive=1))
    plans = sess.plan_network(horizon_h=10.0)
    first = next(p for p in plans if p.pointings)
    sess.observe(first.idx, 0, detected=False)
    assert sess.fallback_active and sess.policy == "sweep"

    store = SessionStore(":memory:")
    store.save_session("s1", sess)
    r = store.load_session("s1")
    assert r.fallback_active and r.policy == "sweep"
    assert r._policy_before_fallback == "infogreedy"
    # controller_choice preserved (reconstructed from the pre-fallback controller).
    assert r.controller_choice is not None


def test_roundtrip_preserves_controller_choice_and_ground_spec():
    sess = _session(policy="auto", ground=GroundAntennaSpec(mount="phased_array", fwhm_deg=2.0))
    sess.plan_network(horizon_h=8.0)
    assert sess.policy == "infogreedy"
    store = SessionStore(":memory:")
    store.save_session("s1", sess)
    r = store.load_session("s1")
    assert r.policy == "infogreedy"
    assert r.controller_choice is not None and r.controller_choice.is_agile


def test_delete_and_listing():
    sess = _session()
    sess.plan_network(horizon_h=6.0)
    store = SessionStore(":memory:")
    store.save_session("s1", sess)
    assert store.session_ids() == ["s1"]
    store.delete_session("s1")
    assert store.session_ids() == []
    assert store.load_session("s1") is None


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_one_bad_snapshot_does_not_abort_recovery():
    # Regression: an un-rebuildable checkpoint must be skipped, not abort the
    # recovery of every other session.
    sess = _session()
    sess.plan_network(horizon_h=6.0)
    store = SessionStore(":memory:")
    store.save_session("good", sess)
    # Inject a corrupt snapshot directly (garbage TLE lines => SGP4/rebuild fails).
    import json

    bad = json.loads(json.dumps({"schema_version": 1, "config": {}}))  # missing everything
    store._conn.execute(
        "INSERT INTO sessions(session_id, tenant_id, updated_ts, snapshot) VALUES(?,?,?,?)",
        ("bad", "default", "2026-01-01T00:00:00+00:00", json.dumps(bad)),
    )
    store._conn.commit()
    loaded = store.load_all_sessions()
    assert "good" in loaded and "bad" not in loaded
    assert "bad" in store.skipped


def test_audit_chain_intact():
    store = SessionStore(":memory:")
    for i in range(5):
        store.append_audit("observe", session_id="s1", detail={"i": i})
    assert store.verify_chain() is None
    assert len(store.audit_log("s1")) == 5


def test_audit_is_append_only():
    store = SessionStore(":memory:")
    store.append_audit("session.create", session_id="s1")
    with pytest.raises(Exception):  # noqa: B017 - sqlite raises on the trigger
        store._conn.execute("UPDATE audit SET action='x' WHERE seq=1")
    with pytest.raises(Exception):  # noqa: B017
        store._conn.execute("DELETE FROM audit WHERE seq=1")


def test_audit_tamper_detected():
    store = SessionStore(":memory:")
    store.append_audit("a", session_id="s1")
    store.append_audit("b", session_id="s1")
    store.append_audit("c", session_id="s1")
    # Bypass the guard and forge a row's payload; the chain no longer verifies.
    store._conn.execute("DROP TRIGGER audit_no_update")
    store._conn.execute("UPDATE audit SET action='forged' WHERE seq=2")
    store._conn.commit()
    assert store.verify_chain() == 2


# ---------------------------------------------------------------------------
# App-level restart recovery
# ---------------------------------------------------------------------------


def test_offline_mode_fails_fast(monkeypatch):
    # On an air-gapped appliance an outbound catalog fetch must fail immediately.
    from acquisition_platform.ingest import catalog

    monkeypatch.setenv("LEOPT_OFFLINE", "1")
    with pytest.raises(OSError, match="offline"):
        catalog._http_get("https://celestrak.org/whatever", timeout=5.0)


def test_app_recovers_sessions_after_restart(tmp_path, monkeypatch):
    import acquisition_platform.service.app as appmod

    db = str(tmp_path / "state.db")
    monkeypatch.setenv("LEOPT_STATE_DB", db)
    # Reset module-level persistence state for a clean slate.
    if appmod._STORE is not None:
        appmod._STORE.close()
    appmod._STORE = None
    appmod._SESSIONS.clear()
    try:
        client = TestClient(appmod.app)
        body = {
            "ingest": {"format": "tle", "text": SSO},
            "network": "single_university",
            "antenna": {"fwhm_deg": 10.0, "dwell_s": 10.0},
            "horizon_h": 8.0,
            "n_particles": 48,
            "rollouts": 20,
            "policy": "auto",
        }
        r = client.post("/api/plan", json=body)
        sid = r.json()["session_id"]
        first = next(p for p in r.json()["passes"] if p["pointings"])
        client.post(
            "/api/observe",
            json={"session_id": sid, "pass_idx": first["idx"], "dwell_idx": 0, "detected": False},
        )
        # Audit recorded and intact.
        audit = client.get(f"/api/audit/{sid}").json()
        assert audit["enabled"] and audit["chain_ok"]
        assert len(audit["entries"]) >= 2  # create + observe

        # Simulate a process restart: drop in-memory state, reopen store, recover.
        appmod._STORE.close()
        appmod._STORE = None
        appmod._SESSIONS.clear()
        assert sid not in appmod._SESSIONS
        appmod._recover_sessions()
        assert sid in appmod._SESSIONS
        # The recovered session still serves belief + accepts observations.
        assert client.get(f"/api/belief/{sid}").status_code == 200
    finally:
        if appmod._STORE is not None:
            appmod._STORE.close()
        appmod._STORE = None
        appmod._SESSIONS.clear()
        monkeypatch.delenv("LEOPT_STATE_DB", raising=False)
