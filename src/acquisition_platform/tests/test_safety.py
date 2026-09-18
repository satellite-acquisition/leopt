"""Tests for the trust & safety layer: envelope, watchdog, authority, wiring."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from acquisition_platform.estimator.ukf import SgpUkf
from acquisition_platform.ingest.tle import parse_tle
from acquisition_platform.planner.session import PlanSession
from acquisition_platform.safety import (
    CLOSED_LOOP,
    FALLBACK_OPEN_LOOP,
    L0_ADVISORY,
    L2_SUPERVISED,
    KeepOutBox,
    PromotionConfig,
    PromotionEvidence,
    SafetyConfig,
    WatchdogConfig,
    WatchdogState,
    can_actuate,
    can_promote,
    classify_pointing,
    evaluate_pointing,
    sun_azel,
    watchdog_step,
)
from acquisition_platform.service.app import app
from antenna_pomdp.config import StationConfig, default_config, network_single_university
from antenna_pomdp.models.particle_filter import ParticleBelief
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle

SSO = (
    "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
    "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
)


# ---------------------------------------------------------------------------
# Safety envelope
# ---------------------------------------------------------------------------


def test_envelope_elevation_floor_and_mask():
    cfg = SafetyConfig()  # floor 3, mask 5
    assert classify_pointing(120.0, 2.0, cfg).status == "reject"
    v = classify_pointing(120.0, 4.0, cfg)
    assert v.status == "rf_inhibit" and v.actuatable and not v.transmit_ok


def test_envelope_keyhole_and_sun():
    cfg = SafetyConfig()
    assert classify_pointing(120.0, 88.5, cfg).status == "keyhole"
    assert classify_pointing(120.0, 45.0, cfg, sun_sep_deg=2.0).status == "reject"
    assert classify_pointing(120.0, 45.0, cfg, sun_sep_deg=12.0).status == "warn"
    assert classify_pointing(120.0, 45.0, cfg, sun_sep_deg=90.0).status == "ok"


def test_envelope_keep_out_box():
    cfg = SafetyConfig(keep_out=(KeepOutBox(100.0, 140.0, 0.0, 30.0, name="GEO"),))
    v = classify_pointing(120.0, 20.0, cfg)
    assert v.status == "rf_inhibit" and not v.transmit_ok
    # Outside the box -> ok.
    assert classify_pointing(200.0, 20.0, cfg).status == "ok"


def test_keep_out_box_wraps_azimuth():
    box = KeepOutBox(350.0, 10.0, 0.0, 90.0)  # crosses 0/360
    assert box.contains(355.0, 20.0) and box.contains(5.0, 20.0)
    assert not box.contains(180.0, 20.0)


def test_full_azimuth_keep_out_matches_all_az():
    # Regression: az_max=360 must not collapse to only az=0.
    ring = KeepOutBox(0.0, 360.0, 0.0, 30.0, name="ring")
    assert ring.contains(0.0, 10.0) and ring.contains(180.0, 10.0) and ring.contains(359.0, 10.0)
    assert not ring.contains(180.0, 40.0)  # still bounded in elevation


def test_hard_keep_out_bars_actuation():
    # Regression: transmit_only=False is a HARD keep-out (not actuatable), not a
    # transmit-only inhibit.
    cfg = SafetyConfig(
        keep_out=(KeepOutBox(100.0, 140.0, 0.0, 30.0, name="no-go", transmit_only=False),)
    )
    v = classify_pointing(120.0, 20.0, cfg)
    assert v.status == "reject" and not v.actuatable and not v.transmit_ok
    # A transmit_only keep-out remains actuatable (track ok, no keying).
    soft = SafetyConfig(keep_out=(KeepOutBox(100.0, 140.0, 0.0, 30.0, transmit_only=True),))
    v2 = classify_pointing(120.0, 20.0, soft)
    assert v2.status == "rf_inhibit" and v2.actuatable and not v2.transmit_ok


def test_mask_cannot_sit_below_floor():
    cfg = SafetyConfig(hard_el_lower_deg=6.0, min_elevation_deg=3.0)
    assert cfg.min_elevation_deg == 6.0  # clamped up to the floor


def test_evaluate_pointing_uses_sun_ephemeris():
    st = Station.from_config(StationConfig(latitude_deg=0.0, longitude_deg=0.0))
    when = datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc)
    s_az, s_el = sun_azel(when, st)
    # Point straight at the Sun -> hard reject (if it's up), else at least defined.
    v = evaluate_pointing(s_az, max(s_el, np.deg2rad(10.0)), when, st, SafetyConfig())
    assert v.status in ("reject", "warn", "keyhole")


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


def _belief(weights):
    l1, l2 = SSO.splitlines()
    parts = [OrbitParticle(SGP4Propagator(l1, l2), float(s)) for s in range(len(weights))]
    w = np.asarray(weights, dtype=float)
    return ParticleBelief(particles=parts, weights=w / w.sum())


def test_watchdog_healthy_stays_closed():
    st = WatchdogState()
    assert (
        watchdog_step(_belief(np.ones(50)), st, WatchdogConfig(), likelihood_total=0.9)
        == CLOSED_LOOP
    )
    assert not st.latched_open_loop


def test_watchdog_latches_on_degeneracy_under_no_detect():
    w = np.full(50, 1e-6)
    w[0] = 1.0
    st = WatchdogState()
    mode = watchdog_step(_belief(w), st, WatchdogConfig(), detected=False, likelihood_total=0.9)
    assert mode == FALLBACK_OPEN_LOOP and "degeneracy" in st.last_reason


def test_watchdog_does_not_latch_on_supported_detection():
    # The same collapsed belief, but arriving as a SUPPORTED detect, is acquisition.
    w = np.full(50, 1e-6)
    w[0] = 1.0
    st = WatchdogState()
    mode = watchdog_step(_belief(w), st, WatchdogConfig(), detected=True, likelihood_total=0.8)
    assert mode == CLOSED_LOOP and st.acquired and not st.latched_open_loop


def test_watchdog_rearm_clears_latch():
    w = np.full(50, 1e-6)
    w[0] = 1.0
    st = WatchdogState()
    watchdog_step(_belief(w), st, WatchdogConfig(), detected=False, likelihood_total=0.9)
    assert st.latched_open_loop
    st.rearm()
    assert not st.latched_open_loop and st.consecutive_divergence == 0


# ---------------------------------------------------------------------------
# Authority ladder
# ---------------------------------------------------------------------------


def test_authority_actuation_gate():
    assert not can_actuate("L1_SHADOW")
    assert can_actuate(L2_SUPERVISED)


def test_promotion_gate():
    cfg = PromotionConfig()
    ok, why = can_promote("L1_SHADOW", PromotionEvidence(10, 0, 0.1, 10.0, True), cfg)
    assert not ok and "need 50" in why
    ok, _ = can_promote("L1_SHADOW", PromotionEvidence(60, 1, 0.1, 10.0, True), cfg)
    assert not ok  # a violation blocks it
    ok, _ = can_promote("L1_SHADOW", PromotionEvidence(60, 0, 1.0, 10.0, True), cfg)
    assert ok


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------


def _session(policy="infogreedy", authority=L0_ADVISORY, safety=None, watchdog_cfg=None):
    base = default_config()
    l1, l2 = SSO.splitlines()
    cfg = replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=l1, nominal_tle_line2=l2),
        antenna=replace(base.antenna, fwhm_deg=10.0, dwell_time_s=10.0),
        filter=replace(base.filter, n_particles=64),
        pomcp=replace(base.pomcp, n_rollouts=20, max_depth=3),
    )
    rng = np.random.default_rng(0)
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, _, _ = parse_tle(SSO, cov_6x6=cov, frame="RTN")
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)
    kw = {}
    if safety is not None:
        kw["safety"] = safety
    if watchdog_cfg is not None:
        kw["watchdog_cfg"] = watchdog_cfg
    return PlanSession(
        config=cfg,
        network=network_single_university(),
        belief=belief,
        policy=policy,
        authority_level=authority,
        leop_start=opm.epoch,
        ukf=ukf,
        rng=rng,
        **kw,
    )


def test_plan_annotates_safety_and_ops():
    sess = _session()
    sess.plan_network(horizon_h=8.0)
    ops = sess.ops_status()
    assert ops["trust"]["level"] == L0_ADVISORY and ops["trust"]["can_actuate"] is False
    assert ops["safety"]["total_dwells"] > 0
    assert ops["watchdog"]["mode"] == CLOSED_LOOP
    for p in sess._plans:
        for pt in p.pointings:
            assert pt.safety_status in ("ok", "warn", "keyhole", "rf_inhibit", "reject")


def test_explain_reflects_no_detect_streak():
    sess = _session()
    plans = sess.plan_network(horizon_h=8.0)
    first = next(p for p in plans if p.pointings)
    sess.observe(first.idx, 0, detected=False)
    text = sess.explain()
    assert "no-detect" in text.lower()


def test_watchdog_fallback_switches_to_sweep_and_rearm_restores():
    # A hair-trigger watchdog: any post-update ESS dip under a no-detect latches.
    sess = _session(
        policy="infogreedy",
        watchdog_cfg=WatchdogConfig(ess_floor_frac=0.999, divergence_consecutive=1),
    )
    plans = sess.plan_network(horizon_h=10.0)
    first = next(p for p in plans if p.pointings)
    sess.observe(first.idx, 0, detected=False)
    assert sess.fallback_active and sess.policy == "sweep"
    assert sess.ops_status()["watchdog"]["latched"] is True
    assert "watchdog" in sess.explain().lower()
    # Re-arm restores the original controller and clears the latch.
    sess.rearm()
    assert not sess.fallback_active and sess.policy == "infogreedy"


def test_high_min_elevation_forces_rf_inhibit():
    sess = _session(safety=SafetyConfig(hard_el_lower_deg=3.0, min_elevation_deg=80.0))
    sess.plan_network(horizon_h=10.0)
    summ = sess.safety_summary()
    # Almost every dwell sits below an 80° transmit mask.
    assert summ["rf_inhibit"] + summ["reject"] > 0


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _plan_body(**over):
    body = {
        "ingest": {"format": "tle", "text": SSO},
        "network": "single_university",
        "antenna": {"fwhm_deg": 10.0, "dwell_s": 10.0},
        "horizon_h": 8.0,
        "n_particles": 64,
        "rollouts": 20,
        "policy": "auto",
    }
    body.update(over)
    return body


def test_api_plan_returns_ops(client):
    r = client.post("/api/plan", json=_plan_body())
    assert r.status_code == 200, r.text
    ops = r.json()["ops"]
    assert ops["trust"]["level"] == "L0_ADVISORY"
    assert "rationale" in ops and "safety" in ops


def test_api_authority_change(client):
    r = client.post("/api/plan", json=_plan_body(authority_level="L0_ADVISORY"))
    sid = r.json()["session_id"]
    r2 = client.post("/api/authority", json={"session_id": sid, "level": "L2_SUPERVISED"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["ops"]["trust"]["can_actuate"] is True


def test_api_rearm_requires_latched(client):
    r = client.post("/api/plan", json=_plan_body())
    sid = r.json()["session_id"]
    r2 = client.post("/api/rearm", json={"session_id": sid})
    assert r2.status_code == 400  # not latched -> nothing to re-arm
