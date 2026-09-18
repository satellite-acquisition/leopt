"""Tests for the prototype antenna-driven controller-selection heuristic.

Covers the selection rule, the PlanSession wiring of the info-greedy / PFT-DPW
controllers, the slew-limited planning clamp, and the two new API seams.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from acquisition_platform.estimator.ukf import SgpUkf
from acquisition_platform.ingest.tle import parse_tle
from acquisition_platform.planner.controller_select import (
    AGILE_SLEW_DEG_S,
    INFO_GREEDY,
    PFT_DPW,
    GroundAntennaSpec,
    list_mounts,
    recommend_controller,
)
from acquisition_platform.planner.session import PlanSession
from acquisition_platform.service.app import app
from antenna_pomdp.config import default_config, network_single_university
from antenna_pomdp.orbit.geometry import angular_separation

SSO = (
    "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
    "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
)


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def test_phased_array_selects_info_greedy():
    c = recommend_controller(GroundAntennaSpec(mount="phased_array", fwhm_deg=2.0))
    assert c.policy == INFO_GREEDY and c.is_agile
    assert c.slew_rate_deg_s >= AGILE_SLEW_DEG_S


def test_slow_dish_short_cadence_selects_pft():
    # Very slow mount, pencil beam, short re-point cadence => dwells couple.
    c = recommend_controller(GroundAntennaSpec(slew_rate_deg_s=0.1, fwhm_deg=1.0, dwell_s=5.0))
    assert c.policy == PFT_DPW and not c.is_agile
    assert c.repoint_budget_deg < c.coupling_span_deg


def test_agile_flag_forces_info_greedy():
    c = recommend_controller(GroundAntennaSpec(mount="legacy_slow", agile=True))
    assert c.policy == INFO_GREEDY and c.is_agile


def test_long_cadence_decouples_a_dish():
    # The same standard dish decouples once given a realistic (hold-scale) cadence.
    coupled = recommend_controller(GroundAntennaSpec(mount="standard_dish"), inter_dwell_s=5.0)
    free = recommend_controller(GroundAntennaSpec(mount="standard_dish"), inter_dwell_s=45.0)
    assert free.repoint_budget_deg > coupled.repoint_budget_deg
    assert free.policy == INFO_GREEDY


def test_list_mounts_shape():
    mounts = list_mounts()
    assert any(m["id"] == "phased_array" and m["agile"] for m in mounts)
    assert all({"id", "name", "slew_rate_deg_s", "agile", "description"} <= set(m) for m in mounts)


# ---------------------------------------------------------------------------
# PlanSession wiring
# ---------------------------------------------------------------------------


def _build_session(policy="auto", ground=None, link_budget=True):
    base = default_config()
    l1, l2 = SSO.splitlines()
    cfg = replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=l1, nominal_tle_line2=l2),
        antenna=replace(
            base.antenna, fwhm_deg=10.0, dwell_time_s=10.0, link_budget_enabled=link_budget
        ),
        filter=replace(base.filter, n_particles=64),
        pomcp=replace(base.pomcp, n_rollouts=20, max_depth=3),
        solver=replace(base.solver, n_iterations=30, max_depth=4, n_belief_particles=16),
    )
    rng = np.random.default_rng(0)
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, _, _ = parse_tle(SSO, cov_6x6=cov, frame="RTN")
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)
    return PlanSession(
        config=cfg,
        network=network_single_university(),
        belief=belief,
        policy=policy,
        ground_antenna=ground,
        leop_start=opm.epoch,
        ukf=ukf,
        rng=rng,
    )


@pytest.mark.parametrize("policy", [INFO_GREEDY, PFT_DPW])
def test_new_controllers_plan_and_observe(policy):
    sess = _build_session(policy=policy)
    plans = sess.plan_network(horizon_h=8.0)
    assert any(len(p.pointings) > 0 for p in plans)
    for p in plans:
        for pt in p.pointings:
            assert 0.0 <= pt.p_detect <= 1.0
    first = next(p for p in plans if p.pointings)
    ent0 = sess.belief_summary()["entropy"]
    sess.observe(first.idx, 0, detected=False)
    assert sess.belief_summary()["entropy"] != ent0


def test_auto_phased_array_resolves_to_info_greedy():
    sess = _build_session(
        policy="auto", ground=GroundAntennaSpec(mount="phased_array", fwhm_deg=2.0)
    )
    assert sess.policy == INFO_GREEDY
    assert sess.controller_choice is not None and sess.controller_choice.is_agile
    plans = sess.plan_network(horizon_h=8.0)
    assert any(len(p.pointings) > 0 for p in plans)


def test_auto_slow_dish_resolves_to_pft():
    sess = _build_session(
        policy="auto", ground=GroundAntennaSpec(slew_rate_deg_s=0.05, fwhm_deg=1.0, dwell_s=5.0)
    )
    assert sess.policy == PFT_DPW
    # The ground spec is authoritative for the slew constraint.
    assert sess.config.antenna.max_slew_rate_deg_s == pytest.approx(0.05)


def test_slew_clamp_bounds_consecutive_boresights():
    # A slow dish: within a pass, successive commanded boresights cannot jump
    # farther than the per-hold slew budget.
    slew = 0.05
    sess = _build_session(
        policy=INFO_GREEDY,
        ground=GroundAntennaSpec(slew_rate_deg_s=slew, fwhm_deg=2.0, dwell_s=5.0),
    )
    plans = sess.plan_network(horizon_h=12.0)
    checked = 0
    for p in plans:
        for a, b in zip(p.pointings, p.pointings[1:]):
            # The move a->b is budgeted by the hold spent parked at a.
            max_move = np.deg2rad(slew) * a.dwell_s
            sep = angular_separation(a.az_rad, a.el_rad, b.az_rad, b.el_rad)
            assert sep <= max_move + 1e-6, (np.rad2deg(sep), np.rad2deg(max_move))
            checked += 1
    assert checked > 0, "expected multi-dwell passes to exercise the clamp"


def test_p_detect_threads_the_link_budget():
    # Regression: _belief_p_detect must use elevation+range (the link budget),
    # not the constant-beam model. Toggling the link budget must change P(detect)
    # for the same geometry — if el/range were dropped, both would be identical.
    from antenna_pomdp.models.observation import Pointing

    def first_pd(link_budget):
        s = _build_session(policy="sweep", link_budget=link_budget)
        plans = s.plan_network(horizon_h=8.0)
        p = next(pp for pp in plans if pp.pointings)
        pt = p.pointings[0]
        return s._belief_p_detect(s.belief, Pointing(pt.az_rad, pt.el_rad), pt.when, p.station)

    assert first_pd(True) != first_pd(False)


def test_watchdog_sees_pre_resample_belief():
    # Regression: the watchdog health belief is the reweighted (pre-resample)
    # posterior, so a discriminating observation makes its weights NON-uniform —
    # the degeneracy signal the post-resample belief would have masked.
    from antenna_pomdp.models.observation import Pointing

    s = _build_session(policy="sweep")
    plans = s.plan_network(horizon_h=8.0)
    p = next(pp for pp in plans if pp.pointings)
    pt = p.pointings[0]
    reweighted, lik = s._prior_reweight(Pointing(pt.az_rad, pt.el_rad), False, pt.when, p.station)
    assert 0.0 <= lik <= 1.0
    assert float(np.std(reweighted.weights)) > 0.0  # not uniform => resample not applied


def test_observe_with_doppler_sharpens_belief():
    # A/B: the SAME observation with a matching Doppler collapses the along-track
    # (launch-slip) spread more than the bare no-detect alone. Two identical
    # sessions isolate the Doppler fold-in from the shared diffusion/angle update.
    from antenna_pomdp.models.doppler import particle_doppler_hz

    def _std_after(with_doppler):
        s = _build_session(policy=INFO_GREEDY)
        plans = s.plan_network(horizon_h=8.0)
        first = next(p for p in plans if p.pointings)
        pt = s._plans[first.idx].pointings[0]
        i = int(np.argmax(s.belief.weights))
        dop = particle_doppler_hz(s.belief.particles[i], pt.when, first.station, s.freq_cfg)
        s.observe(first.idx, 0, detected=False, doppler_hz=(dop if with_doppler else None))
        return float(np.std([p.launch_slip_s for p in s.belief.particles]))

    assert _std_after(with_doppler=True) < _std_after(with_doppler=False)


def test_unconstrained_slew_leaves_schedule_unclamped():
    # Default (huge) slew => no clamp: pointings can be arbitrarily far apart.
    sess = _build_session(policy=INFO_GREEDY, ground=GroundAntennaSpec(agile=True, fwhm_deg=10.0))
    assert sess.config.antenna.max_slew_rate_deg_s >= 1.0e8
    plans = sess.plan_network(horizon_h=8.0)
    assert any(len(p.pointings) > 0 for p in plans)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_api_mounts(client):
    r = client.get("/api/mounts")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()}
    assert "phased_array" in ids and "standard_dish" in ids


def test_api_recommend_controller(client):
    r = client.post(
        "/api/recommend-controller",
        json={"antenna": {"fwhm_deg": 2.0, "mount": "phased_array"}, "sat_antenna": "turnstile"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["policy"] == INFO_GREEDY
    assert body["is_agile"] and body["rationale"]


def test_api_plan_auto_returns_controller(client):
    body = {
        "ingest": {"format": "tle", "text": SSO},
        "network": "single_university",
        "antenna": {"fwhm_deg": 10.0, "dwell_s": 10.0, "mount": "phased_array"},
        "horizon_h": 8.0,
        "n_particles": 64,
        "rollouts": 20,
        "policy": "auto",
    }
    r = client.post("/api/plan", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["active_policy"] == INFO_GREEDY
    assert res["controller"] is not None
    assert res["controller"]["policy"] == INFO_GREEDY
    assert res["passes"]
