"""Map a PlanSession's internal plan + belief into the wire schemas.

All angles are emitted in DEGREES (the boundary convention); times as ISO-8601
strings and seconds-since-rise floats. This is the only place degrees<->radians
conversion happens for the plan output.
"""

from __future__ import annotations

import numpy as np

from acquisition_platform.ingest.schemas import (
    BeliefGrid,
    BeliefSummary,
    ControllerChoiceOut,
    OpsStatusOut,
    PassOut,
    PlanResult,
    PointingOut,
)
from acquisition_platform.planner.session import PassPlan, PlanSession


def pass_to_schema(p: PassPlan) -> PassOut:
    return PassOut(
        idx=p.idx,
        station=p.station_id,
        rise=p.rise.isoformat(),
        set=p.set.isoformat(),
        peak_el_deg=float(p.peak_el_deg),
        pointings=[
            PointingOut(
                t_s=float(pt.t_s),
                az_deg=float(np.rad2deg(pt.az_rad)),
                el_deg=float(np.rad2deg(pt.el_rad)),
                d_tau_s=float(pt.d_tau_s),
                d_theta_deg=float(pt.d_theta_deg),
                p_detect=float(pt.p_detect),
                dwell_s=float(pt.dwell_s),
                safety_status=pt.safety_status,
                transmit_ok=pt.transmit_ok,
                safety_reason=pt.safety_reason,
            )
            for pt in p.pointings
        ],
    )


def belief_summary_to_schema(summary: dict) -> BeliefSummary:
    g = summary["grid"]
    return BeliefSummary(
        entropy=float(summary["entropy"]),
        grid=BeliefGrid(along_s=g["along_s"], cross_deg=g["cross_deg"], weight=g["weight"]),
    )


def controller_choice_to_schema(session: PlanSession) -> ControllerChoiceOut | None:
    choice = session.controller_choice
    if choice is None:
        return None
    return ControllerChoiceOut(**choice.to_dict())


def plan_result(session_id: str, session: PlanSession, plans: list[PassPlan]) -> PlanResult:
    return PlanResult(
        session_id=session_id,
        passes=[pass_to_schema(p) for p in plans],
        belief_summary=belief_summary_to_schema(session.belief_summary()),
        active_policy=session.policy,
        controller=controller_choice_to_schema(session),
        ops=OpsStatusOut(**session.ops_status()),
    )


if __name__ == "__main__":
    from dataclasses import replace

    from acquisition_platform.estimator.ukf import SgpUkf
    from acquisition_platform.ingest.tle import parse_tle
    from antenna_pomdp.config import default_config, network_single_university

    sso_l1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
    sso_l2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    base = default_config()
    cfg = replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=sso_l1, nominal_tle_line2=sso_l2),
        filter=replace(base.filter, n_particles=80),
        pomcp=replace(base.pomcp, n_rollouts=20, max_depth=3),
    )
    rng = np.random.default_rng(0)
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, _, _ = parse_tle(sso_l1 + "\n" + sso_l2, cov_6x6=cov, frame="RTN")
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)
    sess = PlanSession(
        config=cfg,
        network=network_single_university(),
        belief=belief,
        policy="sweep",
        leop_start=opm.epoch,
        rng=rng,
    )
    plans = sess.plan_network(horizon_h=8.0)
    res = plan_result("test-session", sess, plans)
    dumped = res.model_dump_json()
    assert PlanResult.model_validate_json(dumped).session_id == "test-session"
    print(f"[schedule] passes={len(res.passes)}, json bytes={len(dumped)}")
    print("[schedule] PASS")
