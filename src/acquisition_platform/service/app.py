"""LEOPT planning API and browser application.

Request and response models live in ``ingest.schemas``. Sessions are kept in
memory; set ``LEOPT_STATE_DB`` to checkpoint them in SQLite across restarts.
The browser application is served from ``/``.
"""

from __future__ import annotations

import base64
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

from acquisition_platform.estimator.ukf import SgpUkf
from acquisition_platform.ingest.catalog import CelesTrakClient, SpaceTrackClient
from acquisition_platform.ingest.detect import IngestOutcome, parse_any
from acquisition_platform.ingest.opm import OpmState
from acquisition_platform.ingest.schemas import (
    AntennaSpec,
    BeliefResult,
    CatalogQuery,
    CatalogResult,
    ControllerChoiceOut,
    DriverOut,
    IngestRequest,
    IngestResult,
    MountOut,
    NetworkOut,
    ObservationSourceOut,
    ObserveRequest,
    PlanRequest,
    PlanResult,
    RearmRequest,
    RecommendControllerRequest,
    SatAntennaOut,
    ScheduleExportRequest,
    ScheduleExportResult,
    SetAuthorityRequest,
    StationOut,
    TrackExportRequest,
    TrackExportResult,
)
from acquisition_platform.satellite_antennas import (
    get_sat_antenna,
    list_sat_antennas,
    recommended_hold_s,
)
from acquisition_platform.planner.controller_select import (
    GroundAntennaSpec,
    list_mounts,
    recommend_controller,
)
from acquisition_platform.planner.schedule import (
    belief_summary_to_schema,
    plan_result,
)
from acquisition_platform.planner.session import PlanSession
from acquisition_platform.planner.schedule_export import (
    SCHEMA_VERSION as SCHEDULE_SCHEMA_VERSION,
    ScheduleExportError,
    SchedulePdfDependencyError,
    build_schedule_artifacts,
)
from acquisition_platform.service.store import SessionStore
from antenna_pomdp.config import (
    GroundStationNetworkConfig,
    NETWORK_FACTORIES,
    StationConfig,
    default_config,
)

# In-memory session store: uuid -> PlanSession.
_SESSIONS: dict[str, PlanSession] = {}
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()

# Optional checkpoints and audit log, enabled by LEOPT_STATE_DB.
_STORE: SessionStore | None = None


def _get_store() -> SessionStore | None:
    global _STORE
    if _STORE is None:
        path = os.environ.get("LEOPT_STATE_DB")
        if path:
            _STORE = SessionStore(path)
    return _STORE


def _session_lock(session_id: str) -> threading.RLock:
    """Return the per-session lock shared by plan mutations and exports."""
    with _SESSION_LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(session_id, threading.RLock())


def _require_session(session_id: str) -> PlanSession:
    """Resolve an existing session without allocating state for invalid IDs."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return session


def _recover_sessions() -> None:
    """Restore readable session checkpoints on startup."""
    store = _get_store()
    if store is None:
        return
    _SESSIONS.update(store.load_all_sessions())
    try:
        store.append_audit(
            "service.startup",
            detail={"recovered": len(_SESSIONS), "skipped": list(store.skipped)},
        )
    except Exception:  # noqa: BLE001 - a locked/read-only DB must not block boot
        pass


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Persistence failures must degrade to in-memory-only, never kill startup.
    try:
        _recover_sessions()
    except Exception:  # noqa: BLE001 - defensive: recovery is best-effort
        _SESSIONS.clear()
    yield


app = FastAPI(title="LEOPT", lifespan=_lifespan)


def _checkpoint(sid: str, session: PlanSession, action: str, detail: dict | None = None) -> None:
    """Persist a session and append an audit record (no-op if persistence is off)."""
    store = _get_store()
    if store is None:
        return
    store.save_session(sid, session)
    store.append_audit(action, session_id=sid, detail=detail or {})


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


@app.get("/api/networks", response_model=list[NetworkOut])
def list_networks() -> list[NetworkOut]:
    out: list[NetworkOut] = []
    for nid, factory in NETWORK_FACTORIES.items():
        net = factory()
        labels = net.labels()
        out.append(
            NetworkOut(
                id=nid,
                name=net.name,
                n_stations=len(net.stations),
                stations=[
                    StationOut(
                        name=labels[i],
                        lat_deg=s.latitude_deg,
                        lon_deg=s.longitude_deg,
                        alt_m=s.altitude_m,
                        min_el_deg=s.min_elevation_deg,
                    )
                    for i, s in enumerate(net.stations)
                ],
            )
        )
    return out


@app.get("/api/sat-antennas", response_model=list[SatAntennaOut])
def list_satellite_antennas() -> list[SatAntennaOut]:
    return [
        SatAntennaOut(
            id=a.id,
            name=a.name,
            description=a.description,
            coverage=a.coverage,
            base_hold_s=a.base_hold_s,
        )
        for a in list_sat_antennas()
    ]


# ---------------------------------------------------------------------------
# Controller selection
# ---------------------------------------------------------------------------


def _ground_antenna_spec(a: AntennaSpec) -> GroundAntennaSpec:
    """Build the ground-antenna agility spec from a request AntennaSpec."""
    return GroundAntennaSpec(
        mount=a.mount,
        slew_rate_deg_s=a.slew_rate_deg_s,
        agile=a.agile,
        fwhm_deg=float(a.fwhm_deg),
        dwell_s=float(a.dwell_s),
    )


@app.get("/api/mounts", response_model=list[MountOut])
def list_ground_mounts() -> list[MountOut]:
    """Ground-antenna mount presets for the agility picker."""
    return [MountOut(**m) for m in list_mounts()]


@app.post("/api/recommend-controller", response_model=ControllerChoiceOut)
def recommend_controller_endpoint(req: RecommendControllerRequest) -> ControllerChoiceOut:
    """Preview which controller `policy="auto"` would pick for this antenna.

    Lets the operator see the recommendation and its plain-language rationale
    before committing to a plan. The re-point cadence is the spacecraft antenna's
    representative hold (the real schedule cadence).
    """
    spec = _ground_antenna_spec(req.antenna)
    gap = recommended_hold_s(get_sat_antenna(req.sat_antenna), 30.0)
    choice = recommend_controller(spec, inter_dwell_s=gap)
    return ControllerChoiceOut(**choice.to_dict())


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------


def _ingest_outcome(body: dict) -> IngestOutcome:
    """Parse a raw {format?, text, covariance?, frame?, sample_epoch?} body.

    `format` defaults to "auto" (sniffed from the payload) so any of TLE / OPM /
    OEM / NDM-XML / JSON / OMM is accepted on a single seam.
    """
    fmt = body.get("format") or "auto"
    text = body.get("text", "")
    cov = body.get("covariance")
    frame = body.get("frame") or "ECI"
    cov_np = np.asarray(cov, dtype=float) if cov is not None else None
    sample_epoch = None
    if body.get("sample_epoch"):
        from acquisition_platform.ingest.opm import _parse_epoch

        sample_epoch = _parse_epoch(str(body["sample_epoch"]))
    try:
        return parse_any(text, fmt=fmt, cov_6x6=cov_np, frame=frame, sample_epoch=sample_epoch)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"could not parse ingest input: {exc}") from exc


def _ingest_to_state(body: dict) -> tuple[OpmState, str | None, str | None]:
    """Back-compat shim: raw body -> (OpmState, line1, line2)."""
    out = _ingest_outcome(body)
    return out.opm, out.line1, out.line2


def _state_summary(out: IngestOutcome) -> IngestResult:
    """Build the /api/ingest summary (RTN sigmas + along-track angle now)."""
    from acquisition_platform.estimator.ukf import cov_in_eci, rtn_basis

    opm = out.opm
    # Position sigmas in RTN (km).
    A = rtn_basis(opm.r_eci_m, opm.v_eci_m)
    cov_eci = cov_in_eci(opm)
    cov_pos_rtn = A @ cov_eci[:3, :3] @ A.T
    sigma_rtn_km = [float(np.sqrt(max(0.0, cov_pos_rtn[i, i])) / 1e3) for i in range(3)]

    # Along-track position sigma -> angular sigma at the current range (rough:
    # use orbital radius as the lever arm). deg = sigma_along_m / |r| (rad).
    r_norm = float(np.linalg.norm(opm.r_eci_m))
    sigma_along_m = float(np.sqrt(max(0.0, cov_pos_rtn[1, 1])))
    along_deg = float(np.rad2deg(sigma_along_m / max(1.0, r_norm)))

    eph = out.ephemeris
    span_s = float((eph.stop - eph.start).total_seconds()) if eph is not None else None

    return IngestResult(
        epoch=opm.epoch.isoformat(),
        r_eci_m=[float(x) for x in opm.r_eci_m],
        v_eci_m=[float(x) for x in opm.v_eci_m],
        sigma_rtn_km=sigma_rtn_km,
        along_track_sigma_deg_now=along_deg,
        source_format=out.source_format,
        has_ephemeris=eph is not None,
        ephemeris_span_s=span_s,
        warnings=list(opm.warnings),
    )


@app.post("/api/ingest", response_model=IngestResult)
def ingest(req: IngestRequest) -> IngestResult:
    return _state_summary(_ingest_outcome(req.model_dump()))


@app.post("/api/catalog/tle", response_model=CatalogResult)
def catalog_tle(q: CatalogQuery) -> CatalogResult:
    """One catalog fetch attempt (Space-Track / CelesTrak) for a launch.

    Returns the TLE if the object is already catalogued, else `found=false` so
    the frontend can poll on an interval until the element set appears and then
    switch the belief from the OPM/OEM prior to the tracked TLE.
    """
    if q.catnr is None and not q.intldes:
        raise HTTPException(status_code=400, detail="provide catnr or intldes")
    try:
        if q.source == "spacetrack":
            if not q.identity or not q.password:
                raise HTTPException(
                    status_code=400, detail="Space-Track needs identity and password"
                )
            client = SpaceTrackClient(identity=q.identity, password=q.password)
        else:
            client = CelesTrakClient()
        pair = client.tle(catnr=q.catnr, intldes=q.intldes)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"catalog fetch failed: {exc}") from exc
    if pair is None:
        return CatalogResult(found=False, source=q.source, message="not catalogued yet")
    return CatalogResult(found=True, line1=pair[0], line2=pair[1], source=q.source)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _resolve_network(network: object) -> GroundStationNetworkConfig:
    if isinstance(network, str):
        factory = NETWORK_FACTORIES.get(network)
        if factory is None:
            raise HTTPException(status_code=400, detail=f"unknown network {network!r}")
        return factory()
    if isinstance(network, list):
        stations = tuple(
            StationConfig(
                latitude_deg=float(s["lat_deg"]),
                longitude_deg=float(s["lon_deg"]),
                altitude_m=float(s.get("alt_m", 0.0)),
                min_elevation_deg=float(s.get("min_el_deg", 5.0)),
            )
            for s in network
        )
        ids = tuple(str(s.get("name", f"S{i}")) for i, s in enumerate(network))
        return GroundStationNetworkConfig(
            network_id="custom",
            name="Custom network",
            stations=stations,
            station_ids=ids,
        )
    raise HTTPException(status_code=400, detail="network must be an id or station list")


def _build_session(req: PlanRequest) -> tuple[str, PlanSession, list]:
    # The ingest field may be a raw {format,text,...} body. (An IngestResult-only
    # body lacks the original text, so we require the raw form for planning.)
    ing = req.ingest
    if "text" not in ing:
        raise HTTPException(
            status_code=400,
            detail="plan.ingest must be the raw {format?, text, covariance?, frame?} "
            "body so the orbit can be seeded (format defaults to auto-detect).",
        )
    opm, l1, l2 = _ingest_to_state(ing)

    base = default_config()
    # Nominal TLE: from a TLE ingest use its lines; for an OPM, fall back to the
    # engine default nominal TLE as the linearisation point (documented).
    if l1 is not None and l2 is not None:
        orbit_cfg = replace(base.orbit, nominal_tle_line1=l1, nominal_tle_line2=l2)
    else:
        orbit_cfg = base.orbit

    a = req.antenna
    ground = _ground_antenna_spec(a)
    # Map pd_onaxis -> miss_rate = 1 - pd_onaxis; pfa -> false_alarm_rate. The
    # ground-antenna slew rate flows in too (the session treats the ground spec
    # as authoritative, but set it here for a consistent config snapshot).
    antenna_cfg = replace(
        base.antenna,
        fwhm_deg=float(a.fwhm_deg),
        dwell_time_s=float(a.dwell_s),
        miss_rate=float(max(0.0, 1.0 - a.pd_onaxis)),
        false_alarm_rate=float(a.pfa),
        max_slew_rate_deg_s=ground.effective_slew_deg_s(),
    )
    pomcp_cfg = replace(base.pomcp, n_rollouts=int(req.rollouts))
    # PFT-DPW planning simulations per decision: reuse the `rollouts` budget so
    # the web path stays responsive.
    solver_cfg = replace(base.solver, n_iterations=int(req.rollouts))
    filter_cfg = replace(base.filter, n_particles=int(req.n_particles))
    cfg = replace(
        base,
        orbit=orbit_cfg,
        antenna=antenna_cfg,
        pomcp=pomcp_cfg,
        solver=solver_cfg,
        filter=filter_cfg,
    )

    rng = np.random.default_rng(0)
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)
    network = _resolve_network(req.network)
    session = PlanSession(
        config=cfg,
        network=network,
        belief=belief,
        policy=req.policy,
        sat_antenna=get_sat_antenna(req.sat_antenna),
        ground_antenna=ground,
        authority_level=req.authority_level,
        leop_start=opm.epoch,
        ukf=ukf,
        rng=rng,
    )
    plans = session.plan_network(horizon_h=float(req.horizon_h))
    sid = str(uuid.uuid4())
    _SESSIONS[sid] = session
    return sid, session, plans


@app.post("/api/plan", response_model=PlanResult)
def plan(req: PlanRequest) -> PlanResult:
    sid, session, plans = _build_session(req)
    _checkpoint(
        sid,
        session,
        "session.create",
        detail={"policy": req.policy, "active_policy": session.policy},
    )
    return plan_result(sid, session, plans)


# ---------------------------------------------------------------------------
# Observe
# ---------------------------------------------------------------------------


def _snr_obs_from_request(req: ObserveRequest):
    """Build an engine SnrObservation from an observe request, or None.

    Returns None when the report carries no graded strength (a bare detect),
    so the session falls back to the 1-bit detect update.
    """
    from acquisition_platform.observations.base import Observation

    if req.ebn0_db is None and req.snr_db is None and req.locked is None:
        return None  # pure 1-bit detect
    obs = Observation(
        t=datetime.now(timezone.utc),
        detected=req.detected,
        locked=req.locked,
        ebn0_db=req.ebn0_db,
        snr_db=req.snr_db,
        source=req.source,
    )
    return obs.to_snr()


@app.get("/api/observation-sources", response_model=list[ObservationSourceOut])
def observation_sources() -> list[ObservationSourceOut]:
    from acquisition_platform.observations.registry import list_sources

    return [ObservationSourceOut(**s) for s in list_sources()]


@app.post("/api/observe", response_model=PlanResult)
def observe(req: ObserveRequest) -> PlanResult:
    session = _require_session(req.session_id)
    with _session_lock(req.session_id):
        snr_obs = _snr_obs_from_request(req)
        detected = req.detected if req.locked is None else bool(req.locked)
        try:
            plans = session.observe(
                req.pass_idx,
                req.dwell_idx,
                detected,
                snr_obs=snr_obs,
                doppler_hz=req.doppler_hz,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _checkpoint(
            req.session_id,
            session,
            "observe",
            detail={
                "pass_idx": req.pass_idx,
                "dwell_idx": req.dwell_idx,
                "detected": detected,
                "doppler_hz": req.doppler_hz,
                "fallback": session.fallback_active,
            },
        )
        return plan_result(req.session_id, session, plans)


# ---------------------------------------------------------------------------
# Trust & safety: authority level + watchdog re-arm
# ---------------------------------------------------------------------------


@app.post("/api/authority", response_model=PlanResult)
def set_authority(req: SetAuthorityRequest) -> PlanResult:
    """Change a session's graduated-trust authority level (advisory..autonomous)."""
    session = _require_session(req.session_id)
    with _session_lock(req.session_id):
        session.set_authority(req.level)
        _checkpoint(req.session_id, session, "authority.set", detail={"level": req.level})
        return plan_result(req.session_id, session, session._plans)


@app.post("/api/rearm", response_model=PlanResult)
def rearm(req: RearmRequest) -> PlanResult:
    """Operator re-arm after a watchdog fallback: restore the controller and replan."""
    session = _require_session(req.session_id)
    with _session_lock(req.session_id):
        try:
            plans = session.rearm()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _checkpoint(req.session_id, session, "watchdog.rearm", detail={"policy": session.policy})
        return plan_result(req.session_id, session, plans)


@app.get("/api/audit/{session_id}")
def audit_log(session_id: str) -> dict:
    """The hash-chained command history for a session (empty if persistence is off)."""
    store = _get_store()
    if store is None:
        return {"enabled": False, "entries": [], "chain_ok": True}
    return {
        "enabled": True,
        "entries": store.audit_log(session_id=session_id),
        "chain_ok": store.verify_chain() is None,
    }


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------


@app.get("/api/belief/{session_id}", response_model=BeliefResult)
def belief(session_id: str) -> BeliefResult:
    session = _require_session(session_id)
    with _session_lock(session_id):
        summ = belief_summary_to_schema(session.belief_summary())
        return BeliefResult(entropy=summ.entropy, grid=summ.grid)


# ---------------------------------------------------------------------------
# Antenna command out: drivers + exact schedule / continuous-track export
# ---------------------------------------------------------------------------


@app.get("/api/drivers", response_model=list[DriverOut])
def list_antenna_drivers() -> list[DriverOut]:
    from acquisition_platform.drivers.registry import list_drivers

    return [DriverOut(**d) for d in list_drivers()]


@app.post("/api/export/schedule", response_model=ScheduleExportResult)
def export_schedule(req: ScheduleExportRequest, response: Response) -> ScheduleExportResult:
    """Return canonical JSON and its matched PDF rendering in one atomic response.

    The materialized backend plan is captured once while the session is locked.
    The browser therefore cannot receive files from different re-plan revisions.
    """
    session = _require_session(req.session_id)
    try:
        with _session_lock(req.session_id):
            artifacts = build_schedule_artifacts(req.session_id, session)
            store = _get_store()
            if store is not None:
                store.append_audit(
                    "schedule.export",
                    session_id=req.session_id,
                    detail={
                        "schema_version": SCHEDULE_SCHEMA_VERSION,
                        "schedule_id": artifacts.schedule_id,
                        "json_sha256": artifacts.json_sha256,
                        "pdf_sha256": artifacts.pdf_sha256,
                        "pass_count": artifacts.pass_count,
                        "dwell_count": artifacts.dwell_count,
                        "complete": artifacts.complete,
                    },
                )
    except SchedulePdfDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ScheduleExportError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-LEOPT-Schedule-ID"] = artifacts.schedule_id
    response.headers["X-LEOPT-JSON-SHA256"] = artifacts.json_sha256
    response.headers["X-LEOPT-PDF-SHA256"] = artifacts.pdf_sha256
    return ScheduleExportResult(
        schema_version=SCHEDULE_SCHEMA_VERSION,
        schedule_id=artifacts.schedule_id,
        json_filename=artifacts.json_filename,
        json_sha256=artifacts.json_sha256,
        json_size_bytes=len(artifacts.json_bytes),
        json_content=artifacts.json_bytes.decode("utf-8"),
        pdf_filename=artifacts.pdf_filename,
        pdf_sha256=artifacts.pdf_sha256,
        pdf_size_bytes=len(artifacts.pdf_bytes),
        pdf_base64=base64.b64encode(artifacts.pdf_bytes).decode("ascii"),
        pass_count=artifacts.pass_count,
        dwell_count=artifacts.dwell_count,
        complete=artifacts.complete,
    )


@app.post("/api/export/track", response_model=TrackExportResult)
def export_track(req: TrackExportRequest) -> TrackExportResult:
    """Turn a planned pass into a continuous track/ephemeris file for an ACU."""
    from acquisition_platform.drivers.base import densify_track, dwells_from_passplan
    from acquisition_platform.drivers.program_track import (
        to_ccsds_pointing_kvn,
        to_csv,
        to_oem_kvn,
    )

    session = _SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    passplan = next((p for p in session._plans if p.idx == req.pass_idx), None)
    if passplan is None or not passplan.pointings:
        raise HTTPException(status_code=400, detail=f"no planned pointings for pass {req.pass_idx}")

    if req.format == "oem":
        # Orbit ephemeris sampled from the nominal propagator across the pass.
        dwells = dwells_from_passplan(passplan)
        track = densify_track(dwells, rate_hz=req.rate_hz)
        times = [s.t for s in track]
        content = to_oem_kvn(session._nominal, times, object_name=req.object_name)
        n = len(times)
        ext = "oem"
    else:
        track = densify_track(dwells_from_passplan(passplan), rate_hz=req.rate_hz)
        if req.format == "ccsds_pointing":
            content = to_ccsds_pointing_kvn(track, object_name=req.object_name)
            ext = "kvn"
        else:
            content = to_csv(track)
            ext = "csv"
        n = len(track)

    filename = f"pass{req.pass_idx}_{passplan.station_id}.{ext}"
    return TrackExportResult(format=req.format, filename=filename, n_samples=n, content=content)


# ---------------------------------------------------------------------------
# Static dashboard (mounted last so /api/* wins)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    from fastapi.testclient import TestClient

    client = TestClient(app)
    nets = client.get("/api/networks").json()
    assert len(nets) == 4, nets
    sso = (
        "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
        "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    )
    ing = client.post("/api/ingest", json={"format": "tle", "text": sso}).json()
    assert "epoch" in ing and len(ing["r_eci_m"]) == 3
    plan_body = {
        "ingest": {"format": "tle", "text": sso},
        "network": "single_university",
        "antenna": {"fwhm_deg": 10.0, "dwell_s": 10.0, "pd_onaxis": 0.95, "pfa": 0.01},
        "horizon_h": 8.0,
        "n_particles": 80,
        "rollouts": 20,
        "policy": "sweep",
    }
    res = client.post("/api/plan", json=plan_body).json()
    assert res["passes"], res
    print(f"[app] networks=4, plan passes={len(res['passes'])}, sid={res['session_id'][:8]}")
    print("[app] PASS")
