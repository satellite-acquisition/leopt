"""Focused contract tests for the operator schedule JSON/PDF export."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import math
import re
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from acquisition_platform.planner import schedule_export
from acquisition_platform.planner.schedule_export import (
    ScheduleExportError,
    build_schedule_artifacts,
    build_schedule_document,
    canonical_json_bytes,
)
from acquisition_platform.planner.session import PassPlan, PointingPlan
from acquisition_platform.service import app as app_module
from antenna_pomdp.config import StationConfig
from antenna_pomdp.orbit.geometry import Station


FIXED_EXPORT_EPOCH = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
SECRET_TLE = "1 SECRET-TLE-LINE THAT MUST NEVER ENTER A SCHEDULE EXPORT"
SECRET_TOKEN = "super-secret-catalog-token"


class _StubSession:
    """Small duck-typed PlanSession fixture with deliberately sensitive extras."""

    def __init__(self) -> None:
        station = Station.from_config(
            StationConfig(
                latitude_deg=46.0,
                longitude_deg=7.0,
                altitude_m=500.0,
                min_elevation_deg=5.0,
            )
        )
        rise = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)
        pointings = [
            PointingPlan(
                t_s=0.0,
                when=rise,
                az_rad=math.radians(120.0),
                el_rad=math.radians(25.0),
                d_tau_s=-12.5,
                d_theta_deg=0.25,
                p_detect=0.42,
                dwell_s=10.0,
                safety_status="ok",
                transmit_ok=True,
                safety_reason="within envelope",
            ),
            PointingPlan(
                t_s=10.0,
                when=rise + timedelta(seconds=10.0),
                az_rad=math.radians(121.5),
                el_rad=math.radians(26.0),
                d_tau_s=7.5,
                d_theta_deg=-0.25,
                p_detect=0.58,
                dwell_s=10.0,
                safety_status="rf_inhibit",
                transmit_ok=False,
                safety_reason="synthetic transmit keep-out",
            ),
        ]
        self._plans = [
            PassPlan(
                idx=3,
                station_id="TEST-STATION",
                station=station,
                station_index=0,
                rise=rise,
                set=rise + timedelta(seconds=25.0),
                peak_el_deg=42.5,
                pointings=pointings,
            )
        ]
        self.policy = "sweep"
        self.controller_choice = None

        # The exporter must use its explicit allowlist, not serialize a session
        # object or its ingest/config/belief internals.
        self.raw_ingest = SECRET_TLE
        self.catalog_token = SECRET_TOKEN
        self.belief_particles = [{"private": SECRET_TOKEN}]

    def ops_status(self) -> dict:
        return {
            "trust": {
                "level": "L0_ADVISORY",
                "label": "Advisory",
                "description": "Recommend only; nothing auto-actuates.",
                "can_actuate": False,
            },
            "watchdog": {
                "mode": "closed_loop",
                "latched": False,
                "reason": "",
            },
            "fallback_active": False,
            "rationale": "Fixture rationale that is intentionally not exported.",
            "safety": {
                "total_dwells": 2,
                "ok": 1,
                "warn": 0,
                "keyhole": 0,
                "rf_inhibit": 1,
                "reject": 0,
                "first_blocking": {
                    "pass_idx": 3,
                    "reason": "synthetic transmit keep-out",
                },
            },
        }


def _long_schedule_session(n_passes: int = 4, n_dwells: int = 20) -> _StubSession:
    """Build a capped, multipage fixture without exposing private session state."""
    session = _StubSession()
    station = session._plans[0].station
    epoch = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)
    plans: list[PassPlan] = []
    blocked = 0
    for pass_idx in range(n_passes):
        rise = epoch + timedelta(hours=pass_idx)
        pointings: list[PointingPlan] = []
        for dwell_idx in range(n_dwells):
            inhibited = dwell_idx == 7
            blocked += int(inhibited)
            pointings.append(
                PointingPlan(
                    t_s=float(dwell_idx * 10),
                    when=rise + timedelta(seconds=dwell_idx * 10),
                    az_rad=math.radians(100.0 + dwell_idx),
                    el_rad=math.radians(20.0 + 0.25 * dwell_idx),
                    d_tau_s=float(dwell_idx - 10),
                    d_theta_deg=0.1 * ((dwell_idx % 3) - 1),
                    p_detect=0.25 + 0.02 * dwell_idx,
                    dwell_s=10.0,
                    safety_status="rf_inhibit" if inhibited else "ok",
                    transmit_ok=not inhibited,
                    safety_reason="synthetic keep-out" if inhibited else "",
                )
            )
        plans.append(
            PassPlan(
                idx=pass_idx,
                station_id=f"STATION-{pass_idx + 1}",
                station=station,
                station_index=0,
                rise=rise,
                set=rise + timedelta(seconds=n_dwells * 10 + 5),
                peak_el_deg=50.0,
                pointings=pointings,
            )
        )
    session._plans = plans
    total = n_passes * n_dwells
    session.ops_status = lambda: {
        "trust": {
            "level": "L0_ADVISORY",
            "label": "Advisory",
            "description": "Recommend only; nothing auto-actuates.",
            "can_actuate": False,
        },
        "watchdog": {"mode": "closed_loop", "latched": False, "reason": ""},
        "fallback_active": False,
        "rationale": "Multipage fixture.",
        "safety": {
            "total_dwells": total,
            "ok": total - blocked,
            "warn": 0,
            "keyhole": 0,
            "rf_inhibit": blocked,
            "reject": 0,
            "first_blocking": (
                {"pass_idx": 0, "reason": "synthetic keep-out"} if blocked else None
            ),
        },
    }
    return session


@pytest.fixture
def schedule_session() -> _StubSession:
    return _StubSession()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fake_pdf(document: dict, json_sha256: str) -> bytes:
    """Dependency-free static PDF-shaped payload for API/hash plumbing tests."""
    return (
        b"%PDF-1.4\n"
        + f"% schedule={document['schedule_id']}\n".encode()
        + f"% json-sha256={json_sha256}\n".encode()
        + b"%%EOF\n"
    )


def _has_schedule_export_route() -> bool:
    return any(
        getattr(route, "path", None) == "/api/export/schedule" for route in app_module.app.routes
    )


def test_canonical_json_is_sorted_compact_finite_and_lf_terminated() -> None:
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'
    with pytest.raises(ScheduleExportError, match="finite canonical JSON"):
        canonical_json_bytes({"bad": float("nan")})


def test_schedule_id_is_stable_across_export_time_and_binds_payload(
    schedule_session: _StubSession,
) -> None:
    first = build_schedule_document(
        "session-test",
        schedule_session,
        exported_at=FIXED_EXPORT_EPOCH,
    )
    repeated = build_schedule_document(
        "session-test",
        schedule_session,
        exported_at=FIXED_EXPORT_EPOCH + timedelta(hours=1),
    )

    payload_digest = _sha256(canonical_json_bytes(first["schedule"]))
    assert first["schedule_id"] == payload_digest
    assert first["schedule_payload_sha256"] == payload_digest
    assert repeated["schedule_id"] == payload_digest
    assert first["exported_at_utc"] != repeated["exported_at_utc"]
    assert canonical_json_bytes(first) != canonical_json_bytes(repeated)


def test_schedule_id_changes_when_an_exported_command_changes(
    schedule_session: _StubSession,
) -> None:
    baseline = build_schedule_document(
        "session-test",
        schedule_session,
        exported_at=FIXED_EXPORT_EPOCH,
    )
    changed = copy.deepcopy(schedule_session)
    changed._plans[0].pointings[0].az_rad += math.radians(0.01)
    mutated = build_schedule_document(
        "session-test",
        changed,
        exported_at=FIXED_EXPORT_EPOCH,
    )

    assert mutated["schedule_id"] != baseline["schedule_id"]
    assert (
        mutated["schedule"]["commands"][0]["az_deg"]
        != baseline["schedule"]["commands"][0]["az_deg"]
    )


def test_json_export_is_allowlisted_and_does_not_leak_session_secrets(
    schedule_session: _StubSession,
) -> None:
    document = build_schedule_document(
        "session-test",
        schedule_session,
        exported_at=FIXED_EXPORT_EPOCH,
    )
    payload = canonical_json_bytes(document)

    assert SECRET_TLE.encode() not in payload
    assert SECRET_TOKEN.encode() not in payload
    assert b"belief_particles" not in payload
    assert b"raw_ingest" not in payload
    assert document["schedule"]["classification"]["operational_use_permitted"] is False


def test_document_rejects_missing_or_empty_materialized_schedule(
    schedule_session: _StubSession,
) -> None:
    schedule_session._plans = []
    with pytest.raises(ScheduleExportError, match="no materialized plan"):
        build_schedule_document("session-test", schedule_session)

    schedule_session = _StubSession()
    schedule_session._plans[0].pointings = []
    with pytest.raises(ScheduleExportError, match="contains no dwell commands"):
        build_schedule_document("session-test", schedule_session)


def test_artifact_hashes_and_base64_bind_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
    schedule_session: _StubSession,
) -> None:
    monkeypatch.setattr(schedule_export, "_render_schedule_pdf", _fake_pdf)
    artifacts = build_schedule_artifacts(
        "session-test",
        schedule_session,
        exported_at=FIXED_EXPORT_EPOCH,
    )

    assert artifacts.json_sha256 == _sha256(artifacts.json_bytes)
    assert artifacts.pdf_sha256 == _sha256(artifacts.pdf_bytes)
    assert artifacts.json_bytes.endswith(b"\n")
    assert artifacts.pdf_bytes.startswith(b"%PDF-")
    assert artifacts.pdf_bytes.endswith(b"%%EOF\n")
    assert artifacts.json_sha256.encode() in artifacts.pdf_bytes

    encoded_pdf = base64.b64encode(artifacts.pdf_bytes).decode("ascii")
    assert base64.b64decode(encoded_pdf, validate=True) == artifacts.pdf_bytes
    document = json.loads(artifacts.json_bytes)
    assert document["schedule_id"] == artifacts.schedule_id
    assert artifacts.pass_count == 1
    assert artifacts.dwell_count == 2
    assert artifacts.complete is True


def test_every_real_planner_dwell_ends_no_later_than_los() -> None:
    # Keep this on the real planner: an export-only clip would leave the API,
    # UI, and machine-track semantics inconsistent.
    from acquisition_platform.tests.test_platform import _build_session

    session = _build_session("sweep")
    plans = session.plan_network(horizon_h=10.0)
    tolerance = timedelta(microseconds=1)
    assert any(plan.pointings for plan in plans)
    for plan in plans:
        for point in plan.pointings:
            assert point.when + timedelta(seconds=point.dwell_s) <= plan.set + tolerance


@pytest.mark.skipif(
    importlib.util.find_spec("reportlab") is None,
    reason="ReportLab is an app-only dependency pinned in requirements/platform-export.txt",
)
def test_real_planner_session_exports_without_private_state() -> None:
    from acquisition_platform.tests.test_platform import SSO, _build_session

    session = _build_session("sweep")
    session.plan_network(horizon_h=10.0)
    artifacts = build_schedule_artifacts(
        "real-planner-export-test",
        session,
        exported_at=FIXED_EXPORT_EPOCH,
    )
    document = json.loads(artifacts.json_bytes)

    assert artifacts.dwell_count > 0
    assert document["schedule"]["safety_summary"]["total_dwells"] == artifacts.dwell_count
    assert all(
        command["end_utc"]
        <= next(
            item["set_utc"]
            for item in document["schedule"]["passes"]
            if item["pass_idx"] == command["pass_idx"]
        )
        for command in document["schedule"]["commands"]
    )
    assert SSO.splitlines()[0] not in artifacts.json_bytes.decode("utf-8")

    first_pass = next(plan for plan in session._plans if plan.pointings)
    session.observe(first_pass.idx, 0, detected=False)
    replanned = build_schedule_document(
        "real-planner-export-test",
        session,
        exported_at=FIXED_EXPORT_EPOCH,
    )
    assert replanned["schedule_id"] != artifacts.schedule_id


@pytest.mark.skipif(
    importlib.util.find_spec("reportlab") is None,
    reason="ReportLab is an app-only dependency pinned in requirements/platform-export.txt",
)
def test_real_pdf_is_static_and_byte_deterministic(
    schedule_session: _StubSession,
) -> None:
    first = build_schedule_artifacts(
        "session-test",
        schedule_session,
        exported_at=FIXED_EXPORT_EPOCH,
    )
    repeated = build_schedule_artifacts(
        "session-test",
        schedule_session,
        exported_at=FIXED_EXPORT_EPOCH,
    )

    assert first.json_bytes == repeated.json_bytes
    assert first.pdf_bytes == repeated.pdf_bytes
    assert first.pdf_sha256 == repeated.pdf_sha256
    assert first.pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in first.pdf_bytes[-64:]
    assert len(first.pdf_bytes) > 10_000
    for forbidden in (b"/JavaScript", b"/EmbeddedFile", b"/Launch", b"/AcroForm"):
        assert forbidden not in first.pdf_bytes
    assert SECRET_TLE.encode() not in first.pdf_bytes
    assert SECRET_TOKEN.encode() not in first.pdf_bytes


@pytest.mark.skipif(
    importlib.util.find_spec("reportlab") is None,
    reason="ReportLab is an app-only dependency pinned in requirements/platform-export.txt",
)
def test_long_schedule_pdf_is_multipage_without_truncating_commands() -> None:
    artifacts = build_schedule_artifacts(
        "long-schedule-test",
        _long_schedule_session(),
        exported_at=FIXED_EXPORT_EPOCH,
    )
    document = json.loads(artifacts.json_bytes)

    assert artifacts.dwell_count == 80
    assert artifacts.complete is False
    assert len(document["schedule"]["commands"]) == 80
    assert all(item["cap_reached"] for item in document["schedule"]["passes"])
    assert len(re.findall(rb"/Type\s*/Page\b", artifacts.pdf_bytes)) >= 3


@pytest.mark.skipif(
    not _has_schedule_export_route(),
    reason="/api/export/schedule has not been wired into the service yet",
)
def test_api_returns_exact_json_and_base64_pdf_with_matching_hashes(
    monkeypatch: pytest.MonkeyPatch,
    schedule_session: _StubSession,
) -> None:
    monkeypatch.setattr(schedule_export, "_render_schedule_pdf", _fake_pdf)
    sid = "schedule-export-api-test"
    app_module._SESSIONS[sid] = schedule_session
    try:
        response = TestClient(app_module.app).post(
            "/api/export/schedule",
            json={"session_id": sid},
        )
    finally:
        app_module._SESSIONS.pop(sid, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    expected_fields = {
        "schema_version",
        "schedule_id",
        "json_filename",
        "json_content",
        "json_sha256",
        "json_size_bytes",
        "pdf_filename",
        "pdf_base64",
        "pdf_sha256",
        "pdf_size_bytes",
        "pass_count",
        "dwell_count",
        "complete",
    }
    assert set(body) == expected_fields

    json_bytes = body["json_content"].encode("utf-8")
    pdf_bytes = base64.b64decode(body["pdf_base64"], validate=True)
    assert _sha256(json_bytes) == body["json_sha256"]
    assert _sha256(pdf_bytes) == body["pdf_sha256"]
    assert response.headers["x-leopt-schedule-id"] == body["schedule_id"]
    assert response.headers["x-leopt-json-sha256"] == body["json_sha256"]
    assert response.headers["x-leopt-pdf-sha256"] == body["pdf_sha256"]
    assert len(json_bytes) == body["json_size_bytes"]
    assert len(pdf_bytes) == body["pdf_size_bytes"]
    assert body["schema_version"] == "platform-schedule-v1"
    assert json.loads(json_bytes)["schedule_id"] == body["schedule_id"]
    assert body["json_sha256"].encode() in pdf_bytes
    assert body["json_filename"].endswith(".json")
    assert body["pdf_filename"].endswith(".pdf")
    assert body["pass_count"] == 1
    assert body["dwell_count"] == 2
    assert body["complete"] is True
    assert SECRET_TLE not in body["json_content"]
    assert SECRET_TOKEN not in body["json_content"]


@pytest.mark.skipif(
    not _has_schedule_export_route(),
    reason="/api/export/schedule has not been wired into the service yet",
)
def test_api_rejects_invalid_unknown_and_empty_sessions() -> None:
    client = TestClient(app_module.app)

    assert client.post("/api/export/schedule", json={}).status_code == 422
    assert (
        client.post(
            "/api/export/schedule",
            json={"session_id": "definitely-unknown-session"},
        ).status_code
        == 404
    )

    sid = "schedule-export-empty-test"
    empty = _StubSession()
    empty._plans = []
    app_module._SESSIONS[sid] = empty
    try:
        response = client.post("/api/export/schedule", json={"session_id": sid})
    finally:
        app_module._SESSIONS.pop(sid, None)
    assert response.status_code == 409, response.text


def test_api_appends_one_atomic_export_audit_event(
    monkeypatch: pytest.MonkeyPatch,
    schedule_session: _StubSession,
) -> None:
    class AuditCapture:
        def __init__(self) -> None:
            self.events: list[tuple[str, str | None, dict]] = []

        def append_audit(
            self,
            action: str,
            *,
            session_id: str | None = None,
            detail: dict | None = None,
        ) -> None:
            self.events.append((action, session_id, detail or {}))

    audit = AuditCapture()
    monkeypatch.setattr(schedule_export, "_render_schedule_pdf", _fake_pdf)
    monkeypatch.setattr(app_module, "_STORE", audit)
    sid = "schedule-export-audit-test"
    app_module._SESSIONS[sid] = schedule_session
    try:
        response = TestClient(app_module.app).post(
            "/api/export/schedule",
            json={"session_id": sid},
        )
    finally:
        app_module._SESSIONS.pop(sid, None)

    assert response.status_code == 200, response.text
    assert len(audit.events) == 1
    action, event_sid, detail = audit.events[0]
    assert action == "schedule.export"
    assert event_sid == sid
    assert detail["schedule_id"] == response.json()["schedule_id"]
    assert detail["json_sha256"] == response.json()["json_sha256"]
    assert detail["pdf_sha256"] == response.json()["pdf_sha256"]
