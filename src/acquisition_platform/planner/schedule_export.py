"""Canonical JSON and human-readable PDF exports for a planned schedule.

The JSON document is the authoritative representation.  The PDF is rendered
from that exact document and prints its byte-level SHA-256 digest on every page.
Both files are built in one call so an observation or re-plan cannot place the
two representations on different schedule revisions.

This module exports the materialized backend ``PlanSession._plans``.  It never
uses the browser console's separate visual-simulation schedule.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from acquisition_platform.planner.session import MAX_DWELLS_PER_PASS, PlanSession


SCHEMA_VERSION = "platform-schedule-v1"
CANONICALIZATION = "UTF-8 JSON; sorted keys; compact separators; finite numbers; trailing LF"
MAX_EXPORTED_COMMANDS = 2_000
_PDF_LOCK = threading.Lock()


class ScheduleExportError(RuntimeError):
    """The current plan cannot be represented by the export contract."""


class SchedulePdfDependencyError(ScheduleExportError):
    """The app-only PDF dependency is not installed."""


@dataclass(frozen=True)
class ScheduleArtifacts:
    """One atomic JSON/PDF representation pair."""

    schedule_id: str
    json_filename: str
    json_bytes: bytes
    json_sha256: str
    pdf_filename: str
    pdf_bytes: bytes
    pdf_sha256: str
    pass_count: int
    dwell_count: int
    complete: bool


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(document: dict[str, Any]) -> bytes:
    """Serialize a document under the schedule contract's canonical JSON rules."""
    try:
        text = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleExportError(f"schedule is not finite canonical JSON: {exc}") from exc
    return text.encode("utf-8") + b"\n"


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ScheduleExportError("schedule epochs must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _finite(value: float, field: str, digits: int) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ScheduleExportError(f"{field} must be finite")
    return round(number, digits)


def _bounded_text(value: object, field: str, limit: int, *, allow_empty: bool = False) -> str:
    text = str(value)
    if any(unicodedata.category(char).startswith("C") for char in text):
        raise ScheduleExportError(f"{field} contains a control character")
    if text != text.strip():
        raise ScheduleExportError(f"{field} has leading or trailing whitespace")
    if not text and not allow_empty:
        raise ScheduleExportError(f"{field} must not be empty")
    if len(text) > limit:
        raise ScheduleExportError(f"{field} exceeds {limit} characters")
    return text


def _software_version() -> str:
    try:
        return version("leopt")
    except PackageNotFoundError:
        return "0.1.0"


def _controller_record(session: PlanSession) -> dict[str, Any] | None:
    if session.controller_choice is None:
        return None
    record = dict(session.controller_choice.to_dict())
    for field in ("policy", "controller_name", "rationale", "basis"):
        if field in record:
            record[field] = _bounded_text(
                record[field],
                f"controller.{field}",
                800 if field in {"rationale", "basis"} else 120,
            )
    return record


def _command_record(
    *,
    pass_idx: int,
    dwell_idx: int,
    station: str,
    pass_rise: datetime,
    pass_set: datetime,
    point: Any,
) -> dict[str, Any]:
    start = point.when.astimezone(UTC)
    hold_s = _finite(point.dwell_s, "dwell_s", 6)
    if hold_s <= 0.0:
        raise ScheduleExportError("dwell_s must be positive")
    end = start + timedelta(seconds=hold_s)
    tolerance = timedelta(microseconds=1)
    if start < pass_rise.astimezone(UTC) - tolerance:
        raise ScheduleExportError(f"pass {pass_idx} dwell {dwell_idx} starts before rise")
    if end > pass_set.astimezone(UTC) + tolerance:
        raise ScheduleExportError(
            f"pass {pass_idx} dwell {dwell_idx} ends after set; re-plan before export"
        )
    elapsed_s = (start - pass_rise.astimezone(UTC)).total_seconds()
    if abs(float(point.t_s) - elapsed_s) > 1.0e-6:
        raise ScheduleExportError(
            f"pass {pass_idx} dwell {dwell_idx} has inconsistent relative/absolute time"
        )

    az_deg = _finite(math.degrees(point.az_rad) % 360.0, "az_deg", 9)
    el_deg = _finite(math.degrees(point.el_rad), "el_deg", 9)
    if not 0.0 <= az_deg <= 360.0:
        raise ScheduleExportError("az_deg is outside [0, 360]")
    if not -90.0 <= el_deg <= 90.0:
        raise ScheduleExportError("el_deg is outside [-90, 90]")
    probability = _finite(point.p_detect, "p_detect", 12)
    if not 0.0 <= probability <= 1.0:
        raise ScheduleExportError("p_detect is outside [0, 1]")

    status = _bounded_text(point.safety_status, "safety_status", 40)
    if status not in {"ok", "warn", "keyhole", "rf_inhibit", "reject"}:
        raise ScheduleExportError(f"unsupported safety_status {status!r}")
    transmit_ok = bool(point.transmit_ok)
    if status in {"rf_inhibit", "reject"} and transmit_ok:
        raise ScheduleExportError(
            f"pass {pass_idx} dwell {dwell_idx} has an unsafe transmit verdict"
        )
    reason = _bounded_text(point.safety_reason, "safety_reason", 400, allow_empty=True)
    return {
        "dwell_id": f"P{pass_idx:04d}-D{dwell_idx:04d}",
        "pass_idx": int(pass_idx),
        "dwell_idx": int(dwell_idx),
        "station": station,
        "start_utc": _iso_utc(start),
        "end_utc": _iso_utc(end),
        "seconds_since_pass_rise": _finite(
            elapsed_s,
            "seconds_since_pass_rise",
            6,
        ),
        "hold_s": hold_s,
        "az_deg": az_deg,
        "el_deg": el_deg,
        "along_track_offset_s": _finite(point.d_tau_s, "along_track_offset_s", 9),
        "cross_track_offset_deg": _finite(
            point.d_theta_deg,
            "cross_track_offset_deg",
            9,
        ),
        "estimated_p_detect": probability,
        "safety_status": status,
        "transmit_ok": transmit_ok,
        "safety_reason": reason,
    }


def build_schedule_payload(session_id: str, session: PlanSession) -> dict[str, Any]:
    """Capture and validate the current materialized backend schedule."""
    sid = _bounded_text(session_id, "session_id", 128)
    if not session._plans:
        raise ScheduleExportError("session has no materialized plan")

    commands: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []
    warnings = [
        ("Research schedule snapshot only; it is not an actuation authorization or a safety case."),
        (
            "Any observation, re-plan, watchdog transition, authority change, or "
            "service restart can supersede this snapshot."
        ),
        (
            "Rows are planned commands, not an execution log; completed and pending "
            "states are not asserted."
        ),
        (
            "Safety status is evaluated at each dwell start and is not a continuous "
            "hold-interval clearance."
        ),
        (
            "Global sequence is deterministic display order; it does not serialize "
            "commands at independent stations."
        ),
    ]

    seen_pass_indices: set[int] = set()
    previous_rise: datetime | None = None
    for plan in session._plans:
        pass_idx = int(plan.idx)
        if pass_idx < 0 or pass_idx in seen_pass_indices:
            raise ScheduleExportError(f"invalid or duplicate pass index {pass_idx}")
        seen_pass_indices.add(pass_idx)
        station = _bounded_text(plan.station_id, f"pass[{pass_idx}].station", 80)
        rise = plan.rise.astimezone(UTC)
        set_epoch = plan.set.astimezone(UTC)
        if set_epoch <= rise:
            raise ScheduleExportError(f"pass {pass_idx} set must be after rise")
        if previous_rise is not None and rise < previous_rise:
            raise ScheduleExportError("passes are not ordered by rise epoch")
        previous_rise = rise

        pass_commands: list[dict[str, Any]] = []
        previous_start: datetime | None = None
        for dwell_idx, point in enumerate(plan.pointings):
            start = point.when.astimezone(UTC)
            if previous_start is not None and start <= previous_start:
                raise ScheduleExportError(
                    f"pass {pass_idx} dwell starts are not strictly increasing"
                )
            previous_start = start
            command = _command_record(
                pass_idx=pass_idx,
                dwell_idx=dwell_idx,
                station=station,
                pass_rise=rise,
                pass_set=set_epoch,
                point=point,
            )
            pass_commands.append(command)
            commands.append(command)

        if pass_commands:
            last_end = datetime.fromisoformat(pass_commands[-1]["end_utc"].replace("Z", "+00:00"))
        else:
            last_end = rise
        unscheduled_tail_s = max(0.0, (set_epoch - last_end).total_seconds())
        cap_reached = len(pass_commands) >= MAX_DWELLS_PER_PASS and unscheduled_tail_s > 1.0e-6

        latitude_deg = _finite(
            math.degrees(plan.station.latitude_rad),
            "station.latitude_deg",
            9,
        )
        longitude_deg = _finite(
            math.degrees(plan.station.longitude_rad),
            "station.longitude_deg",
            9,
        )
        peak_elevation_deg = _finite(
            plan.peak_el_deg,
            "peak_elevation_deg",
            9,
        )
        if not -90.0 <= latitude_deg <= 90.0:
            raise ScheduleExportError("station latitude is outside [-90, 90]")
        if not -180.0 <= longitude_deg <= 180.0:
            raise ScheduleExportError("station longitude is outside [-180, 180]")
        if not -90.0 <= peak_elevation_deg <= 90.0:
            raise ScheduleExportError("peak elevation is outside [-90, 90]")

        passes.append(
            {
                "pass_idx": pass_idx,
                "station": station,
                "station_geodetic": {
                    "latitude_deg": latitude_deg,
                    "longitude_deg": longitude_deg,
                    "altitude_m": _finite(
                        plan.station.altitude_m,
                        "station.altitude_m",
                        3,
                    ),
                },
                "rise_utc": _iso_utc(rise),
                "set_utc": _iso_utc(set_epoch),
                "peak_elevation_deg": peak_elevation_deg,
                "dwell_count": len(pass_commands),
                "web_dwell_cap": MAX_DWELLS_PER_PASS,
                "cap_reached": cap_reached,
                "unscheduled_tail_s": _finite(
                    unscheduled_tail_s,
                    "unscheduled_tail_s",
                    6,
                ),
                "first_dwell_id": (pass_commands[0]["dwell_id"] if pass_commands else None),
                "last_dwell_id": (pass_commands[-1]["dwell_id"] if pass_commands else None),
            }
        )

    capped_passes = [item for item in passes if item["cap_reached"]]
    if capped_passes:
        warnings.append(
            f"{len(capped_passes)} pass(es) reached the web cap of "
            f"{MAX_DWELLS_PER_PASS} dwells. Those pass tails are incomplete; "
            "see cap_reached and unscheduled_tail_s in the pass summary."
        )
    empty_passes = [item for item in passes if item["dwell_count"] == 0]
    if empty_passes:
        warnings.append(
            f"{len(empty_passes)} pass(es) contain no full-hold dwell command. "
            "The planner does not emit a partial hold at loss of signal."
        )

    if not commands:
        raise ScheduleExportError("session plan contains no dwell commands")
    if len(commands) > MAX_EXPORTED_COMMANDS:
        raise ScheduleExportError(
            f"schedule has {len(commands)} commands; maximum is {MAX_EXPORTED_COMMANDS}"
        )

    commands.sort(
        key=lambda command: (
            command["start_utc"],
            command["station"],
            command["pass_idx"],
            command["dwell_idx"],
        )
    )
    for sequence, command in enumerate(commands, start=1):
        command["sequence"] = sequence

    ops = session.ops_status()
    trust = dict(ops["trust"])
    watchdog = dict(ops["watchdog"])
    for field in ("level", "label", "description"):
        trust[field] = _bounded_text(
            trust[field],
            f"authority.{field}",
            300 if field == "description" else 80,
        )
    for field in ("mode", "reason"):
        watchdog[field] = _bounded_text(
            watchdog[field],
            f"watchdog.{field}",
            300 if field == "reason" else 80,
            allow_empty=field == "reason",
        )

    safety_summary = dict(ops["safety"])
    if int(safety_summary.get("total_dwells", -1)) != len(commands):
        raise ScheduleExportError("safety summary does not match the materialized schedule")
    first_blocking = safety_summary.get("first_blocking")
    if first_blocking is not None:
        safety_summary["first_blocking"] = {
            "pass_idx": int(first_blocking["pass_idx"]),
            "reason": _bounded_text(
                first_blocking.get("reason", ""),
                "safety_summary.first_blocking.reason",
                400,
                allow_empty=True,
            ),
        }

    complete = not any(item["cap_reached"] for item in passes)
    return {
        "artifact_role": "authoritative-current-backend-plan-snapshot",
        "classification": {
            "research_software": True,
            "planned_schedule_snapshot": True,
            "execution_log": False,
            "operational_use_permitted": False,
            "canonical_json_is_authoritative": True,
        },
        "session_id": sid,
        "software_version": _software_version(),
        "active_policy": _bounded_text(session.policy, "active_policy", 80),
        "controller": _controller_record(session),
        "authority": trust,
        "watchdog": watchdog,
        "fallback_active": bool(ops["fallback_active"]),
        "safety_summary": safety_summary,
        "conventions": {
            "time_system": "UTC",
            "epoch_format": "RFC3339 with trailing Z",
            "angle_unit": "degree",
            "duration_unit": "second",
            "coordinate_frame": "topocentric east-north-up at the named station",
            "azimuth_definition": "clockwise from true north in [0, 360)",
            "elevation_definition": "angle above the local horizon",
            "command_epoch_role": "start of dwell hold",
            "sequence_role": (
                "display order by UTC, station, pass, and dwell; concurrent "
                "multi-station commands are not serialized"
            ),
            "safety_evaluation_role": "dwell-start check only",
            "transmit_ok_role": (
                "point-in-time safety-envelope verdict; not authority to actuate or transmit"
            ),
        },
        "completeness": {
            "complete_with_respect_to_web_dwell_cap": complete,
            "pass_count": len(passes),
            "dwell_count": len(commands),
            "first_command_start_utc": commands[0]["start_utc"],
            "last_command_end_utc": max(command["end_utc"] for command in commands),
        },
        "warnings": warnings,
        "passes": passes,
        "commands": commands,
    }


def build_schedule_document(
    session_id: str,
    session: PlanSession,
    *,
    exported_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the self-identifying JSON document used by both output files."""
    schedule = build_schedule_payload(session_id, session)
    schedule_id = _sha256(canonical_json_bytes(schedule))
    at = exported_at or datetime.now(tz=UTC)
    return {
        "schema_version": SCHEMA_VERSION,
        "schedule_id": schedule_id,
        "schedule_payload_sha256": schedule_id,
        "exported_at_utc": _iso_utc(at),
        "canonicalization": CANONICALIZATION,
        "schedule": schedule,
    }


def _pdf_epoch(value: str) -> str:
    date, time = value.replace("T", " ").split(" ", maxsplit=1)
    return f"{escape(date)}<br/>{escape(time)}"


def _pdf_text(value: object) -> str:
    return escape(str(value))


def _render_schedule_pdf(document: dict[str, Any], json_sha256: str) -> bytes:
    try:
        import reportlab
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.platypus import (
            KeepTogether,
            LongTable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise SchedulePdfDependencyError(
            "PDF export requires requirements/platform-export.txt"
        ) from exc

    schedule = document["schedule"]
    schedule_id = document["schedule_id"]
    output = BytesIO()

    with _PDF_LOCK:
        font_root = Path(reportlab.__file__).resolve().parent / "fonts"
        regular_path = font_root / "Vera.ttf"
        bold_path = font_root / "VeraBd.ttf"
        if not regular_path.is_file() or not bold_path.is_file():
            raise SchedulePdfDependencyError("ReportLab's bundled Vera fonts are unavailable")
        if "ScheduleSans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("ScheduleSans", str(regular_path)))
        if "ScheduleSans-Bold" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("ScheduleSans-Bold", str(bold_path)))

        page_size = landscape(letter)
        doc = SimpleDocTemplate(
            output,
            pagesize=page_size,
            leftMargin=0.36 * inch,
            rightMargin=0.36 * inch,
            topMargin=0.48 * inch,
            bottomMargin=0.42 * inch,
            title="Acquisition Schedule",
            author="Acquisition Planning Platform",
            subject=f"Schedule {schedule_id}",
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ScheduleTitle",
            parent=styles["Title"],
            fontName="ScheduleSans-Bold",
            fontSize=20,
            leading=23,
            textColor=colors.HexColor("#182231"),
            spaceAfter=8,
            alignment=TA_LEFT,
        )
        heading_style = ParagraphStyle(
            "ScheduleHeading",
            parent=styles["Heading2"],
            fontName="ScheduleSans-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#20334d"),
            spaceBefore=7,
            spaceAfter=5,
        )
        body_style = ParagraphStyle(
            "ScheduleBody",
            parent=styles["BodyText"],
            fontName="ScheduleSans",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#293442"),
            alignment=TA_LEFT,
        )
        small_style = ParagraphStyle(
            "ScheduleSmall",
            parent=body_style,
            fontSize=6.2,
            leading=7.5,
        )
        tiny_style = ParagraphStyle(
            "ScheduleTiny",
            parent=body_style,
            fontSize=5.4,
            leading=6.5,
        )
        banner_style = ParagraphStyle(
            "ScheduleBanner",
            parent=body_style,
            fontName="ScheduleSans-Bold",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#6f3b00"),
            alignment=TA_CENTER,
        )
        table_header_style = ParagraphStyle(
            "ScheduleTableHeader",
            parent=small_style,
            fontName="ScheduleSans-Bold",
            fontSize=5.6,
            leading=6.8,
            textColor=colors.white,
            alignment=TA_CENTER,
        )

        def canvas_maker(filename: object, **kwargs: Any) -> Canvas:
            kwargs["invariant"] = 1
            kwargs["pageCompression"] = 1
            return Canvas(filename, **kwargs)

        def decorate_page(canvas: Canvas, page_doc: SimpleDocTemplate) -> None:
            canvas.saveState()
            width, _ = page_size
            canvas.setTitle("Acquisition Schedule")
            canvas.setAuthor("Acquisition Planning Platform")
            canvas.setSubject(f"Schedule {schedule_id}")
            canvas.setFont("ScheduleSans", 5.2)
            canvas.setFillColor(colors.HexColor("#596675"))
            footer = (
                f"Schedule {schedule_id} | JSON SHA-256 {json_sha256} | "
                f"Canonical JSON is authoritative | Page {page_doc.page}"
            )
            canvas.drawCentredString(width / 2.0, 0.19 * inch, footer)
            canvas.restoreState()

        story: list[Any] = [
            Paragraph("Acquisition Schedule", title_style),
            Table(
                [
                    [
                        Paragraph(
                            "RESEARCH SCHEDULE SNAPSHOT - NOT AN ACTUATION AUTHORIZATION",
                            banner_style,
                        )
                    ]
                ],
                colWidths=[doc.width],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff2dc")),
                        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d68a22")),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 9),
        ]

        overview_rows = [
            ("Schedule ID", schedule_id),
            ("JSON SHA-256", json_sha256),
            ("Exported UTC", document["exported_at_utc"]),
            ("Session", schedule["session_id"]),
            ("Policy", schedule["active_policy"]),
            (
                "Controller",
                (
                    schedule["controller"]["controller_name"]
                    if schedule["controller"] is not None
                    else schedule["active_policy"]
                ),
            ),
            (
                "Authority",
                f"{schedule['authority']['level']} - {schedule['authority']['label']} "
                f"(can_actuate state: {schedule['authority']['can_actuate']})",
            ),
            (
                "Watchdog",
                f"{schedule['watchdog']['mode']} (latched: {schedule['watchdog']['latched']})",
            ),
            (
                "Coverage",
                f"{schedule['completeness']['pass_count']} passes; "
                f"{schedule['completeness']['dwell_count']} dwell commands; "
                "web-cap complete: "
                f"{schedule['completeness']['complete_with_respect_to_web_dwell_cap']}",
            ),
            (
                "Safety checks",
                (
                    f"{schedule['safety_summary']['ok']} ok; "
                    f"{schedule['safety_summary']['warn']} warning; "
                    f"{schedule['safety_summary']['keyhole']} keyhole; "
                    f"{schedule['safety_summary']['rf_inhibit']} RF inhibit; "
                    f"{schedule['safety_summary']['reject']} rejected"
                ),
            ),
        ]
        overview = Table(
            [
                [
                    Paragraph(f"<b>{_pdf_text(label)}</b>", small_style),
                    Paragraph(_pdf_text(value), small_style),
                ]
                for label, value in overview_rows
            ],
            colWidths=[1.18 * inch, doc.width - 1.18 * inch],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf2f7")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d2dd")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            ),
        )
        story.extend([overview, Paragraph("Use and validity", heading_style)])
        for warning in schedule["warnings"]:
            story.append(Paragraph(f"- {_pdf_text(warning)}", body_style))

        story.extend(
            [
                Paragraph("Conventions", heading_style),
                Paragraph(
                    (
                        "UTC start-of-dwell epochs; topocentric ENU; azimuth clockwise "
                        "from true north; elevation above local horizon; angles in "
                        "degrees and holds in seconds. Every emitted hold ends no later "
                        "than the associated pass set epoch. Global sequence is display "
                        "order; independent-station commands can overlap."
                    ),
                    body_style,
                ),
                Paragraph("Pass summary", heading_style),
            ]
        )
        pass_header = [
            Paragraph("Pass", table_header_style),
            Paragraph("Station / geodetic", table_header_style),
            Paragraph("Rise UTC", table_header_style),
            Paragraph("Set UTC", table_header_style),
            Paragraph("Peak el", table_header_style),
            Paragraph("Dwells", table_header_style),
            Paragraph("Cap", table_header_style),
            Paragraph("Tail (s)", table_header_style),
        ]
        pass_rows: list[list[Any]] = [pass_header]
        for item in schedule["passes"]:
            geodetic = item["station_geodetic"]
            pass_rows.append(
                [
                    str(item["pass_idx"]),
                    Paragraph(
                        (
                            f"{_pdf_text(item['station'])}<br/>"
                            f"{geodetic['latitude_deg']:.5f}, "
                            f"{geodetic['longitude_deg']:.5f}; "
                            f"{geodetic['altitude_m']:.1f} m"
                        ),
                        tiny_style,
                    ),
                    Paragraph(_pdf_epoch(item["rise_utc"]), tiny_style),
                    Paragraph(_pdf_epoch(item["set_utc"]), tiny_style),
                    f"{item['peak_elevation_deg']:.3f}",
                    str(item["dwell_count"]),
                    "REACHED" if item["cap_reached"] else "no",
                    f"{item['unscheduled_tail_s']:.3f}",
                ]
            )
        pass_table = LongTable(
            pass_rows,
            repeatRows=1,
            hAlign="LEFT",
            colWidths=[
                0.38 * inch,
                1.1 * inch,
                1.43 * inch,
                1.43 * inch,
                0.65 * inch,
                0.55 * inch,
                0.65 * inch,
                0.65 * inch,
            ],
            style=TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "ScheduleSans"),
                    ("FONTSIZE", (0, 0), (-1, -1), 6.1),
                    ("LEADING", (0, 0), (-1, -1), 7.2),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20334d")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d2dd")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ALIGN", (4, 1), (-1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            ),
        )
        story.extend(
            [
                pass_table,
                PageBreak(),
                Paragraph("Exact ordered dwell schedule", title_style),
                Paragraph(
                    (
                        "The rows below are rendered directly from the canonical JSON. "
                        "Re-export after any observation or re-plan."
                    ),
                    body_style,
                ),
                Spacer(1, 6),
            ]
        )

        command_header_labels = (
            "Seq",
            "Dwell ID",
            "Start UTC",
            "End UTC",
            "Station",
            "Az",
            "El",
            "Hold",
            "Along (s)",
            "Cross (deg)",
            "P(det)",
            "Safety",
            "TX check",
        )
        command_rows: list[list[Any]] = [
            [Paragraph(label, table_header_style) for label in command_header_labels]
        ]
        row_styles: list[tuple[Any, ...]] = []
        for row_idx, command in enumerate(schedule["commands"], start=1):
            command_rows.append(
                [
                    str(command["sequence"]),
                    command["dwell_id"],
                    Paragraph(_pdf_epoch(command["start_utc"]), tiny_style),
                    Paragraph(_pdf_epoch(command["end_utc"]), tiny_style),
                    Paragraph(_pdf_text(command["station"]), tiny_style),
                    f"{command['az_deg']:.5f}",
                    f"{command['el_deg']:.5f}",
                    f"{command['hold_s']:.3f}",
                    f"{command['along_track_offset_s']:+.3f}",
                    f"{command['cross_track_offset_deg']:+.3f}",
                    f"{command['estimated_p_detect']:.5f}",
                    Paragraph(_pdf_text(command["safety_status"]), tiny_style),
                    "YES" if command["transmit_ok"] else "NO",
                ]
            )
            if not command["transmit_ok"] or command["safety_status"] in {
                "rf_inhibit",
                "reject",
            }:
                row_styles.append(
                    ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#fde8e7"))
                )
            elif command["safety_status"] in {"warn", "keyhole"}:
                row_styles.append(
                    ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#fff5dc"))
                )
            elif row_idx % 2 == 0:
                row_styles.append(
                    ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f7f9fb"))
                )

        command_table = LongTable(
            command_rows,
            repeatRows=1,
            splitByRow=1,
            hAlign="LEFT",
            colWidths=[
                0.33 * inch,
                0.80 * inch,
                1.50 * inch,
                1.50 * inch,
                1.20 * inch,
                0.58 * inch,
                0.58 * inch,
                0.58 * inch,
                0.60 * inch,
                0.62 * inch,
                0.62 * inch,
                0.95 * inch,
                0.42 * inch,
            ],
            style=TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "ScheduleSans"),
                    ("FONTSIZE", (0, 0), (-1, -1), 5.3),
                    ("LEADING", (0, 0), (-1, -1), 6.4),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20334d")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d2dd")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (1, -1), "CENTER"),
                    ("ALIGN", (5, 1), (10, -1), "RIGHT"),
                    ("ALIGN", (12, 1), (12, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
                    *row_styles,
                ]
            ),
        )
        story.append(command_table)

        exceptions = [
            command
            for command in schedule["commands"]
            if not command["transmit_ok"] or command["safety_status"] != "ok"
        ]
        if exceptions:

            def exception_note(command: dict[str, Any]) -> Paragraph:
                detail = command["safety_reason"] or "No additional reason recorded."
                return Paragraph(
                    (
                        f"<b>{_pdf_text(command['dwell_id'])}</b> - "
                        f"{_pdf_text(command['safety_status'])}; "
                        f"TX {'allowed' if command['transmit_ok'] else 'inhibited'}; "
                        f"{_pdf_text(detail)}"
                    ),
                    body_style,
                )

            exception_flowables: list[Any] = [
                Paragraph("Safety notes and exceptions", heading_style),
                *[exception_note(command) for command in exceptions],
            ]
            total_reason_chars = sum(len(command["safety_reason"]) for command in exceptions)
            if len(exceptions) <= 10 and total_reason_chars <= 1_200:
                story.append(KeepTogether(exception_flowables))
            else:
                story.append(KeepTogether(exception_flowables[:2]))
                story.extend(exception_flowables[2:])

        doc.build(
            story,
            onFirstPage=decorate_page,
            onLaterPages=decorate_page,
            canvasmaker=canvas_maker,
        )

    payload = output.getvalue()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-64:]:
        raise ScheduleExportError("PDF renderer returned an invalid document")
    return payload


def build_schedule_artifacts(
    session_id: str,
    session: PlanSession,
    *,
    exported_at: datetime | None = None,
) -> ScheduleArtifacts:
    """Build both output files from one captured schedule document."""
    document = build_schedule_document(session_id, session, exported_at=exported_at)
    json_bytes = canonical_json_bytes(document)
    json_sha256 = _sha256(json_bytes)
    pdf_bytes = _render_schedule_pdf(document, json_sha256)
    pdf_sha256 = _sha256(pdf_bytes)
    schedule_id = document["schedule_id"]
    stem = f"acquisition-schedule-{schedule_id[:16]}"
    completeness = document["schedule"]["completeness"]
    return ScheduleArtifacts(
        schedule_id=schedule_id,
        json_filename=f"{stem}.json",
        json_bytes=json_bytes,
        json_sha256=json_sha256,
        pdf_filename=f"{stem}.pdf",
        pdf_bytes=pdf_bytes,
        pdf_sha256=pdf_sha256,
        pass_count=int(completeness["pass_count"]),
        dwell_count=int(completeness["dwell_count"]),
        complete=bool(completeness["complete_with_respect_to_web_dwell_cap"]),
    )
