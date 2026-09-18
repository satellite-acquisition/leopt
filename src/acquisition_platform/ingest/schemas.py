"""Request and response models for the planning API.

Angles are in degrees, durations in seconds, and epochs in ISO 8601 format.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class StationOut(BaseModel):
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    min_el_deg: float


class NetworkOut(BaseModel):
    id: str
    name: str
    n_stations: int
    stations: list[StationOut]


class SatAntennaOut(BaseModel):
    id: str
    name: str
    description: str
    coverage: float
    base_hold_s: float


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    format: Literal["opm", "tle", "oem", "ndm", "json", "auto"] = "auto"
    text: str
    covariance: list[list[float]] | None = Field(
        default=None, description="Optional 6x6 SI covariance over [r(3), v(3)]."
    )
    frame: Literal["RTN", "ECI"] | None = None
    sample_epoch: str | None = Field(
        default=None,
        description="For an OEM/ephemeris input, the ISO-8601 UTC time to sample "
        "the state at (default: the ephemeris start).",
    )


class IngestResult(BaseModel):
    epoch: str
    r_eci_m: list[float]
    v_eci_m: list[float]
    sigma_rtn_km: list[float]
    along_track_sigma_deg_now: float
    source_format: str = ""
    has_ephemeris: bool = False
    ephemeris_span_s: float | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Catalog (Space-Track / CelesTrak) auto-ingest
# ---------------------------------------------------------------------------


class CatalogQuery(BaseModel):
    """Query the public catalogs for a launch's element set.

    Provide a NORAD catalog number (once assigned) or an international
    designator (e.g. "2024-001A") known at launch. `source` selects the
    provider; Space-Track additionally needs `identity`/`password`.
    """

    source: Literal["celestrak", "spacetrack"] = "celestrak"
    catnr: int | None = None
    intldes: str | None = None
    identity: str | None = None
    password: str | None = None


class CatalogResult(BaseModel):
    found: bool
    line1: str | None = None
    line2: str | None = None
    source: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Antenna / plan request
# ---------------------------------------------------------------------------


class AntennaSpec(BaseModel):
    """Ground-antenna model: beam + detection stats + mechanical agility.

    `mount` selects a slew-rate preset (see planner.controller_select); an
    explicit `slew_rate_deg_s` overrides it and `agile=True` forces the
    phased-array (unconstrained re-point) treatment. These drive both the
    slew-limited planning clamp and the `policy="auto"` controller selection.
    """

    fwhm_deg: float = 10.0
    dwell_s: float = 5.0
    pd_onaxis: float = 0.95
    pfa: float = 0.01
    mount: str = "standard_dish"
    slew_rate_deg_s: float | None = None
    agile: bool = False


class PlanRequest(BaseModel):
    # `ingest` may be a fully-formed IngestResult, or a raw {format, text, ...}
    # body that the service ingests first. Accepted loosely as a dict.
    ingest: dict[str, Any]
    network: Any  # str network id, or a list of station dicts
    antenna: AntennaSpec = Field(default_factory=AntennaSpec)
    horizon_h: float = 12.0
    n_particles: int = 200
    rollouts: int = 150
    # "auto" selects info-greedy or PFT-DPW from the available slew per decision.
    policy: Literal["auto", "pomcp", "sweep", "nominal", "ekf", "infogreedy", "pft"] = "auto"
    sat_antenna: str = "turnstile"  # spacecraft antenna id (drives the hold time)
    # Graduated-trust authority level; advisory (recommend-only) by default.
    authority_level: Literal["L0_ADVISORY", "L1_SHADOW", "L2_SUPERVISED", "L3_AUTONOMOUS"] = (
        "L0_ADVISORY"
    )


class MountOut(BaseModel):
    id: str
    name: str
    slew_rate_deg_s: float | None
    agile: bool
    description: str


class RecommendControllerRequest(BaseModel):
    """Preview the auto-selected controller for an antenna, before planning."""

    antenna: AntennaSpec = Field(default_factory=AntennaSpec)
    sat_antenna: str = "turnstile"


class ControllerChoiceOut(BaseModel):
    policy: str
    controller_name: str
    is_agile: bool
    slew_rate_deg_s: float | None = None
    repoint_budget_deg: float | None = None
    coupling_span_deg: float
    rationale: str
    basis: str


# ---------------------------------------------------------------------------
# Plan / observe results
# ---------------------------------------------------------------------------


class PointingOut(BaseModel):
    t_s: float
    az_deg: float
    el_deg: float
    d_tau_s: float
    d_theta_deg: float
    p_detect: float
    dwell_s: float  # geometry-optimal hold time at this look-angle (seconds)
    # Safety-envelope verdict: ok | warn | keyhole | rf_inhibit | reject.
    safety_status: str = "ok"
    transmit_ok: bool = True
    safety_reason: str = ""


class PassOut(BaseModel):
    idx: int
    station: str
    rise: str
    set: str
    peak_el_deg: float
    pointings: list[PointingOut]


class BeliefGrid(BaseModel):
    along_s: list[float]
    cross_deg: list[float]
    weight: list[list[float]]


class BeliefSummary(BaseModel):
    entropy: float
    grid: BeliefGrid


class TrustOut(BaseModel):
    level: str
    label: str
    description: str
    can_actuate: bool


class WatchdogOut(BaseModel):
    mode: str
    latched: bool
    reason: str = ""


class SafetySummaryOut(BaseModel):
    total_dwells: int
    ok: int = 0
    warn: int = 0
    keyhole: int = 0
    rf_inhibit: int = 0
    reject: int = 0
    first_blocking: dict[str, Any] | None = None


class OpsStatusOut(BaseModel):
    """Trust ladder + belief watchdog + safety-envelope summary + rationale."""

    trust: TrustOut
    watchdog: WatchdogOut
    fallback_active: bool
    rationale: str
    safety: SafetySummaryOut


class PlanResult(BaseModel):
    session_id: str
    passes: list[PassOut]
    belief_summary: BeliefSummary
    # The controller actually driving the schedule, plus the recommendation +
    # rationale (present whenever a modern controller / auto is in play; the
    # recommendation may differ from `active_policy` if the operator overrode it).
    active_policy: str = "pomcp"
    controller: ControllerChoiceOut | None = None
    # Trust/safety status block (graduated-trust level, watchdog, envelope summary).
    ops: OpsStatusOut | None = None


class SetAuthorityRequest(BaseModel):
    session_id: str
    level: Literal["L0_ADVISORY", "L1_SHADOW", "L2_SUPERVISED", "L3_AUTONOMOUS"]


class RearmRequest(BaseModel):
    session_id: str


class ObserveRequest(BaseModel):
    """Fold one dwell outcome into the belief.

    `detected` is the 1-bit detect/no-detect (the manual button). A real RF
    chain can additionally report carrier lock and a graded Eb/N0 (or wideband
    SNR): supply `ebn0_db` (and optionally `locked`) and the belief is updated
    with the SNR-valued likelihood, which collapses the belief far faster.
    """

    session_id: str
    pass_idx: int
    dwell_idx: int
    detected: bool = False
    locked: bool | None = None
    ebn0_db: float | None = None
    snr_db: float | None = None
    # Measured carrier Doppler offset (Hz) from the nominal carrier, if the
    # receiver reports it: folds in the joint angle+Doppler along-track update.
    doppler_hz: float | None = None
    source: str = "manual"


class ObservationSourceOut(BaseModel):
    id: str
    name: str
    transport: str
    needs_hardware: bool
    capabilities: list[str]
    description: str


class BeliefResult(BaseModel):
    entropy: float
    grid: BeliefGrid


# ---------------------------------------------------------------------------
# Antenna command out: drivers + track export
# ---------------------------------------------------------------------------


class DriverOut(BaseModel):
    id: str
    name: str
    transport: str
    capabilities: list[str]
    description: str


class TrackExportRequest(BaseModel):
    """Export a pass's pointing schedule as a continuous track an ACU can load.

    `format` selects the encoding: a CSV / CCSDS-pointing az-el program track
    (densified from the sparse dwells at `rate_hz`), or a CCSDS OEM orbit
    ephemeris sampled from the nominal propagator over the pass.
    """

    session_id: str
    pass_idx: int
    format: Literal["csv", "ccsds_pointing", "oem"] = "csv"
    rate_hz: float = 1.0
    object_name: str = "SPACECRAFT"


class TrackExportResult(BaseModel):
    format: str
    filename: str
    n_samples: int
    content: str


class ScheduleExportRequest(BaseModel):
    """Export one materialized backend plan as a matched JSON/PDF pair."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)


class ScheduleExportResult(BaseModel):
    """Atomic export envelope; both contents represent the same schedule ID."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["platform-schedule-v1"]
    schedule_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    json_filename: str = Field(pattern=r"^[a-z0-9-]+\.json$")
    json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    json_size_bytes: int = Field(ge=1)
    json_content: str
    pdf_filename: str = Field(pattern=r"^[a-z0-9-]+\.pdf$")
    pdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pdf_size_bytes: int = Field(ge=1)
    pdf_base64: str
    pass_count: int = Field(ge=1)
    dwell_count: int = Field(ge=1)
    complete: bool


if __name__ == "__main__":
    req = PlanRequest(
        ingest={"format": "tle", "text": "x"},
        network="single_university",
        antenna=AntennaSpec(fwhm_deg=8.0),
    )
    assert req.policy == "auto" and req.n_particles == 200
    res = PlanResult(
        session_id="abc",
        passes=[
            PassOut(
                idx=0,
                station="STAN",
                rise="2024-01-01T00:00:00+00:00",
                set="2024-01-01T00:10:00+00:00",
                peak_el_deg=45.0,
                pointings=[
                    PointingOut(
                        t_s=0.0,
                        az_deg=10.0,
                        el_deg=20.0,
                        d_tau_s=0.0,
                        d_theta_deg=0.0,
                        p_detect=0.5,
                        dwell_s=30.0,
                    )
                ],
            )
        ],
        belief_summary=BeliefSummary(
            entropy=1.0,
            grid=BeliefGrid(along_s=[0.0], cross_deg=[0.0], weight=[[1.0]]),
        ),
    )
    # Round-trip through JSON to prove the contract serialises.
    dumped = res.model_dump_json()
    assert PlanResult.model_validate_json(dumped).session_id == "abc"
    assert len(dumped) > 0
    print("[schemas] PASS")
