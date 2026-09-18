"""Detect orbit-input formats and normalize them to ``IngestOutcome``.

Supported inputs are TLE, OPM/OEM KVN, NDM XML, and state/ephemeris/OMM JSON.
Each result contains an ``OpmState``, optional TLE lines, and the full
``Ephemeris`` when supplied, allowing later resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from acquisition_platform.ingest.json_ingest import parse_json
from acquisition_platform.ingest.ndm import looks_like_xml, parse_ndm_xml
from acquisition_platform.ingest.oem import Ephemeris, parse_oem
from acquisition_platform.ingest.opm import OpmState, parse_opm
from acquisition_platform.ingest.tle import parse_tle

_KNOWN_FORMATS = ("tle", "opm", "oem", "ndm", "json", "auto")


@dataclass(frozen=True)
class IngestOutcome:
    """Normalised result of ingesting any orbit input.

    `opm` is always present (sampled from an ephemeris at `sample_epoch` when the
    source was an OEM). `line1`/`line2` are the element lines when the source was
    a TLE/OMM (else None). `ephemeris` is the parsed OEM when applicable.
    """

    opm: OpmState
    line1: str | None
    line2: str | None
    ephemeris: Ephemeris | None
    source_format: str


def detect_format(text: str) -> str:
    """Best-effort format sniff for `text` -> one of the concrete formats."""
    s = text.lstrip()
    if not s:
        raise ValueError("empty ingest payload")
    if looks_like_xml(s):
        return "ndm"
    if s[0] in "{[":
        return "json"
    upper = s.upper()
    if "CCSDS_OEM_VERS" in upper:
        return "oem"
    if "CCSDS_OPM_VERS" in upper or ("EPOCH" in upper and "X_DOT" in upper and "=" in s):
        return "opm"
    # Element-line heuristic: a '1 '/'2 ' pair is a TLE. Many bare-number rows
    # with no '=' signs but timestamped leaders is an OEM.
    elem = [ln for ln in s.splitlines() if ln.strip().startswith(("1 ", "2 "))]
    if len(elem) >= 2:
        return "tle"
    if "=" in s:
        return "opm"
    return "oem"


def _from_ephemeris(eph: Ephemeris, sample_epoch: datetime | None, fmt: str) -> IngestOutcome:
    return IngestOutcome(
        opm=eph.to_opm(sample_epoch),
        line1=None,
        line2=None,
        ephemeris=eph,
        source_format=fmt,
    )


def parse_any(
    text: str,
    fmt: str = "auto",
    cov_6x6: np.ndarray | None = None,
    frame: str = "ECI",
    sample_epoch: datetime | None = None,
) -> IngestOutcome:
    """Parse `text` in the given (or auto-detected) format into an IngestOutcome.

    `cov_6x6` / `frame` apply to formats that accept an externally-supplied
    covariance (TLE, OMM JSON). `sample_epoch` selects the OEM sampling time
    (default: the ephemeris start).
    """
    fmt = (fmt or "auto").lower()
    if fmt not in _KNOWN_FORMATS:
        raise ValueError(f"unknown ingest format {fmt!r}; expected one of {_KNOWN_FORMATS}")
    if fmt == "auto":
        fmt = detect_format(text)

    if fmt == "tle":
        opm, l1, l2 = parse_tle(text, cov_6x6=cov_6x6, frame=frame)
        return IngestOutcome(opm, l1, l2, None, "tle")

    if fmt == "opm":
        return IngestOutcome(parse_opm(text), None, None, None, "opm")

    if fmt == "oem":
        return _from_ephemeris(parse_oem(text), sample_epoch, "oem")

    if fmt == "ndm":
        parsed = parse_ndm_xml(text)
        if isinstance(parsed, list):
            # Prefer an OPM (a mean state) if present, else the first ephemeris.
            opm_first = next((p for p in parsed if isinstance(p, OpmState)), None)
            if opm_first is not None:
                return IngestOutcome(opm_first, None, None, None, "ndm")
            parsed = parsed[0]
        if isinstance(parsed, Ephemeris):
            return _from_ephemeris(parsed, sample_epoch, "ndm")
        return IngestOutcome(parsed, None, None, None, "ndm")

    # json
    obj, l1, l2 = parse_json(text, cov_6x6=cov_6x6, frame=frame)
    if isinstance(obj, Ephemeris):
        return _from_ephemeris(obj, sample_epoch, "json")
    return IngestOutcome(obj, l1, l2, None, "json")


if __name__ == "__main__":
    from acquisition_platform.ingest.json_ingest import SAMPLE_OMM_JSON, SAMPLE_STATE_JSON
    from acquisition_platform.ingest.ndm import SAMPLE_OEM_XML, SAMPLE_OPM_XML
    from acquisition_platform.ingest.oem import SAMPLE_OEM
    from acquisition_platform.ingest.opm import SAMPLE_OPM
    from acquisition_platform.ingest.tle import SAMPLE_TLE

    cases = {
        "tle": SAMPLE_TLE,
        "opm": SAMPLE_OPM,
        "oem": SAMPLE_OEM,
        "ndm-opm": SAMPLE_OPM_XML,
        "ndm-oem": SAMPLE_OEM_XML,
        "json-omm": SAMPLE_OMM_JSON,
        "json-state": SAMPLE_STATE_JSON,
    }
    detected = {}
    for name, payload in cases.items():
        out = parse_any(payload, fmt="auto")
        detected[name] = out.source_format
        assert isinstance(out.opm, OpmState), name
        assert out.opm.r_eci_m.shape == (3,), name
        assert np.linalg.norm(out.opm.r_eci_m) > 6.0e6, name  # a real LEO radius

    assert detected["tle"] == "tle"
    assert detected["opm"] == "opm"
    assert detected["oem"] == "oem"
    assert detected["ndm-opm"] == "ndm" and detected["ndm-oem"] == "ndm"
    assert detected["json-omm"] == "json" and detected["json-state"] == "json"

    # TLE / OMM carry element lines; OPM / OEM / state do not.
    assert parse_any(SAMPLE_TLE).line1 is not None
    assert parse_any(SAMPLE_OMM_JSON).line1 is not None
    assert parse_any(SAMPLE_OEM).line1 is None and parse_any(SAMPLE_OEM).ephemeris is not None

    print(f"[detect] formats: {detected}")
    print("[detect] PASS")
