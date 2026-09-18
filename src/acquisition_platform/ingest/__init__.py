"""Ingest layer: parse any orbit input into a common OpmState / Ephemeris.

Formats: TLE, CCSDS OPM (KVN), CCSDS OEM ephemeris (KVN), CCSDS OPM/OEM/NDM
(XML), and JSON (platform state, ephemeris, or CelesTrak/Space-Track OMM). The
`catalog` module pulls the published TLE from CelesTrak / Space-Track after
launch. `parse_any` (in `detect`) is the single autodetecting entry point.
"""

from __future__ import annotations

from acquisition_platform.ingest.catalog import (
    CatalogPoller,
    CelesTrakClient,
    PollResult,
    SpaceTrackClient,
    extract_tle_pair,
)
from acquisition_platform.ingest.detect import (
    IngestOutcome,
    detect_format,
    parse_any,
)
from acquisition_platform.ingest.json_ingest import omm_to_tle, parse_json
from acquisition_platform.ingest.ndm import parse_ndm_xml
from acquisition_platform.ingest.oem import Ephemeris, parse_oem
from acquisition_platform.ingest.opm import OpmState, parse_opm
from acquisition_platform.ingest.tle import parse_tle

__all__ = [
    "OpmState",
    "Ephemeris",
    "IngestOutcome",
    "parse_any",
    "detect_format",
    "parse_opm",
    "parse_oem",
    "parse_tle",
    "parse_ndm_xml",
    "parse_json",
    "omm_to_tle",
    "CelesTrakClient",
    "SpaceTrackClient",
    "CatalogPoller",
    "PollResult",
    "extract_tle_pair",
]
