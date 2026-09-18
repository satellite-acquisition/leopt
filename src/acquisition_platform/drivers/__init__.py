"""Convert planner dwells to antenna tracks and dispatch driver commands.

The pipeline normalizes dwells, samples a track, then exports a file or passes
it to a driver. Angles are in degrees and times are UTC-aware datetimes.
``dry_run=True`` records commands without contacting hardware.
"""

from __future__ import annotations

from acquisition_platform.drivers.base import (
    AntennaDriver,
    TrackSample,
    densify_track,
    dwells_from_passout,
    dwells_from_passplan,
)
from acquisition_platform.drivers.program_track import (
    to_ccsds_pointing_kvn,
    to_csv,
    to_oem_kvn,
)
from acquisition_platform.drivers.registry import (
    DRIVERS,
    get_driver,
    list_drivers,
)
from acquisition_platform.drivers.rotctld import RotctldDriver, SatnogsRotatorDriver
from acquisition_platform.drivers.stream import (
    StreamDriver,
    encode_csv,
    encode_json,
)
from acquisition_platform.drivers.vendors import (
    ComtechDriver,
    GenericRotatorDriver,
    KongsbergDriver,
)

__all__ = [
    # base
    "AntennaDriver",
    "TrackSample",
    "densify_track",
    "dwells_from_passplan",
    "dwells_from_passout",
    # program-track file exporters
    "to_csv",
    "to_ccsds_pointing_kvn",
    "to_oem_kvn",
    # drivers
    "RotctldDriver",
    "SatnogsRotatorDriver",
    "StreamDriver",
    "encode_csv",
    "encode_json",
    "GenericRotatorDriver",
    "KongsbergDriver",
    "ComtechDriver",
    # registry
    "DRIVERS",
    "get_driver",
    "list_drivers",
]
