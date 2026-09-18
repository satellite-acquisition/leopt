"""Observations IN: a live RF-chain feed replacing the manual DETECT button.

One `ObservationSource` interface, one adapter per RF back-end: operator input
(`manual`), a demodulator's lock + Eb/N0 (`modem`, SNMP/serial), an SDR band-
power detector (`sdr`), and a SCPI spectrum-analyzer marker (`spectrum_analyzer`).
Each hardware adapter has a physics-based simulated twin so the whole loop runs
with no radio attached. Reports (`Observation`) map to either the 1-bit detect
update or the SNR-valued update (`Observation.to_snr`), so a real modem's Eb/N0
collapses the belief far faster than a bare detect.
"""

from __future__ import annotations

from acquisition_platform.observations.base import Observation, ObservationSource
from acquisition_platform.observations.manual import ManualSource
from acquisition_platform.observations.modem import (
    SerialModemSource,
    SimulatedModemSource,
    SnmpModemSource,
)
from acquisition_platform.observations.registry import get_source, list_sources
from acquisition_platform.observations.sdr import SimulatedSdrSource, SoapySdrSource
from acquisition_platform.observations.spectrum_analyzer import (
    ScpiSpectrumAnalyzerSource,
    SimulatedSpectrumAnalyzer,
)

__all__ = [
    "Observation",
    "ObservationSource",
    "ManualSource",
    "SnmpModemSource",
    "SerialModemSource",
    "SimulatedModemSource",
    "SoapySdrSource",
    "SimulatedSdrSource",
    "ScpiSpectrumAnalyzerSource",
    "SimulatedSpectrumAnalyzer",
    "get_source",
    "list_sources",
]
