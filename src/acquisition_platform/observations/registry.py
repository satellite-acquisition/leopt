"""Registry of observation sources: id -> class, plus metadata for the UI."""

from __future__ import annotations

from acquisition_platform.observations.base import ObservationSource
from acquisition_platform.observations.manual import ManualSource
from acquisition_platform.observations.modem import (
    SerialModemSource,
    SimulatedModemSource,
    SnmpModemSource,
)
from acquisition_platform.observations.sdr import SimulatedSdrSource, SoapySdrSource
from acquisition_platform.observations.spectrum_analyzer import (
    ScpiSpectrumAnalyzerSource,
    SimulatedSpectrumAnalyzer,
)

# id -> (class, transport, needs-hardware, one-line description)
_ENTRIES: dict[str, tuple[type[ObservationSource], str, bool, str]] = {
    "manual": (ManualSource, "operator", False, "Operator / keyboard reports (the DETECT button)."),
    "modem_snmp": (SnmpModemSource, "snmp", True, "Modem carrier lock + Eb/N0 over SNMP."),
    "modem_serial": (
        SerialModemSource,
        "serial",
        True,
        "Modem lock + Eb/N0 over an ASCII serial line.",
    ),
    "modem_sim": (
        SimulatedModemSource,
        "simulated",
        False,
        "Physics-based modem stand-in (lock + Eb/N0).",
    ),
    "sdr_soapy": (
        SoapySdrSource,
        "soapysdr",
        True,
        "SDR band-power beacon detector (USRP/RTL/Lime).",
    ),
    "sdr_sim": (SimulatedSdrSource, "simulated", False, "Physics-based SDR band-SNR stand-in."),
    "spectrum_scpi": (
        ScpiSpectrumAnalyzerSource,
        "scpi-tcp",
        True,
        "SCPI spectrum-analyzer marker over TCP.",
    ),
    "spectrum_sim": (
        SimulatedSpectrumAnalyzer,
        "simulated",
        False,
        "Physics-based spectrum-analyzer stand-in.",
    ),
}


def list_sources() -> list[dict]:
    """Metadata for every registered source (for the /api/observation-sources UI)."""
    out = []
    for sid, (cls, transport, needs_hw, desc) in _ENTRIES.items():
        out.append(
            {
                "id": sid,
                "name": cls.name,
                "transport": transport,
                "needs_hardware": needs_hw,
                "capabilities": sorted(cls.capabilities),
                "description": desc,
            }
        )
    return out


def get_source(source_id: str, **kwargs) -> ObservationSource:
    """Construct an observation source by id (kwargs forwarded to the class)."""
    entry = _ENTRIES.get(source_id)
    if entry is None:
        raise KeyError(f"unknown observation source {source_id!r}; have {sorted(_ENTRIES)}")
    return entry[0](**kwargs)


if __name__ == "__main__":
    srcs = list_sources()
    ids = {s["id"] for s in srcs}
    assert {"manual", "modem_sim", "sdr_sim", "spectrum_sim"} <= ids
    # Every simulated source is hardware-free; every real transport flags hardware.
    for s in srcs:
        if s["transport"] == "simulated":
            assert not s["needs_hardware"], s
    m = get_source("manual")
    assert m.name == "manual"
    try:
        get_source("nope")
    except KeyError as e:
        assert "unknown observation source" in str(e)
    print(f"[obs.registry] {len(srcs)} sources: {sorted(ids)}")
    print("[obs.registry] PASS")
