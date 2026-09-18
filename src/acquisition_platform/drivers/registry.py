"""Driver registry: id -> class, plus listing/factory helpers for the service.

This is the small lookup table the FastAPI app wires into an "antenna out"
endpoint: enumerate available drivers (`list_drivers`) and instantiate one by id
(`get_driver`). Every value in `DRIVERS` is an `AntennaDriver` subclass.
"""

from __future__ import annotations

from acquisition_platform.drivers.base import AntennaDriver
from acquisition_platform.drivers.rotctld import RotctldDriver, SatnogsRotatorDriver
from acquisition_platform.drivers.stream import StreamDriver
from acquisition_platform.drivers.vendors import (
    ComtechDriver,
    GenericRotatorDriver,
    KongsbergDriver,
)

DRIVERS: dict[str, type[AntennaDriver]] = {
    "rotctld": RotctldDriver,
    "satnogs": SatnogsRotatorDriver,
    "stream": StreamDriver,
    "generic_rotator": GenericRotatorDriver,
    "kongsberg": KongsbergDriver,
    "comtech": ComtechDriver,
}


def list_drivers() -> list[dict]:
    """Return registry metadata: [{id, name, capabilities, transport, description}].

    `name` is read from a throwaway dry-run instance so it reflects the concrete
    driver's `name` property without needing live connection args.
    """
    out: list[dict] = []
    for driver_id, cls in DRIVERS.items():
        try:
            probe = cls(dry_run=True)  # type: ignore[call-arg]
            name = probe.name
        except Exception:  # pragma: no cover - defensive; all drivers accept dry_run
            name = driver_id
        out.append(
            {
                "id": driver_id,
                "name": name,
                "capabilities": sorted(cls.capabilities),
                "transport": cls.transport,
                "description": cls.description,
            }
        )
    return out


def get_driver(driver_id: str, **kwargs: object) -> AntennaDriver:
    """Instantiate a driver by id, forwarding kwargs to its constructor."""
    try:
        cls = DRIVERS[driver_id]
    except KeyError:
        raise KeyError(f"unknown driver {driver_id!r}; known: {sorted(DRIVERS)}") from None
    return cls(**kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    listing = list_drivers()
    ids = {d["id"] for d in listing}
    assert ids == set(DRIVERS), ids
    for d in listing:
        assert d["transport"] in ("file", "tcp"), d
        assert isinstance(d["capabilities"], list)
    drv = get_driver("rotctld", dry_run=True)
    assert drv.name == "rotctld" and drv.dry_run
    try:
        get_driver("nope")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for unknown driver")
    print("[registry] PASS")
