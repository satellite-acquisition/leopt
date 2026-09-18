"""Observation records and interfaces for manual and receiver feedback.

Sources report detection and, where available, Eb/N0 or SNR. ``to_snr`` converts
reports for the graded observation model; ``detected_bit`` supplies binary
feedback. Hardware adapters cover modems, SDRs, and spectrum analyzers.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Observation:
    """One dwell's demodulator / detector report.

    `detected` is the hard yes/no. `locked` is the carrier-lock flag (None if the
    source cannot report it, in which case `detected` stands in). `ebn0_db` is the
    demodulator Eb/N0 estimate when available (None for a bare detector). `snr_db`
    is a wideband SNR (e.g. spectrum peak-to-floor) when Eb/N0 proper is not
    measured. `source` names the producing adapter; `meta` carries raw extras.
    """

    t: datetime
    detected: bool
    locked: bool | None = None
    ebn0_db: float | None = None
    snr_db: float | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    def detected_bit(self) -> bool:
        """Collapse to the legacy 1-bit detect (lock flag wins if present)."""
        return bool(self.locked) if self.locked is not None else bool(self.detected)

    def to_snr(self):  # -> antenna_pomdp...SnrObservation | None
        """Map to the engine's SnrObservation, or None if it can't be graded.

        Imported lazily to keep this layer additive over the engine. Returns:
          * a locked graded report when Eb/N0 (or a usable wideband SNR) is known,
          * a no-lock report when the carrier was not locked,
          * None when the signal was detected but no strength was measured (a bare
            detector) — the caller should then use the 1-bit detect update.
        """
        from antenna_pomdp.models.snr_observation import SnrObservation

        locked = self.detected_bit()
        value = self.ebn0_db if self.ebn0_db is not None else self.snr_db
        if locked and value is not None:
            return SnrObservation(locked=True, ebn0_db=float(value))
        if not locked:
            return SnrObservation(locked=False, ebn0_db=None)
        return None  # detected but ungraded -> use the detect update

    def has_snr(self) -> bool:
        """True when this report carries a usable graded Eb/N0 / SNR value."""
        return self.ebn0_db is not None or self.snr_db is not None


def _utcnow(clock: object | None = None) -> datetime:
    """Deterministic-friendly UTC clock hook (override in tests)."""
    if callable(clock):
        return clock()
    return datetime.now(timezone.utc)


class ObservationSource(abc.ABC):
    """A pluggable live feed of `Observation`s from the RF chain.

    One clean interface, one adapter per RF back-end. `poll()` returns the latest
    report (or None if nothing is available yet); `stream()` yields reports until
    the source is exhausted. `open()`/`close()` manage the underlying transport;
    the object is a context manager.
    """

    name: str = "observation-source"
    capabilities: frozenset[str] = frozenset()

    def open(self) -> None:  # noqa: A003 - matches driver-layer verb
        """Establish the transport / hardware session (no-op by default)."""

    def close(self) -> None:
        """Tear down the transport (no-op by default)."""

    @abc.abstractmethod
    def poll(self) -> Observation | None:
        """Return the next available Observation, or None if none is ready."""

    def stream(self, max_reports: int | None = None) -> Iterator[Observation]:
        """Yield observations from repeated `poll()`s until None / the cap."""
        n = 0
        while max_reports is None or n < max_reports:
            obs = self.poll()
            if obs is None:
                return
            yield obs
            n += 1

    def __enter__(self) -> ObservationSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


if __name__ == "__main__":
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # A graded, locked report -> a real SnrObservation.
    o1 = Observation(t=now, detected=True, locked=True, ebn0_db=8.5, source="modem")
    assert o1.detected_bit() is True and o1.has_snr()
    snr = o1.to_snr()
    assert snr.locked and abs(snr.ebn0_db - 8.5) < 1e-9

    # A bare detector (no Eb/N0) -> legacy bit; cannot be graded (to_snr -> None).
    o2 = Observation(t=now, detected=True, source="sdr")
    assert o2.detected_bit() is True and not o2.has_snr()
    assert o2.to_snr() is None

    # A no-lock report.
    o3 = Observation(t=now, detected=False, locked=False, source="modem")
    assert o3.detected_bit() is False and o3.to_snr().locked is False

    # Wideband SNR stands in for Eb/N0 when the source lacks a proper demod.
    o4 = Observation(t=now, detected=True, locked=True, snr_db=6.0, source="spectrum")
    assert abs(o4.to_snr().ebn0_db - 6.0) < 1e-9

    print("[obs.base] Observation + SnrObservation mapping OK")
    print("[obs.base] PASS")
