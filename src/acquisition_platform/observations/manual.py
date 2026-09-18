"""Manual observation source — the keyboard/operator path (the old button).

Kept as a first-class `ObservationSource` so the manual DETECT workflow and the
automated RF-chain feeds share one interface. An operator (or a test) pushes
reports with `report(...)`; `poll()` pops them FIFO. A report may be a bare
detect/no-detect or carry an Eb/N0, so even the manual path can drive the
SNR-valued belief update when the operator reads a modem off a screen.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone

from acquisition_platform.observations.base import Observation, ObservationSource


class ManualSource(ObservationSource):
    """A FIFO queue of operator-entered observations."""

    name = "manual"
    capabilities = frozenset({"detect", "ebn0"})

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._queue: deque[Observation] = deque()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def report(
        self,
        detected: bool,
        ebn0_db: float | None = None,
        *,
        locked: bool | None = None,
        t: datetime | None = None,
    ) -> Observation:
        """Enqueue one operator report; returns the created Observation."""
        obs = Observation(
            t=t or self._clock(),
            detected=detected,
            locked=locked if locked is not None else (detected if ebn0_db is not None else None),
            ebn0_db=ebn0_db,
            source=self.name,
        )
        self._queue.append(obs)
        return obs

    def poll(self) -> Observation | None:
        return self._queue.popleft() if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)


if __name__ == "__main__":
    clock = iter([datetime(2024, 1, 1, 0, 0, s, tzinfo=timezone.utc) for s in range(10)])
    src = ManualSource(clock=lambda: next(clock))
    src.report(True, ebn0_db=9.0)
    src.report(False)
    assert len(src) == 2
    a = src.poll()
    assert a.detected and a.has_snr() and a.to_snr().locked
    b = src.poll()
    assert not b.detected and b.to_snr().locked is False
    assert src.poll() is None  # drained

    # Streaming interface stops at the first None.
    src.report(True, ebn0_db=7.0)
    got = list(src.stream())
    assert len(got) == 1 and got[0].ebn0_db == 7.0
    print("[obs.manual] queue + stream OK")
    print("[obs.manual] PASS")
