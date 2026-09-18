"""Enumerate a station's passes within a launch and early-orbit window.

Passes above the elevation mask are ordered by rise time and numbered from one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from antenna_pomdp.orbit.geometry import (
    PassWindow,
    Station,
    find_next_pass,
)
from antenna_pomdp.orbit.propagator import SGP4Propagator


@dataclass(frozen=True)
class LeopPassWindow:
    """One pass within the LEOP window."""

    pass_number: int  # 1-based index within the window
    t_rise: datetime
    t_set: datetime
    t_max_el: datetime
    max_elevation_deg: float

    @property
    def duration_s(self) -> float:
        return (self.t_set - self.t_rise).total_seconds()

    @property
    def time_since_start_s(self) -> float:
        """Filled in by `enumerate_leop_passes` via `rebase`."""
        return getattr(self, "_t_since_start_s", 0.0)


def _from_pass_window(pw: PassWindow, idx: int, leop_start: datetime) -> LeopPassWindow:
    out = LeopPassWindow(
        pass_number=idx,
        t_rise=pw.open,
        t_set=pw.close,
        t_max_el=pw.peak,
        max_elevation_deg=float(np.rad2deg(pw.peak_elevation_rad)),
    )
    object.__setattr__(out, "_t_since_start_s", (pw.open - leop_start).total_seconds())
    return out


def enumerate_leop_passes(
    propagator: SGP4Propagator,
    station: Station,
    leop_start: datetime,
    *,
    min_elevation_deg: float = 5.0,
    leop_window_s: float = 86400.0,
    coarse_step_s: float = 30.0,
) -> list[LeopPassWindow]:
    """Return all passes ≥ `min_elevation_deg` within the next `leop_window_s`.

    The pass search continues from the close of each pass plus one coarse
    step (to avoid re-detecting the same pass) until either no more passes
    are found or the LEOP window expires.
    """
    min_el_rad = float(np.deg2rad(min_elevation_deg))
    horizon_end = leop_start + timedelta(seconds=leop_window_s)
    passes: list[LeopPassWindow] = []
    cursor = leop_start
    idx = 1
    while cursor < horizon_end:
        remaining_s = (horizon_end - cursor).total_seconds()
        if remaining_s <= coarse_step_s:
            break
        pw = find_next_pass(
            propagator,
            station,
            cursor,
            min_elevation_rad=min_el_rad,
            search_horizon_s=remaining_s,
            coarse_step_s=coarse_step_s,
        )
        if pw is None:
            break
        # Clip the pass to the LEOP window.
        if pw.open >= horizon_end:
            break
        t_set = min(pw.close, horizon_end)
        clipped = PassWindow(
            open=pw.open,
            close=t_set,
            peak=pw.peak,
            peak_elevation_rad=pw.peak_elevation_rad,
        )
        passes.append(_from_pass_window(clipped, idx, leop_start))
        idx += 1
        cursor = pw.close + timedelta(seconds=coarse_step_s)
    return passes


if __name__ == "__main__":
    from antenna_pomdp.config import default_config

    cfg = default_config()
    station = Station.from_config(cfg.station)
    prop = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    passes = enumerate_leop_passes(
        prop,
        station,
        prop.epoch,
        min_elevation_deg=cfg.station.min_elevation_deg,
        leop_window_s=24 * 3600.0,
    )
    print(f"[leop] {len(passes)} passes in 24h LEOP window")
    for p in passes[:6]:
        print(
            f"  pass {p.pass_number}: rise {p.t_rise.strftime('%H:%M:%S')} "
            f"set {p.t_set.strftime('%H:%M:%S')} "
            f"peak el {p.max_elevation_deg:5.1f}°  dur {p.duration_s:5.0f}s  "
            f"t+{p.time_since_start_s / 3600:5.2f}h"
        )
    assert 1 <= len(passes) <= 20, "implausible LEOP pass count"
    print("[leop] PASS")
