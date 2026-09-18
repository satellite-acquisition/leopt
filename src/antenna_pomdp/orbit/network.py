"""Merge station pass windows into one time-ordered network schedule.

Each pass retains its station index, label, and geometry. The merged list is
numbered from one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from antenna_pomdp.config import GroundStationNetworkConfig
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.leop import LeopPassWindow, enumerate_leop_passes
from antenna_pomdp.orbit.propagator import SGP4Propagator


@dataclass(frozen=True)
class NetworkPassWindow:
    """One pass within the LEOP window, tagged by the station that sees it.

    `network_pass_number` is the 1-based index within the *merged*, time-ordered
    network schedule. `station_pass_number` is the original per-station index
    (the `pass_number` from `enumerate_leop_passes`). `station` is the built
    `Station` object so the runner can drive `PointingEnv`/`update` directly.
    """

    network_pass_number: int
    station_index: int
    station_id: str
    station: Station
    pass_window: LeopPassWindow

    # Convenience pass-throughs (so callers can treat this like a LeopPassWindow).
    @property
    def t_rise(self) -> datetime:
        return self.pass_window.t_rise

    @property
    def t_set(self) -> datetime:
        return self.pass_window.t_set

    @property
    def t_max_el(self) -> datetime:
        return self.pass_window.t_max_el

    @property
    def max_elevation_deg(self) -> float:
        return self.pass_window.max_elevation_deg

    @property
    def duration_s(self) -> float:
        return self.pass_window.duration_s

    @property
    def time_since_start_s(self) -> float:
        return self.pass_window.time_since_start_s

    @property
    def station_pass_number(self) -> int:
        return self.pass_window.pass_number


def enumerate_network_passes(
    propagator: SGP4Propagator,
    network: GroundStationNetworkConfig,
    leop_start: datetime,
    *,
    leop_window_s: float = 86400.0,
    coarse_step_s: float = 30.0,
) -> list[NetworkPassWindow]:
    """Merged, time-ordered list of all passes across every station in `network`.

    Each station's `min_elevation_deg` (from its own `StationConfig`) is honoured
    independently. Passes are clipped to the LEOP window by
    `enumerate_leop_passes` and the merged list is sorted by rise time. Ties
    (identical rise times) are broken by station index for determinism.
    """
    labels = network.labels()
    tagged: list[tuple[datetime, int, Station, str, LeopPassWindow]] = []
    for s_idx, st_cfg in enumerate(network.stations):
        station = Station.from_config(st_cfg)
        passes = enumerate_leop_passes(
            propagator,
            station,
            leop_start,
            min_elevation_deg=st_cfg.min_elevation_deg,
            leop_window_s=leop_window_s,
            coarse_step_s=coarse_step_s,
        )
        for pw in passes:
            tagged.append((pw.t_rise, s_idx, station, labels[s_idx], pw))

    # Sort by rise time, breaking ties by station index for reproducibility.
    tagged.sort(key=lambda item: (item[0], item[1]))

    out: list[NetworkPassWindow] = []
    for i, (_t_rise, s_idx, station, label, pw) in enumerate(tagged, start=1):
        out.append(
            NetworkPassWindow(
                network_pass_number=i,
                station_index=s_idx,
                station_id=label,
                station=station,
                pass_window=pw,
            )
        )
    return out


if __name__ == "__main__":
    from antenna_pomdp.config import (
        default_config,
        network_polar_sso,
        network_single_university,
    )

    cfg = default_config()
    # The packaged nominal TLE is ISS-like (51.6 deg inclination); a polar
    # network at >=72 deg latitude never sees a 51.6 deg orbit. The network
    # study targets a Sun-synchronous orbit (~97.4 deg inclination, ~500 km),
    # for which the polar stations see almost every revolution. We use an SSO
    # TLE here so the smoke check exercises the realistic regime.
    sso_l1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
    sso_l2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    prop = SGP4Propagator(sso_l1, sso_l2)

    single = enumerate_network_passes(
        prop, network_single_university(), prop.epoch, leop_window_s=24 * 3600.0
    )
    polar = enumerate_network_passes(
        prop, network_polar_sso(), prop.epoch, leop_window_s=24 * 3600.0
    )

    print(f"[network] single_university: {len(single)} passes in 24h")
    print(f"[network] polar_sso        : {len(polar)} passes in 24h (should be more than single)")
    for p in polar[:6]:
        print(
            f"  netpass {p.network_pass_number:2d}: {p.station_id:>5s} "
            f"rise {p.t_rise.strftime('%H:%M:%S')} "
            f"peak el {p.max_elevation_deg:5.1f}  t+{p.time_since_start_s / 3600:5.2f}h"
        )

    # Merged list must be time-ordered.
    rises = [p.t_rise for p in polar]
    assert rises == sorted(rises), "network passes not time-ordered"
    # A 3-station polar net should see at least as many passes as 1 station.
    assert len(polar) >= len(single), "polar network saw fewer passes than single"
    # network_pass_number must be 1..M contiguous.
    assert [p.network_pass_number for p in polar] == list(range(1, len(polar) + 1))
    print("[network] PASS")
