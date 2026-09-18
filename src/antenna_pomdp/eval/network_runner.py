"""Evaluate acquisition policies with belief carryover across stations.

A single particle belief follows the time-ordered network pass list. Between
passes it receives process noise through propagate_belief_to. Station-specific
policy callables are cached and selected by each pass's station index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from antenna_pomdp.config import Config, GroundStationNetworkConfig
from antenna_pomdp.eval.leop_runner import (
    PolicyAct,
    PolicyFactory,
    _run_one_pass,
)
from antenna_pomdp.eval.metrics import belief_entropy
from antenna_pomdp.models.particle_filter import (
    make_belief,
    propagate_belief_to,
)
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.network import (
    NetworkPassWindow,
    enumerate_network_passes,
)
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import (
    sample_leop_particles,
    sample_leop_truth,
)


# ---------------------------------------------------------------------------
# Records (analogous to leop_runner's, with station tagging added)
# ---------------------------------------------------------------------------


@dataclass
class NetworkPassRecord:
    network_pass_number: int
    station_index: int
    station_id: str
    t_rise: datetime
    t_set: datetime
    max_elevation_deg: float
    n_steps: int
    detected: bool
    time_within_pass_s: float | None
    belief_entropy_at_start: float


@dataclass
class NetworkEpisodeResult:
    """Outcome of one full multi-station LEOP episode.

    Field names mirror `LeopEpisodeResult` so the existing `leop_metrics`
    helpers (cumulative_acquisition_curve, pass_success_rate, mean_time_to_
    acquire, failure_rate, summarize) work unchanged -- `acquired_pass_number`
    here is the *network* pass index. `acquired_station_id` adds the station
    that made the acquisition.
    """

    acquired: bool
    acquired_pass_number: int | None  # network pass index
    acquired_station_id: str | None
    time_since_separation_s: float | None
    time_within_pass_s: float | None
    n_passes_attempted: int
    n_passes_available: int
    final_belief_entropy: float
    passes: list[NetworkPassRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-station policy cache
# ---------------------------------------------------------------------------


def _build_station_policies(
    cfg: Config,
    network: GroundStationNetworkConfig,
    policy_factory: PolicyFactory,
    rng: np.random.Generator,
) -> list[PolicyAct]:
    """One policy callable per station.

    POMCP's solver is parameterised by a `Station` (it owns a per-station
    propagator + pass geometry), so we instantiate the factory once per station
    against a `Config` whose `.station` field is overridden to that station.
    Stateless policies (sweep / nominal / random) simply ignore the station.
    """
    from dataclasses import replace

    acts: list[PolicyAct] = []
    for st_cfg in network.stations:
        st_cfg_full = replace(
            cfg.station,
            **{
                "latitude_deg": st_cfg.latitude_deg,
                "longitude_deg": st_cfg.longitude_deg,
                "altitude_m": st_cfg.altitude_m,
                "min_elevation_deg": st_cfg.min_elevation_deg,
            },
        )
        station_cfg = replace(cfg, station=st_cfg_full)
        acts.append(policy_factory(station_cfg, rng))
    return acts


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def run_network_episode(
    cfg: Config,
    network: GroundStationNetworkConfig,
    policy_factory: PolicyFactory,
    rng: np.random.Generator,
    *,
    leop_start: datetime | None = None,
) -> NetworkEpisodeResult:
    """Run one multi-station LEOP episode with cross-station belief carryover."""
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    if leop_start is None:
        leop_start = nominal.epoch

    truth = sample_leop_truth(cfg.orbit, cfg.leop, rng)
    particles = sample_leop_particles(cfg.orbit, cfg.leop, n=cfg.filter.n_particles, rng=rng)
    belief = make_belief(particles)

    passes: list[NetworkPassWindow] = enumerate_network_passes(
        nominal,
        network,
        leop_start,
        leop_window_s=cfg.leop.leop_window_s,
    )
    station_policies = _build_station_policies(cfg, network, policy_factory, rng)

    records: list[NetworkPassRecord] = []
    prev_close: datetime | None = None
    for npw in passes:
        station: Station = npw.station
        policy_act = station_policies[npw.station_index]

        # Cross-station belief carryover: diffuse from the previous pass close
        # (at WHATEVER station) to this pass's rise. The belief is global.
        ref = prev_close if prev_close is not None else leop_start
        if npw.t_rise > ref:
            belief = propagate_belief_to(
                belief,
                ref,
                npw.t_rise,
                drift_km_per_min=cfg.leop.inter_pass_drift_km_per_min,
                orbital_speed_km_s=cfg.leop.orbital_speed_km_s,
                rng=rng,
            )

        belief, rec = _run_one_pass(
            cfg, nominal, station, truth, belief, npw.pass_window, policy_act, rng
        )
        records.append(
            NetworkPassRecord(
                network_pass_number=npw.network_pass_number,
                station_index=npw.station_index,
                station_id=npw.station_id,
                t_rise=rec.t_rise,
                t_set=rec.t_set,
                max_elevation_deg=rec.max_elevation_deg,
                n_steps=rec.n_steps,
                detected=rec.detected,
                time_within_pass_s=rec.time_within_pass_s,
                belief_entropy_at_start=rec.belief_entropy_at_start,
            )
        )
        prev_close = npw.t_set
        if rec.detected:
            tsa = (npw.t_rise - leop_start).total_seconds() + (rec.time_within_pass_s or 0.0)
            return NetworkEpisodeResult(
                acquired=True,
                acquired_pass_number=npw.network_pass_number,
                acquired_station_id=npw.station_id,
                time_since_separation_s=tsa,
                time_within_pass_s=rec.time_within_pass_s,
                n_passes_attempted=len(records),
                n_passes_available=len(passes),
                final_belief_entropy=belief_entropy(belief.weights),
                passes=records,
            )

    return NetworkEpisodeResult(
        acquired=False,
        acquired_pass_number=None,
        acquired_station_id=None,
        time_since_separation_s=None,
        time_within_pass_s=None,
        n_passes_attempted=len(records),
        n_passes_available=len(passes),
        final_belief_entropy=belief_entropy(belief.weights),
        passes=records,
    )


def run_network_monte_carlo(
    cfg: Config,
    network: GroundStationNetworkConfig,
    policy_factory: PolicyFactory,
    n_episodes: int,
    seed: int,
) -> list[NetworkEpisodeResult]:
    """Bit-reproducible MC: every episode is seeded from one master generator.

    Mirrors `eval.leop_runner.run_leop_monte_carlo`; not used by the parallel
    experiment driver (which seeds each job explicitly for the same effect).
    """
    master = np.random.default_rng(seed)
    out: list[NetworkEpisodeResult] = []
    for _ in range(n_episodes):
        trial_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        out.append(run_network_episode(cfg, network, policy_factory, trial_rng))
    return out


if __name__ == "__main__":
    from dataclasses import replace

    from antenna_pomdp.config import (
        default_config,
        network_polar_sso,
        network_single_university,
    )
    from antenna_pomdp.eval.leop_runner import make_sweep_policy

    base = default_config()
    # SSO TLE so the polar network actually sees passes (see orbit.network).
    sso_l1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
    sso_l2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    cfg = replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=sso_l1, nominal_tle_line2=sso_l2),
        antenna=replace(base.antenna, fwhm_deg=10.0),
        filter=replace(base.filter, n_particles=48),
        pomcp=replace(base.pomcp, n_rollouts=12, max_depth=3),
        leop=replace(base.leop, leop_window_s=6 * 3600.0),
    )

    rng = np.random.default_rng(0)
    ep_single = run_network_episode(cfg, network_single_university(), make_sweep_policy(), rng)
    rng = np.random.default_rng(0)
    ep_polar = run_network_episode(cfg, network_polar_sso(), make_sweep_policy(), rng)
    print(
        f"[network_runner] single: acquired={ep_single.acquired}, "
        f"n_passes={ep_single.n_passes_available}, records={len(ep_single.passes)}"
    )
    print(
        f"[network_runner] polar : acquired={ep_polar.acquired}, "
        f"station={ep_polar.acquired_station_id}, "
        f"n_passes={ep_polar.n_passes_available}, records={len(ep_polar.passes)}"
    )
    assert ep_polar.n_passes_available >= ep_single.n_passes_available
    # Belief carryover sanity: each pass record must carry a station tag.
    for r in ep_polar.passes:
        assert r.station_id in ("SVAL", "TROLL", "TROM")
    print("[network_runner] PASS")
