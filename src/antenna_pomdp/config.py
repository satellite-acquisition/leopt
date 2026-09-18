"""Configuration for orbit uncertainty, observation models, and acquisition policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Orbit / pass geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitConfig:
    """Nominal orbit + uncertainty envelope (Gaussian on TLE elements).

    Sigmas are one-standard-deviation values applied to the named TLE field.
    The default orbit is a synthetic low-Earth-orbit example.
    """

    # Synthetic nominal TLE used by examples and tests.
    nominal_tle_line1: str = "1 25544U 98067A   24001.00000000  .00012345  00000-0  22588-3 0  9990"
    nominal_tle_line2: str = "2 25544  51.6400 100.0000 0001000  90.0000 270.0000 15.50000000000010"

    # 1-sigma uncertainties applied at the nominal epoch.
    sigma_inclination_deg: float = 0.05
    sigma_raan_deg: float = 0.20
    sigma_mean_anomaly_deg: float = 0.50  # ~along-track timing error
    sigma_eccentricity: float = 1e-4
    sigma_arg_perigee_deg: float = 0.20
    sigma_mean_motion_rev_per_day: float = 5e-4  # ~SMA uncertainty

    # Along-track time-of-flight uncertainty (launch slip), seconds.
    sigma_launch_slip_s: float = 10.0


@dataclass(frozen=True)
class StationConfig:
    """Ground station location (geodetic) and pass selection."""

    latitude_deg: float = 37.4275  # Near Stanford, California
    longitude_deg: float = -122.1697
    altitude_m: float = 30.0

    min_elevation_deg: float = 5.0  # pass-open threshold
    pass_duration_s: float = 600.0  # 10 min default
    pass_step_s: float = 1.0  # truth-state cadence


# ---------------------------------------------------------------------------
# Ground-station networks (multiple stations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundStationNetworkConfig:
    """A named network of ground stations.

    A network is just an ordered tuple of `StationConfig`s plus a short id and
    a human-readable name. Each member station carries its own
    lat/lon/alt/min-elevation, so heterogeneous networks (e.g. a polar dish at
    5 deg plus a mid-lat dish at 10 deg) are expressible. This object is
    deliberately data-only/declarative; all enumeration logic lives in
    `orbit.network`.

    `station_ids` (optional) gives each station a short label for tagging the
    merged pass list; if omitted, "S0", "S1", ... are used.
    """

    network_id: str
    name: str
    stations: tuple[StationConfig, ...]
    station_ids: tuple[str, ...] = ()

    def labels(self) -> tuple[str, ...]:
        """Per-station short labels (fall back to S0, S1, ...)."""
        if self.station_ids and len(self.station_ids) == len(self.stations):
            return tuple(self.station_ids)
        return tuple(f"S{i}" for i in range(len(self.stations)))


# Representative real-world teleport / TT&C sites. Coordinates are approximate
# geodetic positions of the actual antenna farms; min-elevation masks reflect
# typical S-band TT&C horizon masks (polar sites tend to a lower mask thanks to
# clean horizons). These are illustrative-but-realistic, not survey-grade.


def network_single_university() -> GroundStationNetworkConfig:
    """Baseline: one mid-latitude university dish (Stanford-like, ~37.4N).

    Matches the single-station scenario used by the existing LEOP experiment so
    the network study reduces to it exactly when the network has one member.
    """
    return GroundStationNetworkConfig(
        network_id="single_university",
        name="Single university dish (Stanford-like)",
        stations=(
            StationConfig(
                latitude_deg=37.4275,
                longitude_deg=-122.1697,
                altitude_m=20.0,
                min_elevation_deg=5.0,
            ),
        ),
        station_ids=("STAN",),
    )


def network_polar_sso() -> GroundStationNetworkConfig:
    """High-latitude / polar network optimised for SSO (KSAT-like).

    Svalbard (SvalSat, ~78.23N 15.4E) and Troll (TrollSat, Antarctica
    ~72.01S 2.53E) are the two near-polar KSAT sites that see almost every
    revolution of a Sun-synchronous orbit; a mid-latitude site (Tromso,
    ~69.66N) is added for redundancy. Polar clean horizons justify a 5 deg
    mask.
    """
    return GroundStationNetworkConfig(
        network_id="polar_sso",
        name="Polar SSO network (KSAT-like)",
        stations=(
            StationConfig(
                latitude_deg=78.229772,
                longitude_deg=15.407786,
                altitude_m=458.0,
                min_elevation_deg=5.0,
            ),  # SvalSat
            StationConfig(
                latitude_deg=-72.011667,
                longitude_deg=2.535,
                altitude_m=1270.0,
                min_elevation_deg=5.0,
            ),  # TrollSat
            StationConfig(
                latitude_deg=69.662500,
                longitude_deg=18.940278,
                altitude_m=10.0,
                min_elevation_deg=5.0,
            ),  # Tromso
        ),
        station_ids=("SVAL", "TROLL", "TROM"),
    )


def network_distributed_commercial() -> GroundStationNetworkConfig:
    """Globally distributed commercial network (AWS Ground Station / Leaf-like).

    Three antenna farms spread in longitude and latitude:
      - Punta Arenas, Chile (~53.0S 70.85W) -- southern, sees ascending+
        descending nodes (AWS GS / KSAT site).
      - Hawaii (~19.0N 155.66W) -- near-equatorial Pacific (AWS GS Hawaii).
      - Ohio / mid-US (~39.96N 83.0W) -- northern mid-latitude (AWS GS region).
    The spread in longitude maximises the number of distinct revolutions the
    network can catch each day.
    """
    return GroundStationNetworkConfig(
        network_id="distributed_commercial",
        name="Distributed commercial network (AWS/Leaf-like)",
        stations=(
            StationConfig(
                latitude_deg=-53.001111,
                longitude_deg=-70.854722,
                altitude_m=37.0,
                min_elevation_deg=5.0,
            ),  # Punta Arenas
            StationConfig(
                latitude_deg=19.014, longitude_deg=-155.663, altitude_m=110.0, min_elevation_deg=5.0
            ),  # Hawaii
            StationConfig(
                latitude_deg=39.962, longitude_deg=-83.003, altitude_m=275.0, min_elevation_deg=5.0
            ),  # Ohio
        ),
        station_ids=("PUQ", "HNL", "CMH"),
    )


def network_midlat_cluster() -> GroundStationNetworkConfig:
    """Clustered mid-latitude network in one hemisphere (worst case for SSO).

    Three European mid-latitude sites within ~20 deg longitude:
      - Darmstadt / ESOC area (~49.87N 8.62E).
      - Redu, Belgium (~50.0N 5.14E).
      - Villafranca / Madrid (~40.44N 3.95W).
    Because they share roughly the same ground-track geometry, they create
    large coverage gaps and many missed revolutions for an SSO -- the
    deliberate adversarial baseline for the network comparison.
    """
    return GroundStationNetworkConfig(
        network_id="midlat_cluster",
        name="Clustered mid-latitude network (worst-case SSO)",
        stations=(
            StationConfig(
                latitude_deg=49.871111,
                longitude_deg=8.622778,
                altitude_m=144.0,
                min_elevation_deg=5.0,
            ),  # Darmstadt
            StationConfig(
                latitude_deg=50.001944,
                longitude_deg=5.145556,
                altitude_m=385.0,
                min_elevation_deg=5.0,
            ),  # Redu
            StationConfig(
                latitude_deg=40.4425,
                longitude_deg=-3.953056,
                altitude_m=664.0,
                min_elevation_deg=5.0,
            ),  # Villafranca
        ),
        station_ids=("DARM", "REDU", "VILL"),
    )


# Registry: id -> factory. Used by the network experiment driver.
NETWORK_FACTORIES = {
    "single_university": network_single_university,
    "polar_sso": network_polar_sso,
    "distributed_commercial": network_distributed_commercial,
    "midlat_cluster": network_midlat_cluster,
}


def all_networks() -> tuple[GroundStationNetworkConfig, ...]:
    """Return the four representative networks in canonical order."""
    return tuple(f() for f in NETWORK_FACTORIES.values())


# ---------------------------------------------------------------------------
# Antenna / observation model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AntennaConfig:
    """Single-beam Gaussian antenna and acquisition-confirmation model.

    The boresight gain pattern is always Gaussian (`fwhm_deg`). The *on-axis*
    signal-detection probability is either the constant `1 - miss_rate`
    (`link_budget_enabled = False`) or, by default, a geometry-dependent
    `P_D(el, ρ)` computed from a compact self-contained link model
    (`antenna_pomdp.models.observation.on_axis_pd`): free-space path loss vs.
    slant range plus an excess atmospheric loss that grows as 1/sin(el). The
    defaults below are S-band-beacon-realistic and yield P_D ≈ 0.99 at a
    high/near pass and ≈ 0.76 at a low (5°) horizon pass.

    A receiver trigger is not automatically an acquisition. `false_alarm_rate`
    is the probability of a raw off-target trigger over
    `observation_reference_time_s`. Event probabilities are converted to
    cause-specific continuous hazards when `dwell_time_s` differs from that
    reference exposure, so changing the decision interval does not silently
    change receiver strength.
    `confirmation_true_accept_rate` is the probability that a signal-caused
    trigger passes carrier/identity confirmation, while
    `confirmation_false_accept_rate` is the probability that an off-target
    false trigger passes that confirmation. Evaluation keeps correct
    acquisition and false confirmation separate. Paper experiments use the
    default perfect false-trigger rejection.
    """

    fwhm_deg: float = 10.0  # full-width half-max of main beam
    dwell_time_s: float = 5.0  # how long one action points at one cell
    observation_reference_time_s: float = 5.0
    # Maximum midpoint-quadrature interval used to integrate a moving target
    # and moving track-relative command over one dwell.
    temporal_quadrature_step_s: float = 0.25
    false_alarm_rate: float = 0.01  # P(raw false trigger | no signal), per reference exposure
    miss_rate: float = 0.05  # P(no signal detection | boresight), per reference exposure
    confirmation_true_accept_rate: float = 1.0
    confirmation_false_accept_rate: float = 0.0
    min_signal_elevation_deg: float = 0.0  # physical/terrain mask for true signal

    # --- mechanical pointing / slew constraint ---
    # Maximum angular slew rate (deg/s). Between two dwells the boresight can
    # move at most max_slew_rate_deg_s * dwell_time_s; a commanded pointing
    # farther than that is only partially reached this dwell. The default is
    # effectively unconstrained (phased-array / instant re-point) so legacy
    # results are unchanged; a finite value models a mechanically steered dish
    # and gives the acquisition problem genuine sequential (route) structure.
    max_slew_rate_deg_s: float = 1.0e9

    # --- elevation/range-dependent link budget (on-axis P_D) ---
    link_budget_enabled: bool = True
    pd_max: float = 0.99  # asymptotic on-axis P_D at high link margin
    ref_range_m: float = 1.0e6  # reference slant range (~1000 km, mid-LEO pass)
    ref_margin_db: float = 12.0  # link margin (dB) at the reference range & zenith
    zenith_atmos_loss_db: float = 0.5  # one-way excess atmospheric loss at zenith
    pd_slope_per_db: float = 1.0  # logistic steepness of P_D vs. margin (per dB)
    pd_margin50_db: float = 0.0  # margin (dB) at which P_D = pd_max / 2


# ---------------------------------------------------------------------------
# Particle filter / belief
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    """Particle filter knobs."""

    n_particles: int = 500
    ess_resample_threshold: float = 0.5  # resample if ESS / N < threshold
    process_noise_along_track_s: float = 0.5  # diffusion injected on resample


# ---------------------------------------------------------------------------
# POMCP solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PomcpConfig:
    """POMCP / MCTS knobs (Silver & Veness 2010)."""

    n_rollouts: int = 1000
    max_depth: int = 20
    ucb_c: float = 1.0
    gamma: float = 0.95
    n_along_track_offsets: int = 5  # ±T values
    along_track_offset_max_s: float = 30.0
    n_cross_track_levels: int = 3  # K cross-track levels (incl. 0)
    cross_track_offset_max_deg: float = 8.0
    reward_detect: float = 1.0
    reward_step: float = -0.01  # small per-step penalty


# ---------------------------------------------------------------------------
# Continuous-action solver (PFT-DPW)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    """Particle-Filter-Tree with Double Progressive Widening (Sunberg & Kochenderfer 2018).

    An alternative to the fixed-grid POMCP of `PomcpConfig`. Rather than
    discretising the pointing action onto a hand-built grid, PFT-DPW samples
    continuous (Δτ, Δθ) actions and grows the action set with the visit count
    via progressive widening (k_a · N^alpha_a). Belief nodes hold an explicit
    particle set of `n_belief_particles`, so a node's value is estimated from a
    real filtered belief rather than a single root particle.
    """

    n_iterations: int = 1000  # planning simulations per decision (≈ POMCP rollouts)
    max_depth: int = 20
    gamma: float = 0.95
    ucb_c: float = 1.0
    # Action progressive widening: |A(h)| <= k_a * N(h)^alpha_a
    k_action: float = 4.0
    alpha_action: float = 0.25
    # Observation progressive widening (binary obs => effectively off, kept for parity)
    k_obs: float = 1.0
    alpha_obs: float = 0.0
    # Number of particles propagated inside a belief node during planning.
    n_belief_particles: int = 30
    # Continuous action bounds (mirror the PomcpConfig grid extents).
    along_track_offset_max_s: float = 30.0
    cross_track_offset_max_deg: float = 8.0
    reward_detect: float = 1.0
    reward_step: float = -0.01
    # The planner deviates from the myopic greedy anchor only when the best
    # tree action's estimated Q exceeds the greedy action's estimated Q by at
    # least this margin AND that action has been visited at least
    # `deviate_min_visits` times. This guard is a heuristic and does not
    # guarantee improvement over the one-step controller.
    deviate_margin: float = 0.15
    deviate_min_visits: int = 10


# ---------------------------------------------------------------------------
# Heuristic controllers (myopic baselines)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicConfig:
    """Knobs for the strong heuristic controllers in `baselines.heuristics`.

    These controllers are *belief-aware* but non-anticipatory: they choose the
    next dwell from the current posterior without multi-step lookahead.
      - `info_greedy`: maximise expected single-step information gain / detection
        probability over a candidate action set.
      - belief centroid / `mls`: point at the belief mean / most-likely
        particle. The legacy `qmdp` API name is only a compatibility alias for
        the centroid rule and is not a QMDP implementation.
    `n_candidate_actions` is the number of continuous actions sampled when the
    candidate set is not the discrete grid.
    """

    n_candidate_actions: int = 64
    # Objective for the greedy controller: "pdetect" (max expected detections)
    # or "infogain" (max expected belief-entropy reduction).
    greedy_objective: str = "infogain"


# ---------------------------------------------------------------------------
# Koopman optimal open-loop search allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KoopmanConfig:
    """Koopman-inspired non-adaptive search-effort allocation.

    Given the prior belief and an auxiliary per-cell exponential detection law,
    allocate a fixed dwell budget by classical water filling. The allocation is
    optimal for that surrogate only; it is not an open-loop upper bound for the
    overlapping, time-varying Gaussian-beam model.
    """

    n_cells: int = 41  # along-track look-angle cells the budget is spread over
    detection_rate_lambda: float = 1.0  # exponential search-law sensitivity


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    """Monte Carlo evaluation knobs."""

    n_trials: int = 50
    seed: int = 0


# ---------------------------------------------------------------------------
# LEOP (multi-pass) episode configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeopConfig:
    """Multi-pass LEOP episode + Electron-realistic insertion uncertainty.

    The position sigmas are in the RTN (radial / along-track / cross-track)
    frame at separation; they are converted to TLE element perturbations by
    `orbit.sampler.sample_leop_particles`. Launch slip is a *uniform* draw on
    [0, launch_slip_max_s] — this is the dominant along-track uncertainty for
    a first LEOP pass and explicitly is **not** the Gaussian
    `sigma_launch_slip_s` used by the single-pass experiments.
    """

    # Total LEOP window (s); 24h is the operational TT&C horizon.
    leop_window_s: float = 86400.0

    # Launch-slip envelope: uniform on [0, launch_slip_max_s] (seconds).
    launch_slip_max_s: float = 120.0

    # Insertion-error position sigmas (1-sigma, metres).
    sigma_along_track_m: float = 3000.0
    sigma_cross_track_m: float = 1000.0
    sigma_radial_m: float = 500.0

    # Inter-pass process noise: along-track drift growth per minute of dead-
    # reckoning, used by `models.particle_filter.propagate_belief_to`. The
    # default (0.1 km/min) is a conservative SGP4 propagation-error growth
    # rate for an LEO orbit between TLE updates.
    inter_pass_drift_km_per_min: float = 0.1
    # Approximate orbital speed used to translate the km drift into a slip-
    # equivalent (s). For LEO this is ~7.66 km/s.
    orbital_speed_km_s: float = 7.66


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Top-level container; just bundles all sub-configs."""

    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    station: StationConfig = field(default_factory=StationConfig)
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    pomcp: PomcpConfig = field(default_factory=PomcpConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    koopman: KoopmanConfig = field(default_factory=KoopmanConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    leop: LeopConfig = field(default_factory=LeopConfig)


def default_config() -> Config:
    """Return the recommended default configuration."""
    return Config()


# Derived constant helper (not a field — leaves Config frozen-friendly).
def beam_sigma_rad(antenna: AntennaConfig) -> float:
    """Convert FWHM (degrees) to Gaussian sigma in radians."""
    return np.deg2rad(antenna.fwhm_deg) / 2.355


if __name__ == "__main__":
    cfg = default_config()
    print(f"[config] orbit  : {cfg.orbit.nominal_tle_line1[:30]}...")
    print(f"[config] station: ({cfg.station.latitude_deg}, {cfg.station.longitude_deg})")
    print(f"[config] beam sigma = {np.rad2deg(beam_sigma_rad(cfg.antenna)):.3f} deg")
    print(f"[config] N particles = {cfg.filter.n_particles}, rollouts = {cfg.pomcp.n_rollouts}")
    for net in all_networks():
        print(
            f"[config] network {net.network_id:<22s} : {len(net.stations)} stations {net.labels()}"
        )
    assert len(all_networks()) == 4
    print("[config] PASS")
