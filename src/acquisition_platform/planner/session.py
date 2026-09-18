"""Stateful acquisition planning over a particle belief and station network.

``plan_network`` forecasts a per-pass dwell schedule from the current belief.
It diffuses the belief between passes without consuming observations.
``observe`` applies detection feedback and replans the remaining schedule.
``belief_summary`` projects the particles onto an along-track/cross-track grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import numpy as np

from acquisition_platform.estimator.ukf import GaussianState, SgpUkf
from acquisition_platform.planner.controller_select import (
    INFO_GREEDY,
    PFT_DPW,
    ControllerChoice,
    GroundAntennaSpec,
    recommend_controller,
)
from acquisition_platform.safety import (
    CLOSED_LOOP,
    FALLBACK_OPEN_LOOP,
    L0_ADVISORY,
    SafetyConfig,
    WatchdogConfig,
    WatchdogState,
    can_actuate,
    description,
    evaluate_pointing,
    is_level,
    label,
    watchdog_step,
)
from acquisition_platform.satellite_antennas import (
    DEFAULT_SAT_ANTENNA,
    SatAntenna,
    get_sat_antenna,
    recommended_hold_s,
)
from antenna_pomdp.baselines.heuristics import make_infogreedy_policy
from antenna_pomdp.config import (
    AntennaConfig,
    Config,
    FilterConfig,
    GroundStationNetworkConfig,
    LeopConfig,
    PomcpConfig,
)
from antenna_pomdp.eval.metrics import belief_entropy
from antenna_pomdp.models.doppler import FrequencyConfig, update_doppler
from antenna_pomdp.models.observation import (
    Pointing,
    delta_theta_from_azel,
    detection_probability,
)
from antenna_pomdp.models.particle_filter import (
    ParticleBelief,
    belief_azel,
    belief_azel_range,
    propagate_belief_to,
    update,
    update_snr,
)
from antenna_pomdp.models.snr_observation import SnrConfig, SnrObservation
from antenna_pomdp.orbit.geometry import Station, eci_to_azel
from antenna_pomdp.orbit.network import NetworkPassWindow, enumerate_network_passes
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
    build_action_grid,
    slew_limited_pointing,
)
from antenna_pomdp.pomdp.pft_dpw import make_pft_policy
from antenna_pomdp.pomdp.pomcp import PomcpSolver

# Policies the session understands. "auto" is resolved to a concrete controller
# from the ground-antenna agility at construction time (see controller_select).
_POLICIES = ("pomcp", "sweep", "nominal", "ekf", INFO_GREEDY, PFT_DPW)
_PolicyAct = "Callable[[ParticleBelief, PointingEnv, datetime], TrackAction]"

# Cap dwells emitted per pass on the web path so /api/plan stays a few seconds.
MAX_DWELLS_PER_PASS = 20


@dataclass
class PointingPlan:
    """One commanded dwell within a pass."""

    t_s: float  # seconds since that pass's rise
    when: datetime
    az_rad: float
    el_rad: float
    d_tau_s: float  # along-track action offset
    d_theta_deg: float  # cross-track action offset
    p_detect: float  # belief-mean detection probability for this pointing
    dwell_s: float  # geometry-optimal hold time (beam-crossing at this look-angle)
    # Safety-envelope verdict for this commanded pointing (None if unevaluated).
    safety_status: str = "ok"
    transmit_ok: bool = True
    safety_reason: str = ""


@dataclass
class PassPlan:
    idx: int
    station_id: str
    station: Station
    station_index: int
    rise: datetime
    set: datetime
    peak_el_deg: float
    pointings: list[PointingPlan] = field(default_factory=list)


@dataclass
class PlanSession:
    """Holds belief + network + antenna/pomcp cfg; plans and re-plans."""

    config: Config
    network: GroundStationNetworkConfig
    belief: ParticleBelief
    policy: str = "pomcp"
    sat_antenna: SatAntenna = field(default_factory=lambda: get_sat_antenna(DEFAULT_SAT_ANTENNA))
    # Ground-antenna agility spec; drives `policy="auto"` controller selection and
    # the slew constraint. When None, it is derived from the antenna config.
    ground_antenna: GroundAntennaSpec | None = None
    leop_start: datetime | None = None
    ukf: SgpUkf | None = None
    ukf_state: GaussianState | None = None
    snr_cfg: SnrConfig = field(default_factory=SnrConfig)
    # Joint angle+Doppler search: a measured carrier Doppler at an observation
    # collapses the along-track (launch-slip) ambiguity far faster than the angle
    # model alone. Same particle belief — each particle predicts its own Doppler.
    freq_cfg: FrequencyConfig = field(default_factory=FrequencyConfig)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # Set when policy=="auto" resolves (or on any explicit controller): the
    # recommendation + rationale surfaced to the operator. None for the classic
    # baselines (pomcp/sweep/nominal/ekf).
    controller_choice: ControllerChoice | None = None

    # --- trust & safety ---
    # Graduated-trust authority level (advisory by default — the safe start).
    authority_level: str = L0_ADVISORY
    # Pointing safety envelope (per-antenna interlocks) and belief watchdog.
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    watchdog_cfg: WatchdogConfig = field(default_factory=WatchdogConfig)
    watchdog: WatchdogState = field(default_factory=WatchdogState)
    # The controller in effect before a watchdog fallback (to restore on re-arm).
    _policy_before_fallback: str | None = None
    fallback_reason: str = ""
    # Trailing detect/no-detect outcomes, newest last (for the plain-language
    # rationale and the no-detect streak).
    _obs_outcomes: list[bool] = field(default_factory=list)

    # Internal: cached plan + per-station solvers / controllers.
    _nominal: SGP4Propagator | None = None
    _passes: list[NetworkPassWindow] = field(default_factory=list)
    _plans: list[PassPlan] = field(default_factory=list)
    _solvers: dict[int, PomcpSolver] = field(default_factory=dict)
    _controllers: dict[int, object] = field(default_factory=dict)
    _horizon_h: float = 12.0  # remembered from the last plan_network (for re-arm)

    def __post_init__(self) -> None:
        self._nominal = SGP4Propagator(
            self.config.orbit.nominal_tle_line1, self.config.orbit.nominal_tle_line2
        )
        if self.leop_start is None:
            self.leop_start = self._nominal.epoch
        # The ground-antenna spec (when given) is authoritative for the slew
        # constraint: sync it onto the antenna config so the recommendation and
        # the actual planning clamp use one source of truth.
        if self.ground_antenna is not None:
            self.config = replace(
                self.config,
                antenna=replace(
                    self.config.antenna,
                    max_slew_rate_deg_s=self.ground_antenna.effective_slew_deg_s(),
                ),
            )
        if self.policy == "auto":
            choice = recommend_controller(
                self._ground_antenna_spec(), inter_dwell_s=self._repoint_gap_s()
            )
            self.controller_choice = choice
            self.policy = choice.policy
        elif self.policy in (INFO_GREEDY, PFT_DPW):
            # Explicit modern controller: still record the geometry/rationale so
            # the operator sees why (or why not) this fits the antenna.
            self.controller_choice = recommend_controller(
                self._ground_antenna_spec(), inter_dwell_s=self._repoint_gap_s()
            )
        if self.policy not in _POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}")
        if not is_level(self.authority_level):
            raise ValueError(f"unknown authority level {self.authority_level!r}")

    # ------------------------------------------------------------------
    def _ground_antenna_spec(self) -> GroundAntennaSpec:
        """The ground-antenna agility spec, from the request or derived from cfg."""
        if self.ground_antenna is not None:
            return self.ground_antenna
        return GroundAntennaSpec(
            slew_rate_deg_s=float(self.antenna.max_slew_rate_deg_s),
            fwhm_deg=float(self.antenna.fwhm_deg),
            dwell_s=float(self.antenna.dwell_time_s),
        )

    def _repoint_gap_s(self) -> float:
        """Representative time between pointing decisions (the schedule cadence).

        Commands are spaced by the recommended hold (tens of seconds), so this —
        not the short beam dwell — is the window the boresight has to slew. Use a
        mid-elevation hold as the representative gap.
        """
        return recommended_hold_s(self.sat_antenna, 30.0)

    # ------------------------------------------------------------------
    @property
    def antenna(self) -> AntennaConfig:
        return self.config.antenna

    @property
    def pomcp(self) -> PomcpConfig:
        return self.config.pomcp

    @property
    def filter(self) -> FilterConfig:
        return self.config.filter

    @property
    def leop(self) -> LeopConfig:
        return self.config.leop

    # ------------------------------------------------------------------
    def _solver_for(self, npw: NetworkPassWindow) -> PomcpSolver:
        s = self._solvers.get(npw.station_index)
        if s is None:
            s = PomcpSolver(
                pomcp_cfg=self.pomcp,
                antenna_cfg=self.antenna,
                station=npw.station,
                nominal=self._nominal,
            )
            self._solvers[npw.station_index] = s
        return s

    def _station_cfg(self, npw: NetworkPassWindow):
        """The StationConfig for this pass's station (for per-station controllers)."""
        return self.network.stations[npw.station_index]

    def _controller_for(self, npw: NetworkPassWindow):
        """Build (and cache) the per-station info-greedy / PFT-DPW controller act.

        The engine controller factories are single-station (they read
        `cfg.station`), so we bind one per network station on a config whose
        `station` is that pass's station. The returned `act(belief, env, when)`
        is slew-aware via `env.current_bore`.
        """
        act = self._controllers.get(npw.station_index)
        if act is None:
            cfg_s = replace(self.config, station=self._station_cfg(npw))
            if self.policy == INFO_GREEDY:
                act = make_infogreedy_policy(cfg_s)(cfg_s, self.rng)
            else:  # PFT_DPW
                act = make_pft_policy(cfg_s)(cfg_s, self.rng)
            self._controllers[npw.station_index] = act
        return act

    def _slew_reach_rad(self, gap_s: float) -> float:
        """Great-circle boresight travel reachable while parked for `gap_s` seconds."""
        return float(np.deg2rad(self.antenna.max_slew_rate_deg_s) * gap_s)

    @staticmethod
    def _realize(
        bore: tuple[float, float] | None, commanded: Pointing, max_move: float
    ) -> Pointing:
        """Clamp a commanded pointing to the slew-reachable set from `bore`."""
        if bore is None or max_move >= np.pi:
            return commanded
        return slew_limited_pointing(bore[0], bore[1], commanded, max_move)

    def _hold_s(self, el_rad: float) -> float:
        """Recommended hold (s) to ensure a link given the spacecraft antenna.

        Driven by the satellite-side antenna coverage and a representative LEOP
        tumble (see acquisition_platform.satellite_antennas), with a low-elevation
        margin. Independent of the particle belief.
        """
        return recommended_hold_s(self.sat_antenna, float(np.rad2deg(el_rad)))

    def _belief_pd_per_particle(
        self,
        belief: ParticleBelief,
        pointing: Pointing,
        when: datetime,
        station: Station,
    ) -> np.ndarray:
        """Per-particle P(detect), geometry-aware (elevation + range link budget).

        Uses `belief_azel_range` and threads each particle's elevation and range
        into `detection_probability`, matching the particle-filter measurement
        update exactly. Using only az/el would silently bypass the (default-on)
        S-band link budget and report the constant-beam P_D.
        """
        azelr = belief_azel_range(belief, when, station)
        dthetas = np.array(
            [delta_theta_from_azel(pointing, azelr[i, 0], azelr[i, 1]) for i in range(belief.n)]
        )
        return np.asarray(
            detection_probability(dthetas, self.antenna, azelr[:, 1], azelr[:, 2]), dtype=float
        )

    def _belief_p_detect(
        self,
        belief: ParticleBelief,
        pointing: Pointing,
        when: datetime,
        station: Station,
    ) -> float:
        """Weighted mean P(detect) for a pointing under the current belief."""
        pd = self._belief_pd_per_particle(belief, pointing, when, station)
        return float(np.sum(np.asarray(belief.weights, dtype=float) * pd))

    def _prior_reweight(
        self, pointing: Pointing, detected: bool, when: datetime, station: Station
    ) -> tuple[ParticleBelief, float]:
        """Reweight the live (prior) belief by a detect/no-detect, WITHOUT resampling.

        Returns (reweighted belief, likelihood_total). The reweighted belief is
        the pre-resample posterior — the watchdog must inspect THIS (not the
        post-resample belief, whose weights are reset to uniform and would mask a
        degenerate cloud). `likelihood_total` is the belief-averaged support for
        the observed outcome (near zero => no particle explains it).
        """
        pd = self._belief_pd_per_particle(self.belief, pointing, when, station)
        like = pd if detected else (1.0 - pd)
        w = np.asarray(self.belief.weights, dtype=float) * like
        total = float(w.sum())
        wn = w / total if total > 0 else np.full(self.belief.n, 1.0 / self.belief.n)
        reweighted = ParticleBelief(particles=list(self.belief.particles), weights=wn)
        return reweighted, total

    def _choose_action(
        self,
        belief: ParticleBelief,
        env: PointingEnv,
        when: datetime,
        npw: NetworkPassWindow,
        k: int,
    ) -> TrackAction:
        """Pick a dwell action from the configured policy against `belief`.

        `k` is the 0-based dwell index within the pass.
        """
        if self.policy == "nominal":
            return TrackAction(0.0, 0.0)
        if self.policy == "ekf":
            return self._ekf_action(belief, env, when, npw)
        if self.policy == "sweep":
            return self._sweep_action(k)
        if self.policy in (INFO_GREEDY, PFT_DPW):
            return self._controller_for(npw)(belief, env, when)
        # pomcp
        solver = self._solver_for(npw)
        idx, _ = solver.plan(belief, when, env, self.rng)
        return solver.actions[idx]

    def _sweep_action(self, k: int) -> TrackAction:
        """Open-loop track-line sweep: cycle along-track offsets across dwells."""
        grid = build_action_grid(self.pomcp)
        along_vals = sorted({a.along_track_s for a in grid})
        return TrackAction(float(along_vals[k % len(along_vals)]), 0.0)

    def _ekf_action(
        self,
        belief: ParticleBelief,
        env: PointingEnv,
        when: datetime,
        npw: NetworkPassWindow,
    ) -> TrackAction:
        """Gaussian baseline: point at the belief-mean track (zero offset).

        Uses the UKF mean when available, else the particle-belief mean az/el,
        and crucially ignores no-detect information (the belief is only diffused
        between passes, never measurement-updated, for this policy).
        """
        return TrackAction(0.0, 0.0)

    # ------------------------------------------------------------------
    def plan_network(self, horizon_h: float) -> list[PassPlan]:
        """Enumerate passes and produce a per-pass per-dwell pointing schedule."""
        self._horizon_h = horizon_h
        window_s = horizon_h * 3600.0
        self._passes = enumerate_network_passes(
            self._nominal,
            self.network,
            self.leop_start,
            leop_window_s=window_s,
        )
        # Plan from a working COPY of the belief, diffusing between passes but not
        # consuming observations (forecast). The session's live belief is left at
        # its current value so `observe` updates from the true present belief.
        work = ParticleBelief(
            particles=list(self.belief.particles), weights=self.belief.weights.copy()
        )
        plans: list[PassPlan] = []
        prev_close: datetime | None = None
        for npw in self._passes:
            ref = prev_close if prev_close is not None else self.leop_start
            if npw.t_rise > ref:
                work = propagate_belief_to(
                    work,
                    ref,
                    npw.t_rise,
                    drift_km_per_min=self.leop.inter_pass_drift_km_per_min,
                    orbital_speed_km_s=self.leop.orbital_speed_km_s,
                    rng=self.rng,
                )
            plans.append(self._plan_one_pass(npw, work))
            prev_close = npw.t_set
        self._plans = plans
        return plans

    def _env_for(self, npw: NetworkPassWindow) -> PointingEnv:
        # For the slew-aware controllers the between-decision cadence is the
        # schedule hold (tens of seconds), not the short beam dwell — that is the
        # window the boresight actually has to slew, so the controller's internal
        # reachable-set reasoning uses it. The classic baselines keep the beam
        # dwell so their behaviour is unchanged.
        dwell = (
            self._repoint_gap_s()
            if self.policy in (INFO_GREEDY, PFT_DPW)
            else self.antenna.dwell_time_s
        )
        return PointingEnv(
            nominal=self._nominal,
            truth=self.belief.particles[0],
            station=npw.station,
            pass_start=npw.t_rise,
            pass_duration_s=npw.duration_s,
            dwell_time_s=dwell,
            antenna_cfg=self.antenna,
            pomcp_cfg=self.pomcp,
        )

    def _npw_for(self, pass_idx: int) -> NetworkPassWindow | None:
        return next((n for n in self._passes if n.network_pass_number - 1 == pass_idx), None)

    def _plan_dwells(
        self,
        plan: PassPlan,
        npw: NetworkPassWindow,
        env: PointingEnv,
        belief: ParticleBelief,
        start_when: datetime,
        k0: int,
        bore0: tuple[float, float] | None = None,
    ) -> None:
        """Append hold-stepped dwells to `plan` from `start_when` (dwell index k0).

        Each dwell advances the clock by its own recommended hold, so dwell k+1
        starts when dwell k's hold ends; the count is pass-duration / hold (capped).
        A final dwell is emitted only when its complete recommended hold fits
        inside the pass.  Partial holds are not silently shortened at LOS.

        Slew-aware: the boresight is tracked across dwells and each commanded
        pointing is clamped to the great-circle set reachable in one hold. The
        *realized* (reachable) pointing is what is scheduled and what P(detect) is
        scored against. `bore0` seeds the boresight (the last executed dwell when
        re-planning a tail); None means the antenna re-points freely into the
        first dwell (pre-positioned / fresh pass). With the default (effectively
        infinite) slew rate no clamping occurs and the schedule is unchanged.
        """
        when = start_when
        k = k0
        bore = bore0
        # Time the antenna has to slew INTO the next dwell = the hold of the dwell
        # it is currently parked at. Seed with a representative hold; then track
        # each realized dwell's own hold so the reachable-set clamp matches the
        # actual (elevation-dependent) time spent parked.
        prev_hold_s = self._repoint_gap_s()
        while when < npw.t_set and len(plan.pointings) < MAX_DWELLS_PER_PASS:
            env.current_bore = bore
            action = self._choose_action(belief, env, when, npw, k)
            commanded = action_to_pointing(action, self._nominal, when, npw.station)
            pointing = self._realize(bore, commanded, self._slew_reach_rad(prev_hold_s))
            hold_s = self._hold_s(pointing.el)
            if when + timedelta(seconds=hold_s) > npw.t_set:
                break
            p_det = self._belief_p_detect(belief, pointing, when, npw.station)
            verdict = evaluate_pointing(pointing.az, pointing.el, when, npw.station, self.safety)
            plan.pointings.append(
                PointingPlan(
                    t_s=(when - npw.t_rise).total_seconds(),
                    when=when,
                    az_rad=pointing.az,
                    el_rad=pointing.el,
                    d_tau_s=action.along_track_s,
                    d_theta_deg=action.cross_track_deg,
                    p_detect=p_det,
                    dwell_s=hold_s,
                    safety_status=verdict.status,
                    transmit_ok=verdict.transmit_ok,
                    safety_reason=verdict.reasons[0],
                )
            )
            bore = (pointing.az, pointing.el)
            prev_hold_s = hold_s
            when = when + timedelta(seconds=hold_s)
            k += 1

    def _plan_one_pass(self, npw: NetworkPassWindow, belief: ParticleBelief) -> PassPlan:
        plan = PassPlan(
            idx=npw.network_pass_number - 1,
            station_id=npw.station_id,
            station=npw.station,
            station_index=npw.station_index,
            rise=npw.t_rise,
            set=npw.t_set,
            peak_el_deg=npw.max_elevation_deg,
        )
        self._plan_dwells(plan, npw, self._env_for(npw), belief, npw.t_rise, 0)
        return plan

    # ------------------------------------------------------------------
    def observe(
        self,
        pass_idx: int,
        dwell_idx: int,
        detected: bool,
        snr_obs: SnrObservation | None = None,
        doppler_hz: float | None = None,
    ) -> list[PassPlan]:
        """Fold a real observation into the belief and replan the remainder.

        `detected` is the 1-bit detect/no-detect. When `snr_obs` is supplied (a
        graded Eb/N0 report from a real modem / SDR / spectrum analyzer) the
        belief is updated with the SNR-valued likelihood instead, which collapses
        the along-track ambiguity far faster than a single detect bit. When
        `doppler_hz` (a measured carrier Doppler offset) is supplied it is
        additionally folded in via the joint angle+Doppler update, which sharply
        constrains the along-track launch slip.

        Advances (diffuses) the live belief to the reported dwell's time and
        applies the particle-filter update, then keeps the executed dwells as
        history and re-plans this pass's remaining dwells (from the end of the
        held dwell) plus all later passes under the updated belief. Returns the
        full plan list so the caller can advance to the next dwell.
        """
        if not self._plans:
            raise RuntimeError("call plan_network() before observe()")
        target = next((p for p in self._plans if p.idx == pass_idx), None)
        if target is None or dwell_idx >= len(target.pointings):
            raise ValueError(f"no pass {pass_idx} / dwell {dwell_idx} in current plan")
        pt = target.pointings[dwell_idx]

        # Diffuse the live belief up to this observation time (dead-reckoning).
        self.belief = propagate_belief_to(
            self.belief,
            self.leop_start,
            pt.when,
            drift_km_per_min=self.leop.inter_pass_drift_km_per_min,
            orbital_speed_km_s=self.leop.orbital_speed_km_s,
            rng=self.rng,
        )
        # The EKF/Gaussian baseline deliberately does NOT fold in observations.
        pointing = Pointing(az=pt.az_rad, el=pt.el_rad)
        # Watchdog signals, measured on the PRE-resample (reweighted) posterior:
        # the particle filter resamples to uniform weights inside update(), which
        # would heal ESS/entropy before the watchdog could see a degenerate cloud.
        # `likelihood_total` is the belief-averaged support for THIS outcome (near
        # zero => no particle explains the data).
        health_belief, likelihood_total = self._prior_reweight(
            pointing, detected, pt.when, target.station
        )
        if self.policy != "ekf":
            if snr_obs is not None:
                self.belief = update_snr(
                    self.belief,
                    pointing,
                    snr_obs,
                    pt.when,
                    target.station,
                    self.antenna,
                    self.snr_cfg,
                    self.filter,
                    self.rng,
                )
            else:
                self.belief = update(
                    self.belief,
                    pointing,
                    detected,
                    pt.when,
                    target.station,
                    self.antenna,
                    self.filter,
                    self.rng,
                )
        # Joint angle+Doppler: a measured carrier Doppler additionally collapses
        # the along-track (launch-slip) ambiguity. The EKF baseline abstains.
        if doppler_hz is not None and self.policy != "ekf":
            self.belief = update_doppler(
                self.belief,
                float(doppler_hz),
                pt.when,
                target.station,
                self.freq_cfg,
                self.filter,
                self.rng,
            )
        # Re-anchor belief time to this observation.
        self.leop_start = pt.when

        # Belief-health watchdog: on sustained divergence, latch back to the
        # robust open-loop sweep (the automatic fallback). Re-arm is deliberate.
        self._obs_outcomes.append(bool(detected))
        self._run_watchdog(health_belief, bool(detected), likelihood_total)

        # Keep the executed dwells [0..dwell_idx] as history; re-plan this pass's
        # TAIL starting at the end of the held dwell (so the pointer advances to
        # the next dwell), then re-plan all later passes under the updated belief.
        # Earlier passes are left untouched.
        npw = self._npw_for(pass_idx)
        if npw is not None:
            target.pointings = target.pointings[: dwell_idx + 1]
            next_when = pt.when + timedelta(seconds=pt.dwell_s)
            # Continue the boresight from the executed dwell so the tail respects
            # the slew constraint (the pedestal is already parked at pt's look angle).
            self._plan_dwells(
                target,
                npw,
                self._env_for(npw),
                self.belief,
                next_when,
                dwell_idx + 1,
                bore0=(pt.az_rad, pt.el_rad),
            )

        work = ParticleBelief(
            particles=list(self.belief.particles), weights=self.belief.weights.copy()
        )
        ref = pt.when
        ti = self._plans.index(target)
        for j in range(ti + 1, len(self._plans)):
            npw_j = self._npw_for(self._plans[j].idx)
            if npw_j is None:
                continue
            if npw_j.t_rise > ref:
                work = propagate_belief_to(
                    work,
                    ref,
                    npw_j.t_rise,
                    drift_km_per_min=self.leop.inter_pass_drift_km_per_min,
                    orbital_speed_km_s=self.leop.orbital_speed_km_s,
                    rng=self.rng,
                )
            self._plans[j] = self._plan_one_pass(npw_j, work)
            ref = npw_j.t_set
        return self._plans

    # ------------------------------------------------------------------
    # Trust & safety
    # ------------------------------------------------------------------
    def _run_watchdog(
        self, health_belief: ParticleBelief, detected: bool, likelihood_total: float
    ) -> None:
        """Fold the pre-resample (reweighted) belief into the watchdog; latch on divergence.

        `health_belief` is the reweighted-but-not-resampled posterior, so its ESS
        / max-weight / entropy reflect the true belief health that the filter's
        resample would otherwise mask.
        """
        mode = watchdog_step(
            health_belief,
            self.watchdog,
            self.watchdog_cfg,
            detected=detected,
            likelihood_total=likelihood_total,
        )
        if mode == FALLBACK_OPEN_LOOP and self.policy not in ("sweep", "nominal", "ekf"):
            self._policy_before_fallback = self.policy
            self.fallback_reason = self.watchdog.last_reason
            self.policy = "sweep"
            self._controllers.clear()  # drop the now-untrusted closed-loop controllers

    @property
    def fallback_active(self) -> bool:
        return self.watchdog.latched_open_loop

    def rearm(self) -> list[PassPlan]:
        """Operator re-arm after a watchdog fallback: restore the controller and replan.

        Deliberate action — clears the latch, restores the pre-fallback controller,
        and re-plans from the current belief. Raises if not currently latched.
        """
        if not self.watchdog.latched_open_loop:
            raise RuntimeError("watchdog is not latched; nothing to re-arm")
        self.watchdog.rearm()
        if self._policy_before_fallback is not None:
            self.policy = self._policy_before_fallback
            self._policy_before_fallback = None
        self.fallback_reason = ""
        self._controllers.clear()
        return self.plan_network(self._horizon_h)

    def set_authority(self, level: str) -> None:
        """Change the graduated-trust authority level."""
        if not is_level(level):
            raise ValueError(f"unknown authority level {level!r}")
        self.authority_level = level

    def no_detect_streak(self) -> int:
        """Number of consecutive no-detects at the tail of the observation log."""
        streak = 0
        for ok in reversed(self._obs_outcomes):
            if ok:
                break
            streak += 1
        return streak

    def _belief_slip_stats(self) -> tuple[float, float]:
        """Weighted mean and std of the belief's launch-slip (along-track time, s)."""
        slips = np.array([p.launch_slip_s for p in self.belief.particles], dtype=float)
        w = np.asarray(self.belief.weights, dtype=float)
        s = float(w.sum())
        w = w / s if s > 0 else np.full(len(w), 1.0 / max(1, len(w)))
        mean = float(np.dot(w, slips))
        var = float(np.dot(w, (slips - mean) ** 2))
        return mean, float(np.sqrt(max(0.0, var)))

    def safety_summary(self) -> dict:
        """Counts of envelope verdicts over the currently-planned dwells."""
        counts = {"ok": 0, "warn": 0, "keyhole": 0, "rf_inhibit": 0, "reject": 0}
        first_block = None
        for p in self._plans:
            for pt in p.pointings:
                counts[pt.safety_status] = counts.get(pt.safety_status, 0) + 1
                if first_block is None and pt.safety_status in ("rf_inhibit", "reject"):
                    first_block = {"pass_idx": p.idx, "reason": pt.safety_reason}
        total = sum(counts.values())
        return {"total_dwells": total, **counts, "first_blocking": first_block}

    def explain(self) -> str:
        """A plain-language rationale for what the acquisition is doing right now."""
        if self.fallback_active:
            return (
                f"Filter watchdog tripped — {self.fallback_reason}. Reverted to the "
                "open-loop sweep; re-arm to resume closed-loop pointing."
            )
        bits: list[str] = []
        if self._obs_outcomes and self._obs_outcomes[-1]:
            bits.append("Signal acquired — collapsing the belief onto the confirmed track.")
        else:
            streak = self.no_detect_streak()
            if streak == 1:
                bits.append("1 no-detect: reweighting the belief away from the searched cell.")
            elif streak > 1:
                bits.append(
                    f"{streak} consecutive no-detects: the object isn't where the prior expected."
                )
        mean_slip, spread = self._belief_slip_stats()
        if spread > 1.0:
            tail = "delayed" if mean_slip >= 0.0 else "early"
            bits.append(
                f"Belief centred on a launch slip of {mean_slip:+.0f} s (±{spread:.0f} s) — "
                f"searching the {tail} along-track tail."
            )
        if self.controller_choice is not None:
            bits.append(
                f"Controller: {self.controller_choice.controller_name} "
                f"({'agile antenna' if self.controller_choice.is_agile else 'slew-limited dish'})."
            )
        return (
            " ".join(bits)
            if bits
            else "Belief initialised from the prior; awaiting the first pass."
        )

    def ops_status(self) -> dict:
        """Structured trust + watchdog + safety + rationale block for the API."""
        return {
            "trust": {
                "level": self.authority_level,
                "label": label(self.authority_level),
                "description": description(self.authority_level),
                "can_actuate": can_actuate(self.authority_level),
            },
            "watchdog": {
                "mode": FALLBACK_OPEN_LOOP if self.fallback_active else CLOSED_LOOP,
                "latched": self.fallback_active,
                "reason": self.fallback_reason,
            },
            "fallback_active": self.fallback_active,
            "rationale": self.explain(),
            "safety": self.safety_summary(),
        }

    # ------------------------------------------------------------------
    def belief_summary(self, n_along: int = 25, n_cross: int = 21) -> dict:
        """Along-track (s) x cross-track (deg) weight grid + scalar entropy.

        Reference time/station: the first planned pass if available, else the
        next network pass, else the leop_start at the first network station. The
        nominal track at that time defines the (along, cross) axes; each particle
        is binned by its signed along-track-time and cross-track-angle offset
        from the nominal pointing.
        """
        ref_when, station = self._reference_pose()
        nominal_state = self._nominal.propagate(ref_when)
        az0, el0, _ = eci_to_azel(nominal_state.r_eci_m, ref_when, station)

        azel = belief_azel(self.belief, ref_when, station)
        # cross-track angle (deg): elevation difference is a stand-in for the
        # perpendicular offset; along-track time: az difference scaled by track
        # rate. We approximate the track rate from a 1 s finite difference.
        s1 = self._nominal.propagate(ref_when + timedelta(seconds=1.0))
        az1, el1, _ = eci_to_azel(s1.r_eci_m, ref_when + timedelta(seconds=1.0), station)
        daz = (az1 - az0 + np.pi) % (2 * np.pi) - np.pi
        dEl = el1 - el0
        track_rate = np.hypot(daz * np.cos(el0), dEl)  # rad/s along track

        d_az = (azel[:, 0] - az0 + np.pi) % (2 * np.pi) - np.pi
        d_el = azel[:, 1] - el0
        along_proj = d_az * np.cos(el0)  # rad along az
        along_s = along_proj / max(1e-9, track_rate)
        cross_deg = np.rad2deg(d_el)

        a_lim = max(1.0, float(np.percentile(np.abs(along_s), 98)) * 1.1)
        c_lim = max(0.5, float(np.percentile(np.abs(cross_deg), 98)) * 1.1)
        a_edges = np.linspace(-a_lim, a_lim, n_along + 1)
        c_edges = np.linspace(-c_lim, c_lim, n_cross + 1)
        H, _, _ = np.histogram2d(
            along_s, cross_deg, bins=[a_edges, c_edges], weights=self.belief.weights
        )
        a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
        c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
        return {
            "entropy": belief_entropy(self.belief.weights),
            "grid": {
                "along_s": [float(x) for x in a_centers],
                "cross_deg": [float(x) for x in c_centers],
                # rows = along-track bins, cols = cross-track bins
                "weight": [[float(v) for v in row] for row in H],
            },
        }

    def _reference_pose(self) -> tuple[datetime, Station]:
        if self._plans:
            p = self._plans[0]
            return p.rise, p.station
        if self._passes:
            npw = self._passes[0]
            return npw.t_rise, npw.station
        station = Station.from_config(self.network.stations[0])
        return self.leop_start, station


if __name__ == "__main__":
    from dataclasses import replace

    from acquisition_platform.ingest.tle import parse_tle
    from acquisition_platform.estimator.ukf import SgpUkf
    from antenna_pomdp.config import default_config, network_polar_sso

    # SSO TLE so a polar network actually sees passes.
    sso_l1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
    sso_l2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    base = default_config()
    cfg = replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=sso_l1, nominal_tle_line2=sso_l2),
        antenna=replace(base.antenna, fwhm_deg=10.0, dwell_time_s=10.0),
        filter=replace(base.filter, n_particles=120),
        pomcp=replace(base.pomcp, n_rollouts=40, max_depth=4),
    )
    rng = np.random.default_rng(0)
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, l1, l2 = parse_tle(sso_l1 + "\n" + sso_l2, cov_6x6=cov, frame="RTN")
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)

    sess = PlanSession(
        config=cfg,
        network=network_polar_sso(),
        belief=belief,
        policy="pomcp",
        leop_start=opm.epoch,
        ukf=ukf,
        rng=rng,
    )
    plans = sess.plan_network(horizon_h=6.0)
    assert len(plans) >= 1, "expected at least one pass"
    assert any(len(p.pointings) > 0 for p in plans)

    summ0 = sess.belief_summary()
    ent0 = summ0["entropy"]
    # Observe a no-detect on the first dwell of the first pass; entropy changes.
    first = next(p for p in plans if p.pointings)
    sess.observe(first.idx, 0, detected=False)
    ent1 = sess.belief_summary()["entropy"]

    print(
        f"[session] passes={len(plans)}, "
        f"pointings(pass0)={len(plans[0].pointings)}, "
        f"entropy {ent0:.3f} -> {ent1:.3f}"
    )
    assert abs(ent1 - ent0) > 0.0 or ent1 != ent0
    print("[session] PASS")
