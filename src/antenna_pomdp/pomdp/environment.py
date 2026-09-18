"""POMDP environment for antenna pointing during a single pass.

State    : true orbit (represented by an `OrbitParticle`) + current pass time.
Action   : discrete (along-track offset, cross-track offset) pair, drawn from a
           pre-computed grid relative to the nominal-track az/el.
Obs      : Bernoulli detect / no-detect.
Reward   : `reward_detect` on detection, `reward_step` per step otherwise.
Episode  : ends after pass duration / dwell_time steps or upon first detection.

The environment is intentionally a small data class with pure functions so
POMCP can call `transition` and `step` from inside its rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from antenna_pomdp.config import (
    AntennaConfig,
    PomcpConfig,
)
from antenna_pomdp.models.observation import (
    Pointing,
    delta_theta_from_azel,
    sample_observation,
)
from antenna_pomdp.orbit.geometry import (
    Station,
    angular_separation,
    azel_unit_vector,
    eci_to_azel,
)
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle


def slew_limited_pointing(
    bore_az: float,
    bore_el: float,
    target: Pointing,
    max_move_rad: float,
) -> Pointing:
    """Move the boresight from (bore_az, bore_el) toward `target`, capped at
    `max_move_rad` of great-circle travel (one dwell of slewing).

    If the target is within reach the boresight arrives; otherwise it advances
    along the great circle toward the target by exactly `max_move_rad`. Returns
    the realised boresight as a `Pointing`.
    """
    sep = angular_separation(bore_az, bore_el, target.az, target.el)
    if sep <= max_move_rad or sep < 1e-12:
        return target
    # Spherical-linear interpolation between the two ENU unit vectors.
    v0 = azel_unit_vector(bore_az, bore_el)
    v1 = azel_unit_vector(target.az, target.el)
    frac = max_move_rad / sep
    sin_sep = np.sin(sep)
    w0 = np.sin((1.0 - frac) * sep) / sin_sep
    w1 = np.sin(frac * sep) / sin_sep
    v = w0 * v0 + w1 * v1
    v = v / np.linalg.norm(v)
    el = float(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    az = float(np.arctan2(v[0], v[1])) % (2.0 * np.pi)
    return Pointing(az=az, el=el)


# ---------------------------------------------------------------------------
# Action grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackAction:
    """An offset from the nominal along-track / cross-track position.

    `along_track_s` is a time offset along the nominal trajectory; positive =
    look ahead. `cross_track_deg` is an az/el angular offset perpendicular to
    the instantaneous pass direction in the topocentric frame.
    """

    along_track_s: float
    cross_track_deg: float


def build_action_grid(cfg: PomcpConfig) -> list[TrackAction]:
    """Symmetric ± grid in along-track time × cross-track angle."""
    n_T = max(1, cfg.n_along_track_offsets)
    n_K = max(1, cfg.n_cross_track_levels)
    Ts = np.linspace(-cfg.along_track_offset_max_s, cfg.along_track_offset_max_s, n_T)
    Ks = np.linspace(-cfg.cross_track_offset_max_deg, cfg.cross_track_offset_max_deg, n_K)
    return [TrackAction(float(t), float(k)) for t in Ts for k in Ks]


# ---------------------------------------------------------------------------
# Pointing from action
# ---------------------------------------------------------------------------


def action_to_pointing(
    action: TrackAction,
    nominal: SGP4Propagator,
    when: datetime,
    station: Station,
) -> Pointing:
    """Convert a (along-track, cross-track) action into a (az, el) command.

    The along-track coordinate is a time-of-validity (TOV) offset: the nominal
    spacecraft state is propagated to ``when + tau`` and observed in the
    station frame at command epoch ``when``. This deliberately does not rotate
    the station epoch by ``tau``. Cross-track is applied perpendicular to that
    fixed-observer-epoch TOV curve, using a one-second finite difference in
    ``tau``.
    """
    target_time = when + timedelta(seconds=action.along_track_s)
    s0 = nominal.propagate(target_time)
    az0, el0, _ = eci_to_azel(s0.r_eci_m, when, station)

    s1 = nominal.propagate(target_time + timedelta(seconds=1.0))
    az1, el1, _ = eci_to_azel(s1.r_eci_m, when, station)

    # Tangent direction in (az, el) plane (correct az for wraparound).
    daz = (az1 - az0 + np.pi) % (2.0 * np.pi) - np.pi
    dEl = el1 - el0
    tangent = np.array([daz * np.cos(el0), dEl])
    norm = np.linalg.norm(tangent)
    if norm < 1e-9:
        perp = np.array([1.0, 0.0])
    else:
        tangent = tangent / norm
        perp = np.array([-tangent[1], tangent[0]])

    offset_rad = np.deg2rad(action.cross_track_deg)
    new_az = az0 + offset_rad * perp[0] / max(1e-6, np.cos(el0))
    new_el = el0 + offset_rad * perp[1]
    new_el = float(np.clip(new_el, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3))
    new_az = float(new_az % (2.0 * np.pi))
    return Pointing(az=new_az, el=new_el)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class EnvState:
    """Mutable state of a single pass simulation."""

    when: datetime
    step_idx: int
    done: bool
    acquired: bool
    # Current realised boresight (az, el) in radians; None before the first
    # dwell (the antenna is assumed pre-positioned, so the first command is
    # reached freely). Tracks the mechanical slew state across dwells.
    bore_az: float | None = None
    bore_el: float | None = None


@dataclass
class PointingEnv:
    """One-pass POMDP environment.

    `nominal_propagator` is used both to convert actions to az/el and as the
    reference for the search baselines. `truth` is the realised orbit used
    inside `step` to generate observations and rewards.
    """

    nominal: SGP4Propagator
    truth: OrbitParticle
    station: Station
    pass_start: datetime
    pass_duration_s: float
    dwell_time_s: float
    antenna_cfg: AntennaConfig
    pomcp_cfg: PomcpConfig
    # Live realised boresight (az, el) of the current dwell, published by the
    # evaluation loop before each policy call so that slew-aware controllers can
    # restrict themselves to reachable pointings. None => no constraint / first
    # dwell. Not part of the immutable scenario; mutated per step by the runner.
    current_bore: tuple[float, float] | None = None

    @property
    def horizon(self) -> int:
        return max(1, int(self.pass_duration_s // self.dwell_time_s))

    def initial_state(self) -> EnvState:
        return EnvState(when=self.pass_start, step_idx=0, done=False, acquired=False)

    def truth_azel(self, when: datetime) -> tuple[float, float]:
        az, el, _ = self.truth_azel_range(when)
        return az, el

    def truth_azel_range(self, when: datetime) -> tuple[float, float, float]:
        effective = when - timedelta(seconds=self.truth.launch_slip_s)
        state = self.truth.propagator.propagate(effective)
        return eci_to_azel(state.r_eci_m, when, self.station)

    def step(
        self,
        state: EnvState,
        action: TrackAction,
        rng: np.random.Generator,
    ) -> tuple[EnvState, bool, float, Pointing]:
        """Apply one dwell-time step. Returns (next, detected, reward, pointing)."""
        if state.done:
            return state, False, 0.0, Pointing(0.0, 0.0)
        commanded = action_to_pointing(action, self.nominal, state.when, self.station)
        # Apply the mechanical slew limit: the boresight can only travel so far
        # in one dwell. The *realised* pointing is what observes the sky and what
        # the belief must be updated with.
        pointing = self._realize_pointing(state, commanded)
        truth_az, truth_el, truth_range = self.truth_azel_range(state.when)
        d_theta = delta_theta_from_azel(pointing, truth_az, truth_el)
        detected = sample_observation(d_theta, self.antenna_cfg, rng, truth_el, truth_range)

        reward = self.pomcp_cfg.reward_detect if detected else self.pomcp_cfg.reward_step
        next_state = EnvState(
            when=state.when + timedelta(seconds=self.dwell_time_s),
            step_idx=state.step_idx + 1,
            done=detected or (state.step_idx + 1 >= self.horizon),
            acquired=state.acquired or detected,
            bore_az=pointing.az,
            bore_el=pointing.el,
        )
        return next_state, detected, reward, pointing

    def max_slew_rad(self) -> float:
        """Maximum great-circle boresight travel allowed in one dwell (rad)."""
        return np.deg2rad(self.antenna_cfg.max_slew_rate_deg_s) * self.dwell_time_s

    def _realize_pointing(self, state: EnvState, commanded: Pointing) -> Pointing:
        """Clamp a commanded pointing to the slew-reachable set from `state`."""
        if state.bore_az is None or state.bore_el is None:
            return commanded
        return slew_limited_pointing(state.bore_az, state.bore_el, commanded, self.max_slew_rad())


def simulate_observation_for_particle(
    particle: OrbitParticle,
    when: datetime,
    pointing: Pointing,
    station: Station,
    antenna_cfg: AntennaConfig,
    rng: np.random.Generator,
) -> bool:
    """Sample a detect / no-detect outcome under a *hypothetical* truth particle.

    POMCP rollouts use this to generate observations from the belief without
    touching the real environment.
    """
    effective = when - timedelta(seconds=particle.launch_slip_s)
    state = particle.propagator.propagate(effective)
    az, el, rng_m = eci_to_azel(state.r_eci_m, when, station)
    d_theta = delta_theta_from_azel(pointing, az, el)
    return sample_observation(d_theta, antenna_cfg, rng, el, rng_m)


if __name__ == "__main__":
    from antenna_pomdp.config import default_config
    from antenna_pomdp.orbit.sampler import sample_truth

    cfg = default_config()
    rng = np.random.default_rng(0)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    truth = sample_truth(cfg.orbit, rng)
    actions = build_action_grid(cfg.pomcp)
    expected = max(1, cfg.pomcp.n_along_track_offsets) * max(1, cfg.pomcp.n_cross_track_levels)
    assert len(actions) == expected, (len(actions), expected)

    env = PointingEnv(
        nominal=nominal,
        truth=truth,
        station=station,
        pass_start=nominal.epoch + timedelta(minutes=5),
        pass_duration_s=cfg.station.pass_duration_s,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )

    s = env.initial_state()
    n_detect = 0
    for _ in range(env.horizon):
        # Always point at the nominal track (zero offset).
        s, det, _, _ = env.step(s, TrackAction(0.0, 0.0), rng)
        n_detect += int(det)
        if s.done:
            break

    print(
        f"[env] grid size = {len(actions)}, horizon = {env.horizon}, "
        f"acquired = {s.acquired}, detections = {n_detect}"
    )
    print("[env] PASS")
