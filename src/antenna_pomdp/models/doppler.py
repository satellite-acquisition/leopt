"""Doppler predictions and frequency likelihoods for orbit-particle beliefs.

The received carrier offset is f_d = -(v_radial / c) * f_carrier. Range rate
uses the satellite's SGP4 state and the station velocity from Earth rotation.

Oscillator and receiver uncertainties enter the frequency-likelihood width,
along with Doppler-rate smearing over the integration window. A constant
oscillator bias is not estimated as a separate state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from antenna_pomdp.config import FilterConfig
from antenna_pomdp.models.particle_filter import ParticleBelief, _resample
from antenna_pomdp.orbit.geometry import Station, rot_eci_to_ecef
from antenna_pomdp.orbit.propagator import OrbitState
from antenna_pomdp.orbit.sampler import OrbitParticle

OMEGA_EARTH_RAD_S = 7.2921159e-5  # Earth rotation rate (rad/s)
C_LIGHT_M_S = 299_792_458.0

_erf = np.vectorize(math.erf)  # numpy has no erf; vectorize the stdlib scalar one


@dataclass(frozen=True)
class FrequencyConfig:
    """Carrier + oscillator + receiver frequency-search parameters."""

    carrier_hz: float = 2.25e9  # S-band TT&C beacon
    osc_stability_ppm: float = 0.1  # reference oscillator (OCXO ~0.1, TCXO ~2)
    meas_sigma_hz: float = 50.0  # receiver frequency-estimate 1-sigma
    integration_time_s: float = 0.01  # coherent dwell -> FFT bin ~ 1/T

    def osc_sigma_hz(self) -> float:
        return self.osc_stability_ppm * 1e-6 * self.carrier_hz

    def bin_width_hz(self) -> float:
        return 1.0 / max(1e-9, self.integration_time_s)


# ---------------------------------------------------------------------------
# Predicted Doppler per particle
# ---------------------------------------------------------------------------


def range_rate_m_s(state: OrbitState, when: datetime, station: Station) -> float:
    """Line-of-sight range-rate (m/s, receding positive) sat<->station in ECI."""
    rot = rot_eci_to_ecef(when)
    r_stn = rot.T @ station.ecef_m  # station ECI position
    v_stn = np.cross(np.array([0.0, 0.0, OMEGA_EARTH_RAD_S]), r_stn)  # Earth rotation
    los = state.r_eci_m - r_stn
    rng = float(np.linalg.norm(los))
    if rng < 1.0:
        return 0.0
    return float(np.dot(state.v_eci_m_s - v_stn, los / rng))


def particle_doppler_hz(
    particle: OrbitParticle, when: datetime, station: Station, cfg: FrequencyConfig
) -> float:
    """Predicted received-carrier Doppler offset (Hz) for one particle."""
    effective = when - timedelta(seconds=particle.launch_slip_s)
    state = particle.propagator.propagate(effective)
    v_radial = range_rate_m_s(state, when, station)
    return -v_radial / C_LIGHT_M_S * cfg.carrier_hz


def belief_doppler_hz(
    belief: ParticleBelief, when: datetime, station: Station, cfg: FrequencyConfig
) -> np.ndarray:
    """Predicted Doppler (Hz) for every particle at `when`, shape (N,)."""
    return np.array([particle_doppler_hz(p, when, station, cfg) for p in belief.particles])


def effective_sigma_hz(cfg: FrequencyConfig, doppler_rate_hz_s: float = 0.0) -> float:
    """Frequency-likelihood width: receiver + oscillator + Doppler-smear terms."""
    smear = abs(doppler_rate_hz_s) * cfg.integration_time_s
    return float(
        math.sqrt(cfg.meas_sigma_hz**2 + cfg.osc_sigma_hz() ** 2 + (smear / math.sqrt(12.0)) ** 2)
    )


# ---------------------------------------------------------------------------
# Likelihoods + updates
# ---------------------------------------------------------------------------


def freq_bin_detect_likelihood(
    predicted_hz: np.ndarray, bin_lo_hz: float, bin_hi_hz: float, sigma_hz: float
) -> np.ndarray:
    """P(a particle's predicted tone falls inside the scanned bin [lo, hi])."""
    s = math.sqrt(2.0) * sigma_hz
    return 0.5 * (_erf((bin_hi_hz - predicted_hz) / s) - _erf((bin_lo_hz - predicted_hz) / s))


def update_doppler(
    belief: ParticleBelief,
    measured_hz: float,
    when: datetime,
    station: Station,
    cfg: FrequencyConfig,
    filter_cfg: FilterConfig,
    rng: np.random.Generator,
    sigma_hz: float | None = None,
) -> ParticleBelief:
    """Reweight the belief by a graded Doppler measurement; resample on ESS drop."""
    sigma = sigma_hz if sigma_hz is not None else effective_sigma_hz(cfg)
    pred = belief_doppler_hz(belief, when, station, cfg)
    logl = -0.5 * ((pred - measured_hz) / sigma) ** 2
    logw = np.log(np.clip(np.asarray(belief.weights, dtype=float), 1e-300, None)) + logl
    logw -= logw.max()
    w = np.exp(logw)
    total = float(w.sum())
    w = w / total if total > 0 else np.full(belief.n, 1.0 / belief.n)
    new = ParticleBelief(particles=list(belief.particles), weights=w)
    if new.effective_sample_size() < filter_cfg.ess_resample_threshold * new.n:
        new = _resample(new, filter_cfg, rng)
    return new


def update_bin_detect(
    belief: ParticleBelief,
    bin_lo_hz: float,
    bin_hi_hz: float,
    detected: bool,
    when: datetime,
    station: Station,
    cfg: FrequencyConfig,
    filter_cfg: FilterConfig,
    rng: np.random.Generator,
    sigma_hz: float | None = None,
) -> ParticleBelief:
    """Update on a detect / no-detect in a scanned frequency bin (angle held)."""
    sigma = sigma_hz if sigma_hz is not None else effective_sigma_hz(cfg)
    pred = belief_doppler_hz(belief, when, station, cfg)
    p_in = np.clip(freq_bin_detect_likelihood(pred, bin_lo_hz, bin_hi_hz, sigma), 1e-9, 1.0)
    like = p_in if detected else (1.0 - p_in)
    w = np.asarray(belief.weights, dtype=float) * like
    total = float(w.sum())
    w = w / total if total > 0 else np.full(belief.n, 1.0 / belief.n)
    new = ParticleBelief(particles=list(belief.particles), weights=w)
    if new.effective_sample_size() < filter_cfg.ess_resample_threshold * new.n:
        new = _resample(new, filter_cfg, rng)
    return new


if __name__ == "__main__":
    from antenna_pomdp.config import StationConfig, default_config
    from antenna_pomdp.orbit.propagator import SGP4Propagator

    cfg = default_config()
    fcfg = FrequencyConfig()
    fcfg_cfg = FilterConfig()
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(StationConfig())
    when = nominal.epoch + timedelta(minutes=5)

    # Two particles differing ONLY in along-track launch slip.
    p0 = OrbitParticle(nominal, 0.0)
    p30 = OrbitParticle(nominal, 30.0)
    d0 = particle_doppler_hz(p0, when, station, fcfg)
    d30 = particle_doppler_hz(p30, when, station, fcfg)
    assert abs(d0) < 2.0e5 and abs(d30) < 2.0e5, (d0, d30)  # sane S-band LEO Doppler
    assert abs(d0 - d30) > 100.0, "expected a kHz-scale Doppler split across 30 s of slip"

    # A Doppler measurement matching p0 collapses the belief toward p0.
    belief = ParticleBelief(particles=[p0, p30], weights=np.array([0.5, 0.5]))
    rng = np.random.default_rng(0)
    updated = update_doppler(belief, d0, when, station, fcfg, fcfg_cfg, rng)
    assert updated.weights[0] > updated.weights[1], updated.weights

    # Detect-in-bin: a bin centered on p0's tone favours p0 over p30.
    sig = effective_sigma_hz(fcfg)
    liks = freq_bin_detect_likelihood(np.array([d0, d30]), d0 - sig, d0 + sig, sig)
    assert liks[0] > liks[1]

    print(f"[doppler] slip 0s -> {d0 / 1e3:+.2f} kHz, 30s -> {d30 / 1e3:+.2f} kHz")
    print(f"[doppler] posterior weights after freq obs: {updated.weights.round(3)}")
    print("[doppler] PASS")
