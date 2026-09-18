"""Particle filter over orbits, with detect / no-detect updates.

Particles are `OrbitParticle` objects (built in `orbit.sampler`). Each carries
its own SGP4 propagator and a launch-slip offset. Resampling uses systematic
resampling; small along-track jitter is injected on resample so identical
particles diverge over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from antenna_pomdp.config import AntennaConfig, FilterConfig

if TYPE_CHECKING:
    from antenna_pomdp.models.snr_observation import SnrConfig, SnrObservation
from antenna_pomdp.models.observation import (
    Pointing,
    delta_theta_from_azel,
    observation_likelihood,
)
from antenna_pomdp.orbit.geometry import Station, eci_to_azel
from antenna_pomdp.orbit.sampler import OrbitParticle


# ---------------------------------------------------------------------------
# Belief container
# ---------------------------------------------------------------------------


@dataclass
class ParticleBelief:
    """Weighted particle representation of P(orbit | history)."""

    particles: list[OrbitParticle]
    weights: np.ndarray  # shape (N,)

    @property
    def n(self) -> int:
        return len(self.particles)

    def effective_sample_size(self) -> float:
        w = self.weights
        s = float(np.sum(w))
        if s <= 0.0:
            return 0.0
        wn = w / s
        return float(1.0 / np.sum(wn**2))


def make_belief(particles: list[OrbitParticle]) -> ParticleBelief:
    """Build a uniform-weight belief from a list of particles."""
    n = len(particles)
    if n == 0:
        raise ValueError("cannot build a belief from zero particles")
    return ParticleBelief(particles=list(particles), weights=np.full(n, 1.0 / n))


# ---------------------------------------------------------------------------
# Particle az/el computation
# ---------------------------------------------------------------------------


def particle_azel(particle: OrbitParticle, when: datetime, station: Station) -> tuple[float, float]:
    """az/el of one particle at `when`, accounting for its launch-slip offset."""
    az, el, _ = particle_azel_range(particle, when, station)
    return az, el


def particle_azel_range(
    particle: OrbitParticle, when: datetime, station: Station
) -> tuple[float, float, float]:
    """az/el/range of one particle at `when`, accounting for its launch-slip offset."""
    effective = when - timedelta(seconds=particle.launch_slip_s)
    state = particle.propagator.propagate(effective)
    return eci_to_azel(state.r_eci_m, when, station)


def belief_azel(belief: ParticleBelief, when: datetime, station: Station) -> np.ndarray:
    """Stacked (N, 2) array of (az, el) for each particle."""
    return np.array([particle_azel(p, when, station) for p in belief.particles])


def belief_azel_range(belief: ParticleBelief, when: datetime, station: Station) -> np.ndarray:
    """Stacked (N, 3) array of (az, el, range_m) for each particle."""
    return np.array([particle_azel_range(p, when, station) for p in belief.particles])


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------


def update(
    belief: ParticleBelief,
    pointing: Pointing,
    detected: bool,
    when: datetime,
    station: Station,
    antenna_cfg: AntennaConfig,
    filter_cfg: FilterConfig,
    rng: np.random.Generator,
) -> ParticleBelief:
    """Reweight, optionally resample. Returns a *new* belief (no mutation)."""
    azelr = belief_azel_range(belief, when, station)
    dthetas = np.array(
        [delta_theta_from_azel(pointing, azelr[i, 0], azelr[i, 1]) for i in range(belief.n)]
    )
    likelihoods = np.array(
        [
            float(
                observation_likelihood(detected, dthetas[i], antenna_cfg, azelr[i, 1], azelr[i, 2])
            )
            for i in range(belief.n)
        ]
    )
    new_w = belief.weights * likelihoods
    total = float(new_w.sum())
    if total <= 0.0:
        # Degenerate: all particles have zero likelihood. Fall back to uniform
        # over the prior particles to retain support.
        new_w = np.full(belief.n, 1.0 / belief.n)
    else:
        new_w = new_w / total

    candidate = ParticleBelief(particles=list(belief.particles), weights=new_w)
    if candidate.effective_sample_size() < filter_cfg.ess_resample_threshold * candidate.n:
        candidate = _resample(candidate, filter_cfg, rng)
    return candidate


def update_snr(
    belief: ParticleBelief,
    pointing: Pointing,
    obs: SnrObservation,
    when: datetime,
    station: Station,
    antenna_cfg: AntennaConfig,
    snr_cfg: SnrConfig,
    filter_cfg: FilterConfig,
    rng: np.random.Generator,
) -> ParticleBelief:
    """Fold a graded Eb/N0 (SNR-valued) demodulator report into the belief.

    Same reweight-then-maybe-resample shape as `update`, but the likelihood is
    the continuous Eb/N0 model (`models.snr_observation`) rather than the 1-bit
    detect model, so a single locked reading can localise the along-track offset
    that a detect/no-detect update would need many dwells to resolve. Reweighting
    is done in log-space with a max-subtraction for numerical stability.
    """
    from antenna_pomdp.models.snr_observation import snr_log_likelihood

    azelr = belief_azel_range(belief, when, station)
    dthetas = np.array(
        [delta_theta_from_azel(pointing, azelr[i, 0], azelr[i, 1]) for i in range(belief.n)]
    )
    log_like = snr_log_likelihood(obs, dthetas, antenna_cfg, azelr[:, 1], azelr[:, 2], snr_cfg)
    with np.errstate(divide="ignore"):
        log_w = np.log(np.clip(belief.weights, 1e-300, None)) + log_like
    log_w -= float(np.max(log_w))  # stabilise before exponentiating
    new_w = np.exp(log_w)
    total = float(new_w.sum())
    if total <= 0.0 or not np.isfinite(total):
        new_w = np.full(belief.n, 1.0 / belief.n)
    else:
        new_w = new_w / total

    candidate = ParticleBelief(particles=list(belief.particles), weights=new_w)
    if candidate.effective_sample_size() < filter_cfg.ess_resample_threshold * candidate.n:
        candidate = _resample(candidate, filter_cfg, rng)
    return candidate


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _systematic_indices(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # guard against rounding
    return np.clip(np.searchsorted(cumsum, positions), 0, n - 1)


def _resample(
    belief: ParticleBelief, filter_cfg: FilterConfig, rng: np.random.Generator
) -> ParticleBelief:
    idx = _systematic_indices(belief.weights, rng)
    new_particles: list[OrbitParticle] = []
    for j in idx:
        src = belief.particles[j]
        jitter = rng.normal(0.0, filter_cfg.process_noise_along_track_s)
        new_particles.append(
            OrbitParticle(
                propagator=src.propagator,
                launch_slip_s=src.launch_slip_s + jitter,
                weight=1.0,
            )
        )
    return ParticleBelief(particles=new_particles, weights=np.full(belief.n, 1.0 / belief.n))


# ---------------------------------------------------------------------------
# Inter-pass propagation (additive process noise during dead-reckoning)
# ---------------------------------------------------------------------------


def propagate_belief_to(
    belief: ParticleBelief,
    current_time: datetime,
    target_time: datetime,
    *,
    drift_km_per_min: float,
    orbital_speed_km_s: float,
    rng: np.random.Generator,
) -> ParticleBelief:
    """Advance the belief to `target_time` while injecting process noise.

    The underlying SGP4 propagator is stateless wrt wall-clock time (each
    particle already extrapolates from its own TLE epoch). What changes
    between passes is *uncertainty in the model*: SGP4 drifts away from
    truth at roughly `drift_km_per_min` along-track. We convert that
    along-track position-error growth into an equivalent launch-slip
    standard deviation via the orbital speed and add it (in quadrature)
    to each particle's existing slip.

    Particle weights are left unchanged — this represents pure prediction
    with no measurement information.
    """
    dt_s = (target_time - current_time).total_seconds()
    if dt_s <= 0.0:
        return belief
    dt_min = dt_s / 60.0
    sigma_drift_km = drift_km_per_min * dt_min
    sigma_slip_s = sigma_drift_km / max(1e-6, orbital_speed_km_s)
    new_particles: list[OrbitParticle] = []
    for p in belief.particles:
        jitter = float(rng.normal(0.0, sigma_slip_s))
        new_particles.append(
            OrbitParticle(
                propagator=p.propagator,
                launch_slip_s=p.launch_slip_s + jitter,
                weight=p.weight,
            )
        )
    return ParticleBelief(particles=new_particles, weights=belief.weights.copy())


if __name__ == "__main__":
    from antenna_pomdp.config import default_config
    from antenna_pomdp.orbit.geometry import Station
    from antenna_pomdp.orbit.sampler import sample_particles

    cfg = default_config()
    rng = np.random.default_rng(0)
    station = Station.from_config(cfg.station)
    particles = sample_particles(cfg.orbit, n=200, rng=rng)
    belief = make_belief(particles)
    when = particles[0].propagator.epoch + timedelta(minutes=5)
    azel = belief_azel(belief, when, station)
    mean_az = float(np.mean(azel[:, 0]))
    mean_el = float(np.mean(azel[:, 1]))
    pointing = Pointing(az=mean_az, el=mean_el)

    # Apply a "no-detect" update; ESS should drop / particles refocus.
    ess_before = belief.effective_sample_size()
    new_belief = update(
        belief,
        pointing,
        detected=False,
        when=when,
        station=station,
        antenna_cfg=cfg.antenna,
        filter_cfg=cfg.filter,
        rng=rng,
    )
    ess_after = new_belief.effective_sample_size()
    print(f"[pf] ESS before={ess_before:.1f}, after no-detect update={ess_after:.1f}")
    assert ess_after > 0.0
    print("[pf] PASS")
