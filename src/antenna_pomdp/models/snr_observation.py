"""Carrier-lock and Eb/N0 likelihoods for orbit-particle updates.

The predicted Eb/N0 uses the range, atmospheric loss, and Gaussian beam model
from observation.py:

    Eb/N0 = ref_ebn0_db - 20 log10(range / ref_range)
            - zenith_atmos_loss_db (1/sin(el) - 1)
            - (10/ln(10)) (delta_theta / sigma)^2 / 2.

A locked measurement has a Gaussian likelihood with standard deviation
sigma_meas_db. An unlocked measurement uses the probability that Eb/N0 falls
below lock_threshold_db. models.particle_filter.update_snr applies these
likelihoods to the particle weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from antenna_pomdp.config import AntennaConfig, beam_sigma_rad

# Elevation floor for the 1/sin(el) airmass term (mirrors observation.py).
_MIN_LINK_ELEVATION_RAD = np.deg2rad(1.0)
# 10 / ln(10): converts a natural-log power ratio to decibels.
_DB_PER_NAT = 10.0 / np.log(10.0)


@dataclass(frozen=True)
class SnrConfig:
    """Parameters of the Eb/N0 measurement model (dB units).

    Defaults are S-band-beacon realistic: ~12 dB Eb/N0 on-axis at the reference
    range and zenith, a 1 dB modem estimation noise, and lock held down to ~2 dB.
    The free-space / atmospheric terms are read from the `AntennaConfig` so the
    SNR model and the detect model share one link geometry.
    """

    ref_ebn0_db: float = 12.0  # on-axis Eb/N0 at ref_range_m & zenith
    sigma_meas_db: float = 1.0  # modem Eb/N0 estimation 1-sigma
    lock_threshold_db: float = 2.0  # Eb/N0 below which carrier lock is lost


@dataclass(frozen=True)
class SnrObservation:
    """One demodulator report: a lock flag and (if locked) an Eb/N0 estimate."""

    locked: bool
    ebn0_db: float | None = None


def _norm_cdf(z: np.ndarray | float) -> np.ndarray | float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + erf(np.asarray(z, dtype=float) / np.sqrt(2.0)))


def predicted_ebn0_db(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray,
    range_m: float | np.ndarray,
    snr_cfg: SnrConfig,
) -> np.ndarray:
    """Predicted Eb/N0 (dB) for the geometry of one or many particles."""
    sigma = beam_sigma_rad(antenna_cfg)
    el = np.maximum(np.asarray(elevation_rad, dtype=float), _MIN_LINK_ELEVATION_RAD)
    fspl_excess_db = 20.0 * np.log10(np.asarray(range_m, dtype=float) / antenna_cfg.ref_range_m)
    atmos_excess_db = antenna_cfg.zenith_atmos_loss_db * (1.0 / np.sin(el) - 1.0)
    beam_rolloff_db = _DB_PER_NAT * 0.5 * (np.asarray(delta_theta_rad, dtype=float) / sigma) ** 2
    return snr_cfg.ref_ebn0_db - fspl_excess_db - atmos_excess_db - beam_rolloff_db


def snr_log_likelihood(
    obs: SnrObservation,
    delta_theta_rad: np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: np.ndarray,
    range_m: np.ndarray,
    snr_cfg: SnrConfig,
) -> np.ndarray:
    """Per-particle log-likelihood of `obs` (unnormalised; constants dropped).

    Returns log P(obs | particle geometry). For a locked reading this is the
    Gaussian exponent −½z²; for a no-lock report it is log Φ((thr − pred)/σ).
    """
    pred = predicted_ebn0_db(delta_theta_rad, antenna_cfg, elevation_rad, range_m, snr_cfg)
    sigma = max(1e-6, snr_cfg.sigma_meas_db)
    if obs.locked:
        if obs.ebn0_db is None:
            raise ValueError("a locked SnrObservation must carry an ebn0_db value")
        z = (float(obs.ebn0_db) - pred) / sigma
        # A locked reading also implies the carrier was actually above threshold;
        # the Gaussian around `pred` already suppresses particles predicting a
        # sub-threshold Eb/N0, so no extra gate is needed.
        return -0.5 * z**2
    # No lock: the true Eb/N0 fell below the lock threshold.
    cdf = _norm_cdf((snr_cfg.lock_threshold_db - pred) / sigma)
    return np.log(np.clip(cdf, 1e-300, 1.0))


def snr_observation_likelihood(
    obs: SnrObservation,
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray,
    range_m: float | np.ndarray,
    snr_cfg: SnrConfig,
) -> np.ndarray:
    """Linear-scale likelihood (exp of `snr_log_likelihood`)."""
    dth = np.atleast_1d(delta_theta_rad).astype(float)
    el = np.atleast_1d(elevation_rad).astype(float)
    rng = np.atleast_1d(range_m).astype(float)
    return np.exp(snr_log_likelihood(obs, dth, antenna_cfg, el, rng, snr_cfg))


def sample_snr_observation(
    delta_theta_rad: float,
    antenna_cfg: AntennaConfig,
    elevation_rad: float,
    range_m: float,
    snr_cfg: SnrConfig,
    rng: np.random.Generator,
) -> SnrObservation:
    """Draw a realistic demodulator report for a true off-boresight geometry."""
    pred = float(predicted_ebn0_db(delta_theta_rad, antenna_cfg, elevation_rad, range_m, snr_cfg))
    measured = pred + float(rng.normal(0.0, snr_cfg.sigma_meas_db))
    if measured < snr_cfg.lock_threshold_db:
        return SnrObservation(locked=False, ebn0_db=None)
    return SnrObservation(locked=True, ebn0_db=measured)


if __name__ == "__main__":
    from antenna_pomdp.config import default_config

    cfg = default_config()
    snr = SnrConfig()
    rng = np.random.default_rng(0)

    # On-boresight, reference range, zenith -> Eb/N0 ~ ref_ebn0_db.
    pred_on = float(
        predicted_ebn0_db(0.0, cfg.antenna, np.deg2rad(90.0), cfg.antenna.ref_range_m, snr)
    )
    assert abs(pred_on - snr.ref_ebn0_db) < 1e-6, pred_on

    # Off-boresight loses Eb/N0 monotonically.
    pred_off = float(
        predicted_ebn0_db(
            np.deg2rad(8.0), cfg.antenna, np.deg2rad(90.0), cfg.antenna.ref_range_m, snr
        )
    )
    assert pred_off < pred_on, (pred_on, pred_off)

    # The likelihood peaks at the Δθ whose predicted Eb/N0 matches the reading.
    el = np.deg2rad(45.0)
    rng_m = 1.2e6
    true_dth = np.deg2rad(3.0)
    obs = sample_snr_observation(true_dth, cfg.antenna, el, rng_m, snr, rng)
    assert obs.locked and obs.ebn0_db is not None
    grid = np.deg2rad(np.linspace(0.0, 12.0, 200))
    like = snr_observation_likelihood(
        obs, grid, cfg.antenna, np.full_like(grid, el), np.full_like(grid, rng_m), snr
    )
    peak_dth = grid[int(np.argmax(like))]
    assert abs(np.rad2deg(peak_dth - true_dth)) < 1.5, np.rad2deg(peak_dth)

    # A no-lock report favours large Δθ (weak signal) over on-boresight.
    nolock = SnrObservation(locked=False)
    ll = snr_log_likelihood(
        nolock, grid, cfg.antenna, np.full_like(grid, el), np.full_like(grid, rng_m), snr
    )
    assert ll[-1] > ll[0], "no-lock should prefer far-off-boresight particles"

    print(
        f"[snr] Eb/N0 on-axis={pred_on:.1f} dB, off(8deg)={pred_off:.1f} dB, "
        f"meas={obs.ebn0_db:.1f} dB, peak Δθ={np.rad2deg(peak_dth):.1f} deg"
    )
    print("[snr] PASS")
