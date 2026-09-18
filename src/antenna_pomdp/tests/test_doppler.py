"""Tests for the joint angle+Doppler (carrier-frequency) belief-space search."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from antenna_pomdp.config import FilterConfig, StationConfig, default_config
from antenna_pomdp.models.doppler import (
    FrequencyConfig,
    belief_doppler_hz,
    effective_sigma_hz,
    freq_bin_detect_likelihood,
    particle_doppler_hz,
    range_rate_m_s,
    update_bin_detect,
    update_doppler,
)
from antenna_pomdp.models.particle_filter import ParticleBelief
from antenna_pomdp.orbit.geometry import Station
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle


def _setup():
    cfg = default_config()
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(StationConfig())
    when = nominal.epoch + timedelta(minutes=5)
    return nominal, station, when


def _slip_belief(nominal, slips, weights=None):
    parts = [OrbitParticle(nominal, float(s)) for s in slips]
    w = np.full(len(slips), 1.0 / len(slips)) if weights is None else np.asarray(weights, float)
    return ParticleBelief(particles=parts, weights=w / w.sum())


def _mean_slip(belief):
    slips = np.array([p.launch_slip_s for p in belief.particles])
    w = np.asarray(belief.weights)
    return float(np.dot(w / w.sum(), slips))


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def test_doppler_magnitude_is_sane():
    nominal, station, when = _setup()
    d = particle_doppler_hz(OrbitParticle(nominal, 0.0), when, station, FrequencyConfig())
    # S-band LEO Doppler is at most ~±60 kHz.
    assert abs(d) < 1.5e5


def test_range_rate_matches_finite_difference():
    nominal, station, when = _setup()
    dt = 1.0
    s0 = nominal.propagate(when)
    r0 = np.linalg.norm(s0.r_eci_m - _station_eci(station, when))
    s1 = nominal.propagate(when + timedelta(seconds=dt))
    r1 = np.linalg.norm(s1.r_eci_m - _station_eci(station, when + timedelta(seconds=dt)))
    fd = (r1 - r0) / dt
    analytic = range_rate_m_s(s0, when, station)
    # Finite-difference range-rate should match the analytic one within a few %.
    assert abs(fd - analytic) < 0.05 * abs(fd) + 20.0


def _station_eci(station, when):
    from antenna_pomdp.orbit.geometry import rot_eci_to_ecef

    return rot_eci_to_ecef(when).T @ station.ecef_m


def test_different_slips_give_different_doppler():
    nominal, station, when = _setup()
    d = belief_doppler_hz(
        _slip_belief(nominal, [0.0, 30.0, 60.0]), when, station, FrequencyConfig()
    )
    assert np.ptp(d) > 100.0  # kHz-scale spread across a minute of launch slip


# ---------------------------------------------------------------------------
# Likelihood + updates
# ---------------------------------------------------------------------------


def test_freq_bin_likelihood_peaks_in_bin():
    sig = 100.0
    inside = freq_bin_detect_likelihood(np.array([0.0]), -sig, sig, sig)[0]
    outside = freq_bin_detect_likelihood(np.array([1000.0]), -sig, sig, sig)[0]
    assert inside > 0.5 > outside


def test_doppler_update_collapses_along_track():
    nominal, station, when = _setup()
    slips = np.linspace(0.0, 60.0, 60)
    belief = _slip_belief(nominal, slips)  # prior mean 30 s
    truth_slip = 12.0
    measured = particle_doppler_hz(
        OrbitParticle(nominal, truth_slip), when, station, FrequencyConfig()
    )

    prior_mean = _mean_slip(belief)
    post = update_doppler(
        belief, measured, when, station, FrequencyConfig(), FilterConfig(), np.random.default_rng(0)
    )
    post_mean = _mean_slip(post)
    # The frequency measurement pulls the along-track (launch-slip) belief toward truth.
    assert abs(post_mean - truth_slip) < abs(prior_mean - truth_slip)


def test_bin_detect_update_favours_matching_particle():
    nominal, station, when = _setup()
    fcfg = FrequencyConfig()
    d0 = particle_doppler_hz(OrbitParticle(nominal, 0.0), when, station, fcfg)
    d40 = particle_doppler_hz(OrbitParticle(nominal, 40.0), when, station, fcfg)
    belief = _slip_belief(nominal, [0.0, 40.0])
    sig = effective_sigma_hz(fcfg)
    # A detect in a bin around p0's tone should raise p0's weight over p40's.
    post = update_bin_detect(
        belief,
        d0 - sig,
        d0 + sig,
        True,
        when,
        station,
        fcfg,
        FilterConfig(),
        np.random.default_rng(0),
    )
    assert post.weights[0] > post.weights[1]
    assert abs(d0 - d40) > 1.0  # the two are genuinely separable


def test_zero_process_noise_keeps_update_stable():
    # A degenerate (single-particle) belief still updates without error.
    nominal, station, when = _setup()
    belief = _slip_belief(nominal, [10.0])
    d = particle_doppler_hz(belief.particles[0], when, station, FrequencyConfig())
    post = update_doppler(
        belief, d, when, station, FrequencyConfig(), FilterConfig(), np.random.default_rng(0)
    )
    assert pytest.approx(1.0, abs=1e-9) == float(post.weights.sum())
