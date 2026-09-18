"""Tests for the SNR-valued observation model and its particle-filter update."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from antenna_pomdp.config import default_config
from antenna_pomdp.models.observation import Pointing
from antenna_pomdp.models.particle_filter import (
    belief_azel_range,
    make_belief,
    update,
    update_snr,
)
from antenna_pomdp.models.snr_observation import (
    SnrConfig,
    SnrObservation,
    predicted_ebn0_db,
    sample_snr_observation,
    snr_observation_likelihood,
)
from antenna_pomdp.orbit.geometry import Station, find_next_pass
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import sample_leop_particles

SSO_L1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
SSO_L2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"


def test_predicted_ebn0_is_monotone_off_boresight():
    cfg = default_config()
    snr = SnrConfig()
    on = float(predicted_ebn0_db(0.0, cfg.antenna, np.deg2rad(90.0), cfg.antenna.ref_range_m, snr))
    off = float(
        predicted_ebn0_db(
            np.deg2rad(8.0), cfg.antenna, np.deg2rad(90.0), cfg.antenna.ref_range_m, snr
        )
    )
    assert abs(on - snr.ref_ebn0_db) < 1e-6
    assert off < on


def test_locked_likelihood_peaks_at_true_offset():
    cfg = default_config()
    snr = SnrConfig()
    rng = np.random.default_rng(0)
    el, rng_m, true_dth = np.deg2rad(45.0), 1.2e6, np.deg2rad(3.0)
    obs = sample_snr_observation(true_dth, cfg.antenna, el, rng_m, snr, rng)
    assert obs.locked
    grid = np.deg2rad(np.linspace(0.0, 12.0, 300))
    like = snr_observation_likelihood(
        obs, grid, cfg.antenna, np.full_like(grid, el), np.full_like(grid, rng_m), snr
    )
    peak = grid[int(np.argmax(like))]
    assert abs(np.rad2deg(peak - true_dth)) < 1.5


def test_nolock_prefers_far_off_boresight():
    cfg = default_config()
    snr = SnrConfig()
    grid = np.deg2rad(np.linspace(0.0, 12.0, 100))
    el = np.full_like(grid, np.deg2rad(30.0))
    rng_m = np.full_like(grid, 1.5e6)
    like = snr_observation_likelihood(
        SnrObservation(locked=False), grid, cfg.antenna, el, rng_m, snr
    )
    assert like[-1] > like[0]


def _weighted_slip_std(belief) -> float:
    w = belief.weights / belief.weights.sum()
    s = np.array([p.launch_slip_s for p in belief.particles])
    m = float(np.sum(w * s))
    return float(np.sqrt(np.sum(w * (s - m) ** 2)))


def test_snr_update_collapses_more_than_detect():
    base = default_config()
    cfg = replace(
        base, orbit=replace(base.orbit, nominal_tle_line1=SSO_L1, nominal_tle_line2=SSO_L2)
    )
    snr = SnrConfig()
    st = Station.from_config(
        replace(base.station, latitude_deg=78.23, longitude_deg=15.4, min_elevation_deg=5.0)
    )
    nom = SGP4Propagator(SSO_L1, SSO_L2)
    pw = find_next_pass(nom, st, nom.epoch, min_elevation_rad=np.deg2rad(5.0))
    assert pw is not None
    when = pw.peak

    def build():
        rng = np.random.default_rng(3)
        belief = make_belief(sample_leop_particles(cfg.orbit, cfg.leop, 500, rng))
        azelr = belief_azel_range(belief, when, st)
        truth = azelr[250]
        return rng, belief, Pointing(az=truth[0], el=truth[1]), truth

    rng, belief, pointing, truth = build()
    s0 = _weighted_slip_std(belief)
    b_det = update(belief, pointing, True, when, st, cfg.antenna, cfg.filter, rng)

    rng, belief, pointing, truth = build()
    obs = SnrObservation(locked=True, ebn0_db=8.0)
    b_snr = update_snr(belief, pointing, obs, when, st, cfg.antenna, snr, cfg.filter, rng)

    # Both sharpen the along-track spread; the graded Eb/N0 sharpens it more.
    assert _weighted_slip_std(b_det) < s0
    assert _weighted_slip_std(b_snr) < _weighted_slip_std(b_det)
