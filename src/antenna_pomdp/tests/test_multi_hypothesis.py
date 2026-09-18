"""Tests for rideshare multi-hypothesis (multi-object) disambiguation."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from antenna_pomdp.config import FilterConfig, StationConfig, default_config
from antenna_pomdp.models.doppler import FrequencyConfig, particle_doppler_hz
from antenna_pomdp.models.multi_hypothesis import (
    CLUTTER_ID,
    HypothesisConfig,
    disambiguation_report,
    make_multi_hypothesis,
    prune,
    update_multi_doppler,
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


def _cluster(nominal, center_s, n=40, spread=3.0):
    parts = [OrbitParticle(nominal, float(center_s + s)) for s in np.linspace(-spread, spread, n)]
    return ParticleBelief(particles=parts, weights=np.full(n, 1.0 / n))


def test_priors_place_mass_on_own_and_clutter():
    nominal, _, _ = _setup()
    mhb = make_multi_hypothesis(
        [_cluster(nominal, 0.0), _cluster(nominal, 20.0)], ["MINE", "OTHER"]
    )
    p = mhb.hypothesis_probs()
    assert np.isclose(p[0], mhb.cfg.prior_own, atol=1e-6)
    assert np.isclose(p[-1], mhb.cfg.prior_clutter, atol=1e-6)
    assert mhb.labels()[-1] == CLUTTER_ID


def test_resolves_to_true_object():
    nominal, station, when = _setup()
    fcfg, fltr, rng = FrequencyConfig(), FilterConfig(), np.random.default_rng(1)
    mhb = make_multi_hypothesis(
        [_cluster(nominal, 0.0), _cluster(nominal, 25.0), _cluster(nominal, 50.0)],
        ["A", "B", "C"],
    )
    truth = particle_doppler_hz(OrbitParticle(nominal, 25.0), when, station, fcfg)  # object B
    for _ in range(5):
        mhb = update_multi_doppler(mhb, truth, when, station, fcfg, fltr, rng)
    rep = disambiguation_report(mhb)
    assert rep["top"]["id"] == "B"
    assert mhb.candidate_prob("B") > mhb.candidate_prob("A")
    assert mhb.candidate_prob("B") > mhb.candidate_prob("C")


def test_single_obs_does_not_instantly_saturate():
    # Regression guard for the PDA pitfall: with a realistic sigma one observation
    # should NOT jump the top hypothesis to ~1.0 (that signals a too-tight sigma or
    # reading L_k after renormalization).
    nominal, station, when = _setup()
    fcfg, fltr, rng = FrequencyConfig(), FilterConfig(), np.random.default_rng(2)
    mhb = make_multi_hypothesis([_cluster(nominal, 0.0), _cluster(nominal, 25.0)], ["A", "B"])
    truth = particle_doppler_hz(OrbitParticle(nominal, 25.0), when, station, fcfg)
    mhb = update_multi_doppler(mhb, truth, when, station, fcfg, fltr, rng)
    _, top_p = mhb.top_hypothesis()
    assert top_p < 0.999  # evidence accumulates, it does not saturate in one step


def test_entropy_drops_as_it_resolves():
    nominal, station, when = _setup()
    fcfg, fltr, rng = FrequencyConfig(), FilterConfig(), np.random.default_rng(3)
    mhb = make_multi_hypothesis(
        [_cluster(nominal, 0.0), _cluster(nominal, 25.0), _cluster(nominal, 50.0)],
        ["A", "B", "C"],
    )
    h0 = mhb.hypothesis_entropy_bits()
    truth = particle_doppler_hz(OrbitParticle(nominal, 25.0), when, station, fcfg)
    for _ in range(4):
        mhb = update_multi_doppler(mhb, truth, when, station, fcfg, fltr, rng)
    assert mhb.hypothesis_entropy_bits() < h0


def test_pruning_drops_a_far_decoy_but_keeps_own_and_clutter():
    nominal, station, when = _setup()
    fcfg, fltr, rng = FrequencyConfig(), FilterConfig(), np.random.default_rng(4)
    cfg = HypothesisConfig(prune_below=1e-2)
    # Our own object (index 0) is A at +25 s (truth); B is a far decoy.
    mhb = make_multi_hypothesis(
        [_cluster(nominal, 25.0), _cluster(nominal, -40.0)], ["A", "B"], cfg=cfg
    )
    truth = particle_doppler_hz(OrbitParticle(nominal, 25.0), when, station, fcfg)
    for _ in range(6):
        mhb = update_multi_doppler(mhb, truth, when, station, fcfg, fltr, rng)
        mhb = prune(mhb)
    # The decoy is gone; our own candidate and the clutter hypothesis remain.
    assert "B" not in mhb.candidate_ids
    assert "A" in mhb.candidate_ids
    assert mhb.labels()[-1] == CLUTTER_ID


def test_report_probabilities_sum_to_one():
    nominal, _, _ = _setup()
    mhb = make_multi_hypothesis([_cluster(nominal, 0.0), _cluster(nominal, 20.0)], ["A", "B"])
    rep = disambiguation_report(mhb)
    total = sum(c["probability"] for c in rep["candidates"])
    assert np.isclose(total, 1.0, atol=1e-9)
