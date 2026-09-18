"""Tests for the multi-station network enumeration and episode runner.

These exercise the additive network code (orbit.network + eval.network_runner)
on a tiny config. They follow the same smoke/contract style as
test_each_module.py and run in a couple of seconds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from antenna_pomdp.config import (
    Config,
    all_networks,
    default_config,
    network_polar_sso,
    network_single_university,
)
from antenna_pomdp.eval.leop_metrics import (
    cumulative_acquisition_curve,
    failure_rate,
)
from antenna_pomdp.eval.leop_runner import (
    make_nominal_openloop_policy,
    make_sweep_policy,
)
from antenna_pomdp.eval.network_runner import (
    NetworkEpisodeResult,
    run_network_episode,
    run_network_monte_carlo,
)
from antenna_pomdp.orbit.network import enumerate_network_passes
from antenna_pomdp.orbit.propagator import SGP4Propagator


# A 500 km Sun-synchronous orbit (97.4 deg inclination) so polar stations are
# actually visible; the packaged ISS-like TLE never rises at >=72 deg latitude.
_SSO_L1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
_SSO_L2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"


@pytest.fixture
def cfg() -> Config:
    base = default_config()
    return replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=_SSO_L1, nominal_tle_line2=_SSO_L2),
        antenna=replace(base.antenna, fwhm_deg=10.0),
        filter=replace(base.filter, n_particles=48),
        pomcp=replace(base.pomcp, n_rollouts=12, max_depth=3),
        leop=replace(base.leop, leop_window_s=6 * 3600.0),
    )


@pytest.fixture
def nominal(cfg: Config) -> SGP4Propagator:
    return SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)


# ---------------------------------------------------------------------------
# config registry
# ---------------------------------------------------------------------------


def test_four_networks_registered():
    nets = all_networks()
    assert len(nets) == 4
    ids = {n.network_id for n in nets}
    assert ids == {
        "single_university",
        "polar_sso",
        "distributed_commercial",
        "midlat_cluster",
    }


def test_network_labels_match_station_count():
    for net in all_networks():
        assert len(net.labels()) == len(net.stations)


# ---------------------------------------------------------------------------
# orbit.network
# ---------------------------------------------------------------------------


def test_network_passes_time_ordered_and_numbered(cfg, nominal):
    passes = enumerate_network_passes(
        nominal,
        network_polar_sso(),
        nominal.epoch,
        leop_window_s=cfg.leop.leop_window_s,
    )
    assert len(passes) > 0
    rises = [p.t_rise for p in passes]
    assert rises == sorted(rises)
    assert [p.network_pass_number for p in passes] == list(range(1, len(passes) + 1))
    for p in passes:
        assert p.station_id in ("SVAL", "TROLL", "TROM")
        assert p.t_set > p.t_rise


def test_polar_sees_at_least_single(cfg, nominal):
    single = enumerate_network_passes(
        nominal,
        network_single_university(),
        nominal.epoch,
        leop_window_s=24 * 3600.0,
    )
    polar = enumerate_network_passes(
        nominal,
        network_polar_sso(),
        nominal.epoch,
        leop_window_s=24 * 3600.0,
    )
    assert len(polar) >= len(single)


# ---------------------------------------------------------------------------
# eval.network_runner
# ---------------------------------------------------------------------------


def test_network_episode_runs_and_tags_station(cfg):
    rng = np.random.default_rng(0)
    ep = run_network_episode(cfg, network_polar_sso(), make_sweep_policy(), rng)
    assert isinstance(ep, NetworkEpisodeResult)
    assert ep.n_passes_available >= len(ep.passes)
    if ep.acquired:
        assert ep.acquired_station_id in ("SVAL", "TROLL", "TROM")
        assert ep.acquired_pass_number is not None
        assert ep.time_since_separation_s is not None
    for r in ep.passes:
        assert r.station_id in ("SVAL", "TROLL", "TROM")


def test_network_episode_is_reproducible(cfg):
    ep1 = run_network_episode(
        cfg, network_polar_sso(), make_sweep_policy(), np.random.default_rng(7)
    )
    ep2 = run_network_episode(
        cfg, network_polar_sso(), make_sweep_policy(), np.random.default_rng(7)
    )
    assert ep1.acquired == ep2.acquired
    assert ep1.acquired_pass_number == ep2.acquired_pass_number
    assert ep1.acquired_station_id == ep2.acquired_station_id


def test_network_monte_carlo_metrics(cfg):
    eps = run_network_monte_carlo(
        cfg,
        network_polar_sso(),
        make_nominal_openloop_policy(),
        n_episodes=3,
        seed=1,
    )
    assert len(eps) == 3
    # leop_metrics helpers consume these directly (shared field names).
    grid = np.array([600.0, 3600.0, 6 * 3600.0])
    cdf = cumulative_acquisition_curve(eps, grid)
    assert cdf.shape == grid.shape
    assert np.all(np.diff(cdf) >= -1e-9)  # non-decreasing
    fr = failure_rate(eps)
    assert 0.0 <= fr <= 1.0
