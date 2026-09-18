"""Regression tests for the deterministic Brahe SGP4 truth bridge."""

from __future__ import annotations

from datetime import timedelta
import socket

import brahe as bh
import numpy as np

from antenna_pomdp.config import default_config
from antenna_pomdp.orbit.brahe_truth import (
    BraheSgp4Topocentric,
    configure_static_brahe_environment,
)
from antenna_pomdp.orbit.geometry import Station, angular_separation, eci_to_azel
from antenna_pomdp.orbit.propagator import SGP4Propagator


_PARITY_OFFSETS_S = (0.0, 60.0, 600.0, 3600.0, 21600.0, 86400.0)


def test_static_brahe_environment_is_offline(monkeypatch) -> None:
    """The bridge must not invoke Brahe's downloading convenience paths."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network or auto-refresh path was invoked")

    # Cover the public Python entry points that initialize or download EOP
    # data, plus Python socket calls. Static Rust providers need none of them.
    for name in (
        "initialize_eop",
        "initialize_sw",
        "download_standard_eop_file",
        "download_c04_eop_file",
    ):
        monkeypatch.setattr(bh, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    configure_static_brahe_environment()
    cfg = default_config()
    bridge = BraheSgp4Topocentric(
        cfg.orbit.nominal_tle_line1,
        cfg.orbit.nominal_tle_line2,
        cfg.station,
        configure_environment=False,
    )
    state = bridge.azel_range(SGP4Propagator(bridge.tle_line1, bridge.tle_line2).epoch)

    assert bh.get_global_eop_initialization()
    assert np.all(np.isfinite([state.az_rad, state.el_rad, state.range_m]))
    assert 0.0 <= state.az_rad < 2.0 * np.pi
    assert -np.pi / 2.0 <= state.el_rad <= np.pi / 2.0
    assert state.range_m > 0.0


def test_python_sgp4_matches_brahe_topocentric_geometry() -> None:
    """Pinned engines agree after conversion to the same observable frame."""

    cfg = default_config()
    python_sgp4 = SGP4Propagator(
        cfg.orbit.nominal_tle_line1,
        cfg.orbit.nominal_tle_line2,
    )
    station = Station.from_config(cfg.station)
    brahe_sgp4 = BraheSgp4Topocentric(
        cfg.orbit.nominal_tle_line1,
        cfg.orbit.nominal_tle_line2,
        cfg.station,
    )

    for offset_s in _PARITY_OFFSETS_S:
        when = python_sgp4.epoch + timedelta(seconds=offset_s)
        python_state = python_sgp4.propagate(when)
        py_az, py_el, py_range = eci_to_azel(python_state.r_eci_m, when, station)
        brahe_state = brahe_sgp4.azel_range(when)

        separation_deg = float(
            np.rad2deg(
                angular_separation(
                    py_az,
                    py_el,
                    brahe_state.az_rad,
                    brahe_state.el_rad,
                )
            )
        )
        assert separation_deg < 1.0e-3, (offset_s, separation_deg)
        assert abs(py_range - brahe_state.range_m) < 50.0, (
            offset_s,
            py_range,
            brahe_state.range_m,
        )
