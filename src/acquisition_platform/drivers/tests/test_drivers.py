"""Fast, hardware-free tests for the antenna-driver layer.

Real socket round-trips use in-process loopback servers; nothing sleeps and no
external hardware is touched. Style mirrors tests/test_platform.py.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone

import pytest

from acquisition_platform.drivers.base import (
    TrackSample,
    densify_track,
    dwells_from_passout,
)
from acquisition_platform.drivers.program_track import (
    to_ccsds_pointing_kvn,
    to_csv,
    to_oem_kvn,
)
from acquisition_platform.drivers.registry import DRIVERS, get_driver, list_drivers
from acquisition_platform.drivers.rotctld import RotctldDriver, _LoopbackRotctld
from acquisition_platform.drivers.stream import (
    StreamDriver,
    encode_csv,
    encode_json,
)
from acquisition_platform.drivers.vendors import (
    ComtechDriver,
    GenericRotatorDriver,
    KongsbergDriver,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _dwells():
    return [
        {"when": T0, "az_deg": 350.0, "el_deg": 10.0, "dwell_s": 30.0},
        {"when": T0 + timedelta(seconds=30), "az_deg": 10.0, "el_deg": 40.0, "dwell_s": 30.0},
        {"when": T0 + timedelta(seconds=60), "az_deg": 30.0, "el_deg": 20.0, "dwell_s": 30.0},
    ]


def _track(n: int = 2):
    return [TrackSample(T0 + timedelta(seconds=i), 10.0 + i, 20.0 + i, 0.5, 0.1) for i in range(n)]


# ---------------------------------------------------------------------------
# densify_track
# ---------------------------------------------------------------------------


def test_densify_track_length_and_cadence():
    track = densify_track(_dwells(), rate_hz=2.0)
    # 60 s span at 2 Hz -> 120 interior steps + 1 final sample.
    assert len(track) == 121
    assert track[0].t == T0
    assert track[-1].t == T0 + timedelta(seconds=60)
    assert all(0.0 <= s.az_deg < 360.0 for s in track)


def test_densify_track_azimuth_wrap_shortest_path():
    # 350 -> 010 deg must sweep POSITIVE (+20), never the long way (-340).
    track = densify_track(_dwells(), rate_hz=1.0)
    assert track[0].az_rate_deg_s > 0.0
    # Rate magnitude is the short arc: 20 deg over 30 s ~= 0.667 deg/s.
    assert abs(track[0].az_rate_deg_s - 20.0 / 30.0) < 1e-6


def test_densify_track_single_dwell():
    track = densify_track(_dwells()[:1], rate_hz=1.0)
    assert len(track) == 1
    assert track[0].az_rate_deg_s == 0.0 and track[0].el_rate_deg_s == 0.0


def test_densify_track_rejects_bad_input():
    with pytest.raises(ValueError):
        densify_track([], rate_hz=1.0)
    with pytest.raises(ValueError):
        densify_track(_dwells(), rate_hz=0.0)


def test_dwells_from_passout():
    passout = {
        "idx": 0,
        "station": "STAN",
        "rise": "2024-01-01T00:00:00+00:00",
        "set": "2024-01-01T00:01:00+00:00",
        "peak_el_deg": 45.0,
        "pointings": [
            {"t_s": 0.0, "az_deg": 10.0, "el_deg": 20.0, "dwell_s": 30.0},
            {"t_s": 30.0, "az_deg": 15.0, "el_deg": 25.0, "dwell_s": 30.0},
        ],
    }
    dwells = dwells_from_passout(passout)
    assert len(dwells) == 2
    assert dwells[0]["when"] == T0
    assert dwells[1]["when"] == T0 + timedelta(seconds=30)
    assert dwells[1]["az_deg"] == 15.0


# ---------------------------------------------------------------------------
# program_track exporters
# ---------------------------------------------------------------------------


def test_to_csv_header_and_rows():
    csv = to_csv(_track(3))
    lines = csv.strip().splitlines()
    assert lines[0] == "time_utc,az_deg,el_deg,az_rate_deg_s,el_rate_deg_s"
    assert len(lines) == 4  # header + 3 rows
    assert lines[1].startswith("2024-01-01T00:00:00Z,")


def test_to_ccsds_pointing_kvn_structure():
    kvn = to_ccsds_pointing_kvn(_track(2), object_name="TESTSAT", originator="ME")
    assert "META_START" in kvn and "META_STOP" in kvn
    assert "TIME_SYSTEM = UTC" in kvn
    assert "OBJECT_NAME = TESTSAT" in kvn and "ORIGINATOR = ME" in kvn


def test_to_oem_kvn_is_valid_oem():
    l1 = "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990"
    l2 = "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    from antenna_pomdp.orbit.propagator import SGP4Propagator

    prop = SGP4Propagator(l1, l2)
    times = [prop.epoch + timedelta(seconds=60 * i) for i in range(4)]
    oem = to_oem_kvn(prop, times, object_name="TESTSAT", object_id="2024-001A")
    assert "CCSDS_OEM_VERS = 2.0" in oem
    assert "REF_FRAME = TEME" in oem and "CENTER_NAME = EARTH" in oem
    data = [ln for ln in oem.strip().splitlines() if ln[0].isdigit()]
    assert len(data) == 4
    assert len(data[0].split()) == 7  # epoch + xyz + xyz_dot

    with pytest.raises(ValueError):
        to_oem_kvn(prop, [])


# ---------------------------------------------------------------------------
# rotctld
# ---------------------------------------------------------------------------


def test_rotctld_dry_run_records():
    drv = RotctldDriver(dry_run=True)
    with drv:
        drv.send_track(_track(3))
        drv.park()
    assert len(drv.sent) == 3
    assert drv.get_position() == (0.0, 0.0)


def test_rotctld_real_socket_roundtrip():
    with _LoopbackRotctld() as srv:
        drv = RotctldDriver(host=srv.host, port=srv.port)
        with drv:
            drv.command(TrackSample(T0, 123.4, 45.6, 0.0, 0.0))
            az, el = drv.get_position()
            drv.park()
        assert abs(az - 123.4) < 1e-3 and abs(el - 45.6) < 1e-3
        assert abs(srv.az) < 1e-9 and abs(srv.el) < 1e-9


def test_satnogs_is_rotctld_compatible():
    drv = get_driver("satnogs", client_id="gs-1", dry_run=True)
    assert drv.name == "satnogs"
    assert drv.client_id == "gs-1"


# ---------------------------------------------------------------------------
# stream + encoders
# ---------------------------------------------------------------------------


def test_stream_encoders_dry_run():
    for enc in (encode_csv, encode_json):
        drv = StreamDriver(encoder=enc, dry_run=True)
        with drv:
            drv.send_track(_track(2))
        assert len(drv.sent_bytes) == 2
        assert all(b.endswith(b"\n") for b in drv.sent_bytes)
    obj = json.loads(encode_json(_track(1)[0]).decode().strip())
    assert set(obj) == {"t", "az", "el", "az_rate", "el_rate"}


def test_stream_real_socket():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    drv = StreamDriver(host=host, port=port, encoder=encode_json)
    drv.connect()
    conn, _ = srv.accept()
    drv.send_track(_track(3), pacing=False)
    drv.close()
    conn.settimeout(1.0)
    received = b""
    while True:
        try:
            chunk = conn.recv(4096)
        except OSError:
            break
        if not chunk:
            break
        received += chunk
    conn.close()
    srv.close()
    assert received.count(b"\n") == 3


# ---------------------------------------------------------------------------
# vendors
# ---------------------------------------------------------------------------


def test_vendor_encodings():
    gen = GenericRotatorDriver(dry_run=True)
    kon = KongsbergDriver(dry_run=True)
    com = ComtechDriver(dry_run=True)
    for drv in (gen, kon, com):
        with drv:
            drv.send_track(_track(2))
        assert len(drv.sent_bytes) == 2

    assert gen.sent_bytes[0].startswith(b"2024-01-01")
    kobj = json.loads(kon.sent_bytes[0].decode().strip())
    assert kobj["cmd"] == "track" and "az_rate" in kobj
    assert com.sent_bytes[0].startswith(b"TRK,")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_listing_and_factory():
    listing = list_drivers()
    assert {d["id"] for d in listing} == set(DRIVERS)
    for d in listing:
        assert d["transport"] in ("file", "tcp")
        assert isinstance(d["capabilities"], list)
        assert d["description"]


def test_get_driver_unknown_raises():
    with pytest.raises(KeyError):
        get_driver("does_not_exist")


def test_get_driver_all_ids():
    for driver_id in DRIVERS:
        drv = get_driver(driver_id, dry_run=True)
        assert drv.name
