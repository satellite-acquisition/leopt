"""JSON ingest: three operator-friendly JSON encodings of an orbit prior.

1. **Plain state JSON** — the platform-native shape:
       {"epoch": "2024-01-01T00:00:00Z",
        "r_eci_m": [x, y, z], "v_eci_m": [vx, vy, vz],
        "covariance": [[..6x6..]], "frame": "RTN"}
   Position/velocity are SI (metres, m/s). -> OpmState.

2. **Ephemeris JSON** — an OEM expressed as JSON:
       {"ephemeris": [{"epoch": .., "r_eci_m": [..], "v_eci_m": [..]}, ...],
        "covariance": [[..]], "frame": "ECI", "interpolation_degree": 5}
   -> Ephemeris (reuses the OEM container).

3. **CCSDS OMM JSON** — the CelesTrak "GP" / Space-Track OMM record (mean
   Keplerian elements). A single object, or a list of them (we take the first):
       {"NORAD_CAT_ID": 25544, "EPOCH": "..", "MEAN_MOTION": 15.5,
        "ECCENTRICITY": 1e-4, "INCLINATION": 51.6, "RA_OF_ASC_NODE": 100.0,
        "ARG_OF_PERICENTER": 90.0, "MEAN_ANOMALY": 270.0, "BSTAR": 2.2e-4, ...}
   We synthesise the two TLE lines from the mean elements and route through the
   existing `parse_tle`, so an OMM behaves exactly like a pasted TLE.

`parse_json` sniffs which of the three it is and dispatches. Returns the same
`(OpmState, line1|None, line2|None)` shape the TLE parser uses, except an
Ephemeris JSON returns `(Ephemeris, None, None)` in the third slot's place; the
caller (`ingest/detect.py`) normalises that.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from acquisition_platform.ingest.oem import Ephemeris
from acquisition_platform.ingest.opm import (
    DEFAULT_SIGMA_POS_M,
    DEFAULT_SIGMA_VEL_M_S,
    OpmState,
    _default_cov,
    _parse_epoch,
    _RTN_FRAME_KEYS,
)
from acquisition_platform.ingest.tle import parse_tle
from antenna_pomdp.orbit.sampler import _fix_checksum


def _cov_from_json(obj: dict) -> tuple[np.ndarray, str, list[str]]:
    """Return (6x6 SI covariance, frame, warnings) from a JSON body."""
    warnings: list[str] = []
    cov_raw = obj.get("covariance") or obj.get("cov_6x6")
    if cov_raw is None:
        warnings.append(
            "JSON had no covariance; using default isotropic sigma "
            f"({DEFAULT_SIGMA_POS_M:.0f} m pos, {DEFAULT_SIGMA_VEL_M_S:.1f} m/s vel)."
        )
        return _default_cov(), "ECI", warnings
    cov = np.asarray(cov_raw, dtype=float)
    if cov.shape != (6, 6):
        raise ValueError(f"JSON covariance must be 6x6, got {cov.shape}")
    frame_kw = str(obj.get("frame", "ECI")).upper()
    frame = "RTN" if frame_kw in _RTN_FRAME_KEYS else "ECI"
    return cov, frame, warnings


def _parse_state_json(obj: dict) -> OpmState:
    epoch = _parse_epoch(str(obj["epoch"]))
    r = np.asarray(obj["r_eci_m"], dtype=float)
    v = np.asarray(obj["v_eci_m"], dtype=float)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("state JSON needs length-3 r_eci_m and v_eci_m")
    cov, frame, warnings = _cov_from_json(obj)
    return OpmState(
        epoch=epoch, r_eci_m=r, v_eci_m=v, cov_6x6=cov, frame=frame, warnings=tuple(warnings)
    )


def _parse_ephemeris_json(obj: dict) -> Ephemeris:
    recs = obj["ephemeris"]
    epochs: list[datetime] = []
    r_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    for rec in recs:
        epochs.append(_parse_epoch(str(rec["epoch"])))
        r_list.append(np.asarray(rec["r_eci_m"], dtype=float))
        v_list.append(np.asarray(rec["v_eci_m"], dtype=float))
    order = np.argsort([e.timestamp() for e in epochs])
    cov_raw = obj.get("covariance")
    cov = np.asarray(cov_raw, dtype=float) if cov_raw is not None else None
    frame_kw = str(obj.get("frame", "ECI")).upper()
    frame = "RTN" if frame_kw in _RTN_FRAME_KEYS else "ECI"
    return Ephemeris(
        epochs=tuple(epochs[i] for i in order),
        r_eci_m=np.array(r_list)[order],
        v_eci_m=np.array(v_list)[order],
        object_name=str(obj.get("object_name", "")),
        object_id=str(obj.get("object_id", "")),
        cov_6x6=cov,
        frame=frame,
        interp_degree=int(obj.get("interpolation_degree", 5)),
    )


# ---------------------------------------------------------------------------
# OMM (mean Keplerian elements) -> TLE line pair
# ---------------------------------------------------------------------------


def _fmt_tle_epoch(dt: datetime) -> str:
    """Format a datetime as the 14-char TLE epoch YYDDD.DDDDDDDD."""
    dt = dt.astimezone(timezone.utc)
    year2 = dt.year % 100
    start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    doy = (dt - start).total_seconds() / 86400.0 + 1.0  # DOY is 1-based
    return f"{year2:02d}{doy:012.8f}"


def _fmt_tle_exp(value: float) -> str:
    """Format a value in TLE 'assumed-decimal exponent' form, e.g. ' 22588-3'.

    Represents value = mantissa * 10^exp with a leading sign and 5 mantissa
    digits (the leading decimal point is implied). Zero -> ' 00000-0'.
    """
    if value == 0.0:
        return " 00000-0"
    sign = "-" if value < 0 else " "
    a = abs(value)
    exp = 0
    # Normalise so the mantissa is in [0.1, 1.0): 0.dddd form.
    while a >= 1.0:
        a /= 10.0
        exp += 1
    while a < 0.1:
        a *= 10.0
        exp -= 1
    mant = int(round(a * 1e5))
    if mant >= 100000:  # rounded up across a decade
        mant //= 10
        exp += 1
    esign = "-" if exp <= 0 else "+"
    return f"{sign}{mant:05d}{esign}{abs(exp)}"


def omm_to_tle(omm: dict) -> tuple[str, str]:
    """Synthesise the two TLE lines from an OMM / CelesTrak GP record.

    Missing optional fields default sanely (drag terms -> 0, element-set number
    -> 999, rev-at-epoch -> 0). If the record already carries TLE_LINE1 /
    TLE_LINE2 (some feeds include them), those are returned verbatim.
    """
    if omm.get("TLE_LINE1") and omm.get("TLE_LINE2"):
        return str(omm["TLE_LINE1"]).rstrip(), str(omm["TLE_LINE2"]).rstrip()

    norad = int(omm.get("NORAD_CAT_ID", 99999))
    classification = str(omm.get("CLASSIFICATION_TYPE", "U"))[:1] or "U"
    intl = str(omm.get("OBJECT_ID", "")).replace("-", "")
    # Intl designator: 2 digit year + launch number (3) + piece (up to 3).
    intl_field = (intl[2:] if len(intl) > 2 else intl)[:8].ljust(8)

    epoch = _parse_epoch(str(omm["EPOCH"]))
    epoch_str = _fmt_tle_epoch(epoch)

    n_dot = float(omm.get("MEAN_MOTION_DOT", 0.0))  # rev/day^2
    ndot_str = f"{n_dot: .8f}".replace("0.", ".").replace("-0.", "-.")
    ndot_str = (" " + ndot_str.strip())[-10:].rjust(10)
    n_ddot_str = _fmt_tle_exp(float(omm.get("MEAN_MOTION_DDOT", 0.0)))
    bstar_str = _fmt_tle_exp(float(omm.get("BSTAR", 0.0)))
    elem_set = int(omm.get("ELEMENT_SET_NO", 999))
    eph_type = int(omm.get("EPHEMERIS_TYPE", 0))

    line1 = (
        f"1 {norad:05d}{classification} {intl_field}"
        f"{epoch_str} {ndot_str} {n_ddot_str} {bstar_str} {eph_type}"
        f" {elem_set:4d}"
    )
    line1 = _fix_checksum(line1[:68])

    inc = float(omm["INCLINATION"])
    raan = float(omm["RA_OF_ASC_NODE"])
    ecc = float(omm["ECCENTRICITY"])
    argp = float(omm["ARG_OF_PERICENTER"])
    ma = float(omm["MEAN_ANOMALY"])
    n = float(omm["MEAN_MOTION"])
    rev = int(omm.get("REV_AT_EPOCH", 0))
    ecc_str = f"{ecc:.7f}"[2:9]  # drop the "0."
    line2 = (
        f"2 {norad:05d} {inc:8.4f} {raan:8.4f} {ecc_str} {argp:8.4f} {ma:8.4f} {n:11.8f}{rev:5d}"
    )
    line2 = _fix_checksum(line2[:68])
    return line1, line2


def _looks_like_omm(obj: dict) -> bool:
    return "MEAN_MOTION" in obj and "INCLINATION" in obj and "ECCENTRICITY" in obj


def parse_json(
    text: str, cov_6x6: np.ndarray | None = None, frame: str = "ECI"
) -> tuple[OpmState | Ephemeris, str | None, str | None]:
    """Parse a JSON orbit prior, dispatching on its shape.

    Returns `(OpmState, l1, l2)` for a state / OMM body (l1,l2 are the TLE lines
    for an OMM, else None) and `(Ephemeris, None, None)` for an ephemeris body.
    An explicit `cov_6x6` argument overrides any covariance in the body.
    """
    obj = json.loads(text)
    if isinstance(obj, list):
        if not obj:
            raise ValueError("empty JSON array")
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("JSON ingest expects an object or array of objects")

    if "ephemeris" in obj:
        return _parse_ephemeris_json(obj), None, None
    if "r_eci_m" in obj and "v_eci_m" in obj:
        opm = _parse_state_json(obj)
        return opm, None, None
    if _looks_like_omm(obj):
        l1, l2 = omm_to_tle(obj)
        opm, _, _ = parse_tle(f"{l1}\n{l2}", cov_6x6=cov_6x6, frame=frame)
        return opm, l1, l2
    raise ValueError("unrecognised JSON orbit body (need state, ephemeris, or OMM keys)")


SAMPLE_OMM_JSON = """
{"OBJECT_NAME": "DEMOSAT", "OBJECT_ID": "2024-001A", "NORAD_CAT_ID": 25544,
 "EPOCH": "2024-01-01T00:00:00", "MEAN_MOTION": 15.50000000, "ECCENTRICITY": 0.0001000,
 "INCLINATION": 51.6400, "RA_OF_ASC_NODE": 100.0000, "ARG_OF_PERICENTER": 90.0000,
 "MEAN_ANOMALY": 270.0000, "BSTAR": 0.00022588, "ELEMENT_SET_NO": 999}
"""

SAMPLE_STATE_JSON = """
{"epoch": "2024-01-01T00:00:00Z",
 "r_eci_m": [-4453783.586, 3995658.707, 2734025.840],
 "v_eci_m": [-3756.048, -2554.257, -6748.210],
 "frame": "RTN",
 "covariance": [[9e6,0,0,0,0,0],[0,9e6,0,0,0,0],[0,0,1e6,0,0,0],
                [0,0,0,9,0,0],[0,0,0,0,9,0],[0,0,0,0,0,9]]}
"""


if __name__ == "__main__":
    # OMM -> TLE -> OpmState, and the synthesised lines parse under SGP4.
    opm, l1, l2 = parse_json(SAMPLE_OMM_JSON)
    assert l1 is not None and l1.startswith("1 ") and l2.startswith("2 ")
    from antenna_pomdp.orbit.propagator import SGP4Propagator

    prop = SGP4Propagator(l1, l2)
    r = np.linalg.norm(prop.propagate(prop.epoch).r_eci_m) / 1e3
    assert 6.5e3 < r < 8.0e3, r
    assert isinstance(opm, OpmState)

    # Round-trip the inclination/mean-motion back out of the synthesised TLE.
    from antenna_pomdp.orbit.sampler import _parse_tle_fields

    f = _parse_tle_fields(l1, l2)
    assert abs(f["inc_deg"] - 51.64) < 1e-3, f["inc_deg"]
    assert abs(f["mean_motion"] - 15.5) < 1e-6, f["mean_motion"]

    # Plain state JSON with an explicit RTN covariance.
    st, sl1, sl2 = parse_json(SAMPLE_STATE_JSON)
    assert isinstance(st, OpmState) and sl1 is None
    assert st.frame == "RTN"
    assert abs(np.sqrt(st.cov_6x6[0, 0]) - 3000.0) < 1e-6

    # Ephemeris JSON.
    eph_json = (
        '{"ephemeris": ['
        '{"epoch":"2024-01-01T00:00:00","r_eci_m":[-4453783,3995658,2734025],'
        '"v_eci_m":[-3756,-2554,-6748]},'
        '{"epoch":"2024-01-01T00:02:00","r_eci_m":[-4867234,3568120,1893400],'
        '"v_eci_m":[-3120,-4560,-6190]}]}'
    )
    eph, _, _ = parse_json(eph_json)
    assert isinstance(eph, Ephemeris) and eph.n == 2

    # TLE-exponent formatting sanity.
    assert _fmt_tle_exp(0.00022588) == " 22588-3", _fmt_tle_exp(0.00022588)
    assert _fmt_tle_exp(0.0) == " 00000-0"
    assert _fmt_tle_exp(-1.234e-5) == "-12340-4", _fmt_tle_exp(-1.234e-5)

    print(f"[json] OMM->TLE ok (|r|={r:.1f} km), state frame={st.frame}, eph N={eph.n}")
    print("[json] PASS")
