"""Parse CCSDS OPM state vectors and covariance from KVN text.

Positions, velocities, and covariance are converted from km-based units to SI.
Missing covariance uses ``DEFAULT_SIGMA_*``; an unspecified frame defaults to
ECI. ``OpmState`` retains the covariance frame for rotation by the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# Documented fallback 1-sigma values when an OPM omits its covariance. These are
# representative of early-orbit uncertainty (large position error after
# separation): ~3 km position, ~3 m/s velocity, applied isotropically.
DEFAULT_SIGMA_POS_M = 3000.0
DEFAULT_SIGMA_VEL_M_S = 3.0

# Covariance reference-frame keywords the OPM may use for an RTN/RSW block.
_RTN_FRAME_KEYS = {"RTN", "RSW", "RIC", "LVLH"}


@dataclass(frozen=True)
class OpmState:
    """Mean Cartesian state + covariance at an epoch, frame recorded explicitly.

    `r_eci_m` / `v_eci_m` are in TEME-treated-as-ECI metres / m/s (the same
    inertial frame the SGP4 wrapper uses). `cov_6x6` is the SI covariance over
    [x, y, z, vx, vy, vz]; `frame` is "ECI" or "RTN" and tells the estimator
    whether it must rotate the covariance into ECI before propagating.
    """

    epoch: datetime
    r_eci_m: np.ndarray  # shape (3,)
    v_eci_m: np.ndarray  # shape (3,)
    cov_6x6: np.ndarray  # shape (6, 6), SI units (m^2, m^2/s, m^2/s^2)
    frame: str = "ECI"
    warnings: tuple[str, ...] = ()


def _parse_epoch(value: str) -> datetime:
    """Parse a CCSDS epoch (ISO-8601-ish) into a tz-aware UTC datetime."""
    v = value.strip()
    # Python's fromisoformat handles most CCSDS forms; normalise a trailing Z.
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        # Fall back: strip fractional seconds beyond microseconds, retry.
        if "." in v:
            head, _, tail = v.partition(".")
            frac = "".join(ch for ch in tail if ch.isdigit())[:6]
            dt = datetime.fromisoformat(f"{head}.{frac}")
        else:
            raise
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_opm(text: str) -> OpmState:
    """Parse an OPM KVN string into an OpmState (SI units, ECI or RTN cov)."""
    kv: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        # Drop any inline [units] suffix on the value.
        val = val.split("[")[0].strip()
        kv[key.strip().upper()] = val

    warnings: list[str] = []

    # ----- epoch -----
    if "EPOCH" not in kv:
        raise ValueError("OPM is missing the mandatory EPOCH keyword")
    epoch = _parse_epoch(kv["EPOCH"])

    # ----- mean state (km, km/s -> m, m/s) -----
    try:
        r_km = np.array([float(kv["X"]), float(kv["Y"]), float(kv["Z"])])
        v_km_s = np.array([float(kv["X_DOT"]), float(kv["Y_DOT"]), float(kv["Z_DOT"])])
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"OPM is missing a state component: {exc}") from exc
    r_eci_m = r_km * 1e3
    v_eci_m = v_km_s * 1e3

    # ----- frame -----
    cov_frame_kw = (kv.get("COV_REF_FRAME") or kv.get("REF_FRAME") or "ECI").upper()
    frame = "RTN" if cov_frame_kw in _RTN_FRAME_KEYS else "ECI"

    # ----- covariance -----
    cov, have_cov = _parse_covariance(kv)
    if not have_cov:
        warnings.append(
            "OPM had no covariance block; using default isotropic sigma "
            f"({DEFAULT_SIGMA_POS_M:.0f} m pos, {DEFAULT_SIGMA_VEL_M_S:.1f} m/s vel)."
        )
        frame = "ECI"

    return OpmState(
        epoch=epoch,
        r_eci_m=r_eci_m,
        v_eci_m=v_eci_m,
        cov_6x6=cov,
        frame=frame,
        warnings=tuple(warnings),
    )


# Lower-triangular OPM covariance keyword order (CCSDS 502.0-B, km / km/s units).
_COV_KEYS = [
    ("CX_X",),
    ("CY_X", "CY_Y"),
    ("CZ_X", "CZ_Y", "CZ_Z"),
    ("CX_DOT_X", "CX_DOT_Y", "CX_DOT_Z", "CX_DOT_X_DOT"),
    ("CY_DOT_X", "CY_DOT_Y", "CY_DOT_Z", "CY_DOT_X_DOT", "CY_DOT_Y_DOT"),
    (
        "CZ_DOT_X",
        "CZ_DOT_Y",
        "CZ_DOT_Z",
        "CZ_DOT_X_DOT",
        "CZ_DOT_Y_DOT",
        "CZ_DOT_Z_DOT",
    ),
]

# Unit scale per axis to convert OPM km-based covariance to SI: pos rows/cols
# scale by 1e3 (km->m), velocity rows/cols by 1e3 (km/s -> m/s).
_AXIS_SCALE = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3])


def _default_cov() -> np.ndarray:
    diag = np.array([DEFAULT_SIGMA_POS_M] * 3 + [DEFAULT_SIGMA_VEL_M_S] * 3) ** 2
    return np.diag(diag)


def _parse_covariance(kv: dict[str, str]) -> tuple[np.ndarray, bool]:
    """Return (6x6 SI covariance, found_any). Symmetrised from the LT block."""
    cov = np.zeros((6, 6))
    found = False
    for i, row in enumerate(_COV_KEYS):
        for j, key in enumerate(row):
            if key in kv:
                found = True
                val_km = float(kv[key])  # km^2 / km^2 per (km/s) / (km/s)^2
                # Convert to SI by scaling each axis: C_ij_SI = C_ij * s_i * s_j.
                val_si = val_km * _AXIS_SCALE[i] * _AXIS_SCALE[j]
                cov[i, j] = val_si
                cov[j, i] = val_si
    if not found:
        return _default_cov(), False
    # Guard: if the operator gave only a partial block, fill any zero diagonal
    # with the default sigma so the matrix stays positive-definite.
    for d in range(6):
        if cov[d, d] <= 0.0:
            cov[d, d] = (DEFAULT_SIGMA_POS_M if d < 3 else DEFAULT_SIGMA_VEL_M_S) ** 2
    return cov, True


SAMPLE_OPM = """\
CCSDS_OPM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00
ORIGINATOR = ACQ_PLATFORM
OBJECT_NAME = DEMOSAT
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
EPOCH = 2024-01-01T00:00:00.000
X =  -4453.783586 [km]
Y =   3995.658707 [km]
Z =   2734.025840 [km]
X_DOT = -3.756048 [km/s]
Y_DOT = -2.554257 [km/s]
Z_DOT = -6.748210 [km/s]
COV_REF_FRAME = RTN
CX_X = 9.0e-6
CY_X = 0.0
CY_Y = 9.0e-6
CZ_X = 0.0
CZ_Y = 0.0
CZ_Z = 1.0e-6
CX_DOT_X = 0.0
CX_DOT_Y = 0.0
CX_DOT_Z = 0.0
CX_DOT_X_DOT = 9.0e-12
CY_DOT_X = 0.0
CY_DOT_Y = 0.0
CY_DOT_Z = 0.0
CY_DOT_X_DOT = 0.0
CY_DOT_Y_DOT = 9.0e-12
CZ_DOT_X = 0.0
CZ_DOT_Y = 0.0
CZ_DOT_Z = 0.0
CZ_DOT_X_DOT = 0.0
CZ_DOT_Y_DOT = 0.0
CZ_DOT_Z_DOT = 9.0e-12
"""


if __name__ == "__main__":
    st = parse_opm(SAMPLE_OPM)
    assert st.epoch.year == 2024
    assert st.r_eci_m.shape == (3,) and st.v_eci_m.shape == (3,)
    assert st.cov_6x6.shape == (6, 6)
    assert st.frame == "RTN", st.frame
    # RTN radial 1-sigma = sqrt(9e-6 km^2) = 3e-3 km = 3 m.
    assert abs(np.sqrt(st.cov_6x6[0, 0]) - 3.0) < 1e-6, np.sqrt(st.cov_6x6[0, 0])
    # Symmetric.
    assert np.allclose(st.cov_6x6, st.cov_6x6.T)

    # Missing-covariance path falls back to default sigma.
    no_cov = "\n".join(
        line for line in SAMPLE_OPM.splitlines() if not line.startswith(("C", "COV_REF"))
    )
    st2 = parse_opm(no_cov)
    assert st2.frame == "ECI" and len(st2.warnings) == 1
    assert abs(np.sqrt(st2.cov_6x6[0, 0]) - DEFAULT_SIGMA_POS_M) < 1e-6

    r = np.linalg.norm(st.r_eci_m) / 1e3
    print(f"[opm] |r|={r:.1f} km, frame={st.frame}, warnings={len(st.warnings)}")
    print("[opm] PASS")
