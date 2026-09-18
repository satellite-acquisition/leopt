"""Parse CCSDS OEM ephemerides from KVN text.

``Ephemeris.at`` interpolates state vectors; ``to_opm`` samples an ``OpmState``
for the estimator. Interpolation uses the declared Lagrange degree, bounded by
available samples, and is linear between two samples. Inputs are converted
from km and km/s to SI. Missing covariance uses the OPM defaults.

The XML parser in ``ingest.ndm`` shares the ``Ephemeris`` container.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from acquisition_platform.ingest.opm import (
    _AXIS_SCALE,
    _COV_KEYS,
    DEFAULT_SIGMA_POS_M,
    DEFAULT_SIGMA_VEL_M_S,
    OpmState,
    _default_cov,
    _parse_epoch,
    _RTN_FRAME_KEYS,
)

# Cap the Lagrange interpolation degree so a huge ephemeris does not build a
# pathologically high-order polynomial (Runge). CCSDS OEMs typically declare 5-7.
_MAX_INTERP_DEGREE = 8


@dataclass(frozen=True)
class Ephemeris:
    """An ordered time series of ECI Cartesian states (SI units).

    `epochs` is a tuple of tz-aware UTC datetimes (strictly increasing after
    parsing); `r_eci_m` / `v_eci_m` are shape-(N, 3) arrays. `cov_6x6` and
    `frame` carry an OEM covariance if one was present (else a documented
    default), so a sampled state can seed the estimator like an OPM.
    """

    epochs: tuple[datetime, ...]
    r_eci_m: np.ndarray  # (N, 3)
    v_eci_m: np.ndarray  # (N, 3)
    object_name: str = ""
    object_id: str = ""
    cov_6x6: np.ndarray | None = None
    frame: str = "ECI"
    interp_degree: int = 5
    warnings: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.epochs)

    @property
    def start(self) -> datetime:
        return self.epochs[0]

    @property
    def stop(self) -> datetime:
        return self.epochs[-1]

    def _times_s(self) -> np.ndarray:
        """Seconds of each epoch relative to the first (monotone increasing)."""
        t0 = self.epochs[0]
        return np.array([(e - t0).total_seconds() for e in self.epochs])

    def at(self, when: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated (r_eci_m, v_eci_m) at `when` (SI).

        Uses Lagrange interpolation over the window of `interp_degree + 1`
        samples nearest `when`; clamps to the endpoints when `when` is outside
        the covered span (with a recorded caveat handled by the caller).
        """
        ts = self._times_s()
        t0 = self.epochs[0]
        x = (when - t0).total_seconds()
        if self.n == 1:
            return self.r_eci_m[0].copy(), self.v_eci_m[0].copy()
        # Clamp to the covered interval.
        x = float(np.clip(x, ts[0], ts[-1]))
        # Choose a centred window of degree+1 nodes around x.
        deg = int(np.clip(self.interp_degree, 1, min(_MAX_INTERP_DEGREE, self.n - 1)))
        k = deg + 1
        i = int(np.searchsorted(ts, x))
        lo = int(np.clip(i - k // 2, 0, self.n - k))
        idx = np.arange(lo, lo + k)
        r = _lagrange(ts[idx], self.r_eci_m[idx], x)
        v = _lagrange(ts[idx], self.v_eci_m[idx], x)
        return r, v

    def to_opm(self, when: datetime | None = None) -> OpmState:
        """Sample the ephemeris at `when` (default: first epoch) into an OpmState."""
        w = when if when is not None else self.epochs[0]
        r, v = self.at(w)
        warnings = list(self.warnings)
        if not (self.start <= w <= self.stop):
            warnings.append(
                f"requested epoch {w.isoformat()} is outside the OEM span "
                f"[{self.start.isoformat()}, {self.stop.isoformat()}]; clamped."
            )
        cov = self.cov_6x6
        frame = self.frame
        if cov is None:
            cov = _default_cov()
            frame = "ECI"
            warnings.append(
                "OEM had no covariance block; using default isotropic sigma "
                f"({DEFAULT_SIGMA_POS_M:.0f} m pos, {DEFAULT_SIGMA_VEL_M_S:.1f} m/s vel)."
            )
        return OpmState(
            epoch=w,
            r_eci_m=r,
            v_eci_m=v,
            cov_6x6=cov,
            frame=frame,
            warnings=tuple(warnings),
        )


def _lagrange(nodes_t: np.ndarray, nodes_y: np.ndarray, x: float) -> np.ndarray:
    """Lagrange interpolation of vector samples `nodes_y` at abscissa `x`."""
    k = len(nodes_t)
    total = np.zeros(nodes_y.shape[1])
    for j in range(k):
        tj = nodes_t[j]
        basis = 1.0
        for m in range(k):
            if m == j:
                continue
            basis *= (x - nodes_t[m]) / (tj - nodes_t[m])
        total = total + basis * nodes_y[j]
    return total


def _parse_oem_covariance(kv: dict[str, str]) -> tuple[np.ndarray | None, str]:
    """Parse an OEM COVARIANCE keyword block (same LT layout as the OPM)."""
    cov = np.zeros((6, 6))
    found = False
    for i, row in enumerate(_COV_KEYS):
        for j, key in enumerate(row):
            if key in kv:
                found = True
                val_si = float(kv[key]) * _AXIS_SCALE[i] * _AXIS_SCALE[j]
                cov[i, j] = val_si
                cov[j, i] = val_si
    if not found:
        return None, "ECI"
    for d in range(6):
        if cov[d, d] <= 0.0:
            cov[d, d] = (DEFAULT_SIGMA_POS_M if d < 3 else DEFAULT_SIGMA_VEL_M_S) ** 2
    frame_kw = (kv.get("COV_REF_FRAME") or "ECI").upper()
    frame = "RTN" if frame_kw in _RTN_FRAME_KEYS else "ECI"
    return cov, frame


def parse_oem(text: str) -> Ephemeris:
    """Parse an OEM KVN string into an `Ephemeris` (SI units).

    Tolerant of the CCSDS block structure: META_START/META_STOP and
    COVARIANCE_START/COVARIANCE_STOP markers are recognised but not required;
    any `KEY = VALUE` line contributes to the metadata dictionary, and any line
    of >= 7 whitespace-separated numeric-ish tokens is treated as an ephemeris
    record `EPOCH X Y Z X_DOT Y_DOT Z_DOT`.
    """
    kv: dict[str, str] = {}
    epochs: list[datetime] = []
    rows: list[list[float]] = []
    in_cov = False
    cov_kv: dict[str, str] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        upper = line.upper()
        if upper.startswith("COVARIANCE_START"):
            in_cov = True
            continue
        if upper.startswith("COVARIANCE_STOP"):
            in_cov = False
            continue
        if upper in ("META_START", "META_STOP", "DATA_START", "DATA_STOP"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            val = val.split("[")[0].strip()
            target = cov_kv if in_cov else kv
            target[key.strip().upper()] = val
            continue
        # Otherwise: an ephemeris data row "EPOCH X Y Z XD YD ZD".
        parts = line.split()
        if len(parts) >= 7:
            try:
                epoch = _parse_epoch(parts[0])
                vals = [float(p) for p in parts[1:7]]
            except ValueError:
                continue
            epochs.append(epoch)
            rows.append(vals)

    if not epochs:
        raise ValueError("OEM contained no ephemeris state-vector rows")

    # Sort by epoch and convert km/(km/s) -> SI.
    order = np.argsort([e.timestamp() for e in epochs])
    epochs_sorted = tuple(epochs[i] for i in order)
    arr = np.array(rows)[order] * 1e3  # km -> m, km/s -> m/s
    r = arr[:, :3]
    v = arr[:, 3:6]

    cov, cov_frame = _parse_oem_covariance(cov_kv)
    try:
        interp_degree = int(float(kv.get("INTERPOLATION_DEGREE", "5")))
    except ValueError:
        interp_degree = 5

    return Ephemeris(
        epochs=epochs_sorted,
        r_eci_m=r,
        v_eci_m=v,
        object_name=kv.get("OBJECT_NAME", ""),
        object_id=kv.get("OBJECT_ID", ""),
        cov_6x6=cov,
        frame=cov_frame,
        interp_degree=interp_degree,
        warnings=(),
    )


SAMPLE_OEM = """\
CCSDS_OEM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00
ORIGINATOR = LAUNCH_PROVIDER
META_START
OBJECT_NAME = DEMOSAT
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = 2024-01-01T00:00:00.000
STOP_TIME = 2024-01-01T00:10:00.000
INTERPOLATION = LAGRANGE
INTERPOLATION_DEGREE = 5
META_STOP
2024-01-01T00:00:00.000  -4453.783586  3995.658707  2734.025840  -3.756048  -2.554257  -6.748210
2024-01-01T00:02:00.000  -4867.234000  3568.120000  1893.400000  -3.120000  -4.560000  -6.190000
2024-01-01T00:04:00.000  -5150.000000  2990.000000   980.000000  -1.980000  -5.980000  -5.020000
2024-01-01T00:06:00.000  -5270.000000  2300.000000    30.000000  -0.560000  -6.760000  -3.400000
2024-01-01T00:08:00.000  -5210.000000  1550.000000  -910.000000   0.980000  -6.800000  -1.520000
2024-01-01T00:10:00.000  -4980.000000   790.000000 -1790.000000   2.420000  -6.100000   0.420000
"""


if __name__ == "__main__":
    eph = parse_oem(SAMPLE_OEM)
    assert eph.n == 6, eph.n
    assert eph.object_id == "2024-001A"
    assert eph.start.minute == 0 and eph.stop.minute == 10

    # Interpolate at a node -> reproduces that node (Lagrange is exact on nodes).
    r_node, v_node = eph.at(eph.epochs[2])
    assert np.allclose(r_node, eph.r_eci_m[2], atol=1e-3), r_node
    assert np.allclose(v_node, eph.v_eci_m[2], atol=1e-6), v_node

    # Interpolate mid-interval -> stays within the bracketing samples' hull.
    from datetime import timedelta

    mid = eph.epochs[0] + timedelta(seconds=60)
    r_mid, _ = eph.at(mid)
    lo = np.minimum(eph.r_eci_m[0], eph.r_eci_m[1])
    hi = np.maximum(eph.r_eci_m[0], eph.r_eci_m[1])
    assert np.all(r_mid >= lo - 5e3) and np.all(r_mid <= hi + 5e3), r_mid

    # Default-epoch OpmState = first sample; covariance falls back to default.
    opm = eph.to_opm()
    assert opm.epoch == eph.start
    assert opm.cov_6x6.shape == (6, 6)
    assert abs(np.sqrt(opm.cov_6x6[0, 0]) - DEFAULT_SIGMA_POS_M) < 1e-6
    assert any("no covariance" in w for w in opm.warnings)

    # Out-of-span request clamps and warns.
    opm_late = eph.to_opm(eph.stop + timedelta(minutes=30))
    assert any("outside the OEM span" in w for w in opm_late.warnings)

    print(
        f"[oem] N={eph.n}, span={(eph.stop - eph.start)}, |r0|={np.linalg.norm(eph.r_eci_m[0]) / 1e3:.1f} km"
    )
    print("[oem] PASS")
