"""Parse a TLE (two lines) + optional covariance into an OpmState.

We derive the Cartesian mean state by propagating the TLE to its own epoch via
the existing SGP4 wrapper (the single propagator, per the design rules). The
covariance, if supplied, is treated exactly like an OPM covariance: a 6x6 SI
matrix in either ECI or RTN. If omitted we fall back to the same documented
default sigma the OPM parser uses.
"""

from __future__ import annotations

import numpy as np

from acquisition_platform.ingest.opm import (
    DEFAULT_SIGMA_POS_M,
    DEFAULT_SIGMA_VEL_M_S,
    OpmState,
    _default_cov,
)
from antenna_pomdp.orbit.propagator import SGP4Propagator

# An optional title line is tolerated; we keep only the two element lines.
_RTN_FRAMES = {"RTN", "RSW", "RIC", "LVLH"}


def _extract_lines(text: str) -> tuple[str, str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    elem = [ln for ln in lines if ln.startswith(("1 ", "2 "))]
    if len(elem) < 2:
        raise ValueError("could not find two TLE element lines (starting '1 '/'2 ')")
    l1 = next(ln for ln in elem if ln.startswith("1 "))
    l2 = next(ln for ln in elem if ln.startswith("2 "))
    return l1, l2


def parse_tle(
    text: str,
    cov_6x6: np.ndarray | None = None,
    frame: str = "ECI",
) -> tuple[OpmState, str, str]:
    """Parse a TLE -> (OpmState, line1, line2).

    `cov_6x6` is an optional SI 6x6 covariance; `frame` is "ECI" or "RTN".
    Returns the two element lines too, since downstream code (the sampler) wants
    the nominal TLE strings, not just the Cartesian state.
    """
    l1, l2 = _extract_lines(text)
    prop = SGP4Propagator(l1, l2)
    state = prop.propagate(prop.epoch)

    warnings: list[str] = []
    fr = "ECI"
    if cov_6x6 is None:
        cov = _default_cov()
        warnings.append(
            "TLE had no covariance; using default isotropic sigma "
            f"({DEFAULT_SIGMA_POS_M:.0f} m pos, {DEFAULT_SIGMA_VEL_M_S:.1f} m/s vel)."
        )
    else:
        cov = np.asarray(cov_6x6, dtype=float)
        if cov.shape != (6, 6):
            raise ValueError(f"covariance must be 6x6, got {cov.shape}")
        fr = "RTN" if frame.upper() in _RTN_FRAMES else "ECI"

    opm = OpmState(
        epoch=prop.epoch,
        r_eci_m=state.r_eci_m,
        v_eci_m=state.v_eci_m_s,
        cov_6x6=cov,
        frame=fr,
        warnings=tuple(warnings),
    )
    return opm, l1, l2


SAMPLE_TLE = (
    "1 25544U 98067A   24001.00000000  .00012345  00000-0  22588-3 0  9990\n"
    "2 25544  51.6400 100.0000 0001000  90.0000 270.0000 15.50000000000010"
)


if __name__ == "__main__":
    opm, l1, l2 = parse_tle(SAMPLE_TLE)
    r = np.linalg.norm(opm.r_eci_m) / 1e3
    v = np.linalg.norm(opm.v_eci_m) / 1e3
    assert 6.5e3 < r < 8.0e3, r
    assert 6.5 < v < 8.5, v
    assert opm.frame == "ECI" and len(opm.warnings) == 1
    assert l1.startswith("1 ") and l2.startswith("2 ")

    # With an explicit RTN covariance.
    cov = np.diag([3000.0**2] * 3 + [3.0**2] * 3)
    opm2, _, _ = parse_tle(SAMPLE_TLE, cov_6x6=cov, frame="RTN")
    assert opm2.frame == "RTN" and len(opm2.warnings) == 0

    print(f"[tle] |r|={r:.1f} km, |v|={v:.3f} km/s, epoch={opm.epoch.isoformat()}")
    print("[tle] PASS")
