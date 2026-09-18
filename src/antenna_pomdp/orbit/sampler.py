"""Sample orbit particles from independent Gaussian perturbations of TLE fields.

A separate launch-slip term shifts the propagation epoch backward in time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antenna_pomdp.config import OrbitConfig
from antenna_pomdp.orbit.propagator import SGP4Propagator


@dataclass(frozen=True)
class OrbitParticle:
    """One element of the particle prior over the true orbit.

    `launch_slip_s` shifts the *interpretation* of the TLE epoch (i.e. the
    satellite is `launch_slip_s` seconds ahead/behind nominal); the
    propagator itself is built from the perturbed TLE fields.
    """

    propagator: SGP4Propagator
    launch_slip_s: float
    weight: float = 1.0


def _parse_tle_fields(line1: str, line2: str) -> dict[str, float]:
    """Pull the fields we care about out of the standard TLE format."""
    return {
        "inc_deg": float(line2[8:16]),
        "raan_deg": float(line2[17:25]),
        "ecc": float("0." + line2[26:33]),
        "argp_deg": float(line2[34:42]),
        "mean_anomaly_deg": float(line2[43:51]),
        "mean_motion": float(line2[52:63]),
    }


def _format_tle(line1: str, line2: str, fields: dict[str, float]) -> tuple[str, str]:
    """Rewrite line 2 with perturbed fields. Line 1 is unchanged."""
    ecc_str = f"{fields['ecc']:.7f}"[2:9]  # drop "0."
    line2_new = (
        f"{line2[:8]}"
        f"{fields['inc_deg']:8.4f} "
        f"{fields['raan_deg']:8.4f} "
        f"{ecc_str} "
        f"{fields['argp_deg']:8.4f} "
        f"{fields['mean_anomaly_deg']:8.4f} "
        f"{fields['mean_motion']:11.8f}"
        f"{line2[63:]}"
    )
    # TLEs require a checksum; sgp4 tolerates a wrong one for parsing,
    # but to be safe we recompute it.
    line2_new = _fix_checksum(line2_new)
    return line1, line2_new


def _fix_checksum(line: str) -> str:
    body = line[:68]
    s = 0
    for ch in body:
        if ch.isdigit():
            s += int(ch)
        elif ch == "-":
            s += 1
    return body + str(s % 10)


def sample_particles(
    orbit_cfg: OrbitConfig,
    n: int,
    rng: np.random.Generator,
) -> list[OrbitParticle]:
    """Draw `n` IID particles from the Gaussian TLE-uncertainty prior."""
    base_fields = _parse_tle_fields(orbit_cfg.nominal_tle_line1, orbit_cfg.nominal_tle_line2)
    particles: list[OrbitParticle] = []
    for _ in range(n):
        f = dict(base_fields)
        f["inc_deg"] += rng.normal(0.0, orbit_cfg.sigma_inclination_deg)
        f["raan_deg"] = (f["raan_deg"] + rng.normal(0.0, orbit_cfg.sigma_raan_deg)) % 360.0
        f["ecc"] = max(0.0, f["ecc"] + rng.normal(0.0, orbit_cfg.sigma_eccentricity))
        f["argp_deg"] = (f["argp_deg"] + rng.normal(0.0, orbit_cfg.sigma_arg_perigee_deg)) % 360.0
        f["mean_anomaly_deg"] = (
            f["mean_anomaly_deg"] + rng.normal(0.0, orbit_cfg.sigma_mean_anomaly_deg)
        ) % 360.0
        f["mean_motion"] += rng.normal(0.0, orbit_cfg.sigma_mean_motion_rev_per_day)

        line1, line2 = _format_tle(orbit_cfg.nominal_tle_line1, orbit_cfg.nominal_tle_line2, f)
        prop = SGP4Propagator(line1, line2)
        slip = rng.normal(0.0, orbit_cfg.sigma_launch_slip_s)
        particles.append(OrbitParticle(propagator=prop, launch_slip_s=slip))
    return particles


def sample_truth(orbit_cfg: OrbitConfig, rng: np.random.Generator) -> OrbitParticle:
    """One sample serves as the simulated ground truth in each Monte Carlo trial."""
    return sample_particles(orbit_cfg, n=1, rng=rng)[0]


# ---------------------------------------------------------------------------
# LEOP-realistic insertion uncertainty (Electron-like)
# ---------------------------------------------------------------------------
#
# RTN (radial / along-track / cross-track) position sigmas at separation are
# converted to TLE element perturbations as follows (for a near-circular
# orbit of semi-major axis a, mean motion n):
#
#   along-track 1-sigma  ->  Δ mean anomaly  ~  σ_along / a    (rad)
#   cross-track 1-sigma  ->  Δ inclination  ~  σ_cross / a    (rad)
#                                            (half split with RAAN below)
#   radial      1-sigma  ->  Δ semi-major axis  ≈ σ_radial
#                          ->  Δ mean motion / n  =  -1.5 Δa / a
#
# These approximations are valid only to first order and only for circular
# orbits, but they preserve the qualitative scale of the prior covariance
# while keeping the particles compatible with the existing TLE-perturbation
# pipeline. The launch slip is drawn *uniformly* on [0, slip_max] (the
# rocket either launches on time or late, never early).


_EARTH_RADIUS_M = 6_378_137.0


def _approx_sma_m(mean_motion_rev_per_day: float) -> float:
    """Crude Keplerian SMA from mean motion (μ_earth = 3.986e14 m³/s²)."""
    n_rad_s = mean_motion_rev_per_day * 2.0 * np.pi / 86400.0
    return float((3.986004418e14 / n_rad_s**2) ** (1.0 / 3.0))


def _sample_one_leop_particle(
    base_fields: dict[str, float],
    base_a_m: float,
    base_n_rev_day: float,
    leop_cfg,
    orbit_cfg: OrbitConfig,
    rng: np.random.Generator,
    truth: bool,
) -> OrbitParticle:
    f = dict(base_fields)

    # ----- along-track -> mean-anomaly perturbation -----
    sigma_ma_rad = leop_cfg.sigma_along_track_m / base_a_m
    f["mean_anomaly_deg"] = (
        f["mean_anomaly_deg"] + np.rad2deg(rng.normal(0.0, sigma_ma_rad))
    ) % 360.0

    # ----- cross-track -> half to inclination, half to RAAN -----
    sigma_cross_rad = leop_cfg.sigma_cross_track_m / base_a_m
    # 50/50 split keeps the total cross-track covariance correct on average.
    sigma_inc_rad = sigma_cross_rad / np.sqrt(2.0)
    sigma_raan_rad = sigma_cross_rad / np.sqrt(2.0)
    f["inc_deg"] += np.rad2deg(rng.normal(0.0, sigma_inc_rad))
    f["raan_deg"] = (f["raan_deg"] + np.rad2deg(rng.normal(0.0, sigma_raan_rad))) % 360.0

    # ----- radial -> mean-motion perturbation -----
    sigma_a_m = leop_cfg.sigma_radial_m
    da_m = rng.normal(0.0, sigma_a_m)
    dn_over_n = -1.5 * da_m / base_a_m
    f["mean_motion"] = max(1e-3, base_n_rev_day * (1.0 + dn_over_n))

    line1, line2 = _format_tle(orbit_cfg.nominal_tle_line1, orbit_cfg.nominal_tle_line2, f)
    prop = SGP4Propagator(line1, line2)

    # ----- uniform launch slip on [0, slip_max] -----
    slip = float(rng.uniform(0.0, leop_cfg.launch_slip_max_s))
    return OrbitParticle(propagator=prop, launch_slip_s=slip)


def sample_leop_particles(
    orbit_cfg: OrbitConfig,
    leop_cfg,
    n: int,
    rng: np.random.Generator,
) -> list[OrbitParticle]:
    """Draw `n` particles from the Electron-like LEOP insertion prior.

    `leop_cfg` is expected to be an `antenna_pomdp.config.LeopConfig` (or any
    object exposing the same fields); accepted via duck-typing to keep this
    module free of a circular import.
    """
    base_fields = _parse_tle_fields(orbit_cfg.nominal_tle_line1, orbit_cfg.nominal_tle_line2)
    base_n = base_fields["mean_motion"]
    base_a_m = _approx_sma_m(base_n)
    return [
        _sample_one_leop_particle(
            base_fields, base_a_m, base_n, leop_cfg, orbit_cfg, rng, truth=False
        )
        for _ in range(n)
    ]


def sample_leop_truth(
    orbit_cfg: OrbitConfig,
    leop_cfg,
    rng: np.random.Generator,
) -> OrbitParticle:
    """Single-sample variant of `sample_leop_particles` for the simulated truth."""
    return sample_leop_particles(orbit_cfg, leop_cfg, n=1, rng=rng)[0]


if __name__ == "__main__":
    from datetime import timedelta

    from antenna_pomdp.config import default_config

    cfg = default_config()
    rng = np.random.default_rng(0)
    parts = sample_particles(cfg.orbit, n=64, rng=rng)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    when = nominal.epoch + timedelta(minutes=10)
    r_nom = nominal.propagate(when).r_eci_m
    spread = np.std([np.linalg.norm(p.propagator.propagate(when).r_eci_m - r_nom) for p in parts])
    assert spread > 0.0, "particles collapsed onto the nominal"
    print(f"[sampler] {len(parts)} particles, 1-sigma position spread = {spread / 1e3:.2f} km")
    print("[sampler] PASS")
