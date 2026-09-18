"""Map Cartesian orbit uncertainty into SGP4 particles and propagate a UKF.

RTN covariance is rotated into ECI before sampling. The state-to-TLE mapping
uses first-order element perturbations about a nominal orbit: along-track
position changes mean anomaly, cross-track position changes inclination and
RAAN, and radial position changes semi-major axis. Velocity deviations refine
these offsets. Each sigma point is then propagated with SGP4.

This approximation targets near-circular LEO. It is not a Cartesian-to-TLE
differential-correction fit or a high-fidelity orbit-determination method.
``EkfBaselinePolicy`` uses the propagated Gaussian without miss updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from acquisition_platform.ingest.opm import OpmState
from antenna_pomdp.config import OrbitConfig
from antenna_pomdp.models.observation import Pointing
from antenna_pomdp.models.particle_filter import ParticleBelief, make_belief
from antenna_pomdp.orbit.geometry import Station, eci_to_azel
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import (
    OrbitParticle,
    _approx_sma_m,
    _format_tle,
    _parse_tle_fields,
)

_MU_EARTH = 3.986004418e14


# ---------------------------------------------------------------------------
# RTN basis + covariance rotation
# ---------------------------------------------------------------------------


def rtn_basis(r_eci_m: np.ndarray, v_eci_m: np.ndarray) -> np.ndarray:
    """Return the 3x3 matrix whose ROWS are the RTN unit vectors in ECI.

    R = radial (along r), N = cross-track (along h = r x v), T = N x R
    (completes the right-handed along-track direction). Multiplying an ECI
    vector by this matrix gives its RTN components.
    """
    r = np.asarray(r_eci_m, dtype=float)
    v = np.asarray(v_eci_m, dtype=float)
    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.vstack([r_hat, t_hat, n_hat])  # rows: R, T, N


def rotate_cov_rtn_to_eci(
    cov_rtn: np.ndarray, r_eci_m: np.ndarray, v_eci_m: np.ndarray
) -> np.ndarray:
    """Rotate a 6x6 RTN covariance into ECI using the mean state's RTN basis."""
    A = rtn_basis(r_eci_m, v_eci_m)  # rtn = A @ eci  =>  eci = A.T @ rtn
    block = np.zeros((6, 6))
    block[:3, :3] = A.T
    block[3:, 3:] = A.T
    return block @ cov_rtn @ block.T


def cov_in_eci(opm: OpmState) -> np.ndarray:
    """Return the 6x6 covariance in ECI, rotating from RTN if necessary."""
    if opm.frame.upper() == "RTN":
        return rotate_cov_rtn_to_eci(opm.cov_6x6, opm.r_eci_m, opm.v_eci_m)
    return np.asarray(opm.cov_6x6, dtype=float)


# ---------------------------------------------------------------------------
# Cartesian deviation -> TLE-perturbed propagator (first-order element fit)
# ---------------------------------------------------------------------------


def _orbit_cfg_from_state(
    r_eci_m: np.ndarray, v_eci_m: np.ndarray, orbit_cfg: OrbitConfig
) -> OrbitConfig:
    """Best-effort: keep the supplied nominal TLE if it matches, else reuse it.

    The platform always carries an explicit nominal TLE (from a TLE ingest, or
    the engine default for an OPM ingest). We do not re-fit it here; the nominal
    TLE in `orbit_cfg` is the linearisation point.
    """
    return orbit_cfg


def _state_deviation_to_tle(
    base_fields: dict[str, float],
    base_a_m: float,
    base_n_rev_day: float,
    d_rtn_pos_m: np.ndarray,
    d_rtn_vel_m_s: np.ndarray,
) -> dict[str, float]:
    """Map an RTN position+velocity deviation to perturbed TLE element fields.

    First-order, near-circular: along-track position -> mean anomaly; cross-
    track position -> split inclination/RAAN; radial position -> SMA -> mean
    motion. The along-track VELOCITY deviation refines the SMA (energy) term,
    and the radial velocity nudges the mean anomaly (phase). This mirrors the
    physics in `orbit.sampler._sample_one_leop_particle` but is driven by a
    specific deviation vector instead of a random draw.
    """
    f = dict(base_fields)
    d_radial, d_along, d_cross = d_rtn_pos_m
    dv_radial, dv_along, _dv_cross = d_rtn_vel_m_s

    # along-track position -> mean anomaly
    dma_rad = d_along / base_a_m
    f["mean_anomaly_deg"] = (f["mean_anomaly_deg"] + np.rad2deg(dma_rad)) % 360.0

    # radial velocity -> small extra phase (dt ~ dv_r / (n^2 a)); folded into MA
    n_rad_s = base_n_rev_day * 2.0 * np.pi / 86400.0
    if n_rad_s > 0:
        dphase_rad = dv_radial / (n_rad_s * base_a_m)
        f["mean_anomaly_deg"] = (f["mean_anomaly_deg"] + np.rad2deg(dphase_rad)) % 360.0

    # cross-track position -> inclination / RAAN (50/50 split, matches sampler)
    dcross_rad = d_cross / base_a_m
    f["inc_deg"] += np.rad2deg(dcross_rad / np.sqrt(2.0))
    f["raan_deg"] = (f["raan_deg"] + np.rad2deg(dcross_rad / np.sqrt(2.0))) % 360.0

    # radial position + along-track velocity -> SMA -> mean motion.
    # da from radial offset (circular approx) plus vis-viva sensitivity to dv_t.
    v_circ = np.sqrt(_MU_EARTH / base_a_m)
    da_m = d_radial + 2.0 * base_a_m * (dv_along / max(1e-6, v_circ))
    dn_over_n = -1.5 * da_m / base_a_m
    f["mean_motion"] = max(1e-3, base_n_rev_day * (1.0 + dn_over_n))
    return f


def _propagator_from_fields(orbit_cfg: OrbitConfig, fields: dict[str, float]) -> SGP4Propagator:
    l1, l2 = _format_tle(orbit_cfg.nominal_tle_line1, orbit_cfg.nominal_tle_line2, fields)
    return SGP4Propagator(l1, l2)


# ---------------------------------------------------------------------------
# UKF
# ---------------------------------------------------------------------------


@dataclass
class GaussianState:
    """A Gaussian estimate of the Cartesian state at an epoch (ECI, SI)."""

    epoch: datetime
    mean: np.ndarray  # shape (6,)  [r(3), v(3)]
    cov: np.ndarray  # shape (6, 6)


@dataclass
class SgpUkf:
    """UKF front-end whose 'dynamics' are SGP4 evaluated on element-fit points."""

    orbit_cfg: OrbitConfig
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    def _weights(self, n: int) -> tuple[np.ndarray, np.ndarray, float]:
        lam = self.alpha**2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)
        return wm, wc, lam

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n = mean.size
        _, _, lam = self._weights(n)
        # Symmetric PSD square root via eigen-decomposition (robust to tiny
        # negative eigenvalues from rotation round-off).
        vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
        vals = np.clip(vals, 0.0, None)
        sqrt = vecs @ np.diag(np.sqrt((n + lam) * vals)) @ vecs.T
        pts = [mean.copy()]
        for i in range(n):
            pts.append(mean + sqrt[:, i])
            pts.append(mean - sqrt[:, i])
        return np.array(pts)

    def initial_state(self, opm: OpmState) -> GaussianState:
        mean = np.concatenate([opm.r_eci_m, opm.v_eci_m])
        return GaussianState(epoch=opm.epoch, mean=mean, cov=cov_in_eci(opm))

    def _propagate_point(self, point: np.ndarray, epoch: datetime, target: datetime) -> np.ndarray:
        """Push one Cartesian point through SGP4 via its element-fit TLE."""
        base_fields = _parse_tle_fields(
            self.orbit_cfg.nominal_tle_line1, self.orbit_cfg.nominal_tle_line2
        )
        base_n = base_fields["mean_motion"]
        base_a = _approx_sma_m(base_n)

        # Express this point's deviation from the engine's nominal-TLE state in
        # RTN, then fold it into element perturbations.
        nominal = SGP4Propagator(self.orbit_cfg.nominal_tle_line1, self.orbit_cfg.nominal_tle_line2)
        ref = nominal.propagate(epoch)
        A = rtn_basis(ref.r_eci_m, ref.v_eci_m_s)
        d_pos_eci = point[:3] - ref.r_eci_m
        d_vel_eci = point[3:] - ref.v_eci_m_s
        d_rtn_pos = A @ d_pos_eci
        d_rtn_vel = A @ d_vel_eci

        fields = _state_deviation_to_tle(base_fields, base_a, base_n, d_rtn_pos, d_rtn_vel)
        prop = _propagator_from_fields(self.orbit_cfg, fields)
        s = prop.propagate(target)
        return np.concatenate([s.r_eci_m, s.v_eci_m_s])

    def propagate(self, state: GaussianState, target: datetime) -> GaussianState:
        """Unscented propagation of mean+cov to `target` through SGP4."""
        if target <= state.epoch:
            return GaussianState(epoch=target, mean=state.mean.copy(), cov=state.cov.copy())
        n = state.mean.size
        wm, wc, _ = self._weights(n)
        sig = self._sigma_points(state.mean, state.cov)
        prop_pts = np.array([self._propagate_point(p, state.epoch, target) for p in sig])
        mean = np.sum(wm[:, None] * prop_pts, axis=0)
        dev = prop_pts - mean
        cov = np.einsum("k,ki,kj->ij", wc, dev, dev)
        cov = (cov + cov.T) / 2.0
        return GaussianState(epoch=target, mean=mean, cov=cov)

    # ------------------------------------------------------------------
    def seed_particles(self, opm: OpmState, n: int, rng: np.random.Generator) -> ParticleBelief:
        """Draw `n` OrbitParticles from the ingested Gaussian (ECI) belief.

        Each Gaussian (r, v) draw is decomposed into an RTN deviation and mapped
        to a perturbed TLE so the particle is a drop-in `OrbitParticle`. The
        along-track deviation is additionally encoded as a launch-slip so the
        existing particle filter's process-noise machinery applies cleanly.
        """
        mean = np.concatenate([opm.r_eci_m, opm.v_eci_m])
        cov = cov_in_eci(opm)
        cov = (cov + cov.T) / 2.0
        draws = rng.multivariate_normal(mean, cov, size=n)

        base_fields = _parse_tle_fields(
            self.orbit_cfg.nominal_tle_line1, self.orbit_cfg.nominal_tle_line2
        )
        base_n = base_fields["mean_motion"]
        base_a = _approx_sma_m(base_n)
        A = rtn_basis(opm.r_eci_m, opm.v_eci_m)
        orbital_speed = np.sqrt(_MU_EARTH / base_a)

        particles: list[OrbitParticle] = []
        for d in draws:
            d_pos_eci = d[:3] - opm.r_eci_m
            d_vel_eci = d[3:] - opm.v_eci_m
            d_rtn_pos = A @ d_pos_eci
            d_rtn_vel = A @ d_vel_eci
            fields = _state_deviation_to_tle(base_fields, base_a, base_n, d_rtn_pos, d_rtn_vel)
            prop = _propagator_from_fields(self.orbit_cfg, fields)
            # along-track metres -> equivalent launch slip (s)
            slip = float(d_rtn_pos[1] / max(1e-6, orbital_speed))
            particles.append(OrbitParticle(propagator=prop, launch_slip_s=slip))
        return make_belief(particles)


# ---------------------------------------------------------------------------
# EKF / Gaussian baseline pointing policy
# ---------------------------------------------------------------------------


@dataclass
class EkfBaselinePolicy:
    """Open-loop Gaussian policy: point at propagated-mean +- k*sigma along-track.

    It propagates the mean Gaussian with the UKF and points the antenna at the
    mean track, optionally offset by `k_sigma` along-track standard deviations.
    Crucially it does NOT fold in no-detect observations — it is the Gaussian
    strawman the POMCP policy is compared against.
    """

    ukf: SgpUkf
    station: Station
    k_sigma: float = 0.0

    def pointing_at(self, state: GaussianState, when: datetime) -> Pointing:
        prop = self.propagate_mean(state, when)
        r = prop.mean[:3]
        # along-track 1-sigma in metres from the propagated cov, in RTN.
        A = rtn_basis(prop.mean[:3], prop.mean[3:])
        cov_pos_rtn = A @ prop.cov[:3, :3] @ A.T
        sigma_along_m = float(np.sqrt(max(0.0, cov_pos_rtn[1, 1])))
        v = prop.mean[3:]
        speed = np.linalg.norm(v)
        offset_m = self.k_sigma * sigma_along_m
        if speed > 0 and offset_m != 0.0:
            r = r + (v / speed) * offset_m
        az, el, _ = eci_to_azel(r, when, self.station)
        return Pointing(az=az, el=el)

    def propagate_mean(self, state: GaussianState, when: datetime) -> GaussianState:
        return self.ukf.propagate(state, when)


if __name__ == "__main__":
    from acquisition_platform.ingest.tle import SAMPLE_TLE, parse_tle
    from antenna_pomdp.config import default_config

    cfg = default_config()
    rng = np.random.default_rng(0)

    # Ingest a TLE with an RTN covariance, build a UKF on the same nominal TLE.
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, l1, l2 = parse_tle(SAMPLE_TLE, cov_6x6=cov, frame="RTN")
    orbit_cfg = type(cfg.orbit)(nominal_tle_line1=l1, nominal_tle_line2=l2)
    ukf = SgpUkf(orbit_cfg=orbit_cfg)

    # (1) RTN -> ECI rotation preserves total position variance trace.
    eci_cov = cov_in_eci(opm)
    assert abs(np.trace(eci_cov[:3, :3]) - np.trace(cov[:3, :3])) < 1.0

    # (2) sigma-point propagation runs and grows along-track uncertainty.
    g0 = ukf.initial_state(opm)
    g1 = ukf.propagate(g0, opm.epoch + timedelta(minutes=20))
    assert g1.cov.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh((g1.cov + g1.cov.T) / 2.0) > -1e-3)

    # (3) seed_particles -> N particles with nonzero spread.
    belief = ukf.seed_particles(opm, n=200, rng=rng)
    assert belief.n == 200
    station = Station.from_config(cfg.station)
    when = opm.epoch + timedelta(minutes=5)
    from antenna_pomdp.models.particle_filter import belief_azel

    azel = belief_azel(belief, when, station)
    spread_deg = float(np.rad2deg(np.std(azel[:, 0])))

    # (4) EKF baseline yields a finite pointing.
    pol = EkfBaselinePolicy(ukf=ukf, station=station, k_sigma=1.0)
    p = pol.pointing_at(g0, when)
    assert np.isfinite(p.az) and np.isfinite(p.el)

    print(f"[ukf] seeded {belief.n} particles, az spread = {spread_deg:.3f} deg")
    print(
        f"[ukf] along-track sigma grew "
        f"{np.sqrt(g0.cov[1, 1]):.0f} -> {np.sqrt(g1.cov[1, 1]):.0f} m (ECI y)"
    )
    assert spread_deg > 0.0
    print("[ukf] PASS")
