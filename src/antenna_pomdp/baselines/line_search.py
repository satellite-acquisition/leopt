"""Scalar along-track search and a matched two-axis comparator.

make_line_search_policy evaluates a one-dimensional time-offset grid using
particle geometry. It maximizes correct-acquisition probability or expected
entropy reduction at the dwell midpoint. make_track_grid_search_policy also
varies cross-track offset; the schedule evaluator integrates selected dwells
on a finer temporal grid.

The scalar approximation depends on projected uncertainty, station geometry,
beamwidth, and mount limits. belief_track_coordinates returns each particle's
(tau, theta) projection onto the nominal-track polyline; anisotropy alone does
not establish that the one-dimensional approximation is adequate.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from antenna_pomdp.config import Config
from antenna_pomdp.models.observation import (
    Pointing,
    false_confirmation_probability,
    true_acquisition_probability,
)
from antenna_pomdp.models.particle_filter import ParticleBelief, belief_azel_range
from antenna_pomdp.orbit.geometry import Station, eci_to_azel
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.pomdp.environment import PointingEnv, TrackAction, slew_limited_pointing

# ---------------------------------------------------------------------------
# Vectorised az/el helpers
# ---------------------------------------------------------------------------


def _unit_vectors(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """(K, 3) ENU unit vectors for arrays of az/el (radians)."""
    cos_el = np.cos(el)
    return np.stack([np.sin(az) * cos_el, np.cos(az) * cos_el, np.sin(el)], axis=-1)


def _nominal_track_azel(
    nominal: SGP4Propagator,
    when: datetime,
    station: Station,
    taus_s: np.ndarray,
) -> np.ndarray:
    """(K, 2) az/el of the nominal track sampled at `when + tau` for each tau.

    Matches `action_to_pointing` with a zero cross-track offset: the pointing
    for action (tau, 0) is the nominal satellite position at `when + tau`, seen
    from the station at `when`.
    """
    out = np.empty((taus_s.size, 2))
    for k, tau in enumerate(taus_s):
        s = nominal.propagate(when + timedelta(seconds=float(tau)))
        az, el, _ = eci_to_azel(s.r_eci_m, when, station)
        out[k, 0] = az
        out[k, 1] = el
    return out


def _track_grid_azel(
    nominal: SGP4Propagator,
    when: datetime,
    station: Station,
    taus_s: np.ndarray,
    cross_track_deg: np.ndarray,
) -> np.ndarray:
    """Exact action-model pointings for a dense tau/cross-track product grid.

    The expensive orbit propagation is performed once per along-track offset;
    all cross-track levels reuse the local tangent, matching
    :func:`action_to_pointing` algebra without a Python loop over the full
    Cartesian grid.
    """
    track0 = _nominal_track_azel(nominal, when, station, taus_s)
    track1 = np.empty_like(track0)
    for k, tau in enumerate(taus_s):
        state = nominal.propagate(when + timedelta(seconds=float(tau) + 1.0))
        # Match action_to_pointing exactly: differentiate the time-of-validity
        # coordinate with respect to tau while holding the observer epoch fixed.
        az, el, _ = eci_to_azel(state.r_eci_m, when, station)
        track1[k] = (az, el)

    daz = (track1[:, 0] - track0[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
    delv = track1[:, 1] - track0[:, 1]
    tangent = np.stack([daz * np.cos(track0[:, 1]), delv], axis=1)
    norm = np.linalg.norm(tangent, axis=1)
    safe = np.where(norm > 1e-9, norm, 1.0)
    tangent = tangent / safe[:, None]
    tangent[norm <= 1e-9] = (0.0, 1.0)
    perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    offset = np.deg2rad(cross_track_deg)[None, :]
    az = track0[:, 0, None] + offset * perp[:, 0, None] / np.maximum(
        1e-6, np.cos(track0[:, 1, None])
    )
    el = track0[:, 1, None] + offset * perp[:, 1, None]
    az = az % (2.0 * np.pi)
    el = np.clip(el, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
    return np.stack([az, el], axis=-1).reshape(-1, 2)


# ---------------------------------------------------------------------------
# The one-axis reduction: per-particle track coordinates
# ---------------------------------------------------------------------------


def belief_track_coordinates(
    belief: ParticleBelief,
    when: datetime,
    station: Station,
    nominal: SGP4Propagator,
    *,
    tau_span_s: float = 180.0,
    tau_step_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project every particle onto the nominal track: (tau_i, theta_i) arrays.

    tau_i is the along-track time offset whose nominal-track pointing is
    angularly closest to particle i (parabolic refinement between grid nodes);
    theta_i is the residual off-track angle (radians, unsigned). The
    (tau, theta) cloud makes the one-axis structure measurable: its anisotropy
    ratio std(angular tau extent) / RMS(theta) is what justifies searching on
    the line. RMS is used because an unsigned distance does not have a
    physically meaningful signed mean to subtract.
    """
    taus = np.arange(-tau_span_s, tau_span_s + tau_step_s, tau_step_s)
    track = _nominal_track_azel(nominal, when, station, taus)
    track_u = _unit_vectors(track[:, 0], track[:, 1])  # (K, 3)

    azelr = belief_azel_range(belief, when, station)
    part_u = _unit_vectors(azelr[:, 0], azelr[:, 1])  # (N, 3)

    # (N, K) angular separations via the dot-product matrix.
    dots = np.clip(part_u @ track_u.T, -1.0, 1.0)
    seps = np.arccos(dots)
    j = np.argmin(seps, axis=1)

    # Parabolic refinement of the minimum over (j-1, j, j+1).
    tau_i = taus[j].astype(float)
    theta_i = seps[np.arange(len(j)), j]
    interior = (j > 0) & (j < taus.size - 1)
    if np.any(interior):
        ji = j[interior]
        rows = np.arange(len(j))[interior]
        y0 = seps[rows, ji - 1]
        y1 = seps[rows, ji]
        y2 = seps[rows, ji + 1]
        denom = y0 - 2.0 * y1 + y2
        ok = np.abs(denom) > 1e-15
        shift = np.zeros_like(y1)
        shift[ok] = 0.5 * (y0[ok] - y2[ok]) / denom[ok]
        shift = np.clip(shift, -1.0, 1.0)
        tau_i[interior] = taus[ji] + shift * tau_step_s
        # Refined minimum value of the parabola.
        theta_i[interior] = y1 - 0.25 * (y0 - y2) * shift
    return tau_i, np.maximum(theta_i, 0.0)


def belief_anisotropy(
    belief: ParticleBelief,
    when: datetime,
    station: Station,
    nominal: SGP4Propagator,
) -> tuple[float, float, float]:
    """(sigma_along_rad, rms_offtrack_rad, ratio) of the projected belief cloud.

    sigma_along is the weighted std of the *angular* along-track extent
    (tau_i mapped through the local track rate), while rms_offtrack is the
    weighted root-mean-square residual angle from the nominal track. The ratio
    >> 1 is the one-axis premise, quantified.
    """
    tau_i, theta_i = belief_track_coordinates(belief, when, station, nominal)
    w = np.asarray(belief.weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(belief.n, 1.0 / belief.n)

    # Local angular rate of the track (rad of look angle per second of tau).
    track = _nominal_track_azel(nominal, when, station, np.array([-1.0, 1.0]))
    u = _unit_vectors(track[:, 0], track[:, 1])
    rate = float(np.arccos(np.clip(u[0] @ u[1], -1.0, 1.0))) / 2.0  # rad/s

    mean_tau = float(np.dot(w, tau_i))
    sigma_along = float(np.sqrt(np.dot(w, (tau_i - mean_tau) ** 2))) * rate
    rms_offtrack = float(np.sqrt(np.dot(w, theta_i**2)))
    ratio = sigma_along / max(rms_offtrack, 1e-12)
    return sigma_along, rms_offtrack, ratio


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------


def make_line_search_policy(cfg: Config | None = None, *, n_tau: int | None = None):
    """Scalar line-search controller: info-greedy restricted to the tau axis.

    Same one-step objective as `make_infogreedy_policy` (reads
    `cfg.heuristic.greedy_objective`), but the candidate set is a dense 1-D
    grid of along-track offsets at zero cross-track, and the whole evaluation
    is one (n_tau, N) matrix computation. Slew-aware: candidate pointings are
    clamped to the reachable set before scoring, exactly like the 2-D rule.

    Returns a `PolicyFactory` matching the `leop_runner` contract.
    """

    def factory(_cfg: Config, rng: np.random.Generator):
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)
        antenna = _cfg.antenna
        objective = _cfg.heuristic.greedy_objective
        t_max = _cfg.pomcp.along_track_offset_max_s
        n_tau_resolved = (
            max(1, int(_cfg.pomcp.n_along_track_offsets)) if n_tau is None else max(1, int(n_tau))
        )
        taus = np.linspace(-t_max, t_max, n_tau_resolved)

        def act(belief: ParticleBelief, env: PointingEnv, when: datetime) -> TrackAction:
            # Score the exposure at its midpoint. The evaluator subsequently
            # uses 0.25-s (configurable) within-dwell quadrature, so this remains
            # an explicitly labeled one-point controller surrogate rather than
            # an exact integrated action optimizer.
            bore = getattr(env, "current_bore", None)
            max_move = np.deg2rad(antenna.max_slew_rate_deg_s) * env.dwell_time_s
            uses_endpoint_fallback = bore is not None and max_move < np.pi
            score_when = (
                when if uses_endpoint_fallback else when + timedelta(seconds=0.5 * env.dwell_time_s)
            )
            azelr = belief_azel_range(belief, score_when, station)
            w = np.asarray(belief.weights, dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.full(belief.n, 1.0 / belief.n)
            part_u = _unit_vectors(azelr[:, 0], azelr[:, 1])  # (N, 3)
            track = _nominal_track_azel(nominal, score_when, station, taus)  # (K, 2)

            # Slew clamp: replace unreachable candidates by their reachable
            # great-circle limit (same physics as heuristics.realized_pointing).
            if uses_endpoint_fallback:
                for k in range(track.shape[0]):
                    p = slew_limited_pointing(
                        bore[0], bore[1], Pointing(track[k, 0], track[k, 1]), max_move
                    )
                    track[k, 0] = p.az
                    track[k, 1] = p.el

            track_u = _unit_vectors(track[:, 0], track[:, 1])  # (K, 3)
            dots = np.clip(track_u @ part_u.T, -1.0, 1.0)  # (K, N)
            dpsi = np.arccos(dots)
            p_true = np.asarray(
                true_acquisition_probability(
                    dpsi,
                    antenna,
                    azelr[None, :, 1],
                    azelr[None, :, 2],
                ),
                dtype=float,
            )
            p_false = np.asarray(
                false_confirmation_probability(
                    dpsi,
                    antenna,
                    azelr[None, :, 1],
                    azelr[None, :, 2],
                ),
                dtype=float,
            )
            p_terminal = np.clip(p_true + p_false, 0.0, 1.0)
            p_acquire = p_true @ w

            if objective == "pdetect":
                # Mission success is correct spacecraft acquisition, not a
                # false terminal confirmation.
                best = int(np.argmax(p_acquire))
            else:
                # One-step expected information gain, vectorised over the grid.
                best = int(np.argmax(_expected_information_gain_rows(w, p_terminal)))
            return TrackAction(float(taus[best]), 0.0)

        return act

    return factory


def make_track_grid_search_policy(
    cfg: Config | None = None,
    *,
    n_tau: int = 33,
    n_cross: int = 9,
):
    """Two-axis posterior detection-greedy comparator on a dense track grid.

    This controller exists to test the validity boundary of the scalar
    reduction. It uses the same correct-acquisition objective and vectorized
    particle scoring as the line controller, but admits cross-track actions.
    It is not an online tree-search planner.
    """

    def factory(_cfg: Config, rng: np.random.Generator):
        del rng
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)
        antenna = _cfg.antenna
        taus = np.linspace(
            -_cfg.pomcp.along_track_offset_max_s,
            _cfg.pomcp.along_track_offset_max_s,
            n_tau,
        )
        crosses = np.linspace(
            -_cfg.pomcp.cross_track_offset_max_deg,
            _cfg.pomcp.cross_track_offset_max_deg,
            n_cross,
        )

        def act(belief: ParticleBelief, env: PointingEnv, when: datetime) -> TrackAction:
            bore = getattr(env, "current_bore", None)
            max_move = np.deg2rad(antenna.max_slew_rate_deg_s) * env.dwell_time_s
            uses_endpoint_fallback = bore is not None and max_move < np.pi
            score_when = (
                when if uses_endpoint_fallback else when + timedelta(seconds=0.5 * env.dwell_time_s)
            )
            azelr = belief_azel_range(belief, score_when, station)
            w = np.asarray(belief.weights, dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.full(belief.n, 1.0 / belief.n)
            part_u = _unit_vectors(azelr[:, 0], azelr[:, 1])
            pointings = _track_grid_azel(
                nominal,
                score_when,
                station,
                taus,
                crosses,
            )

            if uses_endpoint_fallback:
                for k in range(pointings.shape[0]):
                    p = slew_limited_pointing(
                        bore[0],
                        bore[1],
                        Pointing(pointings[k, 0], pointings[k, 1]),
                        max_move,
                    )
                    pointings[k] = (p.az, p.el)

            bore_u = _unit_vectors(pointings[:, 0], pointings[:, 1])
            dpsi = np.arccos(np.clip(bore_u @ part_u.T, -1.0, 1.0))
            p_true = np.asarray(
                true_acquisition_probability(
                    dpsi,
                    antenna,
                    azelr[None, :, 1],
                    azelr[None, :, 2],
                ),
                dtype=float,
            )
            best = int(np.argmax(p_true @ w))
            ti, ci = divmod(best, n_cross)
            return TrackAction(float(taus[ti]), float(crosses[ci]))

        return act

    return factory


def _entropy_rows(unnormalised: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each row after normalisation; 0 for zero rows."""
    s = unnormalised.sum(axis=1, keepdims=True)
    safe = np.where(s > 0.0, s, 1.0)
    p = unnormalised / safe
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0.0, np.log(p), 0.0)
    h = -np.sum(p * logp, axis=1)
    return np.where(s[:, 0] > 0.0, h, 0.0)


def _expected_information_gain_rows(
    weights: np.ndarray,
    detection_likelihoods: np.ndarray,
) -> np.ndarray:
    """Expected one-step entropy reduction for each binary-sensor action.

    ``detection_likelihoods[k, i]`` is ``P(Y=1 | X=i, a=k)``.
    Under a hard-FOV channel whose likelihood is constant inside and outside
    the field of view, this particle-level expression reduces exactly to
    Ho et al. (2021), Eq. (20), as covered by the line-search regression tests.
    """
    w = np.asarray(weights, dtype=float)
    likelihoods = np.asarray(detection_likelihoods, dtype=float)
    if likelihoods.ndim == 1:
        likelihoods = likelihoods[None, :]
    if w.ndim != 1 or likelihoods.ndim != 2 or likelihoods.shape[1] != w.size:
        raise ValueError("expected weights (N,) and detection_likelihoods (K, N)")

    total = float(w.sum())
    if total <= 0.0:
        w = np.full(w.size, 1.0 / w.size)
    else:
        w = w / total
    likelihoods = np.clip(likelihoods, 0.0, 1.0)

    p_detect = likelihoods @ w
    wd = w[None, :] * likelihoods
    wn = w[None, :] * (1.0 - likelihoods)
    expected_posterior_entropy = p_detect * _entropy_rows(wd) + (1.0 - p_detect) * _entropy_rows(wn)
    prior_entropy = _entropy_rows(w[None, :])[0]
    return prior_entropy - expected_posterior_entropy


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import time
    from dataclasses import replace

    from antenna_pomdp.config import default_config
    from antenna_pomdp.models.particle_filter import make_belief
    from antenna_pomdp.orbit.sampler import sample_leop_particles

    base = default_config()
    cfg = replace(
        base,
        filter=replace(base.filter, n_particles=300),
        pomcp=replace(base.pomcp, along_track_offset_max_s=120.0),
        antenna=replace(base.antenna, fwhm_deg=1.0),
    )
    rng = np.random.default_rng(0)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_leop_particles(cfg.orbit, cfg.leop, n=cfg.filter.n_particles, rng=rng)
    belief = make_belief(particles)
    when = nominal.epoch + timedelta(hours=9, minutes=30)

    sa, sc, ratio = belief_anisotropy(belief, when, station, nominal)
    print(
        f"[line_search] belief anisotropy: along={np.rad2deg(sa):.2f} deg, "
        f"cross={np.rad2deg(sc):.2f} deg, ratio={ratio:.1f}"
    )

    env = PointingEnv(
        nominal=nominal,
        truth=particles[0],
        station=station,
        pass_start=when,
        pass_duration_s=cfg.station.pass_duration_s,
        dwell_time_s=cfg.antenna.dwell_time_s,
        antenna_cfg=cfg.antenna,
        pomcp_cfg=cfg.pomcp,
    )

    act = make_line_search_policy(cfg)(cfg, rng)
    t0 = time.perf_counter()
    a = act(belief, env, when)
    dt_ms = (time.perf_counter() - t0) * 1e3
    assert isinstance(a, TrackAction)
    assert abs(a.cross_track_deg) < 1e-12
    t_max = cfg.pomcp.along_track_offset_max_s
    assert -t_max - 1e-9 <= a.along_track_s <= t_max + 1e-9, a
    print(f"[line_search] action: tau={a.along_track_s:.2f} s (decision {dt_ms:.1f} ms)")
    print("[line_search] PASS")
