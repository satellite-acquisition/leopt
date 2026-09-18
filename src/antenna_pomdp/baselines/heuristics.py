"""Myopic pointing policies evaluated against the current particle belief.

InfoGreedy maximizes expected detection probability or one-step Shannon
information gain over the POMCP grid and sampled continuous actions. The
belief-centroid policy points at the weighted mean direction; MLS points at
the highest-weight particle.

Factories follow the PolicyFactory interface in eval.leop_runner. Angles are
radians internally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np

from antenna_pomdp.config import Config, default_config
from antenna_pomdp.models.observation import (
    Pointing,
    delta_theta_from_azel,
    detection_probability,
    true_acquisition_probability,
)
from antenna_pomdp.models.particle_filter import ParticleBelief, belief_azel_range
from antenna_pomdp.orbit.geometry import Station, angular_separation, azel_unit_vector
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.pomdp.environment import (
    PointingEnv,
    TrackAction,
    action_to_pointing,
    build_action_grid,
    slew_limited_pointing,
)

PolicyAct = Callable[[ParticleBelief, PointingEnv, datetime], TrackAction]
PolicyFactory = Callable[[Config, np.random.Generator], PolicyAct]


# ---------------------------------------------------------------------------
# Candidate action set
# ---------------------------------------------------------------------------


def build_candidate_actions(cfg: Config, rng: np.random.Generator) -> list[TrackAction]:
    """Discrete POMCP grid + uniformly-sampled continuous actions.

    The continuous actions are drawn uniformly over
    (Δτ ∈ ±along_track_offset_max_s, Δθ ∈ ±cross_track_offset_max_deg), so the
    myopic controllers are not confined to the hand-built grid the planner uses.
    """
    actions = list(build_action_grid(cfg.pomcp))
    n_extra = max(0, int(cfg.heuristic.n_candidate_actions))
    if n_extra:
        t_max = cfg.pomcp.along_track_offset_max_s
        k_max = cfg.pomcp.cross_track_offset_max_deg
        taus = rng.uniform(-t_max, t_max, size=n_extra)
        thetas = rng.uniform(-k_max, k_max, size=n_extra)
        actions.extend(TrackAction(float(t), float(k)) for t, k in zip(taus, thetas))
    return actions


# ---------------------------------------------------------------------------
# Belief helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(weights: np.ndarray) -> float:
    """Shannon entropy (nats) of a normalised weight vector."""
    w = weights[weights > 0.0]
    if w.size == 0:
        return 0.0
    return float(-np.sum(w * np.log(w)))


def _belief_mean_azel(azelr: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted-mean (az, el) via averaging unit vectors (az-wraparound safe)."""
    vecs = np.array([azel_unit_vector(az, el) for az, el in azelr[:, :2]])
    mean_vec = np.average(vecs, axis=0, weights=weights)
    norm = np.linalg.norm(mean_vec)
    if norm < 1e-12:
        # Degenerate (antipodal cancellation); fall back to max-weight particle.
        i = int(np.argmax(weights))
        return float(azelr[i, 0]), float(azelr[i, 1])
    u = mean_vec / norm
    el = float(np.arcsin(np.clip(u[2], -1.0, 1.0)))
    az = float(np.arctan2(u[0], u[1]) % (2.0 * np.pi))
    return az, el


def realized_pointing(
    action: TrackAction,
    nominal: SGP4Propagator,
    when: datetime,
    station: Station,
    env: PointingEnv,
) -> Pointing:
    """Commanded pointing, clamped to the slew-reachable set from the live
    boresight (`env.current_bore`). Falls back to the unconstrained pointing
    when there is no boresight yet or the slew rate is effectively infinite, so
    slew-aware controllers degrade to ordinary myopic ones in the legacy regime.
    """
    p = action_to_pointing(action, nominal, when, station)
    bore = getattr(env, "current_bore", None)
    if bore is None:
        return p
    max_move = np.deg2rad(env.antenna_cfg.max_slew_rate_deg_s) * env.dwell_time_s
    if max_move >= np.pi:
        return p
    return slew_limited_pointing(bore[0], bore[1], p, max_move)


def _dthetas_for_pointing(pointing: Pointing, azelr: np.ndarray) -> np.ndarray:
    """Per-particle off-boresight angle Δθ_i for a commanded pointing."""
    return np.array(
        [delta_theta_from_azel(pointing, azelr[i, 0], azelr[i, 1]) for i in range(azelr.shape[0])]
    )


# ---------------------------------------------------------------------------
# Closest-pointing controllers (belief centroid / MLS share the same core)
# ---------------------------------------------------------------------------


def _closest_action_to_azel(
    target_az: float,
    target_el: float,
    candidates: list[TrackAction],
    nominal: SGP4Propagator,
    when: datetime,
    station: Station,
) -> TrackAction:
    """Pick the candidate whose pointing minimises angular sep to (target_az, el)."""
    best_action = candidates[0]
    best_sep = np.inf
    for a in candidates:
        p = action_to_pointing(a, nominal, when, station)
        sep = angular_separation(p.az, p.el, target_az, target_el)
        if sep < best_sep:
            best_sep = sep
            best_action = a
    return best_action


# ---------------------------------------------------------------------------
# InfoGreedy
# ---------------------------------------------------------------------------


def make_infogreedy_policy(cfg: Config | None = None) -> PolicyFactory:
    """Myopic information-greedy controller.

    Reads `cfg.heuristic.greedy_objective` ("pdetect" or "infogain") to choose
    the per-action score. The `cfg` argument is accepted for signature parity
    with the other factories; the live config passed by the runner wins.
    """

    def factory(_cfg: Config, rng: np.random.Generator) -> PolicyAct:
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)
        objective = _cfg.heuristic.greedy_objective
        antenna = _cfg.antenna

        def act(belief: ParticleBelief, env: PointingEnv, when: datetime) -> TrackAction:
            candidates = build_candidate_actions(_cfg, rng)
            azelr = belief_azel_range(belief, when, station)
            w = np.asarray(belief.weights, dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.full(belief.n, 1.0 / belief.n)
            el = azelr[:, 1]
            rng_m = azelr[:, 2]
            h_prior = _shannon_entropy(w) if objective == "infogain" else 0.0

            best_action = candidates[0]
            best_score = -np.inf
            for a in candidates:
                # Evaluate the *reachable* pointing so the controller is
                # slew-aware (only the planner additionally routes its dwells).
                p = realized_pointing(a, nominal, when, station, env)
                dthetas = _dthetas_for_pointing(p, azelr)
                if objective == "pdetect":
                    p_true_i = np.asarray(
                        true_acquisition_probability(dthetas, antenna, el, rng_m),
                        dtype=float,
                    )
                    score = float(np.dot(w, p_true_i))
                else:
                    # One-step expected entropy reduction (Bayesian experimental
                    # design). Build both posteriors and weight by p_det.
                    pd_i = np.asarray(
                        detection_probability(dthetas, antenna, el, rng_m),
                        dtype=float,
                    )
                    p_det = float(np.dot(w, pd_i))
                    lik_det = pd_i
                    lik_nodet = 1.0 - pd_i
                    score = h_prior - _expected_posterior_entropy(w, lik_det, lik_nodet, p_det)

                if score > best_score:
                    best_score = score
                    best_action = a
            return best_action

        return act

    return factory


def _expected_posterior_entropy(
    w: np.ndarray,
    lik_det: np.ndarray,
    lik_nodet: np.ndarray,
    p_det: float,
) -> float:
    """p_det·H(w|detect) + (1-p_det)·H(w|no-detect) (nats)."""
    wd = w * lik_det
    sd = float(wd.sum())
    h_det = _shannon_entropy(wd / sd) if sd > 0.0 else 0.0
    wn = w * lik_nodet
    sn = float(wn.sum())
    h_nodet = _shannon_entropy(wn / sn) if sn > 0.0 else 0.0
    return p_det * h_det + (1.0 - p_det) * h_nodet


# ---------------------------------------------------------------------------
# Belief centroid — point at the posterior mean direction
# ---------------------------------------------------------------------------


def make_belief_centroid_policy(cfg: Config | None = None) -> PolicyFactory:
    """Certainty-equivalent controller pointing at the weighted mean direction.

    This is not QMDP: it does not evaluate fully observable MDP action values.
    """

    def factory(_cfg: Config, rng: np.random.Generator) -> PolicyAct:
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)

        def act(belief: ParticleBelief, env: PointingEnv, when: datetime) -> TrackAction:
            candidates = build_candidate_actions(_cfg, rng)
            azelr = belief_azel_range(belief, when, station)
            w = np.asarray(belief.weights, dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.full(belief.n, 1.0 / belief.n)
            az, el = _belief_mean_azel(azelr, w)
            return _closest_action_to_azel(az, el, candidates, nominal, when, station)

        return act

    return factory


def make_qmdp_policy(cfg: Config | None = None) -> PolicyFactory:
    """Deprecated compatibility alias for :func:`make_belief_centroid_policy`."""
    return make_belief_centroid_policy(cfg)


# ---------------------------------------------------------------------------
# MLS — point at most-likely (max-weight) particle
# ---------------------------------------------------------------------------


def make_mls_policy(cfg: Config | None = None) -> PolicyFactory:
    """Most-likely-state controller: point at the max-weight particle direction."""

    def factory(_cfg: Config, rng: np.random.Generator) -> PolicyAct:
        nominal = SGP4Propagator(_cfg.orbit.nominal_tle_line1, _cfg.orbit.nominal_tle_line2)
        station = Station.from_config(_cfg.station)

        def act(belief: ParticleBelief, env: PointingEnv, when: datetime) -> TrackAction:
            candidates = build_candidate_actions(_cfg, rng)
            azelr = belief_azel_range(belief, when, station)
            w = np.asarray(belief.weights, dtype=float)
            i = int(np.argmax(w))
            az, el = float(azelr[i, 0]), float(azelr[i, 1])
            return _closest_action_to_azel(az, el, candidates, nominal, when, station)

        return act

    return factory


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from dataclasses import replace
    from datetime import timedelta

    from antenna_pomdp.models.particle_filter import make_belief
    from antenna_pomdp.orbit.sampler import sample_particles

    base = default_config()
    cfg = replace(
        base,
        filter=replace(base.filter, n_particles=120),
        heuristic=replace(base.heuristic, n_candidate_actions=16),
    )
    rng = np.random.default_rng(0)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    particles = sample_particles(cfg.orbit, n=cfg.filter.n_particles, rng=rng)
    belief = make_belief(particles)
    when = nominal.epoch + timedelta(minutes=5)

    # A throwaway env (controllers only use it for parity; they read the belief).
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

    t_max = cfg.pomcp.along_track_offset_max_s
    k_max = cfg.pomcp.cross_track_offset_max_deg

    for name, maker in [
        (
            "infogain",
            lambda: replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective="infogain")),
        ),
        (
            "pdetect",
            lambda: replace(cfg, heuristic=replace(cfg.heuristic, greedy_objective="pdetect")),
        ),
    ]:
        c = maker()
        act = make_infogreedy_policy(c)(c, np.random.default_rng(1))
        a = act(belief, env, when)
        assert isinstance(a, TrackAction)
        assert -t_max - 1e-9 <= a.along_track_s <= t_max + 1e-9, a
        assert -k_max - 1e-9 <= a.cross_track_deg <= k_max + 1e-9, a
        print(
            f"[heuristics] infogreedy/{name}: a=({a.along_track_s:.2f}s, {a.cross_track_deg:.2f}deg)"
        )

    for name, mk in [
        ("belief_centroid", make_belief_centroid_policy),
        ("mls", make_mls_policy),
    ]:
        act = mk(cfg)(cfg, np.random.default_rng(2))
        a = act(belief, env, when)
        assert isinstance(a, TrackAction)
        assert -t_max - 1e-9 <= a.along_track_s <= t_max + 1e-9, a
        assert -k_max - 1e-9 <= a.cross_track_deg <= k_max + 1e-9, a
        print(f"[heuristics] {name}: a=({a.along_track_s:.2f}s, {a.cross_track_deg:.2f}deg)")

    # Mean-direction recovery: a concentrated belief yields a candidate
    # pointing close to the posterior mean direction.
    azelr = belief_azel_range(belief, when, station)
    az_t, el_t = _belief_mean_azel(azelr, np.full(belief.n, 1.0 / belief.n))
    centroid_action = make_belief_centroid_policy(cfg)(cfg, np.random.default_rng(3))(
        belief, env, when
    )
    qp = action_to_pointing(centroid_action, nominal, when, station)
    sep = np.rad2deg(angular_separation(qp.az, qp.el, az_t, el_t))
    print(f"[heuristics] belief centroid within {sep:.2f} deg of belief-mean direction")

    print("[heuristics] PASS")
