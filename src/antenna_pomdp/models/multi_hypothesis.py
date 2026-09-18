"""Multi-object association using one particle belief per candidate orbit.

Each observation updates a candidate's log weight with its marginal likelihood,
L_k = sum_i w_i * p(obs | particle_i), evaluated before normalizing that
candidate's particle weights. The sub-belief then receives the same observation.

A clutter hypothesis has a flat likelihood. Hypothesis weights are normalized
with softmax, and candidates below the configured floor are pruned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from antenna_pomdp.config import FilterConfig
from antenna_pomdp.models.doppler import (
    FrequencyConfig,
    belief_doppler_hz,
    effective_sigma_hz,
    update_doppler,
)
from antenna_pomdp.models.particle_filter import ParticleBelief
from antenna_pomdp.orbit.geometry import Station

CLUTTER_ID = "h0_clutter"


@dataclass(frozen=True)
class HypothesisConfig:
    """Priors + resolution/pruning thresholds for the hypothesis mixture."""

    prior_own: float = 0.5  # prior mass on the operator's own candidate (index 0)
    prior_clutter: float = 0.02  # prior mass on the clutter hypothesis
    clutter_log_likelihood: float = -6.0  # flat per-obs log-likelihood for clutter
    resolved_threshold: float = 0.9  # top hypothesis probability to call it resolved
    resolved_margin: float = 3.0  # ... and top/second odds ratio must exceed this
    prune_below: float = 1e-3  # drop candidates below this posterior probability


@dataclass
class MultiHypothesisBelief:
    """K candidate sub-beliefs + a clutter hypothesis, with log hypothesis weights.

    `candidate_ids[k]` labels sub-belief `subbeliefs[k]`; the clutter hypothesis
    has no sub-belief and always sits at the LAST index of `hyp_log_weights`.
    Particle weights *inside* each sub-belief stay normalized (they localize that
    object's along-track); P(object = k) lives entirely in `hyp_log_weights`.
    """

    subbeliefs: list[ParticleBelief]
    candidate_ids: list[str]
    hyp_log_weights: np.ndarray  # shape (K + 1,), last entry = clutter
    cfg: HypothesisConfig = field(default_factory=HypothesisConfig)

    @property
    def n_candidates(self) -> int:
        return len(self.subbeliefs)

    def hypothesis_probs(self) -> np.ndarray:
        w = self.hyp_log_weights - self.hyp_log_weights.max()
        p = np.exp(w)
        return p / p.sum()

    def labels(self) -> list[str]:
        return [*self.candidate_ids, CLUTTER_ID]

    def top_hypothesis(self) -> tuple[str, float]:
        p = self.hypothesis_probs()
        i = int(np.argmax(p))
        return self.labels()[i], float(p[i])

    def hypothesis_entropy_bits(self) -> float:
        p = self.hypothesis_probs()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def resolved(self) -> bool:
        p = np.sort(self.hypothesis_probs())[::-1]
        if p[0] < self.cfg.resolved_threshold:
            return False
        second = p[1] if len(p) > 1 else 1e-9
        return (p[0] / max(second, 1e-12)) >= self.cfg.resolved_margin

    def candidate_prob(self, candidate_id: str) -> float:
        p = self.hypothesis_probs()
        return float(p[self.labels().index(candidate_id)])


def make_multi_hypothesis(
    subbeliefs: list[ParticleBelief],
    candidate_ids: list[str],
    cfg: HypothesisConfig | None = None,
) -> MultiHypothesisBelief:
    """Build a mixture with prior mass `prior_own` on candidate 0, the rest split."""
    cfg = cfg or HypothesisConfig()
    k = len(subbeliefs)
    priors = np.empty(k + 1)
    priors[-1] = cfg.prior_clutter
    if k == 1:
        priors[0] = 1.0 - cfg.prior_clutter
    else:
        priors[0] = cfg.prior_own
        priors[1:k] = (1.0 - cfg.prior_own - cfg.prior_clutter) / (k - 1)
    priors = np.clip(priors, 1e-12, None)
    return MultiHypothesisBelief(
        subbeliefs=list(subbeliefs),
        candidate_ids=list(candidate_ids),
        hyp_log_weights=np.log(priors / priors.sum()),
        cfg=cfg,
    )


def update_multi_doppler(
    mhb: MultiHypothesisBelief,
    measured_hz: float,
    when: datetime,
    station: Station,
    freq_cfg: FrequencyConfig,
    filter_cfg: FilterConfig,
    rng: np.random.Generator,
) -> MultiHypothesisBelief:
    """PDA update on a measured Doppler: score each candidate, advance, accumulate."""
    sigma = effective_sigma_hz(freq_cfg)
    new_logw = mhb.hyp_log_weights.copy()
    new_subs: list[ParticleBelief] = []
    for k, sub in enumerate(mhb.subbeliefs):
        pred = belief_doppler_hz(sub, when, station, freq_cfg)
        # Per-particle Gaussian Doppler likelihood, stabilized for the marginal.
        d = (pred - measured_hz) / sigma
        loglik = -0.5 * d * d
        w = np.asarray(sub.weights, dtype=float)
        m = float(loglik.max())
        # L_k = Σ_i w_i · exp(loglik_i)  — the candidate's MARGINAL likelihood,
        # computed from the PRIOR sub-belief weights (before the sub-belief is
        # renormalized by update_doppler below). This is the crux of PDA.
        marginal = float(np.dot(w, np.exp(loglik - m)))
        log_lk = m + np.log(max(marginal, 1e-300))
        new_logw[k] += log_lk
        new_subs.append(
            update_doppler(
                sub, measured_hz, when, station, freq_cfg, filter_cfg, rng, sigma_hz=sigma
            )
        )
    # Clutter hypothesis: a flat per-observation likelihood.
    new_logw[-1] += mhb.cfg.clutter_log_likelihood
    return MultiHypothesisBelief(
        subbeliefs=new_subs,
        candidate_ids=list(mhb.candidate_ids),
        hyp_log_weights=new_logw,
        cfg=mhb.cfg,
    )


def prune(mhb: MultiHypothesisBelief) -> MultiHypothesisBelief:
    """Drop candidates below the posterior floor; always keep candidate 0 + clutter."""
    p = mhb.hypothesis_probs()
    keep = [0]  # always keep the operator's own candidate
    for k in range(1, mhb.n_candidates):
        if p[k] >= mhb.cfg.prune_below:
            keep.append(k)
    if keep == list(range(mhb.n_candidates)):
        return mhb  # nothing to prune
    subs = [mhb.subbeliefs[k] for k in keep]
    ids = [mhb.candidate_ids[k] for k in keep]
    logw = np.array([mhb.hyp_log_weights[k] for k in keep] + [mhb.hyp_log_weights[-1]])
    return MultiHypothesisBelief(
        subbeliefs=subs, candidate_ids=ids, hyp_log_weights=logw, cfg=mhb.cfg
    )


def disambiguation_report(mhb: MultiHypothesisBelief) -> dict:
    """Operator-facing readout: per-candidate probability + resolution status."""
    p = mhb.hypothesis_probs()
    top_id, top_p = mhb.top_hypothesis()
    return {
        "resolved": mhb.resolved(),
        "top": {"id": top_id, "probability": top_p},
        "entropy_bits": mhb.hypothesis_entropy_bits(),
        "candidates": [
            {"id": cid, "probability": float(p[i])} for i, cid in enumerate(mhb.candidate_ids)
        ]
        + [{"id": CLUTTER_ID, "probability": float(p[-1])}],
    }


if __name__ == "__main__":
    from datetime import timedelta

    from antenna_pomdp.config import StationConfig, default_config
    from antenna_pomdp.models.doppler import particle_doppler_hz
    from antenna_pomdp.orbit.propagator import SGP4Propagator
    from antenna_pomdp.orbit.sampler import OrbitParticle

    cfg = default_config()
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(StationConfig())
    when = nominal.epoch + timedelta(minutes=5)
    fcfg = FrequencyConfig()
    fltr = FilterConfig()
    rng = np.random.default_rng(0)

    def cluster(center_s):
        parts = [OrbitParticle(nominal, float(center_s + s)) for s in np.linspace(-3, 3, 40)]
        return ParticleBelief(particles=parts, weights=np.full(40, 1.0 / 40))

    # Three along-track-separated candidates; ours is candidate "SAT-B" at +20 s.
    mhb = make_multi_hypothesis(
        [cluster(20.0), cluster(0.0), cluster(45.0)], ["SAT-B", "SAT-A", "SAT-C"]
    )
    truth = particle_doppler_hz(OrbitParticle(nominal, 20.0), when, station, fcfg)

    p0 = mhb.candidate_prob("SAT-B")
    for _ in range(4):
        mhb = update_multi_doppler(mhb, truth, when, station, fcfg, fltr, rng)
        mhb = prune(mhb)
    rep = disambiguation_report(mhb)
    assert rep["top"]["id"] == "SAT-B", rep
    assert mhb.candidate_prob("SAT-B") > p0
    print(f"[multi_hyp] resolved={rep['resolved']} top={rep['top']}")
    print("[multi_hyp] PASS")
