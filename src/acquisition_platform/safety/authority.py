"""Authority levels and evidence checks for controller operation.

L0 is advisory; L1 records shadow decisions; L2 permits supervised commands;
L3 permits autonomous commands. Promotion requires operator sign-off, a minimum
pass count, no envelope violations, and acceptable shadow pointing error.
These checks define permission state; this module does not actuate hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

L0_ADVISORY = "L0_ADVISORY"
L1_SHADOW = "L1_SHADOW"
L2_SUPERVISED = "L2_SUPERVISED"
L3_AUTONOMOUS = "L3_AUTONOMOUS"

_LADDER = (L0_ADVISORY, L1_SHADOW, L2_SUPERVISED, L3_AUTONOMOUS)
_RANK = {level: i for i, level in enumerate(_LADDER)}

_META = {
    L0_ADVISORY: (
        "Advisory",
        "Recommend only — a human or the ACU executes. Nothing auto-actuates.",
    ),
    L1_SHADOW: (
        "Shadow",
        "Runs closed-loop alongside live ops; logs what it would do, commands nothing.",
    ),
    L2_SUPERVISED: (
        "Supervised closed-loop",
        "Commands the antenna with an operator watching and able to abort.",
    ),
    L3_AUTONOMOUS: ("Autonomous", "Machine-to-machine; exception-only human involvement."),
}


def is_level(level: str) -> bool:
    return level in _RANK


def rank(level: str) -> int:
    return _RANK[level]


def can_actuate(level: str) -> bool:
    """Does this level command the antenna (vs recommend/shadow only)?"""
    return _RANK[level] >= _RANK[L2_SUPERVISED]


def label(level: str) -> str:
    return _META[level][0]


def description(level: str) -> str:
    return _META[level][1]


def next_level(level: str) -> str | None:
    i = _RANK[level]
    return _LADDER[i + 1] if i + 1 < len(_LADDER) else None


@dataclass(frozen=True)
class PromotionConfig:
    """Evidence gate to advance one authority level."""

    min_passes: int = 50  # passes at the current level before eligible
    max_pointing_delta_frac_hpbw: float = 0.5  # 95th-pct |cand-incumbent| < ½ HPBW
    require_operator_signoff: bool = True


@dataclass(frozen=True)
class PromotionEvidence:
    """The record checked against the gate for a promotion decision."""

    passes_completed: int
    envelope_violations: int
    p95_pointing_delta_deg: float
    hpbw_deg: float
    operator_signoff: bool = False


def can_promote(level: str, ev: PromotionEvidence, cfg: PromotionConfig) -> tuple[bool, str]:
    """Return (eligible, human-readable reason) for advancing from `level`."""
    if next_level(level) is None:
        return False, "already at the top authority level"
    if ev.passes_completed < cfg.min_passes:
        return (
            False,
            f"need {cfg.min_passes} passes at {label(level)}; {ev.passes_completed} completed",
        )
    if ev.envelope_violations > 0:
        return False, f"{ev.envelope_violations} safety-envelope violation(s) in the sample"
    limit = cfg.max_pointing_delta_frac_hpbw * ev.hpbw_deg
    if ev.p95_pointing_delta_deg > limit:
        return (
            False,
            f"95th-pct shadow pointing delta {ev.p95_pointing_delta_deg:.2f}° "
            f"exceeds {limit:.2f}° (½ HPBW)",
        )
    if cfg.require_operator_signoff and not ev.operator_signoff:
        return False, "awaiting operator sign-off"
    return True, f"eligible to promote to {label(next_level(level))}"


if __name__ == "__main__":
    assert can_actuate(L2_SUPERVISED) and not can_actuate(L1_SHADOW)
    assert rank(L3_AUTONOMOUS) > rank(L0_ADVISORY)
    cfg = PromotionConfig()
    # Not enough passes.
    ok, why = can_promote(L1_SHADOW, PromotionEvidence(10, 0, 0.1, 10.0, True), cfg)
    assert not ok and "need 50" in why, why
    # A violation blocks promotion.
    ok, why = can_promote(L1_SHADOW, PromotionEvidence(60, 1, 0.1, 10.0, True), cfg)
    assert not ok and "violation" in why, why
    # Too-large pointing delta blocks it.
    ok, why = can_promote(L1_SHADOW, PromotionEvidence(60, 0, 9.0, 10.0, True), cfg)
    assert not ok, why
    # Clean sample + sign-off -> eligible.
    ok, why = can_promote(L1_SHADOW, PromotionEvidence(60, 0, 1.0, 10.0, True), cfg)
    assert ok, why
    print("[authority]", why)
    print("[authority] PASS")
