"""Safety layer: the hard envelope and the belief watchdog.

These sit *below* the controller. The planner may recommend any pointing, but
the envelope (`envelope.py`) is the authority on what the pedestal is allowed to
slew to or transmit through, and the watchdog (`watchdog.py`) is the authority
on whether the belief is healthy enough to keep the loop closed. An operator
flips autonomy on only once these guardrails exist.
"""

from acquisition_platform.safety.authority import (
    L0_ADVISORY,
    L1_SHADOW,
    L2_SUPERVISED,
    L3_AUTONOMOUS,
    PromotionConfig,
    PromotionEvidence,
    can_actuate,
    can_promote,
    description,
    is_level,
    label,
    next_level,
)
from acquisition_platform.safety.envelope import (
    EnvelopeVerdict,
    KeepOutBox,
    SafetyConfig,
    classify_pointing,
    evaluate_pointing,
    sun_azel,
)
from acquisition_platform.safety.watchdog import (
    CLOSED_LOOP,
    FALLBACK_OPEN_LOOP,
    WatchdogConfig,
    WatchdogState,
    watchdog_step,
)

__all__ = [
    "EnvelopeVerdict",
    "KeepOutBox",
    "SafetyConfig",
    "classify_pointing",
    "evaluate_pointing",
    "sun_azel",
    "WatchdogConfig",
    "WatchdogState",
    "watchdog_step",
    "CLOSED_LOOP",
    "FALLBACK_OPEN_LOOP",
    "L0_ADVISORY",
    "L1_SHADOW",
    "L2_SUPERVISED",
    "L3_AUTONOMOUS",
    "PromotionConfig",
    "PromotionEvidence",
    "can_actuate",
    "can_promote",
    "description",
    "is_level",
    "label",
    "next_level",
]
