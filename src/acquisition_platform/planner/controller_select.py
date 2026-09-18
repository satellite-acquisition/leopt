"""Select a controller using a ground antenna's available slew per decision.

The heuristic compares slew rate times inter-dwell time with the estimated
angular search span. It chooses information-greedy or PFT-DPW; it is not a
validated policy-selection rule or an operational-performance guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ground-antenna mount presets: id -> (label, typical slew rate deg/s, note).
# Slew rates are representative S-band TT&C figures: phased arrays steer
# electronically (effectively instantaneous); small fast pedestals reach
# 10-20 deg/s; typical dishes 3-6 deg/s; large apertures 0.5-2 deg/s.
_MOUNT_PRESETS: dict[str, tuple[str, float, str]] = {
    "phased_array": (
        "Phased array (electronically steered)",
        1.0e9,
        "Beam is steered electronically, so re-pointing is effectively instantaneous.",
    ),
    "agile_pedestal": (
        "Agile pedestal (fast 3-axis)",
        15.0,
        "Fast mechanical pedestal that can re-point across a pass within a dwell.",
    ),
    "standard_dish": (
        "Standard S-band dish",
        5.0,
        "Typical mid-size TT&C az-el dish.",
    ),
    "large_dish": (
        "Large aperture dish",
        1.5,
        "Big-aperture, higher-gain dish; slower to slew.",
    ),
    "legacy_slow": (
        "Legacy / heavy mount",
        0.5,
        "Slow, heavy mount — successive dwells are strongly coupled.",
    ),
}

DEFAULT_MOUNT = "standard_dish"

# At/above this slew rate a mount is treated as agile regardless of geometry:
# a phased array or a very fast pedestal can always re-point across the belief.
AGILE_SLEW_DEG_S = 10.0

# Controller ids used by PlanSession.policy.
INFO_GREEDY = "infogreedy"
PFT_DPW = "pft"


@dataclass(frozen=True)
class GroundAntennaSpec:
    """Ground-antenna agility, the input to controller selection.

    `mount` selects a preset slew rate; an explicit `slew_rate_deg_s` overrides
    it. `agile` forces the phased-array (unconstrained) treatment. `fwhm_deg`
    and `dwell_s` describe the beam and the per-decision cadence used to size the
    re-point budget.
    """

    mount: str = DEFAULT_MOUNT
    slew_rate_deg_s: float | None = None
    agile: bool = False
    fwhm_deg: float = 10.0
    dwell_s: float = 5.0

    def effective_slew_deg_s(self) -> float:
        if self.agile:
            return 1.0e9
        if self.slew_rate_deg_s is not None:
            return float(self.slew_rate_deg_s)
        preset = _MOUNT_PRESETS.get(self.mount, _MOUNT_PRESETS[DEFAULT_MOUNT])
        return preset[1]

    def mount_label(self) -> str:
        return _MOUNT_PRESETS.get(self.mount, _MOUNT_PRESETS[DEFAULT_MOUNT])[0]


@dataclass(frozen=True)
class ControllerChoice:
    """The recommendation plus the numbers and prose that justify it."""

    policy: str  # "infogreedy" | "pft"
    controller_name: str
    is_agile: bool
    slew_rate_deg_s: float
    repoint_budget_deg: float  # how far the boresight can travel per decision
    coupling_span_deg: float  # how far the search must reach
    rationale: str
    basis: str  # scope and evidentiary basis for the prototype heuristic

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "controller_name": self.controller_name,
            "is_agile": self.is_agile,
            "slew_rate_deg_s": (
                None if self.slew_rate_deg_s >= 1.0e8 else round(self.slew_rate_deg_s, 3)
            ),
            "repoint_budget_deg": (
                None if self.repoint_budget_deg >= 1.0e8 else round(self.repoint_budget_deg, 2)
            ),
            "coupling_span_deg": round(self.coupling_span_deg, 2),
            "rationale": self.rationale,
            "basis": self.basis,
        }


_CONTROLLER_NAMES = {
    INFO_GREEDY: "One-step information-greedy",
    PFT_DPW: "PFT-DPW online planner",
}

_BASIS = (
    "Heuristic: compare the available slew per decision with an estimated search "
    "span. Controller selection has no optimality or operational-performance guarantee."
)


def recommend_controller(
    spec: GroundAntennaSpec,
    coupling_span_deg: float | None = None,
    inter_dwell_s: float | None = None,
) -> ControllerChoice:
    """Recommend a controller for this antenna.

    `coupling_span_deg` is the angular extent the search must cover (default: a
    few beamwidths, the typical early-LEOP belief width). `inter_dwell_s` is the
    time between decisions the boresight has to slew (default: the antenna dwell
    time).
    """
    slew = spec.effective_slew_deg_s()
    dt = float(inter_dwell_s if inter_dwell_s is not None else spec.dwell_s)
    # Belief is a few beamwidths wide early in LEOP; default the coupling span to
    # 4 FWHM (bounded so a pencil beam does not read as "already decoupled").
    span = float(
        coupling_span_deg if coupling_span_deg is not None else max(8.0, 4.0 * spec.fwhm_deg)
    )
    budget = slew * dt

    # Agile if electronically steered / very fast, OR if the per-decision
    # re-point budget comfortably (2x) exceeds the span the search must cover.
    is_agile = spec.agile or slew >= AGILE_SLEW_DEG_S or budget >= 2.0 * span

    if is_agile:
        policy = INFO_GREEDY
        if spec.agile or slew >= AGILE_SLEW_DEG_S:
            why = (
                f"{spec.mount_label()} re-points effectively freely "
                f"({'electronic steering' if slew >= 1.0e8 else f'{slew:.0f} deg/s'}), so the "
                "prototype treats successive dwell placement as effectively uncoupled "
                "and selects the lower-complexity information-greedy policy."
            )
        else:
            why = (
                f"The antenna can slew {budget:.0f} deg between decisions "
                f"({slew:.1f} deg/s x {dt:.0f} s), comfortably covering the ~{span:.0f} deg "
                "search span. The prototype therefore selects the lower-complexity "
                "information-greedy policy."
            )
    else:
        policy = PFT_DPW
        why = (
            f"{spec.mount_label()} can only slew {budget:.1f} deg between decisions "
            f"({slew:.1f} deg/s x {dt:.0f} s) against a ~{span:.0f} deg search span, so "
            "successive dwells may be coupled. The prototype selects PFT-DPW to explore "
            "that route-planning regime; no validated performance advantage is implied."
        )

    return ControllerChoice(
        policy=policy,
        controller_name=_CONTROLLER_NAMES[policy],
        is_agile=is_agile,
        slew_rate_deg_s=slew,
        repoint_budget_deg=budget,
        coupling_span_deg=span,
        rationale=why,
        basis=_BASIS,
    )


def list_mounts() -> list[dict]:
    """Registry metadata for the frontend mount picker."""
    return [
        {
            "id": mid,
            "name": label,
            "slew_rate_deg_s": (None if slew >= 1.0e8 else slew),
            "agile": slew >= AGILE_SLEW_DEG_S,
            "description": note,
        }
        for mid, (label, slew, note) in _MOUNT_PRESETS.items()
    ]


if __name__ == "__main__":
    # Phased array -> info-greedy (electronic steering; always agile).
    pa = recommend_controller(GroundAntennaSpec(mount="phased_array", fwhm_deg=2.0))
    assert pa.policy == INFO_GREEDY and pa.is_agile, pa
    # Slow legacy dish, pencil beam, short re-point cadence -> coupled -> PFT.
    dish = recommend_controller(GroundAntennaSpec(mount="legacy_slow", fwhm_deg=1.0, dwell_s=5.0))
    assert dish.policy == PFT_DPW and not dish.is_agile, dish
    # A standard dish, re-pointing over a realistic ~40 s hold, decouples -> agile.
    std = recommend_controller(GroundAntennaSpec(mount="standard_dish"), inter_dwell_s=40.0)
    assert std.is_agile and std.policy == INFO_GREEDY, std
    # Explicit very-slow slew with a short cadence and pencil beam -> coupled -> PFT.
    slow = recommend_controller(GroundAntennaSpec(slew_rate_deg_s=0.2, fwhm_deg=1.0, dwell_s=5.0))
    assert slow.policy == PFT_DPW, slow
    # agile flag forces info-greedy regardless of mount.
    forced = recommend_controller(GroundAntennaSpec(mount="legacy_slow", agile=True))
    assert forced.policy == INFO_GREEDY, forced
    for m in list_mounts():
        assert "id" in m and "name" in m
    print("[controller_select] phased-array ->", pa.policy)
    print(
        "[controller_select] legacy dish  ->",
        dish.policy,
        f"(budget {dish.repoint_budget_deg:.2f} deg)",
    )
    print("[controller_select] std dish/40s ->", std.policy)
    print("[controller_select] PASS")
