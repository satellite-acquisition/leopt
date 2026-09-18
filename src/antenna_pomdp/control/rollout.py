"""Feasible prefix improvement against a retained full-window continuation.

Only a miss continues acquisition, so the suffix contributes a per-particle
survival likelihood. Multiplying the current survival mass by that likelihood
lets a short beam search evaluate complete-window acquisition without building
an observation tree. The retained incumbent supplies a model-level safeguard;
finite-width search is still approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuationResult:
    """A full-length feasible prefix and its complete-window miss mass."""

    sequence: tuple[int, ...]
    terminal_miss_mass: float
    incumbent_miss_mass: float
    improved: bool


def continuation_beam_search(
    survival_mass: np.ndarray,
    probability_tensor: np.ndarray,
    initial_feasible: np.ndarray,
    transition_feasible: np.ndarray,
    terminal_feasible: np.ndarray,
    tail_survival: np.ndarray,
    incumbent_sequence: tuple[int, ...],
    *,
    beam_width: int,
    improvement_tolerance: float = 1.0e-12,
) -> ContinuationResult:
    """Improve a prefix only if its feasible full continuation is better.

    ``probability_tensor`` is ``(H, actions, particles)``; the transition array
    is ``(H-1, actions, actions)``. ``terminal_feasible[a]`` checks the boundary
    from final prefix action ``a`` to the first retained suffix command (or is
    true for all actions when no suffix remains). ``tail_survival`` contains
    the product of miss likelihoods through that fixed suffix.

    The incumbent must be a feasible H-step prefix. Backward viability removes
    dead ends before beam pruning, so no truncated prefix can be returned.
    Prefixes are ranked using their captured mass after accounting for the
    fixed suffix; the safeguard compares the complete remaining-window value.
    The guarantee applies to the supplied planning model, not unseen outcomes.
    """
    mass = np.asarray(survival_mass, dtype=float)
    probabilities = np.asarray(probability_tensor, dtype=float)
    tail = np.asarray(tail_survival, dtype=float)
    if mass.ndim != 1 or probabilities.ndim != 3 or probabilities.shape[2] != mass.size:
        raise ValueError("probability_tensor must have shape (H, actions, particles)")
    horizon, actions, _ = probabilities.shape
    if horizon < 1 or actions < 1 or beam_width < 1:
        raise ValueError("horizon, action count, and beam_width must be positive")
    initial = np.asarray(initial_feasible, dtype=bool)
    transitions = np.asarray(transition_feasible, dtype=bool)
    terminal = np.asarray(terminal_feasible, dtype=bool)
    if initial.shape != (actions,) or terminal.shape != (actions,):
        raise ValueError("initial and terminal feasibility must have one entry per action")
    if transitions.shape != (horizon - 1, actions, actions):
        raise ValueError("transition_feasible has the wrong shape")
    if tail.shape != mass.shape:
        raise ValueError("tail_survival must have one entry per particle")
    if (
        not np.all(np.isfinite(mass))
        or np.any(mass < 0)
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.all(np.isfinite(tail))
        or np.any((tail < 0) | (tail > 1))
        or not np.isfinite(improvement_tolerance)
        or improvement_tolerance < 0
    ):
        raise ValueError("invalid survival mass, probabilities, or improvement tolerance")
    incumbent = tuple(int(action) for action in incumbent_sequence)
    if len(incumbent) != horizon or any(action < 0 or action >= actions for action in incumbent):
        raise ValueError("incumbent_sequence must be a full-length prefix")
    if (
        not initial[incumbent[0]]
        or not terminal[incumbent[-1]]
        or any(
            not transitions[stage, incumbent[stage], incumbent[stage + 1]]
            for stage in range(horizon - 1)
        )
    ):
        raise ValueError("the incumbent prefix and suffix boundary must be feasible")

    effective_mass = mass * tail
    incumbent_residual = effective_mass.copy()
    for stage, action in enumerate(incumbent):
        incumbent_residual *= 1.0 - probabilities[stage, action]
    incumbent_miss = float(incumbent_residual.sum())

    viable = np.empty((horizon, actions), dtype=bool)
    viable[-1] = terminal
    for stage in range(horizon - 2, -1, -1):
        viable[stage] = np.any(transitions[stage] & viable[stage + 1][None, :], axis=1)

    nodes = []
    for action in np.flatnonzero(initial & viable[0]):
        residual = effective_mass * (1.0 - probabilities[0, action])
        nodes.append((float(residual.sum()), (int(action),), residual))
    nodes.sort(key=lambda node: (node[0], node[1]))
    nodes = nodes[:beam_width]
    for stage in range(1, horizon):
        expanded = []
        for _, sequence, node_mass in nodes:
            for action in np.flatnonzero(transitions[stage - 1, sequence[-1]] & viable[stage]):
                residual = node_mass * (1.0 - probabilities[stage, action])
                expanded.append((float(residual.sum()), sequence + (int(action),), residual))
        expanded.sort(key=lambda node: (node[0], node[1]))
        nodes = expanded[:beam_width]
    # The validated incumbent implies at least one viable path at every depth.
    best_miss, best_sequence, _ = nodes[0]
    if best_miss < incumbent_miss - improvement_tolerance:
        return ContinuationResult(best_sequence, best_miss, incumbent_miss, True)
    return ContinuationResult(incumbent, incumbent_miss, incumbent_miss, False)
