"""Core algebra and finite-horizon search routines for Bayesian acquisition.

The only continuing observation is a miss.  We therefore carry unnormalized
``survival_mass`` rather than constructing an observation tree.  The sum of
that vector is the probability that the search has reached the current stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


def posterior_after_miss(weights: np.ndarray, detection_probability: np.ndarray) -> np.ndarray:
    """Return the normalized particle posterior after one non-detection."""
    weights_array = np.asarray(weights, dtype=float)
    probability = np.asarray(detection_probability, dtype=float)
    if weights_array.shape != probability.shape:
        raise ValueError("weights and detection_probability must have the same shape")
    residual = weights_array * (1.0 - np.clip(probability, 0.0, 1.0))
    normalizer = float(residual.sum())
    if normalizer <= 0.0:
        raise ValueError("the miss event has zero probability")
    return residual / normalizer


def advance_survival_mass(
    survival_mass: np.ndarray, detection_probability: np.ndarray
) -> np.ndarray:
    """Apply one miss likelihood without normalization."""
    mass = np.asarray(survival_mass, dtype=float)
    probability = np.asarray(detection_probability, dtype=float)
    if mass.shape != probability.shape:
        raise ValueError("survival_mass and detection_probability must have the same shape")
    return mass * (1.0 - np.clip(probability, 0.0, 1.0))


def sequence_acquisition_probability(
    weights: np.ndarray, detection_probabilities: np.ndarray
) -> float:
    """Probability of at least one acquisition along a fixed dwell sequence."""
    weights_array = np.asarray(weights, dtype=float)
    probabilities = np.asarray(detection_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != weights_array.size:
        raise ValueError("detection_probabilities must have shape (stages, particles)")
    normalized = weights_array / weights_array.sum()
    survival = np.prod(1.0 - np.clip(probabilities, 0.0, 1.0), axis=0)
    return float(1.0 - normalized @ survival)


def recursive_acquisition_probability(
    weights: np.ndarray, detection_probabilities: np.ndarray
) -> float:
    """Same probability computed through sequential Bayes updates."""
    posterior = np.asarray(weights, dtype=float)
    posterior = posterior / posterior.sum()
    survival_probability = 1.0
    acquisition_probability = 0.0
    for probability in np.asarray(detection_probabilities, dtype=float):
        step_probability = float(posterior @ probability)
        acquisition_probability += survival_probability * step_probability
        survival_probability *= 1.0 - step_probability
        if survival_probability <= 1.0e-15:
            break
        posterior = posterior_after_miss(posterior, probability)
    return float(acquisition_probability)


def greedy_action(
    survival_mass: np.ndarray,
    action_probabilities: np.ndarray,
    feasible: np.ndarray | None = None,
) -> tuple[int, float]:
    """Choose the feasible action with largest immediate captured mass."""
    mass = np.asarray(survival_mass, dtype=float)
    probabilities = np.asarray(action_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != mass.size:
        raise ValueError("action_probabilities must have shape (actions, particles)")
    scores = probabilities @ mass
    if feasible is not None:
        feasible_array = np.asarray(feasible, dtype=bool)
        if feasible_array.shape != (probabilities.shape[0],):
            raise ValueError("feasible must have one entry per action")
        scores = np.where(feasible_array, scores, -np.inf)
    if not np.isfinite(scores).any():
        raise ValueError("no feasible action")
    action = int(np.argmax(scores))
    return action, float(scores[action])


@dataclass(frozen=True)
class BeamSearchResult:
    """First action and planned survival-path prefix from finite-horizon search."""

    first_action: int
    sequence: tuple[int, ...]
    captured_mass: float


def beam_search_action(
    survival_mass: np.ndarray,
    probability_tensor: np.ndarray,
    initial_feasible: np.ndarray,
    transition_feasible: np.ndarray,
    *,
    horizon: int,
    beam_width: int,
) -> BeamSearchResult:
    """Maximize captured survival mass over a short feasible action sequence.

    ``probability_tensor`` has shape ``(available_stages, actions, particles)``.
    ``transition_feasible[h-1, a, b]`` states whether action ``b`` at local
    stage ``h`` is reachable after action ``a`` at local stage ``h-1``.
    Mechanics are hard constraints; no arbitrarily scaled motion penalty is
    mixed into acquisition probability.
    """
    mass = np.asarray(survival_mass, dtype=float)
    tensor = np.asarray(probability_tensor, dtype=float)
    if tensor.ndim != 3 or tensor.shape[2] != mass.size:
        raise ValueError("probability_tensor must have shape (stages, actions, particles)")
    resolved_horizon = min(max(1, int(horizon)), tensor.shape[0])
    first_feasible = np.asarray(initial_feasible, dtype=bool)
    if first_feasible.shape != (tensor.shape[1],):
        raise ValueError("initial_feasible must have one entry per action")
    transitions = np.asarray(transition_feasible, dtype=bool)
    expected_shape = (max(0, resolved_horizon - 1), tensor.shape[1], tensor.shape[1])
    if transitions.shape != expected_shape:
        raise ValueError(f"transition_feasible must have shape {expected_shape}")

    # The H=1 path deliberately delegates to the same implementation as the
    # public greedy controller; the equivalence is exact, not approximate.
    if resolved_horizon == 1:
        action, captured = greedy_action(mass, tensor[0], first_feasible)
        return BeamSearchResult(action, (action,), captured)

    initial_total = float(mass.sum())
    nodes: list[tuple[float, np.ndarray, int, tuple[int, ...]]] = []
    for action in np.flatnonzero(first_feasible):
        residual = advance_survival_mass(mass, tensor[0, action])
        captured = initial_total - float(residual.sum())
        nodes.append((captured, residual, int(action), (int(action),)))
    if not nodes:
        raise ValueError("no feasible first action")
    nodes.sort(key=lambda node: (-node[0], node[3]))
    nodes = nodes[: max(1, int(beam_width))]

    for stage in range(1, resolved_horizon):
        expanded: list[tuple[float, np.ndarray, int, tuple[int, ...]]] = []
        for _, node_mass, previous, sequence in nodes:
            for action in np.flatnonzero(transitions[stage - 1, previous]):
                residual = advance_survival_mass(node_mass, tensor[stage, action])
                captured = initial_total - float(residual.sum())
                expanded.append((captured, residual, int(action), sequence + (int(action),)))
        if not expanded:
            break
        expanded.sort(key=lambda node: (-node[0], node[3]))
        nodes = expanded[: max(1, int(beam_width))]

    best = min(nodes, key=lambda node: (-node[0], node[3]))
    return BeamSearchResult(best[3][0], best[3], float(best[0]))


def set_acquisition_value(
    weights: np.ndarray,
    action_probabilities: np.ndarray,
    selected_actions: Iterable[int],
) -> float:
    """Relaxed order-independent acquisition set function used in theory tests."""
    selected = tuple(dict.fromkeys(int(action) for action in selected_actions))
    if not selected:
        return 0.0
    return sequence_acquisition_probability(
        weights,
        np.asarray(action_probabilities)[list(selected)],
    )


@dataclass(frozen=True)
class OracleResult:
    """Exact perfect-cell line-search result for a small diagnostic instance."""

    probability: float
    sequence: tuple[int, ...]


def exact_line_oracle(
    cell_probability: np.ndarray,
    cell_position: np.ndarray,
    *,
    start_position: float,
    maximum_step: float,
    horizon: int,
) -> OracleResult:
    """Exact DP for perfect detection in disjoint cells on a line.

    A cell contributes its probability only on its first visit.  This small
    oracle is intentionally separate from the continuous MPC approximation and
    is used to quantify a known finite-horizon planning gap.
    """
    probability = np.asarray(cell_probability, dtype=float)
    position = np.asarray(cell_position, dtype=float)
    if probability.ndim != 1 or position.shape != probability.shape:
        raise ValueError("cell_probability and cell_position must be one-dimensional peers")
    if np.any(probability < 0.0) or probability.sum() > 1.0 + 1.0e-12:
        raise ValueError("cell probabilities must be nonnegative and sum to at most one")

    @lru_cache(maxsize=None)
    def solve(step: int, previous: int, visited_mask: int) -> tuple[float, tuple[int, ...]]:
        if step >= horizon:
            return 0.0, ()
        previous_position = start_position if previous < 0 else float(position[previous])
        best_value = 0.0
        best_sequence: tuple[int, ...] = ()
        for action in range(probability.size):
            if abs(float(position[action]) - previous_position) > maximum_step + 1.0e-12:
                continue
            new_bit = 1 << action
            gain = 0.0 if visited_mask & new_bit else float(probability[action])
            future, suffix = solve(step + 1, action, visited_mask | new_bit)
            candidate_value = gain + future
            candidate_sequence = (action,) + suffix
            if candidate_value > best_value + 1.0e-15 or (
                abs(candidate_value - best_value) <= 1.0e-15
                and (not best_sequence or candidate_sequence < best_sequence)
            ):
                best_value = candidate_value
                best_sequence = candidate_sequence
        return best_value, best_sequence

    value, sequence = solve(0, -1, 0)
    return OracleResult(float(value), sequence)


def greedy_line_search(
    cell_probability: np.ndarray,
    cell_position: np.ndarray,
    *,
    start_position: float,
    maximum_step: float,
    horizon: int,
) -> OracleResult:
    """Immediate-mass greedy policy for the same perfect-cell oracle model."""
    probability = np.asarray(cell_probability, dtype=float)
    position = np.asarray(cell_position, dtype=float)
    current = float(start_position)
    visited: set[int] = set()
    sequence: list[int] = []
    value = 0.0
    for _ in range(horizon):
        feasible = [
            action
            for action in range(probability.size)
            if abs(float(position[action]) - current) <= maximum_step + 1.0e-12
        ]
        if not feasible:
            break
        action = min(
            feasible,
            key=lambda candidate: (
                -(0.0 if candidate in visited else float(probability[candidate])),
                candidate,
            ),
        )
        if action not in visited:
            value += float(probability[action])
        visited.add(action)
        sequence.append(action)
        current = float(position[action])
    return OracleResult(float(value), tuple(sequence))
