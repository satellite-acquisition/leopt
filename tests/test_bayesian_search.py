"""Theory-level invariants for finite-horizon Bayesian acquisition search."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from antenna_pomdp.control.bayesian_search import (
    advance_survival_mass,
    beam_search_action,
    exact_line_oracle,
    greedy_action,
    greedy_line_search,
    posterior_after_miss,
    recursive_acquisition_probability,
    sequence_acquisition_probability,
    set_acquisition_value,
)


def test_miss_update_and_odds_identity() -> None:
    weights = np.array([0.2, 0.3, 0.5])
    probability = np.array([0.8, 0.2, 0.5])
    posterior = posterior_after_miss(weights, probability)
    assert posterior.sum() == pytest.approx(1.0)
    expected_odds = weights[0] / weights[1] * (1.0 - probability[0]) / (1.0 - probability[1])
    assert posterior[0] / posterior[1] == pytest.approx(expected_odds)


def test_sequence_product_equals_recursive_bayes_and_incidence_sum() -> None:
    weights = np.array([0.15, 0.35, 0.5])
    probabilities = np.array([[0.8, 0.1, 0.2], [0.1, 0.7, 0.3], [0.2, 0.2, 0.6]])
    product_value = sequence_acquisition_probability(weights, probabilities)
    recursive_value = recursive_acquisition_probability(weights, probabilities)
    survival_mass = weights.copy()
    incidence = 0.0
    for probability in probabilities:
        incidence += float(survival_mass @ probability)
        survival_mass = advance_survival_mass(survival_mass, probability)
    assert recursive_value == pytest.approx(product_value)
    assert incidence == pytest.approx(product_value)


def test_horizon_one_mpc_is_exactly_greedy() -> None:
    mass = np.array([0.4, 0.6])
    actions = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.3]])
    feasible = np.array([True, True, False])
    greedy_index, greedy_value = greedy_action(mass, actions, feasible)
    result = beam_search_action(
        mass,
        actions[None, :, :],
        feasible,
        np.empty((0, 3, 3), dtype=bool),
        horizon=1,
        beam_width=1,
    )
    assert result.first_action == greedy_index
    assert result.captured_mass == greedy_value


def test_relaxed_acquisition_objective_is_submodular() -> None:
    weights = np.array([0.3, 0.7])
    actions = np.array([[0.8, 0.1], [0.2, 0.7], [0.5, 0.4]])
    for small, large, candidate in [((0,), (0, 1), 2), ((), (1,), 0)]:
        small_gain = set_acquisition_value(
            weights, actions, (*small, candidate)
        ) - set_acquisition_value(weights, actions, small)
        large_gain = set_acquisition_value(
            weights, actions, (*large, candidate)
        ) - set_acquisition_value(weights, actions, large)
        assert small_gain >= large_gain - 1.0e-15


def test_small_oracle_matches_brute_force_and_exposes_greedy_gap() -> None:
    probability = np.array([0.34, 0.33, 0.33])
    position = np.array([-1.0, 1.0, 2.0])
    oracle = exact_line_oracle(
        probability,
        position,
        start_position=0.0,
        maximum_step=1.0,
        horizon=2,
    )
    brute_values = []
    for sequence in itertools.product(range(3), repeat=2):
        locations = (0.0, float(position[sequence[0]]), float(position[sequence[1]]))
        if all(abs(right - left) <= 1.0 for left, right in zip(locations, locations[1:])):
            brute_values.append(sum(probability[action] for action in set(sequence)))
    greedy = greedy_line_search(
        probability,
        position,
        start_position=0.0,
        maximum_step=1.0,
        horizon=2,
    )
    assert oracle.probability == pytest.approx(max(brute_values))
    assert oracle.probability == pytest.approx(0.66)
    assert greedy.probability == pytest.approx(0.34)
