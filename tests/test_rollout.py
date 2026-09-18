"""Full-window value, feasibility, and incumbent safeguards for continuation MPC."""

from __future__ import annotations


import numpy as np
import pytest

from antenna_pomdp.control.rollout import continuation_beam_search


def test_tail_value_changes_the_short_horizon_choice() -> None:
    # The retained suffix already detects particle 0. Search particle 1 now,
    # although its immediate mass is smaller, to acquire both by the deadline.
    result = continuation_beam_search(
        np.array([0.6, 0.4]),
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        np.ones(2, dtype=bool),
        np.empty((0, 2, 2), dtype=bool),
        np.ones(2, dtype=bool),
        np.array([0.0, 1.0]),
        (0,),
        beam_width=40,
    )
    assert result.sequence == (1,)
    assert result.terminal_miss_mass == pytest.approx(0.0)
    assert result.incumbent_miss_mass == pytest.approx(0.4)
    assert result.improved


def test_suffix_boundary_rejects_a_high_value_unreachable_rejoin() -> None:
    result = continuation_beam_search(
        np.array([1.0]),
        np.array([[[0.2], [0.99]]]),
        np.ones(2, dtype=bool),
        np.empty((0, 2, 2), dtype=bool),
        np.array([True, False]),
        np.array([0.5]),
        (0,),
        beam_width=40,
    )
    assert result.sequence == (0,)
    assert result.terminal_miss_mass == pytest.approx(0.4)
    assert not result.improved


def test_dead_end_cannot_be_returned_as_a_shortened_prefix() -> None:
    transitions = np.array([[[True, False], [False, False]]])
    result = continuation_beam_search(
        np.array([1.0]),
        np.array([[[0.2], [0.99]], [[0.3], [0.9]]]),
        np.ones(2, dtype=bool),
        transitions,
        np.ones(2, dtype=bool),
        np.ones(1),
        (0, 0),
        beam_width=1,
    )
    assert result.sequence == (0, 0)
    assert result.terminal_miss_mass == pytest.approx(0.8 * 0.7)


def test_incumbent_survives_when_narrow_beam_prunes_its_better_path() -> None:
    # Width one prefers action 0's immediate gain. The retained incumbent's
    # lower-gain first action leads to a much better two-stage result.
    result = continuation_beam_search(
        np.array([0.6, 0.4]),
        np.array([[[1.0, 0.0], [0.0, 0.8]], [[0.0, 0.0], [1.0, 0.0]]]),
        np.ones(2, dtype=bool),
        np.array([np.eye(2, dtype=bool)]),
        np.ones(2, dtype=bool),
        np.ones(2),
        (1, 1),
        beam_width=1,
    )
    assert result.sequence == (1, 1)
    assert result.terminal_miss_mass == pytest.approx(0.08)
    assert not result.improved


def test_infeasible_incumbent_is_rejected_instead_of_claiming_a_safeguard() -> None:
    with pytest.raises(ValueError, match="incumbent.*feasible"):
        continuation_beam_search(
            np.ones(1),
            np.array([[[0.5]]]),
            np.ones(1, dtype=bool),
            np.empty((0, 1, 1), dtype=bool),
            np.zeros(1, dtype=bool),
            np.ones(1),
            (0,),
            beam_width=40,
        )


def test_zero_survival_mass_retains_the_exact_incumbent() -> None:
    result = continuation_beam_search(
        np.zeros(2),
        np.array([[[0.8, 0.1], [0.1, 0.7]]]),
        np.ones(2, dtype=bool),
        np.empty((0, 2, 2), dtype=bool),
        np.ones(2, dtype=bool),
        np.ones(2),
        (1,),
        beam_width=40,
    )
    assert result.sequence == (1,)
    assert result.terminal_miss_mass == 0.0
    assert not result.improved


def test_full_width_matches_exhaustive_feasible_prefix_value() -> None:
    import itertools

    rng = np.random.default_rng(1827)
    probabilities = rng.uniform(0.0, 0.8, (3, 3, 4))
    mass = np.array([0.1, 0.2, 0.3, 0.4])
    tail = np.array([0.8, 0.5, 0.2, 0.9])
    transitions = np.array(
        [
            [[True, True, False], [False, True, True], [True, False, True]],
            [[True, False, True], [True, True, False], [False, True, True]],
        ]
    )
    initial = np.array([True, True, False])
    terminal = np.array([True, False, True])
    values = []
    for sequence in itertools.product(range(3), repeat=3):
        if (
            initial[sequence[0]]
            and terminal[sequence[-1]]
            and all(transitions[t, sequence[t], sequence[t + 1]] for t in range(2))
        ):
            values.append(
                float(
                    np.sum(
                        mass * tail * np.prod(1.0 - probabilities[np.arange(3), sequence], axis=0)
                    )
                )
            )
    result = continuation_beam_search(
        mass, probabilities, initial, transitions, terminal, tail, (0, 0, 0), beam_width=27
    )
    assert result.terminal_miss_mass == pytest.approx(min(values))
