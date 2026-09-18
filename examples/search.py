"""Plan three synthetic dwells, update after a miss, and replan."""

import numpy as np

from antenna_pomdp.control.bayesian_search import beam_search_action, posterior_after_miss
from antenna_pomdp.control.mount import MountLimits, transition_feasible


def main() -> None:
    # Three possible target directions at one azimuth, with a nonuniform prior.
    elevations = np.array([40.0, 44.0, 48.0])
    weights = np.array([0.2, 0.5, 0.3])
    radians = np.deg2rad(elevations)
    directions = np.column_stack((np.zeros(3), np.cos(radians), np.sin(radians)))
    limits = MountLimits(
        max_rate_rad_s=np.deg2rad(3.0),
        max_accel_rad_s2=np.deg2rad(2.0),
        settle_s=0.5,
        dwell_s=2.0,
    )
    interval_s = 6.0
    feasible = np.array(
        [
            [transition_feasible(a, b, limits, interval_s) for b in directions]
            for a in directions
        ]
    )

    # Rows are pointing actions; columns are target hypotheses.
    separation = elevations[:, None] - elevations[None, :]
    probability = 0.9 * np.exp(-0.5 * (separation / 2.0) ** 2)
    horizon = 3
    tensor = np.repeat(probability[None, :, :], horizon, axis=0)
    transitions = np.repeat(feasible[None, :, :], horizon - 1, axis=0)
    current = 1

    for label in ("Prior", "After one miss"):
        plan = beam_search_action(
            weights,
            tensor,
            feasible[current],
            transitions,
            horizon=horizon,
            beam_width=32,
        )
        print(f"{label}: weights = {np.round(weights, 3)}")
        print(f"  Planned elevations: {elevations[list(plan.sequence)]} degrees")
        print(f"  Acquisition probability: {plan.captured_mass:.1%}")
        current = plan.first_action
        weights = posterior_after_miss(weights, probability[current])


if __name__ == "__main__":
    main()
