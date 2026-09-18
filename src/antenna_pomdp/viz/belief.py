"""Belief visualisation: particle scatter in (az, el) at a given epoch."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from antenna_pomdp.models.particle_filter import ParticleBelief, belief_azel
from antenna_pomdp.orbit.geometry import Station


def plot_belief_azel(
    belief: ParticleBelief,
    when: datetime,
    station: Station,
    ax=None,
    *,
    truth_azel: tuple[float, float] | None = None,
):
    """Scatter the particle cloud in (az, el) at time `when`.

    Returns the `matplotlib.axes.Axes`. If `ax` is None, a new figure is created.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    azel = belief_azel(belief, when, station)
    az_deg = np.rad2deg(azel[:, 0])
    el_deg = np.rad2deg(azel[:, 1])
    w = belief.weights
    ax.scatter(az_deg, el_deg, s=12.0 + 200.0 * w, alpha=0.5, label="particles")
    if truth_azel is not None:
        ax.scatter(
            [np.rad2deg(truth_azel[0])],
            [np.rad2deg(truth_azel[1])],
            marker="x",
            color="red",
            s=80,
            label="truth",
        )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title(f"Belief @ {when.isoformat(timespec='seconds')}")
    ax.legend(loc="best")
    return ax


if __name__ == "__main__":
    from datetime import timedelta

    import matplotlib

    matplotlib.use("Agg")

    from antenna_pomdp.config import default_config
    from antenna_pomdp.models.particle_filter import make_belief
    from antenna_pomdp.orbit.propagator import SGP4Propagator
    from antenna_pomdp.orbit.sampler import sample_particles

    cfg = default_config()
    rng = np.random.default_rng(0)
    nominal = SGP4Propagator(cfg.orbit.nominal_tle_line1, cfg.orbit.nominal_tle_line2)
    station = Station.from_config(cfg.station)
    belief = make_belief(sample_particles(cfg.orbit, n=50, rng=rng))
    ax = plot_belief_azel(belief, nominal.epoch + timedelta(minutes=5), station)
    assert ax is not None
    print("[viz.belief] PASS")
