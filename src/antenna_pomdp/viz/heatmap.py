"""POMCP action-visitation heatmap on the (along-track, cross-track) grid."""

from __future__ import annotations

import numpy as np

from antenna_pomdp.config import PomcpConfig


def plot_action_visits(action_visits: np.ndarray, pomcp_cfg: PomcpConfig, ax=None):
    """2D heatmap of POMCP visit counts over the action grid.

    `action_visits` is expected to be flattened in (along-track outer, cross-track
    inner) order, matching `build_action_grid`.
    """
    import matplotlib.pyplot as plt

    n_T = max(1, pomcp_cfg.n_along_track_offsets)
    n_K = max(1, pomcp_cfg.n_cross_track_levels)
    grid = np.asarray(action_visits, dtype=float).reshape(n_T, n_K)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            -pomcp_cfg.along_track_offset_max_s,
            pomcp_cfg.along_track_offset_max_s,
            -pomcp_cfg.cross_track_offset_max_deg,
            pomcp_cfg.cross_track_offset_max_deg,
        ],
    )
    ax.set_xlabel("Along-track offset (s)")
    ax.set_ylabel("Cross-track offset (deg)")
    ax.set_title("POMCP root-action visit counts")
    plt.colorbar(im, ax=ax, label="visits")
    return ax


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    from antenna_pomdp.config import default_config

    cfg = default_config()
    n = max(1, cfg.pomcp.n_along_track_offsets) * max(1, cfg.pomcp.n_cross_track_levels)
    visits = np.random.default_rng(0).integers(0, 100, size=n)
    ax = plot_action_visits(visits, cfg.pomcp)
    assert ax is not None
    print("[viz.heatmap] PASS")
