"""Stone search-pattern recovery: per-condition heatmaps + belief scatter.

Three rows × two columns figure:
    row 0 - Uniform prior      (expected pattern: expanding ellipses)
    row 1 - Elongated prior    (expected pattern: along-track sweep arrows)
    row 2 - Post-detection     (expected pattern: tight spiral)

Left column  : cumulative POMCP root-action visit counts on the action grid,
               summed over all steps and normalised; viridis colormap.
Right column : belief particle scatter in the same (along_s, cross_deg)
               coordinates at the mid-episode reference step, coloured by
               weight.

Each heatmap is overlaid with a white schematic of the *expected* Stone
search pattern for that belief structure so that the qualitative match
between learned visitation and the classical heuristic is visible at a
glance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antenna_pomdp.config import PomcpConfig


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoneConditionResult:
    """All artifacts the figure needs for one belief condition."""

    label: str  # row title
    visit_grid: np.ndarray  # shape (n_T, n_K), normalised sum over steps
    particles_along_s: np.ndarray  # shape (N,) particle along-track coord (s)
    particles_cross_deg: np.ndarray  # shape (N,) particle cross-track coord (deg)
    particles_weight: np.ndarray  # shape (N,) particle weights (sum to 1)
    pattern: str  # "ellipses" | "arrows" | "spiral"


# ---------------------------------------------------------------------------
# Compact manuscript-figure style
# ---------------------------------------------------------------------------


def _set_manuscript_style() -> None:
    import matplotlib as mpl

    wong = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.prop_cycle": mpl.cycler(color=wong),
        }
    )


# ---------------------------------------------------------------------------
# Stone pattern overlays
# ---------------------------------------------------------------------------


def _overlay_ellipses(ax, extent: tuple[float, float, float, float]) -> None:
    """Expanding concentric ellipses centred on the origin."""
    from matplotlib.patches import Ellipse

    x_max = extent[1]
    y_max = extent[3]
    for k, frac in enumerate((0.3, 0.55, 0.85)):
        ax.add_patch(
            Ellipse(
                (0.0, 0.0),
                width=2.0 * frac * x_max,
                height=2.0 * frac * y_max,
                facecolor="none",
                edgecolor="white",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
            )
        )


def _overlay_arrows(ax, extent: tuple[float, float, float, float]) -> None:
    """Horizontal arrows along the along-track axis, at a few cross-track levels."""
    x_min, x_max, y_min, y_max = extent
    span = 0.78 * (x_max - x_min) / 2.0
    for y_frac in (-0.0,):
        y = y_frac * y_max
        ax.annotate(
            "",
            xy=(span, y),
            xytext=(-span, y),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.4, alpha=0.9),
        )


def _overlay_spiral(ax, extent: tuple[float, float, float, float]) -> None:
    """Tight Archimedean spiral around the origin (creeping line / square spiral)."""
    x_max = extent[1]
    y_max = extent[3]
    r_max = 0.55
    theta = np.linspace(0.0, 4.0 * np.pi, 240)
    r = r_max * theta / theta[-1]
    x = r * np.cos(theta) * x_max
    y = r * np.sin(theta) * y_max
    ax.plot(x, y, color="white", linewidth=1.1, linestyle="-", alpha=0.85)


_PATTERN_DISPATCH = {
    "ellipses": _overlay_ellipses,
    "arrows": _overlay_arrows,
    "spiral": _overlay_spiral,
}


# ---------------------------------------------------------------------------
# Main plotting entry point
# ---------------------------------------------------------------------------


def plot_stone_recovery(
    conditions: list[StoneConditionResult],
    pomcp_cfg: PomcpConfig,
    out_png,
    out_pdf=None,
) -> None:
    """Render the 3×2 stone-recovery figure to disk.

    Parameters
    ----------
    conditions : 3 ``StoneConditionResult`` instances in row order
        (uniform, elongated, post-detect).
    pomcp_cfg : the ``PomcpConfig`` used to build the action grid (used here
        only to set the heatmap extent).
    out_png : path for the PNG output. ``out_pdf`` optional.
    """
    import matplotlib.pyplot as plt

    _set_manuscript_style()

    n_rows = len(conditions)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(7.16, 2.4 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    extent = (
        -pomcp_cfg.along_track_offset_max_s,
        pomcp_cfg.along_track_offset_max_s,
        -pomcp_cfg.cross_track_offset_max_deg,
        pomcp_cfg.cross_track_offset_max_deg,
    )

    for row, cond in enumerate(conditions):
        # ---- left: visit heatmap -------------------------------------------
        ax_h = axes[row, 0]
        # grid shape (n_T, n_K); imshow expects (n_y, n_x) = (n_K, n_T).
        grid_img = np.asarray(cond.visit_grid, dtype=float).T
        # Normalise per-panel for visibility (cumulative sum may concentrate).
        vmax = float(grid_img.max()) if grid_img.max() > 0.0 else 1.0
        im = ax_h.imshow(
            grid_img,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=extent,
            vmin=0.0,
            vmax=vmax,
        )
        # Nominal track = cross-track = 0 line.
        ax_h.axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
        # Stone pattern schematic overlay.
        overlay = _PATTERN_DISPATCH.get(cond.pattern)
        if overlay is not None:
            overlay(ax_h, extent)
        ax_h.set_xlabel("Along-track offset (s)")
        ax_h.set_ylabel("Cross-track offset (deg)")
        ax_h.set_title(f"{cond.label}: POMCP visitation")
        fig.colorbar(im, ax=ax_h, label="normalised visits", fraction=0.046, pad=0.02)

        # ---- right: belief scatter -----------------------------------------
        ax_s = axes[row, 1]
        w = np.asarray(cond.particles_weight, dtype=float)
        w_norm = w / w.max() if w.max() > 0 else w
        sc = ax_s.scatter(
            cond.particles_along_s,
            cond.particles_cross_deg,
            c=w_norm,
            cmap="magma",
            s=8.0,
            edgecolors="none",
            alpha=0.85,
            vmin=0.0,
            vmax=1.0,
        )
        ax_s.axhline(0.0, color="0.4", linestyle="--", linewidth=0.6, alpha=0.7)
        ax_s.axvline(0.0, color="0.4", linestyle="--", linewidth=0.6, alpha=0.7)
        ax_s.set_xlim(extent[0], extent[1])
        ax_s.set_ylim(extent[2], extent[3])
        ax_s.set_xlabel("Along-track offset (s)")
        ax_s.set_ylabel("Cross-track offset (deg)")
        ax_s.set_title(f"{cond.label}: belief at reference step")
        ax_s.grid(True, alpha=0.3, linewidth=0.3)
        fig.colorbar(sc, ax=ax_s, label="weight (normalised)", fraction=0.046, pad=0.02)

    fig.suptitle(
        "Stone-pattern recovery: POMCP visitation vs classical optimal-search heuristic",
        fontsize=10,
        fontweight="bold",
    )

    fig.savefig(out_png)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


__all__ = ["StoneConditionResult", "plot_stone_recovery"]
