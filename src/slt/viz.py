"""Shared plotting style and reusable panels.

Design parameters (palette, sequential ramp) follow the validated reference
palette; see CONSTITUTION.md 7b for the figure-quality rules.

Pure module: importing this never creates a figure or writes a file.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Categorical slots, in fixed order. Never cycled, never reordered. ---------
CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

# Single-hue sequential ramps, light -> dark. Used for loss magnitude.
_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_ORANGE_STEPS = [
    "#fde3d3", "#fbcdb2", "#f8b591", "#f49d72", "#f08557", "#eb6834",
    "#d95926", "#c04d1f", "#a54219", "#8a3614", "#702b10", "#57210c",
]

SEQ_BLUE = LinearSegmentedColormap.from_list("slt_blue", _BLUE_STEPS)
SEQ_ORANGE = LinearSegmentedColormap.from_list("slt_orange", _ORANGE_STEPS)

# For 3-D surfaces the lightest steps vanish into the page and the valley floor —
# the part we most want to read — loses its shading. Drop them.
SEQ_BLUE_SURFACE = LinearSegmentedColormap.from_list("slt_blue_surf", _BLUE_STEPS[2:])


@dataclass(frozen=True)
class Theme:
    mode: str
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    categorical: list

    @property
    def seq(self) -> LinearSegmentedColormap:
        # Light surface -> ramp runs light(low loss is *not* the point here);
        # we always render magnitude dark = large, so the ramp is used as-is on
        # light and reversed on dark so "large loss" stays the high-contrast end.
        return SEQ_BLUE if self.mode == "light" else SEQ_BLUE.reversed()


LIGHT = Theme(
    mode="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#7a7975",
    grid="#e4e3df",
    categorical=CATEGORICAL_LIGHT,
)
DARK = Theme(
    mode="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#8f8e85",
    grid="#33332f",
    categorical=CATEGORICAL_DARK,
)

THEMES = {"light": LIGHT, "dark": DARK}


def use_style(theme: Theme = LIGHT) -> None:
    """Apply the project rcParams. Call once at the top of a script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink_secondary,
            "axes.edgecolor": theme.grid,
            "axes.titlecolor": theme.ink,
            "xtick.color": theme.ink_secondary,
            "ytick.color": theme.ink_secondary,
            "xtick.labelcolor": theme.ink_secondary,
            "ytick.labelcolor": theme.ink_secondary,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "mathtext.fontset": "cm",
        }
    )


# --- Panels -------------------------------------------------------------------


def loss_contours(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    theme: Theme = LIGHT,
    log: bool = True,
    floor: float | None = None,
    n_levels: int = 12,
    line_levels: int = 8,
):
    """Filled contours of a 2-D loss surface, plus recessive contour lines.

    ``log=True`` renders ``log10(Z + floor)``, which is the only way the
    structure near a degenerate minimum is visible at all: on a linear scale a
    singular valley is a featureless flat region.  The floor used is recorded on
    the returned object so log entries can state it (CONSTITUTION 3c).
    """
    if log:
        if floor is None:
            positive = Z[Z > 0]
            floor = float(np.percentile(positive, 0.5)) if positive.size else 1e-12
        field = np.log10(Z + floor)
        label = r"$\log_{10}\,K(w)$"
    else:
        field = Z
        floor = 0.0
        label = r"$K(w)$"

    levels = np.linspace(field.min(), field.max(), n_levels + 1)
    cf = ax.contourf(X, Y, field, levels=levels, cmap=theme.seq, extend="neither")
    cf.set_edgecolor("face")  # kill the hairline seams between filled bands
    ax.contour(
        X, Y, field,
        levels=np.linspace(field.min(), field.max(), line_levels + 1),
        colors=theme.surface, linewidths=0.5, alpha=0.55,
    )
    cf._slt_floor = floor  # noqa: SLF001 - carried for logging
    cf._slt_label = label  # noqa: SLF001
    return cf


def add_colorbar(fig, mappable, ax, label: str, theme: Theme = LIGHT):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, color=theme.ink_secondary, fontsize=8)
    cb.ax.tick_params(labelsize=7, color=theme.grid, labelcolor=theme.ink_secondary)
    cb.outline.set_edgecolor(theme.grid)
    return cb


def mark_zero_set(ax, segments, *, theme: Theme = LIGHT, label: str | None = None):
    """Overlay the true-parameter set W0 as dashed lines in surface colour."""
    for i, (xs, ys) in enumerate(segments):
        ax.plot(
            xs, ys,
            color=theme.surface, linestyle=(0, (4, 3)), linewidth=2.2,
            solid_capstyle="round", zorder=4,
            label=label if (label and i == 0) else None,
        )


def annotate(ax, text: str, *, xy=(0.03, 0.97), theme: Theme = LIGHT,
             va="top", ha="left", **kw):
    """Takeaway annotation. Text wears ink tokens, never a series colour."""
    return ax.text(
        *xy, text, transform=ax.transAxes, va=va, ha=ha,
        fontsize=8, color=theme.ink_secondary, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.42", facecolor=theme.surface,
                  edgecolor=theme.grid, linewidth=0.8, alpha=0.92),
        zorder=6, **kw,
    )


def tidy(ax, *, theme: Theme = LIGHT, grid: str | None = None):
    if grid:
        ax.grid(True, axis=grid, color=theme.grid, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
    return ax
