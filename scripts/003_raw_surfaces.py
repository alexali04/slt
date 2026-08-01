"""003 — DRAFT (raw figures, CONSTITUTION 8a) — 3-D loss surfaces, unannotated.

Just the surfaces. One file per 2-D model, each showing K(w) under two
z-scalings, plus a contact sheet and a panel of the 1-D models.

No annotation boxes, no overlays, no derived quantities: rule 8a. Anything done
to make the surface visible (log scaling, z-clipping) is stated in the axis
label, since there is nowhere else to put it.

Produces:
  figures/raw/003_contact_sheet.png
  figures/raw/003_1d_curves.png
  figures/raw/003_surface_<key>.png      (one per 2-D model)

See logs/003-raw-surfaces.md.
"""

import _bootstrap  # noqa: F401  (path side effect)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from _bootstrap import save

from slt import models as M
from slt import viz as V

THEME = V.LIGHT
V.use_style(THEME)

# Every 2-D model in the zoo, in the order they were introduced.
SURFACES = [
    M.LINEAR_2D,
    M.LINEAR_2D_ILLCOND,
    M.LINEAR_2D_COLLINEAR,
    M.PRODUCT,
    M.monomial(2, 1),
    M.monomial(2, 2),
    M.monomial(3, 1),
    M.TANH_1D,
]

LOG_FLOOR = 1e-8  # only used for the log-z rendering; stated in every axis label


def _style_3d(ax):
    ax.set_facecolor(THEME.surface)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(THEME.surface)
        axis.pane.set_edgecolor(THEME.grid)
        axis.pane.set_alpha(1.0)
        axis.line.set_color(THEME.grid)
        axis._axinfo["grid"]["color"] = THEME.grid
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors=THEME.ink_secondary, labelsize=7, pad=-1)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(r"$w_1$", labelpad=-4)
    ax.set_ylabel(r"$w_2$", labelpad=-4)


def surface(ax, ls, *, log: bool, n: int = 220, elev: float = 30, azim: float = -125):
    """One 3-D surface of K. ``log`` switches the z-axis to log10(K + floor)."""
    X, Y, Z = ls.grid(n)
    if log:
        field = np.log10(Z + LOG_FLOOR)
        zlabel = rf"$\log_{{10}}(K + 10^{{{int(np.log10(LOG_FLOOR))}}})$"
    else:
        field = Z
        zlabel = r"$K(w)$"

    ax.plot_surface(
        X, Y, field,
        cmap=V.SEQ_BLUE_SURFACE, linewidth=0, antialiased=True,
        rcount=n, ccount=n, shade=True,
    )
    _style_3d(ax)
    ax.set_zlabel(zlabel, labelpad=-2, fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    return field


def per_model_files():
    for ls in SURFACES:
        fig = plt.figure(figsize=(11.0, 4.8))
        for i, log in enumerate([False, True]):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            surface(ax, ls, log=log, n=240)
            ax.set_title("linear $z$" if not log else "log $z$", fontsize=9,
                         color=THEME.ink_secondary, pad=0)
        fig.suptitle(f"{ls.title}\n${ls.formula}$", fontsize=11, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, f"003_surface_{ls.key}.png", kind="raw")
        plt.close(fig)


def contact_sheet(log: bool, name: str):
    fig = plt.figure(figsize=(15.5, 7.4))
    for i, ls in enumerate(SURFACES):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        surface(ax, ls, log=log, n=200)
        ax.set_title(f"${ls.formula}$", fontsize=9, pad=-2)
    scale = ("log $z$: " rf"$\log_{{10}}(K+10^{{{int(np.log10(LOG_FLOOR))}}})$"
             if log else r"linear $z$: $K(w)$")
    fig.suptitle(f"Loss surfaces $K(w)$ over the $(w_1, w_2)$ plane — {scale}",
                 fontsize=12, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, name, kind="raw")
    plt.close(fig)


def one_d_curves():
    models = [M.LINEAR_1D, M.SQUARE_1D]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, ls in zip(axes, models):
        w, k = ls.line(3000)
        ax.plot(w, k, color=THEME.categorical[0])
        ax.set_xlabel(r"$w$")
        ax.set_ylabel(r"$K(w)$")
        ax.set_title(f"${ls.formula}$", fontsize=10)
        V.tidy(ax, theme=THEME, grid="y")
    fig.suptitle("Loss curves $K(w)$ for the $d=1$ models", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "003_1d_curves.png", kind="raw")
    plt.close(fig)


if __name__ == "__main__":
    print("003 — raw surfaces")
    contact_sheet(log=False, name="003_contact_sheet_linear.png")
    contact_sheet(log=True, name="003_contact_sheet_log.png")
    one_d_curves()
    per_model_files()
