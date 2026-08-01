"""002 — DRAFT (diagram figures, CONSTITUTION 8b) — Loss landscapes of singular models.

The models Alex named at kickoff: f(w) = w_1 w_2, f(w) = w_1^n w_2^m,
f(x) = w_2 tanh(w_1 x), and f(w) = w^2.

Produces:
  figures/diagram/002a_singular_landscapes.png
  figures/diagram/002b_singular_vs_regular.png

See logs/002-singular-landscapes.md.
"""

import _bootstrap  # noqa: F401  (path side effect)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import save

from slt import dynamics as D
from slt import models as M
from slt import viz as V

THEME = V.LIGHT
V.use_style(THEME)
C = THEME.categorical

CROSS = [([0, 0], [-9, 9]), ([-9, 9], [0, 0])]  # W_0 for every crossing model


def singular_contours(ax, ls, letter, note, *, zero_set=CROSS, n=600):
    X, Y, Z = ls.grid(n)
    cf = V.loss_contours(ax, X, Y, Z, theme=THEME, log=True, n_levels=14)
    V.mark_zero_set(ax, zero_set, theme=THEME, label=r"$W_0$")
    (x0, x1), (y0, y1) = ls.box
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title(f"{letter}.  ${ls.formula}$")
    ax.legend(loc="lower right")
    V.annotate(ax, note, theme=THEME)
    return cf


def rlct_line(ls) -> str:
    return (f"$\\lambda = {ls.rlct_theory:g}$,  $m = {ls.multiplicity_theory}$"
            f"   (vs $d/2 = {ls.dim_over_two:g}$)")


def one_d_family(ax):
    """Panel F: K = w^{2k}/2. W_0 is a single point for every k, yet the model
    is singular for every k > 1 -- degeneracy with no flat direction at all."""
    w = np.linspace(-1.5, 1.5, 2000)
    for i, k in enumerate([1, 2, 3]):
        K = 0.5 * w ** (2 * k)
        lam = 1.0 / (2 * k)
        ax.plot(w, K, color=C[i],
                label=rf"$K=\frac{{1}}{{2}}w^{{{2 * k}}}$:  $\lambda={lam:.3g}$")
        ax.annotate(rf"$\lambda={lam:.3g}$", xy=(1.5, 0.5 * 1.5 ** (2 * k)),
                    xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color=C[i], va="center")
    ax.plot([0], [0], marker="o", ms=6, color=THEME.surface, zorder=5,
            markeredgecolor=THEME.ink, markeredgewidth=1.2)
    ax.set_xlim(-1.7, 1.85)
    ax.set_ylim(-0.15, 3.0)
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$K(w)$")
    ax.set_title(r"F.  $d=1$:  $f_w = w^k$ fitting $f_0 = 0$")
    ax.legend(loc="upper center")
    V.annotate(
        ax,
        "$W_0=\\{0\\}$ is a *point* in all three,\n"
        "but $K^{(2)}(0)=0$ once $k>1$:\n"
        "the Fisher information vanishes.\n"
        "$\\lambda = 1/(2k)$, and only $k=1$ hits $d/2$.\n"
        "Singularity $\\neq$ flat direction.",
        xy=(0.03, 0.03), va="bottom", ha="left", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="y")


def figure_a():
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0))

    specs = [
        (M.PRODUCT, "A",
         "the canonical singularity.\n"
         "$W_0$ is a *cross*: two branches\nmeeting at the origin.\n"
         + rlct_line(M.PRODUCT)),
        (M.monomial(2, 1), "B",
         "unequal exponents tilt the cross.\n"
         "the $w_1$ branch is 'wider' and\nsets $\\lambda$ on its own.\n"
         + rlct_line(M.monomial(2, 1))),
        (M.monomial(2, 2), "C",
         "raising both exponents widens\nboth branches: $\\lambda$ halves,\n"
         "and the tie restores $m=2$.\n"
         + rlct_line(M.monomial(2, 2))),
        (M.monomial(3, 1), "D",
         "higher order $\\Rightarrow$ flatter valley\n"
         "$\\Rightarrow$ smaller $\\lambda$ $\\Rightarrow$ *less*\n"
         "effective complexity.\n"
         + rlct_line(M.monomial(3, 1))),
    ]
    for ax, (ls, letter, note) in zip(axes.ravel()[:4], specs):
        cf = singular_contours(ax, ls, letter, note)
        V.add_colorbar(fig, cf, ax, r"$\log_{10}K$", THEME)

    cf = singular_contours(
        axes[1, 1], M.TANH_1D, "E",
        "a real (tiny) neural network.\n"
        "$w_2=0$ kills the unit; $w_1=0$\nkills its input. Same cross.\n"
        r"$K \approx \frac{1}{2}w_1^2w_2^2 + O(|w|^6)$" "\n"
        + rlct_line(M.TANH_1D),
    )
    V.add_colorbar(fig, cf, axes[1, 1], r"$\log_{10}K$", THEME)

    one_d_family(axes[1, 2])

    fig.suptitle(
        "Singular models: $W_0$ is not a point, or the curvature on it vanishes — "
        "either way $\\lambda < d/2$, so the Gaussian (Laplace) approximation to the "
        "volume near the minimum fails",
        fontsize=12, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "002a_singular_landscapes.png")
    plt.close(fig)


def figure_b():
    """Head-to-head, and the first look at what this means for dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    lr, steps = 0.12, 4000
    inits = [(1.30, 0.45), (1.15, 0.85), (1.05, 1.00)]

    # --- A: regular reference, same GD, same learning rate --------------------
    ls = M.LINEAR_2D
    X, Y, Z = ls.grid(500)
    ax = axes[0]
    V.loss_contours(ax, X, Y, Z, theme=THEME, log=True, n_levels=14)
    for i, w0 in enumerate(inits):
        tr = D.gradient_descent(ls.K, w0, lr=lr, steps=steps)
        ax.plot(tr.w[:, 0], tr.w[:, 1], color=C[i], lw=1.8, zorder=5,
                path_effects=[pe.withStroke(linewidth=3.6,
                                            foreground=THEME.surface)])
        ax.plot(*w0, marker="o", ms=5, color=C[i], zorder=6,
                markeredgecolor=THEME.surface, markeredgewidth=1.2)
    ax.plot([0], [0], marker="*", ms=13, color=THEME.surface, zorder=7,
            markeredgecolor=THEME.ink, markeredgewidth=1.0)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title("A.  Regular:  $K=\\frac{1}{2}\\|w\\|^2$")
    V.annotate(
        ax,
        "every initialisation reaches the\n"
        "*same* minimum, with the *same*\n"
        "curvature $\\nabla^2K=I$ there.\n"
        "sharpness is a property of the model.",
        xy=(0.03, 0.03), va="bottom", ha="left", theme=THEME,
    )

    # --- B: the crossing, same GD ---------------------------------------------
    ls = M.PRODUCT
    X, Y, Z = ls.grid(600)
    ax = axes[1]
    V.loss_contours(ax, X, Y, Z, theme=THEME, log=True, n_levels=14)
    V.mark_zero_set(ax, CROSS, theme=THEME, label=r"$W_0$")
    endpoints = []
    for i, w0 in enumerate(inits):
        tr = D.gradient_descent(ls.K, w0, lr=lr, steps=steps)
        ax.plot(tr.w[:, 0], tr.w[:, 1], color=C[i], lw=1.8, zorder=5,
                path_effects=[pe.withStroke(linewidth=3.6,
                                            foreground=THEME.surface)])
        ax.plot(*w0, marker="o", ms=5, color=C[i], zorder=6,
                markeredgecolor=THEME.surface, markeredgewidth=1.2)
        end = tr.w[-1]
        endpoints.append((w0, end, tr.sharpness[-1], C[i]))
        ax.plot(end[0], end[1], marker="*", ms=12, color=C[i], zorder=7,
                markeredgecolor=THEME.surface, markeredgewidth=1.0)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title("B.  Singular:  $K=\\frac{1}{2}w_1^2w_2^2$")
    ax.legend(loc="lower right")
    V.annotate(
        ax,
        "gradient flow conserves $w_1^2-w_2^2$,\n"
        "so each initialisation slides to a\n"
        "*different* point of $W_0$ — and, by\n"
        "panel C, to a different sharpness.",
        xy=(0.03, 0.03), va="bottom", ha="left", theme=THEME,
    )

    # --- C: sharpness varies along W_0 ---------------------------------------
    ax = axes[2]
    a = np.linspace(-1.6, 1.6, 400)
    sharp = np.array([D.sharpness(ls.K, np.array([ai, 0.0])) for ai in a])
    ax.plot(a, sharp, color=THEME.ink_secondary, lw=1.8, zorder=3,
            label=r"$\lambda_{\max}\nabla^2K$ along $W_0$")
    ax.plot(a, a**2, color=THEME.ink_muted, lw=1.0, ls=(0, (4, 3)), zorder=2,
            label=r"analytic: $\lambda_{\max}=w_1^2$")
    for w0, end, sh, colour in endpoints:
        ax.plot(end[0], sh, marker="*", ms=12, color=colour, zorder=6,
                markeredgecolor=THEME.surface, markeredgewidth=1.0)
        ax.annotate(
            rf"init $({w0[0]:.2f},{w0[1]:.2f})$" "\n"
            rf"$w_1^2-w_2^2={w0[0] ** 2 - w0[1] ** 2:.2f}$",
            xy=(end[0], sh), xytext=(6, 10), textcoords="offset points",
            fontsize=7, color=colour,
        )
    ax.axvline(0, color=THEME.grid, lw=0.9, zorder=1)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.12, 3.7)
    ax.set_xlabel(r"position along $W_0$   ($w = (w_1, 0)$)")
    ax.set_ylabel(r"sharpness  $\lambda_{\max}(\nabla^2 K)$")
    ax.set_title("C.  On a singular $W_0$, sharpness is not one number")
    ax.legend(loc="upper left")
    V.annotate(
        ax,
        "the origin — the most degenerate point,\n"
        "the one with the smallest local $\\lambda$ —\n"
        "is also the *flattest*. Curvature and\n"
        "degeneracy point in opposite directions here.",
        xy=(0.5, 0.78), va="top", ha="center", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="y")

    fig.suptitle(
        "What changes when the model is singular: the minimum you reach, and the "
        "curvature you find there, both depend on where you started",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "002b_singular_vs_regular.png")
    plt.close(fig)

    for w0, end, sh, _ in endpoints:
        print(f"  init {w0} -> w={np.round(end, 4)}  sharpness={sh:.4f}  "
              f"conserved w1^2-w2^2={w0[0] ** 2 - w0[1] ** 2:.4f}")


if __name__ == "__main__":
    print("002 — singular landscapes")
    figure_a()
    figure_b()
