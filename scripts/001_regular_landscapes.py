"""001 — DRAFT (diagram figures, CONSTITUTION 8b) — Loss landscapes of regular (non-singular) statistical models.

Reference figures for what "no singularity" looks like, so that the singular
cases in 002 have something to be compared against.

Produces:
  figures/diagram/001a_regular_landscapes.png
  figures/diagram/001b_illconditioned_is_not_singular.png

See logs/001-regular-landscapes.md.
"""

import _bootstrap  # noqa: F401  (path side effect)
import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import save

from slt import models as M
from slt import rlct as R
from slt import viz as V

THEME = V.LIGHT
V.use_style(THEME)
C = THEME.categorical
SEED = 0
N_SAMPLES = 4_000_000


def sublevel_widths(ax):
    """Panel A: K(w) = w^2/2 with sub-level sets. The width of {K < eps}
    scales like eps^{1/2}; the exponent 1/2 IS the RLCT of this model."""
    ls = M.LINEAR_1D
    w, k = ls.line(2000)
    ax.plot(w, k, color=C[0], zorder=3)

    for i, eps in enumerate([0.5, 0.125, 0.03125]):
        half = np.sqrt(2 * eps)
        ax.fill_between(
            [-half, half], 0, eps,
            color=C[0], alpha=0.10 + 0.06 * i, linewidth=0, zorder=1,
        )
        ax.annotate(
            "", xy=(-half, eps), xytext=(half, eps),
            arrowprops=dict(arrowstyle="<->", color=THEME.ink_muted, lw=0.9),
        )
        ax.text(
            0, eps + 0.035, rf"$2\sqrt{{2\epsilon}}$" if i == 0 else "",
            ha="center", fontsize=8, color=THEME.ink_secondary,
        )
        ax.text(
            half + 0.06, eps, rf"$\epsilon={eps:g}$",
            va="center", fontsize=7.5, color=THEME.ink_muted,
        )

    ax.plot([0], [0], marker="o", ms=6, color=C[0], zorder=5,
            markeredgecolor=THEME.surface, markeredgewidth=1.4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 1.32)
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$K(w)$")
    ax.set_title("A.  Regular, $d=1$:  $K(w)=\\frac{1}{2} w^2$")
    V.annotate(
        ax,
        "$W_0=\\{0\\}$ is a single point\n"
        "$K''(0)=1>0$  (Fisher info positive)\n"
        r"$\mathrm{Vol}\{K<\epsilon\}\propto\epsilon^{1/2}$"
        "\n" r"$\Rightarrow\ \lambda=1/2=d/2$",
        xy=(0.5, 0.99), ha="center", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="y")


def contour_panel(ax, ls, title, note):
    X, Y, Z = ls.grid(500)
    cf = V.loss_contours(ax, X, Y, Z, theme=THEME, log=True, n_levels=14)
    ax.plot([0], [0], marker="o", ms=6, color=THEME.surface, zorder=5,
            markeredgecolor=THEME.ink, markeredgewidth=1.2)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title(title)
    ax.set_aspect("equal")
    V.annotate(ax, note, theme=THEME)
    return cf


def hessian_spectra(ax):
    """Panel D: the Hessian of a regular model is positive definite, and it is
    the SAME everywhere (these losses are exactly quadratic)."""
    entries = [
        ("isotropic\n$\\Sigma=I$", np.linalg.eigvalsh(np.eye(2)), C[0]),
        ("correlated\n$\\kappa=100$", np.linalg.eigvalsh(M._ILL), C[0]),
        ("collinear\n(rank 1)", np.linalg.eigvalsh(np.outer(M._V, M._V)), C[0]),
    ]
    floor = 1e-4
    for row, (name, ev, colour) in enumerate(entries):
        for j, lam in enumerate(sorted(ev, reverse=True)):
            y = row + (0.18 if j == 0 else -0.18)
            drawn = max(lam, floor)
            ax.barh(y, drawn, height=0.28, color=colour,
                    alpha=1.0 if j == 0 else 0.55, linewidth=0)
            txt = f"{lam:.3g}" if lam > 1e-12 else "0"
            ax.text(drawn * 1.35, y, txt, va="center", fontsize=7.5,
                    color=THEME.ink_secondary)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e[0] for e in entries], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(floor, 40)
    ax.set_ylim(-0.55, 3.45)
    ax.set_xlabel(r"Hessian eigenvalue  $\mathrm{eig}(\nabla^2 K)$")
    ax.set_title("D.  Curvature at the minimum")
    ax.axvline(floor, color=THEME.ink_muted, lw=0.8, ls=":")
    V.annotate(
        ax,
        "a zero eigenvalue (no bar drawn) means a flat\n"
        "direction $\\Rightarrow$ $W_0$ is not a point $\\Rightarrow$ singular.\n"
        "$\\kappa=100$ is small curvature, not zero curvature.",
        xy=(0.02, 0.98), va="top", ha="left", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="x")


def volume_scaling_panel(ax, specs, title, note, *, legend_loc="upper left",
                         note_xy=(0.97, 0.05), note_va="bottom", note_ha="right"):
    """Vol{K < eps} on log-log. The slope is lambda."""
    handles = []
    for ls, colour, mult, label in specs:
        vs = R.volume_scaling(ls.K, ls.box, n_samples=N_SAMPLES, seed=SEED)
        fit = R.fit_rlct(vs, multiplicity=mult, max_fraction=0.03)
        mask = vs.usable(max_fraction=0.03)
        (h,) = ax.plot(
            vs.eps[mask], vs.volume[mask], color=colour, marker="o", ms=3,
            linestyle="none", alpha=0.9,
        )
        xs = vs.eps[mask]
        ax.plot(xs, np.exp(fit.intercept) * xs**fit.rlct, color=colour, lw=1.4,
                alpha=0.75, zorder=1)
        handles.append(
            (h, f"{label}:  $\\hat\\lambda$={fit.rlct:.3f}"
                f"   (theory {ls.rlct_theory:g})")
        )
        print(f"  {ls.key:24s} theory lambda={ls.rlct_theory}  {fit}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\mathrm{Vol}\,\{w:\,K(w)<\epsilon\}$")
    ax.set_title(title)
    ax.legend([h for h, _ in handles], [t for _, t in handles], loc=legend_loc)
    V.annotate(ax, note, xy=note_xy, va=note_va, ha=note_ha, theme=THEME)
    V.tidy(ax, theme=THEME, grid="both")


def figure_a():
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))

    sublevel_widths(axes[0, 0])

    cf = contour_panel(
        axes[0, 1], M.LINEAR_2D,
        "B.  Regular, $d=2$:  isotropic design",
        "circular level sets\nunique minimum\n$\\lambda = d/2 = 1$",
    )
    V.add_colorbar(fig, cf, axes[0, 1], r"$\log_{10}K$", THEME)

    cf = contour_panel(
        axes[0, 2], M.LINEAR_2D_ILLCOND,
        "C.  Regular, $d=2$:  correlated design",
        "elongated but still elliptical\nstill a unique minimum\n"
        "$\\lambda = d/2 = 1$ regardless of $\\kappa$",
    )
    V.add_colorbar(fig, cf, axes[0, 2], r"$\log_{10}K$", THEME)

    hessian_spectra(axes[1, 0])

    # Self-similarity of the sub-level sets of a quadratic.
    ax = axes[1, 1]
    X, Y, Z = M.LINEAR_2D_ILLCOND.grid(500)
    eps_levels = [0.0128, 0.0032, 0.0008, 0.0002]
    for i, eps in enumerate(eps_levels):
        ax.contourf(X, Y, Z, levels=[0, eps],
                    colors=[V.SEQ_BLUE(0.25 + 0.18 * i)], zorder=i)
        ax.contour(X, Y, Z, levels=[eps], colors=[THEME.surface],
                   linewidths=0.8, zorder=i)
    ax.plot([0], [0], marker="o", ms=5, color=THEME.surface, zorder=9,
            markeredgecolor=THEME.ink, markeredgewidth=1.2)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title(r"E.  Sub-level sets $\{K<\epsilon\}$ shrink self-similarly")
    V.annotate(
        ax,
        r"$\epsilon = 0.0128 \to 0.0002$, each $4\times$ smaller"
        "\n"
        "every drop of $4\\times$ in $\\epsilon$\n"
        "shrinks *both* axes by $2\\times$\n"
        r"$\Rightarrow\ \mathrm{Vol}\propto\epsilon^{d/2}=\epsilon$",
        xy=(0.03, 0.03), va="bottom", ha="left", theme=THEME,
    )

    volume_scaling_panel(
        fig.axes[5],
        [(M.LINEAR_1D, C[0], 1, "$d{=}1$"),
         (M.LINEAR_2D, C[1], 1, "$d{=}2$ isotropic"),
         (M.LINEAR_2D_ILLCOND, C[2], 1, "$d{=}2$ correlated")],
        "F.  Volume scaling recovers $\\lambda=d/2$",
        "slope of the log-log line $=\\lambda$\n"
        "regular models sit exactly at $d/2$\n"
        "(points: Monte-Carlo, $4\\times10^6$ samples)",
    )

    fig.suptitle(
        "Regular models: the minimum is a point, the curvature is nondegenerate, "
        "and the learning coefficient is always $\\lambda = d/2$",
        fontsize=12, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "001a_regular_landscapes.png")
    plt.close(fig)


def figure_b():
    """The guardrail figure (CONSTITUTION 6a): ill-conditioned is NOT singular."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    cf = contour_panel(
        axes[0], M.LINEAR_2D_ILLCOND,
        "A.  Ill-conditioned but regular  ($\\kappa=100$)",
        "$\\Sigma\\succ0$: eigenvalues $1,\\ 0.01$\n"
        "narrow valley, but it has a bottom\n"
        "$W_0=\\{0\\}$,  $\\lambda=1$",
    )
    V.add_colorbar(fig, cf, axes[0], r"$\log_{10}K$", THEME)

    ls = M.LINEAR_2D_COLLINEAR
    X, Y, Z = ls.grid(500)
    ax = axes[1]
    cf = V.loss_contours(ax, X, Y, Z, theme=THEME, log=True, n_levels=14)
    V.mark_zero_set(ax, [([-2, 2], [2, -2])], theme=THEME, label=r"$W_0$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title("B.  Genuinely degenerate  (rank-1 $\\Sigma$)")
    ax.legend(loc="lower left")
    V.annotate(
        ax,
        "$\\Sigma$ singular: eigenvalues $1,\\ 0$\n"
        "$W_0$ is a *line*, not a point\n"
        "$\\lambda = \\mathrm{rank}/2 = 1/2 < d/2$",
        theme=THEME,
    )
    V.add_colorbar(fig, cf, ax, r"$\log_{10}K$", THEME)

    volume_scaling_panel(
        axes[2],
        [(M.LINEAR_2D_ILLCOND, C[0], 1, "correlated, full rank"),
         (M.LINEAR_2D_COLLINEAR, C[1], 1, "collinear, rank 1")],
        "C.  Only degeneracy moves $\\lambda$",
        "$\\kappa=100$ shifts the *intercept*\n"
        "(a narrower valley = less volume)\n"
        "but only rank deficiency\nchanges the *slope*",
        legend_loc="lower right", note_xy=(0.03, 0.97), note_va="top", note_ha="left",
    )

    fig.suptitle(
        "A small Hessian eigenvalue is not a singularity: conditioning moves the "
        "intercept, degeneracy moves the exponent",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "001b_illconditioned_is_not_singular.png")
    plt.close(fig)


if __name__ == "__main__":
    print("001 — regular landscapes")
    figure_a()
    figure_b()
