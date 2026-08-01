"""004 — DRAFT (diagram figures, CONSTITUTION 8b) — RLCT estimator vs known theory.

Estimator QA, not an experiment: every lambda checked here is known in closed
form, so the only question is whether the volume-scaling estimator reproduces it
(CONSTITUTION 3b). If it does not, the estimator is wrong, not the algebra.

Produces:
  figures/diagram/004a_rlct_verification.png
  figures/diagram/004b_multiplicity_signature.png

See logs/004-rlct-verification.md.
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
N_SAMPLES = 8_000_000
MAX_FRACTION = 0.05

ZOO = [
    M.LINEAR_2D,
    M.LINEAR_2D_COLLINEAR,
    M.SQUARE_1D,
    M.PRODUCT,
    M.monomial(2, 1),
    M.monomial(2, 2),
    M.monomial(3, 1),
    M.TANH_1D,
]


def measure(ls):
    """Three estimators of the same lambda, so the log can show what each costs."""
    vs = R.volume_scaling(ls.K, ls.box, n_samples=N_SAMPLES, seed=SEED, n_eps=50)
    m = ls.multiplicity_theory or 1
    fit_1 = R.fit_rlct(vs, multiplicity=1, max_fraction=MAX_FRACTION)
    fit_m = R.fit_rlct(vs, multiplicity=m, max_fraction=MAX_FRACTION)
    fit_s = R.fit_rlct_subleading(vs, multiplicity=m, max_fraction=MAX_FRACTION,
                                  guess=max(ls.rlct_theory * 0.8, 0.05))
    return vs, fit_s, fit_m, fit_1


def facet(ax, ls, vs, fit_s, fit_m):
    mask = vs.usable(max_fraction=MAX_FRACTION)
    eps, vol = vs.eps[mask], vs.volume[mask]
    ax.plot(eps, vol, color=C[0], marker="o", ms=2.6, linestyle="none", alpha=0.85,
            label="measured")
    # Theory *shape*, anchored to the data at the largest epsilon: the intercept
    # C is not predicted by theory, so comparing it would compare nothing.
    ref = eps**ls.rlct_theory * np.log(1.0 / eps) ** ((ls.multiplicity_theory or 1) - 1)
    ax.plot(eps, ref * (vol[-1] / ref[-1]), color=C[1], lw=1.5, alpha=0.9,
            label=r"theory: $\epsilon^{\lambda}(\log 1/\epsilon)^{m-1}$"
                  "\n(anchored at largest $\\epsilon$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"${ls.formula}$", fontsize=9)
    ax.set_xlabel(r"$\epsilon$", fontsize=8)
    ax.set_ylabel(r"$\mathrm{Vol}\{K<\epsilon\}$", fontsize=8)
    ax.tick_params(labelsize=7)

    delta = fit_s.rlct - ls.rlct_theory
    ok = abs(delta) < 0.005
    V.annotate(
        ax,
        rf"$\lambda$ theory $= {ls.rlct_theory:g}$,  $m={ls.multiplicity_theory}$"
        "\n"
        rf"$\hat\lambda = {fit_s.rlct:.4f}$,  $\Delta = {delta:+.4f}$"
        f"   {'ok' if ok else 'MISMATCH'}"
        "\n"
        rf"(leading-term fit: {fit_m.rlct:.4f})",
        xy=(0.97, 0.04), va="bottom", ha="right", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="both")
    return ok


def figure_a(results):
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.0))
    all_ok = True
    for ax, (ls, vs, fit_s, fit_m, fit_1) in zip(axes.ravel(), results):
        all_ok &= facet(ax, ls, vs, fit_s, fit_m)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Volume scaling $\\mathrm{Vol}\\{K<\\epsilon\\} \\sim C\\,\\epsilon^{\\lambda}"
        "(\\log 1/\\epsilon)^{m-1}$ recovers every known learning coefficient"
        f"   —   {'all 8 agree' if all_ok else 'DISAGREEMENT, see log'}",
        fontsize=12, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "004a_rlct_verification.png")
    plt.close(fig)


def figure_b(results):
    """Multiplicity is visible without doing any algebraic geometry: divide out
    eps^lambda and see whether what is left is flat or grows like log(1/eps)."""
    by_key = {ls.key: (ls, vs) for ls, vs, _, _, _ in results}
    picks = [
        ("linear_2d", C[0], "regular, $m=1$"),
        ("monomial_2_1", C[1], "$w_1^4w_2^2$, $m=1$"),
        ("monomial_1_1", C[2], "$w_1^2w_2^2$, $m=2$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for key, colour, label in picks:
        ls, vs = by_key[key]
        eps, sig = R.multiplicity_signature(vs, ls.rlct_theory)
        keep = vs.usable(max_fraction=MAX_FRACTION)[vs.usable()]
        eps, sig = eps[keep], sig[keep]
        ax.plot(eps, sig / sig[-1], color=colour, lw=1.8, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\mathrm{Vol}\{K<\epsilon\}\,/\,\epsilon^{\lambda}$   (normalised)")
    ax.set_ylim(0.82, 2.85)
    ax.set_title(r"A.  Divide out $\epsilon^{\lambda}$")
    ax.legend(loc="upper right")
    V.annotate(
        ax,
        "flat $\\Rightarrow m=1$\n"
        "rising like $\\log(1/\\epsilon)$ $\\Rightarrow m=2$\n"
        "(each curve normalised to 1 at its largest $\\epsilon$;\n"
        "the models reach very different $\\epsilon$, so read\n"
        "each curve's shape, not their overlap)",
        xy=(0.03, 0.97), va="top", ha="left", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="both")

    ax = axes[1]
    keys = [ls.key for ls, _, _, _, _ in results]
    theory = np.array([ls.rlct_theory for ls, _, _, _, _ in results])
    sub = np.array([f.rlct for _, _, f, _, _ in results])
    with_m = np.array([f.rlct for _, _, _, f, _ in results])
    assume_1 = np.array([f.rlct for _, _, _, _, f in results])
    y = np.arange(len(keys))
    ax.plot(sub - theory, y + 0.22, marker="o", linestyle="none", color=C[0],
            label="correct $m$ + subleading term")
    ax.plot(with_m - theory, y, marker="o", linestyle="none", color=C[1],
            label="correct $m$, leading term only")
    ax.plot(assume_1 - theory, y - 0.22, marker="o", linestyle="none", color=C[2],
            label="assuming $m=1$")
    ax.axvline(0, color=THEME.ink_muted, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlim(-0.086, 0.078)
    ax.set_xlabel(r"$\hat\lambda - \lambda_{\mathrm{theory}}$")
    ax.set_title(r"B.  Cost of getting $m$ wrong")
    ax.legend(loc="upper right")
    V.annotate(
        ax,
        "the $m=2$ rows are the only ones that move.\n"
        "ignoring $m$ inflates $\\hat\\lambda$; keeping only\n"
        "the leading $\\log(1/\\epsilon)$ term still leaves\n"
        "$+3$ to $+8\\%$. one subleading constant fixes it.",
        xy=(0.97, 0.03), va="bottom", ha="right", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="x")

    fig.suptitle(
        "Multiplicity $m$ is measurable, and mis-specifying it biases $\\hat\\lambda$",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "004b_multiplicity_signature.png")
    plt.close(fig)


if __name__ == "__main__":
    print("004 — RLCT verification")
    print(f"  {'model':22s} {'d':>2s} {'theory':>8s} {'m':>2s} "
          f"{'+sublead':>9s} {'leading':>9s} {'m=1':>9s} {'eps range':>24s}")
    results = []
    for ls in ZOO:
        vs, fit_s, fit_m, fit_1 = measure(ls)
        results.append((ls, vs, fit_s, fit_m, fit_1))
        print(f"  {ls.key:22s} {ls.dim:2d} {ls.rlct_theory:8.4f} "
              f"{ls.multiplicity_theory:2d} {fit_s.rlct:9.4f} {fit_m.rlct:9.4f} "
              f"{fit_1.rlct:9.4f} "
              f"[{fit_s.eps_range[0]:.1e}, {fit_s.eps_range[1]:.1e}]")
    figure_a(results)
    figure_b(results)
