# 001 — DRAFT — Loss landscapes of regular (non-singular) models

- **Date:** 2026-07-31
- **Type:** DRAFT (diagram)
- **Code:** `scripts/001_regular_landscapes.py`
- **Figures:** `figures/diagram/001a_regular_landscapes.png`,
  `figures/diagram/001b_illconditioned_is_not_singular.png`
- **Seeds / settings:** RNG seed 0; volume scaling with $4\times10^6$ uniform samples
  per model; fit restricted to $\epsilon$ with $\ge 200$ samples below threshold and
  $\le 3\%$ of samples below threshold.

## What this draws

The reference case. Linear regression $y = w \cdot x + N(0,1)$ with $x \sim N(0,\Sigma)$,
for which the KL divergence is exactly $K(w) = \frac12 (w-w^*)^\top \Sigma (w-w^*)$.
Three variants — $\Sigma = I$, $\Sigma$ with condition number 100, and $\Sigma$ of
rank 1 — plus the 1-parameter case.

The point of having these is to make "singular" mean something by contrast. Everything
in 002 is a departure from this picture.

001b exists specifically to nail down CONSTITUTION 6a, which is the distinction I
expect to be the most slippery one in this project: **an ill-conditioned model is not
a singular model.** A condition number of 100 makes a narrow valley, and the narrow
valley has a bottom.

## Rendering choices that change what you see

- All contour panels plot $\log_{10}(K + \text{floor})$, not $K$. The floor is the
  0.5th percentile of the positive values on the grid, chosen per panel. On a linear
  scale these landscapes are visually identical to each other and to the singular ones
  in 002 — everything looks like a smooth bowl. **The log scale is doing most of the
  work in every contour figure in this repo.**
- Panel A's sub-level shading is drawn analytically ($\pm\sqrt{2\epsilon}$), not
  measured.
- Panel D clips a zero Hessian eigenvalue to the left edge of a log axis, so the
  "collinear" row shows one bar and a `0` label rather than two bars.
- Panel E uses $\epsilon \in \{0.0128, 0.0032, 0.0008, 0.0002\}$, chosen so the
  largest sub-level ellipse fits inside the plotted window. Larger $\epsilon$ gets
  clipped by the box and the self-similarity stops being visible, which would be an
  artifact of the window rather than of the model.
- Volume-scaling panels cap the fit at 3% of samples for the same reason: above that
  the sub-level set of the ill-conditioned model runs into the walls of the sampling
  box and the measured slope bends. This cap is a real constraint of the estimator,
  not a cosmetic choice.

## Notes

Volume scaling recovers $\lambda = d/2$ to three digits in every regular case:

| model | $d$ | $\lambda$ (theory) | $\hat\lambda$ (empirical) |
|---|---|---|---|
| `linear_1d` | 1 | 0.5 | 0.4997 ± 0.0003 |
| `linear_2d` | 2 | 1.0 | 1.0015 ± 0.0007 |
| `linear_2d_illcond` | 2 | 1.0 | 0.9953 ± 0.0009 |
| `linear_2d_collinear` | 2 | 0.5 | 0.4981 ± 0.0003 |

$R^2 = 1.0000$ on all four. This satisfies CONSTITUTION 3b for the estimator: it works
where we know the answer, so it is worth something where we do not. The quoted errors
are the WLS standard errors on the slope and are almost certainly optimistic — they
capture scatter around the fitted line, not the systematic error from the choice of fit
window. The window sensitivity is the thing to check before trusting a $\hat\lambda$ to
better than ~1%.

The `linear_2d_collinear` case is the useful one. It is degenerate — $W_0$ is a line,
$\Sigma$ is rank 1 — but its degeneracy is the boring kind: $W_0$ is a smooth manifold
and $K$ is quadratic transverse to it, so $\lambda = \text{rank}/2$ and there is no
normal-crossing structure. It sits between 001 and 002 and is a good sanity check that
$\lambda < d/2$ on its own does not imply anything interesting is happening.

## Open threads

- The claim "the quoted standard error understates the real uncertainty" is asserted,
  not measured. A fit-window sensitivity sweep would settle it and is cheap. Not an
  experiment in the CONSTITUTION 1 sense — it is estimator QA — but worth doing before
  any $\hat\lambda$ is used in an argument.
- Nothing here touches dynamics. The regular models are the null case for the whole
  project: every initialisation reaches the same minimum with the same curvature, so
  there is nothing for SLT or anything else to explain.
