# 002 — DRAFT — Loss landscapes of singular models

- **Date:** 2026-07-31
- **Type:** DRAFT (diagram)
- **Code:** `scripts/002_singular_landscapes.py`
- **Figures:** `figures/diagram/002a_singular_landscapes.png`,
  `figures/diagram/002b_singular_vs_regular.png`
- **Seeds / settings:** deterministic (no RNG). GD in 002b: learning rate 0.12, 4000
  steps, finite-difference gradients with $h = 10^{-5}$, Hessians with $h = 10^{-4}$.

## What this draws

The models named at kickoff:

| model | $K(w)$ | $W_0$ | $\lambda$ | $m$ | $d/2$ |
|---|---|---|---|---|---|
| `monomial_1_1` | $\frac12 w_1^2 w_2^2$ | cross | 1/2 | 2 | 1 |
| `monomial_2_1` | $\frac12 w_1^4 w_2^2$ | cross | 1/4 | 1 | 1 |
| `monomial_2_2` | $\frac12 w_1^4 w_2^4$ | cross | 1/4 | 2 | 1 |
| `monomial_3_1` | $\frac12 w_1^6 w_2^2$ | cross | 1/6 | 1 | 1 |
| `tanh_1d` | $\frac12 \mathbb{E}_x[w_2\tanh(w_1 x)]^2$ | cross | 1/2 | 2 | 1 |
| `square_1d` | $\frac12 w^4$ | $\{0\}$ | 1/4 | 1 | 1/2 |

All $\lambda$ values in this table are **theory**, from the normal-crossing formula
$\lambda = \min_i 1/(2k_i)$ for $K = \prod_i |w_i|^{2k_i}$, with $m$ the number of
indices attaining the minimum. Empirical confirmation is 004's job, not this entry's —
nothing in this entry has been numerically verified.

Two facts worth separating, because the figures show both and they are usually
conflated:

1. **`monomial_1_1` and `tanh_1d` are the same singularity.** Expanding
   $\tanh(w_1 x) = w_1 x - (w_1 x)^3/3 + \dots$ gives
   $K = \frac12 w_1^2 w_2^2 \,\mathbb{E}[x^2] + O(|w|^6)$. The tanh unit's degeneracy
   *is* the product degeneracy; the neural network adds nothing to the local geometry
   at the origin. This is a point in SLT's favour as a descriptive language — one
   normal crossing covers both.
2. **`square_1d` is singular with $W_0$ a single point.** Degeneracy is not the same
   thing as a flat direction. $K = \frac12 w^4$ has an isolated minimum, no valley,
   and $\lambda = 1/4 < d/2 = 1/2$ purely because the contact order at the minimum is
   4 rather than 2. Panel F of 002a is the figure for this.

## Rendering choices that change what you see

- Contours are $\log_{10}(K + \text{floor})$ as in 001, floor = 0.5th percentile of
  positive grid values, per panel. On a linear scale these surfaces show the cross as
  a faint dark band and nothing else; the log scale is what makes the branch structure
  legible. See `figures/raw/003_contact_sheet_linear.png` vs `..._log.png` for the
  unannotated version of the same contrast.
- The dashed $W_0$ overlay is drawn from the known algebraic answer, not detected from
  the grid.
- 002b panel C's ylim was extended to 3.7 purely to make room for the annotation; the
  sharpness curve itself only reaches ~2.6 in the plotted range.
- 002a panel F plots three curves on a shared linear axis, which flatters $k=1$ and
  squashes $k=3$ near the origin. That is the honest linear picture, but the visual
  impression "higher $k$ is flatter near 0" is doing the work that the number
  $\lambda = 1/(2k)$ does precisely.

## Notes

**Sharpness varies along $W_0$, and GD picks which value it gets.** For
$K = \frac12 w_1^2 w_2^2$ the Hessian at $(a, 0)$ is $\mathrm{diag}(0, a^2)$, so
$\lambda_{\max} = a^2$: zero at the origin, growing away from it. Gradient *flow*
conserves $w_1^2 - w_2^2$, so an initialisation $(a_0, b_0)$ with $a_0^2 > b_0^2$ lands
at $(\sqrt{a_0^2 - b_0^2}, 0)$ and finds sharpness exactly $a_0^2 - b_0^2$.

Measured, at lr 0.12 / 4000 steps:

| init | conserved $w_1^2 - w_2^2$ | final $w_1$ | final sharpness |
|---|---|---|---|
| (1.30, 0.45) | 1.4875 | 1.2114 | 1.4676 |
| (1.15, 0.85) | 0.6000 | 0.7569 | 0.5729 |
| (1.05, 1.00) | 0.1025 | 0.3103 | 0.0963 |

Final sharpness tracks the conserved quantity to ~1–6%, with discrete GD undershooting
the gradient-flow prediction — expected, since the conservation law is exact only for
the flow, and the gap grows as the run spends longer near the origin where the drift is
slowest. This is a **discretisation artifact, not a finding**; do not read the residual
as structure.

**The thing I want to flag as a trap.** It is tempting to narrate this as "the most
degenerate point has the smallest $\lambda$ and is also the flattest, therefore
degeneracy and sharpness are linked". That narration is close to empty. In this model
the *most degenerate* point is the origin, and the origin is where both $w_1$ and $w_2$
are small — so of course the curvature is small there too. Both quantities are
functions of $|w|$ near the crossing, and the correlation is an artifact of that shared
dependence, not evidence that one explains the other. A cheap baseline — "sharpness at
convergence $= a_0^2 - b_0^2$, from a conservation law with no SLT content" — predicts
the entire table above without mentioning the RLCT once. Per CONSTITUTION 0a, the
baseline wins here and the SLT framing has earned nothing yet.

The gradient-flow conservation law for the product model is standard (it is the
"balancedness" invariant of two-layer linear/diagonal networks), so this is not a new
observation — it is the baseline that any SLT story about this model has to beat.

## Open threads

Proposed experiments, none run, hypotheses left blank for Alex per CONSTITUTION 1:

1. **Does $\lambda$ predict anything about sharpening that the conservation law does
   not?** The product model has a conserved quantity that fixes the answer exactly. A
   model *without* such an invariant (`monomial_2_1`, the unequal-exponent case, or the
   tanh net past its quadratic regime) would separate the two explanations. Hypothesis:
   _(blank)_.
2. **Edge of stability on a singular model.** Nothing above uses a learning rate near
   $2/\lambda_{\max}$; at lr 0.12 with sharpness ≤ 2.6 these runs are nowhere near the
   stability boundary. Running the same models at large lr is the obvious next step,
   and it is the first thing in this project that is genuinely an experiment. It needs
   a hypothesis before it runs. Hypothesis: _(blank)_.
3. **Local vs global $\lambda$.** Every $\lambda$ in the table is the global minimum
   over $W_0$, attained at the origin. GD does not converge to the origin from generic
   inits — it converges to a branch point where the *local* learning coefficient is
   different (1/2 on a branch of `monomial_1_1`, vs 1/2 at the origin with $m=2$). If
   the local coefficient is what dynamics sees, the global one may be the wrong number
   to be quoting at all. Worth resolving before it causes a mistake.
