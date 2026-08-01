# 004 — DRAFT — Volume-scaling RLCT estimator vs known theory

- **Date:** 2026-07-31
- **Type:** DRAFT (diagram) — estimator QA, not an experiment. Every $\lambda$ here is
  known in closed form; the only question is whether the code reproduces it
  (CONSTITUTION 3b).
- **Code:** `scripts/004_rlct_verification.py`
- **Figures:** `figures/diagram/004a_rlct_verification.png`,
  `figures/diagram/004b_multiplicity_signature.png`
- **Seeds / settings:** seed 0, $8\times10^6$ uniform samples per model, 50 $\epsilon$
  points, fit window = thresholds with $\ge 200$ samples below and $\le 5\%$ of samples
  below.

## Method

Measure $V(\epsilon) = \mathrm{Vol}\{w \in B : K(w) < \epsilon\}$ by uniform Monte Carlo
over the model's box, then fit

$$V(\epsilon) = A\,\epsilon^{\lambda}\,\bigl(\log(1/\epsilon) + c\bigr)^{m-1}$$

with $m$ taken from theory and $(\log A, \lambda, c)$ fitted, weighted by sample count.
Three estimators are reported per model so their costs are visible:

1. `m=1` — plain power law, ignoring multiplicity.
2. `leading` — correct $m$, leading term only ($c$ forced to 0).
3. `+sublead` — correct $m$, one subleading constant $c$ fitted.

## Result

`empirical`, all three estimators, against `theory`:

| model | $d$ | $\lambda$ theory | $m$ | +sublead | leading | $m{=}1$ |
|---|---|---|---|---|---|---|
| `linear_2d` | 2 | 1.0000 | 1 | **0.9966** | 0.9966 | 0.9966 |
| `linear_2d_collinear` | 2 | 0.5000 | 1 | **0.4990** | 0.4990 | 0.4990 |
| `square_1d` | 1 | 0.2500 | 1 | **0.2501** | 0.2501 | 0.2501 |
| `monomial_1_1` | 2 | 0.5000 | 2 | **0.4997** | 0.5152 | 0.4331 |
| `monomial_2_1` | 2 | 0.2500 | 1 | **0.2484** | 0.2484 | 0.2484 |
| `monomial_2_2` | 2 | 0.2500 | 2 | **0.2498** | 0.2589 | 0.2166 |
| `monomial_3_1` | 2 | 0.1667 | 1 | **0.1661** | 0.1661 | 0.1661 |
| `tanh_1d` | 2 | 0.5000 | 2 | **0.5008** | 0.5395 | 0.4453 |

All eight agree with theory to $|\Delta| < 0.004$ using the full estimator. The
estimator is fit for purpose.

## The one real finding here

**The $m=2$ models are biased by $+3\%$ to $+8\%$ if you fit only the leading term, and
that bias does not go away as $\epsilon \to 0$.**

Over a box, $V(\epsilon)$ is a *sum*: the most degenerate point contributes
$\epsilon^{\lambda}(\log 1/\epsilon)^{m-1}$, and the branches of $W_0$ away from it
contribute $\epsilon^{\lambda}$. Their ratio decays like $1/\log(1/\epsilon)$ — so at
$\epsilon = 10^{-11}$ the "subleading" term is still ~4% of the leading one, and a
leading-order fit absorbs it into the slope. `tanh_1d` shows $\hat\lambda = 0.5395$
against a true 0.5 even over a fit window spanning seven decades of $\epsilon$.

Three things follow, all of which matter more later than now:

1. **Extending the $\epsilon$ range is not a fix.** The correction is logarithmic. Going
   from $10^{-11}$ to $10^{-22}$ buys a factor of two in the bias, and $10^{-22}$ is
   already at the floor of float64 for these models. Anyone who reports an RLCT from a
   power-law fit and claims 1% accuracy on a model with $m > 1$ is wrong by more than
   they think.
2. **The bias is upward, i.e. toward $d/2$**, which is the direction that makes a
   singular model look less singular. That is the wrong direction to be biased in for a
   project trying to decide whether SLT quantities carry information: it works against
   detecting a difference, so a *null* result from this estimator is more trustworthy
   than a positive one.
3. **This is a warning about MCMC-based local learning coefficient estimators too.**
   They face the same finite-$\epsilon$ (finite-$n$, finite-$\beta$) truncation. This
   run does not test them, but it says the failure mode is real and has the same
   logarithmic slowness.

## Rendering choices that change what you see

- 004a's orange reference curve is the theory *shape*
  $\epsilon^{\lambda}(\log 1/\epsilon)^{m-1}$ anchored to the data at the largest
  $\epsilon$. The constant $C$ is not predicted by theory, so anchoring is the only
  honest way to overlay it; the curve is a shape comparison, not a prediction of
  height.
- 004b panel A normalises each curve to 1 at its own largest $\epsilon$. The three
  models reach wildly different $\epsilon$ ranges ($10^{-4}$ vs $10^{-19}$), so the
  curves do not overlap in $x$ and their apparent left-to-right positions carry no
  meaning. Read each curve's slope, not the comparison between them.
- Each panel's $\epsilon$ window is auto-selected per model, so the x-ranges in 004a
  differ by up to 20 orders of magnitude between panels. They are not comparable
  across panels.
- The `+/-` standard errors printed by the code are WLS slope errors. They are
  optimistic: they describe scatter about the fitted line and not the systematic error
  from the fit window or from a mis-specified $m$, which is the error that actually
  bit here. **Do not quote them as the uncertainty on $\lambda$.**

## Notes

- Multiplicity is directly measurable (004b panel A): divide $V(\epsilon)$ by
  $\epsilon^{\lambda}$ and see whether the remainder is flat ($m=1$) or grows like
  $\log(1/\epsilon)$ ($m=2$). No algebraic geometry required. This is a genuinely
  useful thing that SLT's framing gives you, and it is worth noting as a point *for*
  the theory's practical content — it is a falsifiable, cheap measurement.
- `tanh_1d` returning 0.5008 against theory 0.5 is the most informative row: the theory
  value came from a Taylor expansion of $\tanh$, and the measurement is of the actual
  expectation computed by Gauss-Hermite quadrature. Agreement to $8\times10^{-4}$ says
  the expansion argument in 002 is sound.

## Open threads

- No experiment. Nothing here has a hypothesis; it is all verification against known
  answers, and that is the correct classification (CONSTITUTION 1b).
- The estimator has only been tested at $d \le 2$. Uniform Monte Carlo over a box dies
  quickly with dimension — the fraction of samples in the sub-level set falls off fast
  — and every model in the eventual zoo is higher dimensional. Testing it at $d = 4$–$6$
  on a reduced-rank regression, where the RLCT is also known in closed form (Aoyagi &
  Watanabe), is the natural next QA step and should happen before any $d > 2$ number is
  believed.
- The subleading-constant fix is ad hoc: it works because these models have exactly one
  competing stratum. A model with several strata of different local $\lambda$ will need
  more terms, and there is no principled stopping point. Worth knowing the limit of the
  trick before relying on it.
