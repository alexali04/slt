# LOG

Index of every draft and experiment. One line each, newest last. Entries live in
`logs/NNN-slug.md` and are append-only once written (CONSTITUTION 2).

Templates: [`logs/TEMPLATE-draft.md`](logs/TEMPLATE-draft.md),
[`logs/TEMPLATE-experiment.md`](logs/TEMPLATE-experiment.md).

| # | date | type | entry |
|---|------|------|-------|
| 001 | 2026-07-31 | DRAFT · diagram | [Regular (non-singular) loss landscapes](logs/001-regular-landscapes.md) — the reference case, and why $\kappa=100$ is not a singularity |
| 002 | 2026-07-31 | DRAFT · diagram | [Singular loss landscapes](logs/002-singular-landscapes.md) — $w_1w_2$, $w_1^nw_2^m$, $w_2\tanh(w_1x)$, $w^2$; and the conservation law that already explains the sharpness story without SLT |
| 003 | 2026-07-31 | DRAFT · raw | [Raw 3-D loss surfaces](logs/003-raw-surfaces.md) — the unannotated objects, linear and log $z$ |
| 004 | 2026-07-31 | DRAFT · diagram | [RLCT estimator vs known theory](logs/004-rlct-verification.md) — all 8 recovered to <0.004; leading-order fits are biased $+3$–$8\%$ when $m>1$ |
| 005 | 2026-07-31 | DRAFT · raw | [Interactive tanh loss surface](logs/005-tanh-interactive.md) — extended domain, linear $z$; $K = \frac12 w_2^2\,g(w_1)$ exactly. ⚠️ far-field claims superseded by 006 |
| 006 | 2026-07-31 | DRAFT · raw | [Tanh far field + quadrature bug](logs/006-tanh-far-field-correction.md) — Gauss–Hermite faked a saturation; truth is $g = 1-\sqrt{2/\pi}/\|w_1\|$. The two branches of $W_0$ have the same RLCT and wildly different curvature |
| 007 | 2026-07-31 | DRAFT · diagram | [EoS animation on the noisy tanh surface](logs/007-eos-animation.md) — progressive sharpening pins $\lambda_{\max}$ at $2/\eta$; minibatch noise sustains the oscillation and opens a sharpness gap |
| 008 | 2026-07-31 | EXPERIMENT · **not yet run** | [Grokking on modular addition](logs/008-grokking.md) — Nanda replication, code ready and pre-registered; hypothesis slot empty, so it has not started (CONSTITUTION 1) |

## Standing questions

Live threads, oldest first. Move to an entry when one gets answered.

- **Does any SLT invariant beat a cheap baseline at a dynamical question?** Nothing so
  far does, and 006 is a point actively against: the two branches of the tanh model's
  $W_0$ have identical local RLCT ($1/2$) but curvature that saturates at 1 on one and
  diverges like $w_2^2$ on the other. The RLCT cannot tell them apart; the Hessian
  can. On the product model the gradient-flow conservation law $w_1^2 - w_2^2$
  predicts final sharpness exactly, again with no reference to $\lambda$ (002).
- **Local vs global $\lambda$.** Every quoted $\lambda$ is the global min over $W_0$,
  attained at the origin, but GD converges elsewhere on $W_0$ from generic inits. If
  dynamics sees a local coefficient, the global one may be the wrong number entirely
  (002).
- **Does the volume-scaling estimator survive $d > 2$?** Untested. Reduced-rank
  regression is the natural test since its RLCT is also closed-form (004).
- **What is the standard visualisation once $d > 2$?** Surfaces stop being an option
  immediately and every planned model is higher dimensional (003).

## Not yet run

Experiments that need a hypothesis from Alex before they run (CONSTITUTION 1):

- **Grokking on modular addition** (008). Code is written and the run directory,
  figures and cost estimate are all specified in the entry; nothing has been trained.
  Alex starts it (CONSTITUTION 9c). If the answer is "no hypothesis, just show me the
  plot", that is a valid answer and it gets recorded as exploratory — but the entry
  should not be cited later as evidence about SLT, and it says so.
- **Edge of stability on a singular model.** All GD so far is at lr 0.12 with sharpness
  $\le 2.6$ — nowhere near $2/\eta$. Running these models at large lr is the first real
  experiment in the project.
- **A model with no conservation law.** `monomial_2_1` or the tanh net outside its
  quadratic regime, to separate "SLT explains it" from "the invariant explains it".
- **Central flows on the tanh model** (raised 2026-07-31). Setup is ready in 007 —
  two parameters, analytic derivatives, exact EoS fixed point $w_2 = \sqrt{2/\eta}$.
  Needs a hypothesis, a quantitative bar for "replicated", and a statement of what it
  has to beat: plain gradient flow plus the two-line self-stabilisation argument
  already predicts the fixed point.
