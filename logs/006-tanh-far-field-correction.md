# 006 — DRAFT — Correction to 005: the tanh far field, and a quadrature bug

- **Date:** 2026-07-31
- **Type:** DRAFT (raw) — **corrects [005](005-tanh-interactive.md)**, per CONSTITUTION 2
  (entries are append-only; corrections go in a new entry that links back).
- **Code:** `src/slt/models.py` (`tanh_g`), `figures/raw/005_tanh_surface_interactive.html`
- **Trigger:** Alex asked what $K$ looks like as $w_1, w_2$ get much larger, and why
  that is hard.

## What 005 got wrong

005 claimed $g(w_1) = \mathbb{E}_x[\tanh^2(w_1x)]$ saturates to 1 at machine precision
past $|w_1| \approx 30$, and that the surface is therefore "bit-for-bit independent of
$w_1$" out there. **That was an artifact of the quadrature, not a property of the
model.**

Both the artifact page (20-node) and `models.py` (64-node) used **Gauss–Hermite**. As
$w_1$ grows, $\tanh^2(w_1x)$ approaches a step function of $x$ — flat at 1 except in a
window $|x| \lesssim 1/w_1$ — and no polynomial rule tracks that. The rule reports the
saturation it cannot resolve:

| $w_1$ | true $g$ | GH-20 | GH-64 |
|---|---|---|---|
| 1 | 0.394294 | 0.394376 | 0.394294 |
| 3 | 0.745167 | 0.791833 | 0.747373 |
| 10 | 0.920537 | 0.997982 | 0.976440 |
| 100 | 0.992021 | **1.000000** | **1.000000** |

## What is actually true

$$g(a) \;=\; 1 - \sqrt{2/\pi}\,/\,|a| \;+\; O(a^{-2})$$

The approach to 1 is **algebraic, not exponential**. Verified to 4 significant figures
at $a = 100, 300, 1000, 10^4$. Derivation: $1-g(a) = \mathbb{E}[\mathrm{sech}^2(ax)]$,
and for large $a$ the $\mathrm{sech}^2$ bump has width $1/a$ and area 2, so
$1-g(a) \approx \varphi(0)\cdot 2/a = \sqrt{2/\pi}/a$.

So **$K$ never stops depending on $w_1$.** At $w_1 = 100$ the dependence is still 0.8%.
There is a real, slowly-decaying tail out there — 005 said there was nothing.

## The fix

Integrate the complement, where the structure is, after substituting $u = ax$:

$$1 - g(a) \;=\; \frac{2}{a}\int_0^{U} \mathrm{sech}^2(u)\,\varphi(u/a)\,du,
\qquad U = \min(20,\, 8a)$$

$\mathrm{sech}^2$ decays like $4e^{-2u}$, so $U = 20$ truncates below $10^{-17}$, and
the integrand is smooth on $[0,U]$ for every $a$. 48-node Gauss–Legendre holds ~$10^{-14}$
from $a = 0.01$ to $a = 10^4$, checked against `scipy.integrate.quad` (which itself
fails at $a = 1000$ — it misses the narrow feature — so the asymptotic was used as the
third opinion there).

Applied in both `models.py::tanh_g` and the artifact page. `_tanh_K` now uses the exact
factorisation $K = \frac12 w_2^2 g(w_1)$ rather than re-quadrating the whole square.

## Blast radius

- **004's RLCT for `tanh_1d` is unchanged at 0.5008** (theory 0.5). Re-run after the
  fix, byte-identical. Expected: the RLCT is a local property at the origin, where
  $|w_1|$ is small and Gauss–Hermite was accurate. The bug only ever touched the far
  field.
- 002 and 003 render `tanh_1d` on $\pm 2.5$ where the GH-64 error was $\lesssim 10^{-3}$;
  their figures move by less than a line width. Regenerated anyway.
- No other model used Gauss–Hermite.

## The actual answer to "what happens further out"

The two branches of the cross behave **completely differently** far from the origin,
which is not visible in the $\pm 2.5$ window everything before this was drawn in.
Hessian on $W_0$, measured numerically:

| point | $\lambda_{\max}(\nabla^2K)$ | limit |
|---|---|---|
| $(a, 0)$, $a = 1, 10, 100$ | 0.394, 0.921, 0.992 | $g(a) \to 1$, **bounded** |
| $(0, b)$, $b = 1, 10, 100$ | 1.00, 100.0, 10000 | $b^2$, **diverges** |

Both branches are exactly zero-loss all the way out. Both have local RLCT $1/2$. But
one has curvature saturating at 1 and the other has curvature growing without bound.

**The RLCT does not distinguish them; the Hessian does.** That is a point against SLT
carrying dynamical information here, and it is the sharpest thing the project has
produced so far — it belongs on the skeptic's side of the ledger (CONSTITUTION 0a).

## Why it is hard to *see*, as opposed to compute

Nothing about computing $K$ far out is hard. Displaying it is, for one reason: on a
linear $z$ axis a single window has to span the whole range of $K$, and that range is
enormous. Over $|w| \le 1000$, $\max K = 5\times10^5$ while the pinch region near the
origin sits at $K \sim 10^{-5}$ — a ratio of $10^{10}$. The pinch is then $10^{-10}$ of
the plot height: not small, *invisible*. Zooming the camera does not help, because the
camera scales $x$, $y$, and $z$ together.

The three things that trade off, none of which can be had at once on one linear-$z$
surface plot:

1. **Domain** — how far out in $w$ you can see.
2. **Vertical resolution** — whether the near-origin structure is more than one pixel.
3. **Mesh resolution** — whether $|w_1| \lesssim 3$ gets enough grid points to be drawn
   by $K$ rather than by the grid (005's table).

The page has controls for all three and a warning for (3), which lets you pick two. It
cannot give all three, and that is a property of linear $z$ rather than a limitation of
the renderer. A log-$z$ toggle would collapse (1) and (2) into one view at the cost of
no longer being the linear picture Alex asked for — not added unilaterally.

## Notes on process

This bug survived a check that looked convincing: 005 recorded the Gauss–Hermite nodes
and weights as "verified against numpy to $2\times10^{-15}$." That was true and
irrelevant — it verified the *nodes*, not that the *rule* converges for this integrand.
Worth remembering as a template for how numerical QA gives false confidence: matching a
reference implementation of a subroutine says nothing about whether the subroutine is
the right one for the problem. CONSTITUTION 3b says every analytic value gets a
numerical check; this entry says the check has to be against the quantity, not against
another implementation of the same method.
