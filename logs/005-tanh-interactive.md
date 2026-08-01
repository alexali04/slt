# 005 — DRAFT — Interactive tanh loss surface, extended domain

> ⚠️ **Partly wrong. Superseded by [006](006-tanh-far-field-correction.md).**
> Everything below about $g(w_1)$ saturating past $|w_1|\approx 30$ is a Gauss–Hermite
> quadrature artifact. The true behaviour is $g(a) = 1 - \sqrt{2/\pi}/|a| + O(a^{-2})$:
> algebraic, not exponential, so $K$ never stops depending on $w_1$. Left unedited per
> CONSTITUTION 2 so the record of what we believed survives.

- **Date:** 2026-07-31
- **Type:** DRAFT (raw)
- **Code / figure:** `figures/raw/005_tanh_surface_interactive.html` — self-contained,
  hand-authored, no generation step (the page computes $K$ itself in JS). Tracked in
  git; only `figures/**/*.png` is ignored.
- **Published:** https://claude.ai/code/artifact/07ef4ef3-e1c2-4084-a16d-5d2dc15c134a
- **Seeds / settings:** none. 20-node Gauss–Hermite quadrature, verified against
  `numpy.polynomial.hermite.hermgauss` to $2\times10^{-15}$.

## What this draws

$K(w)$ for $f_w(x) = w_2\tanh(w_1 x)$ over an adjustable domain (default $\pm 8$,
against $\pm 2.5$ in 002/003), **linear $z$ throughout**. Drag to rotate, scroll to
zoom, sliders for domain half-width, vertical scale, mesh density, elevation.

## The factorisation that makes it cheap

$$K(w_1, w_2) = \tfrac12\,\mathbb{E}_x\!\left[w_2\tanh(w_1x)\right]^2
             = \tfrac12\,w_2^2\;g(w_1), \qquad g(w_1)=\mathbb{E}_x[\tanh^2(w_1x)]$$

$K$ is **exactly quadratic in $w_2$** — no approximation. Only the 1-D function $g$
needs quadrature, which is why the page can recompute the whole surface live at any
domain. Checked: $g(w_1) \to 1$ as $|w_1| \to \infty$, $g(w_1) \approx w_1^2$ as
$w_1 \to 0$ ($g(0.01) = 1.00\times10^{-4}$, $g(30) = 1.000000$).

## Rendering choices that change what you see

- Linear $z$, as asked. No floor, no clipping.
- The $z$ axis is rescaled so the surface fills a comparable fraction of the frame at
  any domain half-width. **Widening the domain does not make the surface taller on
  screen even though $\max K$ grows like (half-width)$^2$** — the readout prints
  $\max K$ so the true scale is visible. The vertical-scale slider is an explicit
  exaggeration factor on top of that.
- Colour tracks height only and carries no extra information.
- Painter's algorithm on a quad mesh, so at low mesh settings the near-vertical walls
  at large $|w_2|$ can show sorting artifacts. Raise the mesh slider if a face looks
  wrong.

## Notes

Extending past $\pm 2.5$ makes the global shape obvious in a way the tight window hid:
away from $w_1 = 0$ the surface is just a **parabolic trough in $w_2$ with uniform
curvature**, because $g$ saturates at 1 by about $|w_1| \approx 3$. All of the
interesting geometry is the pinch where that trough's curvature collapses to zero as
$w_1 \to 0$ — outside a small neighbourhood of the origin, the model is boring and
looks regular in the $w_2$ direction.

Worth holding onto for the skeptic's side of the ledger: on a linear $z$ scale, at a
domain that a training run would actually visit, this landscape reads as an ordinary
trough. The singular structure is a small feature near the origin, and 003 already
noted it takes ~8 orders of magnitude of log scaling to see clearly.

## Revision, same day — how far out can you go

$w_1$ and $w_2$ now have independent half-widths on log sliders, 1 to 1000, plus an
origin-concentrated mesh option.

**There is no interesting limit in $w_1$.** $g$ saturates monotonically and is done:
$1-g(3) = 2.1\times10^{-1}$, $1-g(10) = 2.0\times10^{-3}$, $1-g(30) = 1.9\times10^{-9}$,
and $g(100) = 1$ to double precision. Past $|w_1| \approx 30$ the surface is
*bit-for-bit* independent of $w_1$ — a perfect parabolic cylinder $K = \frac12 w_2^2$.
Nothing is hiding further out; there is nothing left to be a function of.

**There is no limit in $w_2$ either**, in a more boring way: $K$ is exactly
$\frac12 w_2^2 g(w_1)$, so widening $w_2$ scales the surface quadratically and changes
nothing about its shape.

**The real limit is the mesh, not the model.** The entire feature lives in
$|w_1| \lesssim 3$. On a uniform grid that region gets 45 points at half-width 8, 12 at
half-width 30, and 3 at half-width 100 — past which the pinch is being drawn by the
grid rather than by $K$. Measured (mirrored in numpy, `res=120`):

| half-width | uniform mesh | concentrated mesh |
|---|---|---|
| 8 | 45 pts | 87 pts |
| 30 | 12 pts | 55 pts |
| 100 | **3 pts** | 37 pts |
| 1000 | **1 pt** | 17 pts |

The page counts these live and prints a warning below the controls when the count drops
under 12. The concentrated mesh (cubic in the normalised coordinate) keeps the pinch
resolved out to $\pm 1000$ and is the only way to view both scales at once.

Caveat on the concentrated mesh: it is a *rendering* choice with no statistical
meaning. It puts $\Delta w_1 \approx 5\times10^{-3}$ at the origin and $\approx 25$ at
the edge, so apparent surface roughness varies across the plot for reasons that have
nothing to do with $K$.

## Open threads

- The $w_1$ direction being flat-to-machine-precision for $|w_1| \gtrsim 30$ means GD
  started far out has *exactly* zero gradient in $w_1$, not just a small one. Whether
  that matters for the dynamics questions is untested.
