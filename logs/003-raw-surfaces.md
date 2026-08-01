# 003 — DRAFT — Raw 3-D loss surfaces

- **Date:** 2026-07-31
- **Type:** DRAFT (raw)
- **Code:** `scripts/003_raw_surfaces.py`
- **Figures:** `figures/raw/003_contact_sheet_linear.png`,
  `figures/raw/003_contact_sheet_log.png`, `figures/raw/003_1d_curves.png`,
  `figures/raw/003_surface_<key>.png` (8 files, one per 2-D model)
- **Seeds / settings:** deterministic. Surfaces on a $240\times240$ grid
  (contact sheets $200\times200$), view `elev=30, azim=-125`.

## What this draws

$K(w)$ as a 3-D surface over the $(w_1, w_2)$ plane, for all eight 2-D models, with no
annotation of any kind (CONSTITUTION 8a). Each model also gets its own file showing the
same surface twice, linear-$z$ and log-$z$.

This entry exists because 001 and 002 were built as diagrams before anyone had looked
at the plain surface — the wrong order, now written into the constitution as rule 8c.

## Rendering choices that change what you see

- **The log-$z$ panels plot $\log_{10}(K + 10^{-8})$.** The floor $10^{-8}$ is
  arbitrary and it sets the depth of the funnels: a smaller floor makes every singular
  valley look deeper without changing anything about the model. Do not read funnel
  depth as a property of the landscape. It is stated in every $z$-axis label.
- The surface colormap is truncated to drop its two lightest steps, so the valley floor
  keeps some shading instead of washing out into the page. Colour is redundant with
  height here; it carries no extra information.
- Grid resolution ($200$–$240$) undersamples the valley floor of the higher-order
  monomials, where $K$ drops through many orders of magnitude between adjacent grid
  points. The visible faceting near the axes in `monomial_3_1` is the mesh, not the
  function.
- Linear-$z$ panels are dominated by the corners of the box, which is exactly why the
  log panels exist. Both are kept because the linear one is the honest default and the
  log one is the informative transform.

## Notes

The log contact sheet is the single most useful image produced so far. Read left to
right, top to bottom:

- The three regular/degenerate linear models each show **one** feature — a single point
  funnel (isotropic), a single point funnel in a tilted bowl (correlated), and a
  straight trench (collinear, where $W_0$ is a line).
- The five singular models all show the **same qualitative object**: two orthogonal
  trenches meeting at a funnel at the origin. Raising the monomial exponents widens the
  trenches; it does not change the topology.
- `tanh_1d` is visually indistinguishable from `monomial_1_1` near the origin, which
  is what the Taylor expansion in 002 says it should be. Away from the origin the tanh
  surface saturates and the monomial does not — the outer parts of the two surfaces
  look nothing alike. Worth remembering that the RLCT is a statement about a germ at a
  point and says nothing about the rest of the picture.
- The linear-$z$ contact sheet shows almost none of this. Every singular surface looks
  like a smooth four-lobed bowl. If someone eyeballed these landscapes on a linear
  scale they would not notice the cross at all.

That last point cuts both ways and is worth being honest about: the structure SLT cares
about is only *visible* here under a log transform spanning eight orders of magnitude in
$K$. Whether structure that requires that much magnification to see is structure that
matters to a training run is an open question, not a settled one.

## Open threads

- No 3-D surface exists for anything above $d=2$, and every model in the eventual zoo
  (normal mixtures, reduced-rank regression, HMMs) is higher dimensional. Surfaces stop
  being an option almost immediately; 2-D slices and sub-level-set volume are what
  survive. Worth deciding early what the standard visualisation is for $d > 2$ rather
  than improvising it per model.
- The floor $10^{-8}$ should probably be derived per model (e.g. from the grid spacing
  and the local order of vanishing) rather than fixed, so that funnel depth becomes
  comparable across panels. Currently it is not comparable and the contact sheet should
  not be read as if it were.
