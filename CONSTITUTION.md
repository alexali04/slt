# CONSTITUTION

Standing agreements between Alex and Claude for this codebase. This is the document
of record for *how we work*, not *what we found* (that is `LOG.md`).

**Read this file at the start of every session.** When a new working agreement is
established in conversation, add it here as a numbered rule with the date.

---

## 0. Purpose of the project

Investigate the relationship between:

- **Optimization phenomenology** — progressive sharpening, edge of stability (EoS),
  gradient descent confining itself to a tiny subspace (the "top-$k$ Hessian
  eigenspace" phenomenon), and
- **Singular learning theory** (SLT) — Watanabe's framework, where the map from
  parameters to distributions is non-injective and the Fisher information matrix is
  degenerate, so the free energy is governed by the **real log canonical threshold**
  (RLCT / learning coefficient) $\lambda$ rather than by $d/2$.

Working thesis to be tested, not assumed: *the degeneracy structure that SLT measures
(the geometry of the zero set of the loss, its RLCT and multiplicity) is the same
structure that governs sharpening dynamics and the low-dimensional subspace GD lives
in.* We do not yet know if this is true.

Deliberately **not** deep neural networks. The model zoo is small singular models where
the RLCT is known analytically or computable: normal mixtures, binomial mixtures,
reduced-rank regression, Bayesian networks, HMMs, tiny tanh networks, monomial toy
losses.

**Declared exceptions.** A model that breaks the rule above is listed here or it is not
in the repo. Listing it is the point: the reason it is worth the exception has to be
written down where the rule is, not buried in a log entry.

- **008, the grokking transformer.** A one-layer transformer on $(a+b) \bmod 113$. Its
  RLCT is not known in closed form and it is a real, if small, neural network. It is
  here because grokking is a **delayed dynamical transition that the training loss does
  not see** — the thing this project is about — and it cannot be studied on a model that
  does not exhibit it. The cost of the exception is that rule 3b does not apply to it:
  there is no theory value to check the machinery against, so anything measured on this
  model is uncalibrated until it is reproduced on a model where there is one.

### 0a. Epistemic stance — SLT is the thing on trial

Alex's prior is **skeptical of SLT's practical utility**, and the codebase is built to
be capable of returning that verdict. Concretely:

- The live question is not "how does SLT explain sharpening?" but "**does SLT buy
  anything here that a simpler description does not?**" Every SLT quantity we compute
  should be pitted against a cheap baseline (Hessian rank, top eigenvalue, parameter
  norm, a plain quadratic model of the loss), and the baseline is allowed to win.
- **"SLT predicts X" is only interesting if something else predicts not-X.** A result
  that both SLT and a quadratic model explain is not evidence for SLT.
- Watanabe's mathematics is not in dispute; the theorems are theorems. What is in
  dispute is whether the RLCT is a *useful* description of anything a practitioner
  cares about, especially anything about **optimization dynamics**, which SLT (a
  Bayesian, equilibrium theory) does not actually make claims about.
- Claude must not launder SLT vocabulary into explanation. Saying a valley is "flatter
  because $\lambda$ is smaller" is circular — $\lambda$ is *defined* by the volume
  scaling. State the measurement, not the mysticism.
- Watch for the confirmation-bias failure mode where a toy model is chosen *because*
  it makes SLT look good. Toy models here are chosen because their RLCT is known in
  closed form, which is a different reason, and the two must not be quietly conflated.

---

## 1. The Method Rule

> **When Alex asks Claude to run an experiment, Claude must first ask Alex to state the
> hypothesis being tested and/or the potential conclusions that could be drawn.**
>
> It is fine for the answer to be "there is no hypothesis, I'm just looking." In that
> case Claude records the run as **exploratory** and moves on — but Claude must still
> have asked. The reminder is the point.

Rationale: this codebase will accumulate runs quickly. A run with no recorded question
attached to it is a run whose output cannot be interpreted six weeks later.

**Corollary 1a — the drafting exemption.** Producing a *visualization* of a known
object (plotting $L(w) = (w_1 w_2)^2$, rendering a landscape, making a figure prettier)
is **drafting**, not an experiment. No hypothesis is required. Drafting is still
logged, just under a lighter template.

**Corollary 1b — the classification is declared, not inferred.** Every log entry
declares itself `DRAFT` or `EXPERIMENT` in its header. If Claude is unsure which a task
is, it asks.

**Corollary 1c — negative and null results are logged.** An experiment whose answer was
"no effect" or "the estimator didn't converge" gets the same log entry as a successful
one. Deleting failed runs is how a research codebase starts lying to you.

---

## 2. Logging

- `LOG.md` at the repo root is the **index**: one line per entry, newest last.
- `logs/NNN-slug.md` is the entry itself. `NNN` is a zero-padded serial, never reused.
- Every entry carries: serial, date, `DRAFT`/`EXPERIMENT`, the code that produced it,
  the figures it produced, and (for experiments) hypothesis → result → conclusion.
- Templates live in `logs/TEMPLATE-draft.md` and `logs/TEMPLATE-experiment.md`.
- An entry is **append-only once written**. Corrections go in a new entry that links
  back, so the record of what we believed at the time survives.

---

## 3. Claims and numbers

3a. **Theory values are cited; empirical values are measured.** Any $\lambda$ quoted in
a log or docstring is labelled either `theory` (with a derivation or reference) or
`empirical` (with the estimator and its settings). Never blur the two.

3b. **Every analytically-known RLCT gets a numerical check.** If we know
$\lambda = 1/4$ for $L(w) = w^4$, the code must also estimate it from volume scaling and
the two must be shown side by side. Agreement is evidence the estimator works;
disagreement is a finding.

3c. **No silent fudging.** If a plot needed clipping, a log scale, a jitter, or an
epsilon to look right, that is stated in the log entry.

---

## 4. Code layout

```
slt/
├── CONSTITUTION.md      # this file
├── LOG.md               # index of log entries
├── logs/                # one file per draft/experiment
├── src/slt/             # importable library — no side effects on import
│   ├── models.py        # loss landscapes, as pure functions of parameters
│   ├── rlct.py          # RLCT / learning-coefficient estimators
│   ├── dynamics.py      # GD, sharpness tracking, Hessian spectra
│   ├── grokking.py      # rule 0 exception — needs torch, so kept out of __init__
│   └── viz.py           # shared plotting style + reusable panels
├── scripts/             # NNN_name.py — one script per log entry, runnable, reproducible
├── notebooks/           # rule 4e — thin runners for hosted GPUs, never implementations
├── runs/                # rule 9 — metrics and checkpoints from long runs; git-ignored
└── figures/             # generated output; regenerable, safe to delete
    ├── raw/             # rule 8a — the object, unannotated
    └── diagram/         # rule 8b — the object plus the argument
```

4a. **Library is pure, scripts have effects.** `src/slt/` never writes files or shows
plots at import time. `scripts/` does the I/O.

4b. **One script per log entry**, named with the same serial. `scripts/002_foo.py`
produces the figures logged in `logs/002-foo.md`.

4c. **Figures are disposable, scripts are not.** `figures/` is regenerable by running
the scripts. Never hand-edit a figure.

4d. **Seeds are fixed and recorded** for anything stochastic.

4e. **Notebooks drive, they do not implement.** A notebook in `notebooks/` exists to get a
script onto hardware this laptop does not have — a hosted GPU — and may contain setup,
invocation, and display. It must not contain a model, an estimator, or a training loop.
The moment a cell holds logic the repo does not, the run stops being reproducible from
the repo, and the notebook's outputs become the only record of what was run.

Notebooks are named for the entry they run (`notebooks/008_grokking_colab.ipynb`) and
are committed **without output cells** — a notebook full of stale outputs is a figure
nobody can regenerate (4c), with the added problem that its outputs get read as results.

---

## 5. Mathematical conventions

- $w \in W \subseteq \mathbb{R}^d$ is the parameter, $d$ its dimension.
- $K(w)$ is the **population** loss normalised so $\min_w K(w) = 0$ (in SLT, the KL
  divergence $K(w) = \mathrm{KL}(q \,\|\, p_w)$). $L_n(w)$ is the empirical loss.
- $W_0 = \{w : K(w) = 0\}$ is the set of true parameters. A model is **regular** iff
  $W_0$ is a single point and the Fisher information there is positive definite;
  otherwise **singular**.
- $\lambda$ is the RLCT (learning coefficient), $m$ its multiplicity. Free energy
  asymptotic: $F_n = n L_n(\hat w) + \lambda \log n - (m-1)\log\log n + O_p(1)$.
- Regular models have $\lambda = d/2$, $m = 1$. **$\lambda \le d/2$ always.**
- "Sharpness" means $\lambda_{\max}(\nabla^2 L)$ unless stated otherwise. Note the
  unfortunate collision with $\lambda$ for the RLCT: in code, `rlct` and `sharpness`
  are always spelled out, never `lambda`.

---

## 6. Conceptual guardrails

These are distinctions that are easy to lose and expensive to lose.

6a. **Ill-conditioned ≠ singular.** A regular model with a tiny Hessian eigenvalue has
a narrow valley but still a *point* minimum and still $\lambda = d/2$. Singularity is
about the minimum set having positive dimension or a non-quadratic contact order, not
about small numbers. Figures should make this contrast explicitly.

6b. **Degeneracy is a property of $K$ near $W_0$, not of a single point.** The RLCT is
determined by the germ of $K$ along its zero set.

6c. **The Hessian fails for Bayesian volume. It does not follow that it fails for
dynamics.** These are two different claims and only the first is established.

*What is actually true:* the Laplace approximation to the marginal likelihood assumes an
invertible Hessian at the minimum. At a singular point the Hessian is rank-deficient (or
zero), the Gaussian integral does not converge to the right thing, and the volume of the
near-minimal set scales like $\epsilon^{\lambda}(\log 1/\epsilon)^{m-1}$ instead of
$\epsilon^{d/2}$. That is Watanabe's theorem and it is not in question. It is a
statement about **free energy and model selection**.

*What is not established, and what we must not assume:* that a second-order model of the
loss is therefore a bad description of **training dynamics**. Empirically it is often a
good one. Quadratic and second-order-regression models reproduce progressive sharpening
and the $2/\eta$ edge-of-stability plateau; the top-$k$ Hessian eigenspace does capture
where most of the gradient lives in real networks. This works in regimes where the
Bayesian Laplace approximation is invalid, including LLM training. The theoretical
gap has not stopped the Hessian from being predictive.

So the honest framing of this project's central question is:

> The Hessian is a bad tool for Bayesian volume and an empirically decent tool for
> dynamics. SLT's invariants are the right tool for Bayesian volume. **Is there any
> dynamical question where they beat the Hessian?** If the answer is no, that is a
> result, and it goes in the log.

Corollary: whenever a figure or claim in this repo contrasts "Hessian" with "RLCT", it
must say which of the two questions — volume or dynamics — it is contrasting them on.

6d. **Local vs global.** The RLCT of a model is a min over $W_0$; the *local* learning
coefficient at a given $w^*$ is what dynamics actually sees. Always say which.

---

## 7. Interaction defaults

7a. Claude proposes, Alex disposes: for anything beyond a figure, state the plan
briefly before writing a lot of code.

7b. Prefer few, dense figures over many thin ones — *within the diagram type* defined
in rule 8. Never mix the two types in one file.

7c. Analytical results preferred where available; simulation used to confirm, not to
substitute.

7d. Dependencies stay minimal (`numpy`, `scipy`, `matplotlib`). Adding one is a
conversation. `torch` is expected eventually for MCMC-based LLC estimation and is
pre-approved for that purpose, and from 2026-07-31 also for the grokking model of entry
008 (rule 0, declared exceptions).

`torch` stays **optional**: `import slt` must keep working for someone who installed only
numpy/scipy/matplotlib. Any module that needs torch is therefore left out of
`src/slt/__init__.py` and imported by name (`from slt import grokking`), and a script's
plotting path must not import it — you should be able to draw a figure from a run
directory on a machine that cannot run the model.

---

## 8. Two kinds of figure

Every figure produced in this repo is exactly one of these. The type is declared in the
script docstring and in the log entry, and it determines where the file is written.

### 8a. Raw figures — `figures/raw/`

**Just the object.** A 3-D surface of $K(w)$, a curve, a heatmap. No annotation boxes,
no arrows, no takeaway text, no derived quantities overlaid. Title names the model and
the formula; axes are labelled; that is all.

The purpose is to let Alex *look at the thing* without being told what to see. A raw
figure should be reusable in a context where the surrounding argument is different, or
where there is no argument yet. If a raw figure needs a paragraph to be intelligible,
it is a failed raw figure — fix the rendering, don't add prose to the image.

Rules specific to raw figures:

- One model per file where practical; contact sheets are allowed for side-by-side
  comparison but keep the same no-annotation rule.
- Where the dynamic range demands it, show the same surface under more than one
  $z$-scaling (linear and $\log$). State the scaling in the axis label — that is a
  label, not an annotation.
- **Any transform applied to make the surface visible (log, clipping, floor,
  $z$-limit) must appear in the axis label or title**, since there is no annotation
  box to put it in. This is rule 3c applied to raw figures.

### 8b. Diagram figures — `figures/diagram/`

**The object plus the argument.** Multi-panel, annotated, with the takeaway written on
the figure. These are the ones that carry an explanation: sub-level sets with their
scaling called out, side-by-side contrasts, fitted exponents, overlaid trajectories.

Rules specific to diagram figures:

- Every panel must be readable without the surrounding prose: title states the model,
  axes labelled with meaning, annotation carries the takeaway.
- An annotation states a **measurement or a definition**, never an interpretation that
  the figure does not show. If the claim needs an experiment, it belongs in a log
  entry, not on a figure (rule 0a).

### 8c. Derivation order

Draft the **raw** figure first, look at it, and only then decide what the diagram
should argue. A diagram built before anyone has looked at the raw surface is a diagram
arguing for a preconception.

---

## 9. Long runs and run data

Rule 8 covers figures. This rule covers the other thing an entry can produce: a run
measured in hours, whose real output is a stream of numbers that a figure is derived
from later. Figures are disposable (4c). **Run data is not** — regenerating it costs the
hours again.

9a. **The run directory.** Everything a long run produces that is not a figure goes in
`runs/NNN-slug/`:

- `config.json` — every hyperparameter, the seed, the device, library versions. Written
  **before the first step**, not after the last one, so an interrupted run is still
  self-describing.
- `metrics.csv` — one row per recorded step, **flushed as it is written**. Never
  buffered until the end; the end may not arrive.
- `ckpt-NNNNNN.pt`, `last.pt` — checkpoints, where there are any.
- `tb/` — TensorBoard event files, where a run streams them.

**Live dashboards are a view, not the record.** A number may be read out of TensorBoard
only if it is also in `metrics.csv`; nothing is logged, plotted or claimed from a
dashboard alone. Event files are as disposable as figures — deleting `tb/` must cost
nothing, and a run that writes only to a dashboard has not recorded anything.

9b. **Train and plot are separate entry points of the same script.** Rule 4b still holds
— one script per log entry — so this is `python scripts/NNN_x.py train` and
`python scripts/NNN_x.py plot`, not two files. Plotting must never re-train. A run killed
at step 3,000 of 40,000 must still plot, and the figure must say where it stopped.

9c. **Claude sets long runs up; Alex starts them.** Claude does not begin a run measured
in hours unless asked to. Before handing over it states the expected wall-clock cost, the
device that estimate assumes, and the exact paths the output will appear at. A smoke run
of a few hundred steps to prove the pipeline is not a long run and needs no ceremony.

9d. **`runs/` is git-ignored**, like `figures/`. Any number that matters is therefore
copied into the log entry. A finding that exists only inside a run directory is one
`rm -rf` from being lost.

9e. **Estimated cost before, actual cost after**, both in the log entry. That is the only
way the next estimate gets better.

9f. **Partial is a legitimate state.** A run still going, or stopped early, is logged as
partial with the step it reached. Same spirit as 1c: the record says what happened.

---

## Amendment record

| Date | Rule | Change |
|------|------|--------|
| 2026-07-31 | 0–7 | Initial constitution drafted from project kickoff. |
| 2026-07-31 | 0a | Added. Alex is a skeptic of SLT's practical utility; the codebase must be able to return a negative verdict, and SLT quantities must be pitted against cheap baselines. |
| 2026-07-31 | 6c | **Corrected.** Previously claimed "the Hessian is the wrong tool at a singularity". Overstated: that is true for Bayesian volume (Laplace) and not established for dynamics, where second-order models empirically track progressive sharpening and EoS, including in LLMs. Rewritten as a scope distinction. |
| 2026-07-31 | 8, 7b | Added the raw / diagram figure split, with `figures/raw/` and `figures/diagram/`. |
| 2026-07-31 | 9 | Added. Long runs write a `runs/NNN-slug/` directory; train and plot are separate entry points of one script; Alex starts anything measured in hours. Prompted by entry 008. |
| 2026-07-31 | 0 | Added the **declared exceptions** clause and listed 008's grokking transformer, which is a neural network with no known RLCT and so is outside the model zoo rule 0 describes. |
| 2026-07-31 | 7d | `torch` pre-approval widened from MCMC-based LLC estimation to also cover 008, and the rule that it stays an *optional* dependency written down. |
| 2026-07-31 | 4, 4e | Added `notebooks/` and `runs/` to the layout, and rule 4e: notebooks drive scripts onto hosted GPUs, never implement anything, and are committed without output cells. |
| 2026-07-31 | 9a | Added `tb/` to the run directory, and the rule that a live dashboard is a view and never the record: nothing is claimed from TensorBoard that is not also in `metrics.csv`. |
