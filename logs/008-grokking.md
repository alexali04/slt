# 008 — EXPERIMENT — Grokking on modular addition (Nanda et al. replication)

- **Date:** 2026-07-31
- **Type:** EXPERIMENT — **pre-registered, NOT YET RUN.** The hypothesis section below
  is empty and stays empty until Alex fills it (CONSTITUTION 1). Claude does not write
  it, and it is not backfilled after the curves are seen.
- **Code:** `scripts/008_grokking.py` (`train` / `plot`), `src/slt/grokking.py`,
  `notebooks/008_grokking_colab.ipynb` (a runner only — no model code, CONSTITUTION 4e)
- **Figures:** `figures/raw/008_grokking_curves.png` (8a),
  `figures/diagram/008_grokking.png` (8b)
- **Run data:** `runs/008-grokking/` (CONSTITUTION 9)
- **Seeds / settings:** seed 0. The seed drives two independent streams: the train/test
  split (`seed`) and the parameter init (`seed + 1`), both on explicit CPU generators so
  the run is reproducible across devices.

## Scope note — why this model is in the repo at all

CONSTITUTION 0 says deliberately not deep neural networks. This is a neural network and
its RLCT is not known in closed form, so it is a **declared exception**, now listed in
rule 0. The reason for the exception: grokking is a delayed dynamical transition that
the *training* loss does not see, which is the shape of thing this project exists to
ask about, and it does not occur in any model in the current zoo.

The cost of the exception is real and is stated in rule 0: **3b does not apply here.**
There is no theory $\lambda$ to check against, so any SLT quantity later measured on
this model is uncalibrated until the same measurement is reproduced somewhere with a
known answer.

## Hypothesis

_(blank — Alex, CONSTITUTION 1)_

- **If true, we expect:**
- **If false, we expect:**

Prompts, if useful. This entry as written is a **replication**, and a replication whose
stated purpose is "see the plot" is legitimately exploratory — say so and it gets
recorded as exploratory, which is a valid answer under rule 1. What it is *not* is
evidence about SLT, and the next entry is where that has to be decided:

- what counts as "replicated"? (train ≥99% inside ~1k steps and test ≥99% by ~15k, with
  test flat at chance in between, is the published shape)
- does anything need to hold for a **second seed**, or is one run the deliverable?

## Baseline it is being tested against

**None, and that is the honest answer for this entry.** Nothing here is an SLT
measurement, so there is nothing for a cheap baseline to beat. Recording it explicitly
so that no later entry can cite 008 as if it were SLT evidence.

What the run does provide is the cheap baseline for whatever comes next: panel C of the
diagram figure is $\|w\|_2$ over training, logged every recorded step. Weight-norm decay
under `weight_decay=1.0` is the standard non-SLT account of why the transition lands
where it lands, and per rule 0a any SLT invariant proposed later has to beat *that*, not
just correlate with the transition.

## Method

Nanda, Chan, Lieberum, Smith, Steinhardt, *Progress Measures for Grokking via
Mechanistic Interpretability* (ICLR 2023), modular-addition setup.

| | |
|---|---|
| task | $(a+b) \bmod 113$, tokens `[a, b, =]`, answer read at the last position |
| data | all $113^2 = 12{,}769$ pairs; 3,830 train (30%) / 8,939 test, fixed split |
| model | 1 layer, $d_\text{model}=128$, 4 heads, $d_\text{head}=32$, $d_\text{mlp}=512$, ReLU, learned positional embeddings, **no LayerNorm**, 226,816 parameters |
| optimizer | full-batch AdamW, lr $10^{-3}$, weight decay 1.0, $\beta=(0.9, 0.98)$, 10-step linear warmup |
| steps | 40,000 |
| loss | cross-entropy at the final position, computed in float64 (see below) |

Deviations from the reference, all marked `NOTE` in `src/slt/grokking.py`: `W_E` is
stored transposed (identical maths, matters only if reference weights are ever loaded).

Recording schedule: every step up to 100, then every 10 (`--eval-every`). The plot is
read on a log-x axis, where uniform sampling puts almost nothing in the first decade.

**Every number in the table above is a CLI default, not a hardcoded constant** — depth,
width, heads, modulus, train fraction, optimizer, learning rate, weight decay, betas,
momentum, warmup and seed are all flags (`train --help`). A run that changes any of them
gets its own run directory and its own figure filenames, derived from what changed, so a
sweep cannot overwrite this entry's results.

**Sweeps are not this entry.** 008 is the replication at the reference configuration.
"What happens at 2 blocks / higher lr / lower weight decay" is a different question and
therefore a different log entry with its own hypothesis (CONSTITUTION 1). This entry's
numbers must stay the reference ones.

**Numerical caveat, stated in advance (rule 3c).** Loss is computed in float64 because
after grokking the train loss reaches ~$10^{-8}$ and a float32 `log_softmax` has already
lost the digits that distinguish that from $10^{-6}$. **MPS has no float64**, so on that
device the computation stays float32 and the deep tail of the loss curve is not
trustworthy below roughly $10^{-6}$. The accuracy curves — which is what the grokking
plot is — are unaffected. Run on `--device cpu` if the loss tail is the object of
interest; that is the slower path.

**Rendering choices (rule 3c).** Both figures use a log x-axis, so step 0 is not
plotted; the first recorded point is step 1. The loss panels are additionally log-y.
Panel A of the diagram marks the two milestone steps (train and test crossing 99%) on
all three panels so they can be read against each other; both are measurements, not
interpretations.

**Estimated cost (rule 9e):** ~630 TFLOP total. Estimated 20–40 min on the M5 GPU
(`--device mps`), 1.5–3 h on CPU, 10–25 min on a Colab T4. **Estimates, not
measurements** — nothing has been timed, because torch is not installed locally. The
script prints a live step/s rate and ETA from the first `--print-every` block, so the
real number is known within a minute of starting; record it below.

Note that the Colab path uses CUDA and therefore *does* get float64, so the loss tail is
trustworthy there in a way it is not on MPS.

**Actual cost:** _(pending)_

## Result

_(pending — run not started)_

## Conclusion

_(pending)_

## What this does not show

Written before the run, because it is true regardless of how the run comes out:

- Nothing about SLT. No RLCT, no learning coefficient, no volume scaling is computed
  anywhere in this entry.
- Nothing about **why** the delay happens. The entry produces the phenomenon; it does
  not explain it, and the diagram figure's annotations are restricted to measured
  crossing points for that reason.
- Nothing seed-general. One seed, one split. Grokking's transition step is known to move
  substantially between seeds, so a single run fixes the *shape* and not the *number*.
- Nothing that transfers to the rest of the zoo. This model is the declared exception in
  rule 0, and the exception does not widen just because the run succeeds.

## Next, once this exists

The reason to have a grokking model in the repo is the question in LOG.md's standing
list: *does any SLT invariant beat a cheap baseline at a dynamical question?* Grokking is
the strongest available test case, because the transition is invisible to the training
loss, which is what the cheap baselines are built out of. The checkpoints written by
`--ckpt-every` exist so that a local-learning-coefficient estimate can be run along the
saved trajectory later without retraining.

That is a separate entry and needs its own hypothesis. Flagging one trap in advance:
$\hat\lambda$ moving at the grokking step would **not** on its own be evidence for SLT —
the parameter norm moves there too, and so does everything else. The question is whether
it moves *earlier*, or *more sharply*, or predicts the transition step **before** it
happens. Anything less is rule 0a's "both SLT and the baseline explain it", which is not
evidence.
