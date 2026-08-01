# slt

Investigating whether singular learning theory says anything useful about optimization
dynamics — progressive sharpening, edge of stability, and gradient descent living in a
tiny subspace.

The models are deliberately small and singular, not deep networks: monomial crossings,
tiny tanh nets, and (planned) normal mixtures, binomial mixtures, reduced-rank
regression, Bayesian networks, HMMs. They are chosen because their RLCTs are known in
closed form, which makes it possible to check the machinery against a right answer.

**SLT is the thing on trial here, not the assumed framework.** See CONSTITUTION 0a.

## Read these first

| file | what it is |
|---|---|
| [CONSTITUTION.md](CONSTITUTION.md) | working agreements between Alex and Claude. Read at the start of every session. |
| [LOG.md](LOG.md) | index of every draft and experiment, plus the standing open questions |

## Layout

```
src/slt/      importable library, no side effects on import
  models.py     loss landscapes as pure functions of parameters
  rlct.py       volume-scaling RLCT / multiplicity estimators
  dynamics.py   GD, Hessians, sharpness
  viz.py        shared style and reusable panels
  grokking.py   1-layer transformer on modular addition (needs torch; not
                re-exported from __init__, so `import slt` works without it)
scripts/      NNN_name.py, one per log entry, runnable and reproducible
notebooks/    thin Colab runners for the entries that want a GPU (CONSTITUTION 4e)
logs/         NNN-slug.md, one per draft or experiment
figures/
  raw/          the object, unannotated (CONSTITUTION 8a)
  diagram/      the object plus the argument (CONSTITUTION 8b)
runs/         NNN-slug/, metrics and checkpoints from long runs (CONSTITUTION 9)
```

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
.venv/bin/python scripts/001_regular_landscapes.py
```

Every script writes into `figures/` and prints its numbers to stdout. `figures/` is
regenerable and safe to delete.

Entry 008 is the exception: it trains a model for tens of minutes, so it splits into a
`train` step that writes `runs/008-grokking/` and a `plot` step that reads it
(CONSTITUTION 9b). It needs torch, which the other entries do not.

```bash
.venv/bin/pip install torch
.venv/bin/python scripts/008_grokking.py train --device auto
.venv/bin/python scripts/008_grokking.py plot --run 008-grokking
```

`train --help` lists the model and optimizer flags (depth, width, lr, weight decay,
optimizer, seed). Changing any of them gives the run its own **run id**, which is its
directory name and also names its figures — `--n-layers 2` is `008-grokking-L2`. `train`
prints the id, `list` lists them all, `plot --run <id>` draws one and
`compare --runs <ids>` overlays several. Add `--tensorboard` to watch a run live
(`tensorboard --logdir runs`) — a view only, `metrics.csv` stays the record.

On a GPU instead, open [`notebooks/008_grokking_colab.ipynb`](notebooks/008_grokking_colab.ipynb)
in Colab. It clones this repo and drives the same script — no model code lives in the
notebook (CONSTITUTION 4e).
