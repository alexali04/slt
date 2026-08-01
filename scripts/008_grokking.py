"""008 — EXPERIMENT — Grokking on modular addition (Nanda et al. replication).

**NOT YET RUN.** The hypothesis slot in `logs/008-grokking.md` is empty and
CONSTITUTION 1 says it is filled by Alex before the run, not after. This file
exists so the run is ready to start, not so it starts.

A one-layer transformer on $(a+b) \\bmod 113$, trained full-batch with AdamW and
weight decay 1.0 on 30% of the pairs. Train accuracy saturates within ~1k steps;
test accuracy stays at chance for roughly a further decade of steps and then
rises to ~100%. Reproducing that gap is the whole deliverable.

Two entry points, because the run is measured in hours and the plot in seconds
(CONSTITUTION 9b):

    python scripts/008_grokking.py train --device auto
    python scripts/008_grokking.py plot

`train` writes `runs/008-grokking/` and needs torch. `plot` reads that directory
and needs only numpy + matplotlib. A run killed halfway still plots.

Figures:
  figures/raw/008_grokking_curves.png   — the curves, unannotated (CONSTITUTION 8a)
  figures/diagram/008_grokking.png      — with the transition measured (8b)

See logs/008-grokking.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (path side effect)
import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import ROOT, run_dir, save

from slt import viz as V

THEME = V.LIGHT
V.use_style(THEME)
C = THEME.categorical

RUN_NAME = "008-grokking"
ACC_THRESHOLD = 0.99  # what counts as "solved", for both memorisation and grokking


# --- Run directory ------------------------------------------------------------


def paths(tag: str | None) -> dict[str, Path]:
    directory = run_dir(RUN_NAME if not tag else f"{RUN_NAME}-{tag}")
    return {
        "dir": directory,
        "config": directory / "config.json",
        "metrics": directory / "metrics.csv",
        "last": directory / "last.pt",
    }


def read_metrics(path: Path) -> np.ndarray:
    """Load metrics.csv as a structured array, sorted by step, one row per step.

    A resumed run can leave the CSV with rows past the checkpoint it restarted
    from, so later rows win on a duplicate step.
    """
    if not path.exists():
        sys.exit(f"no metrics at {path} — run `train` first")
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if rows.size == 0:
        sys.exit(f"{path} has a header and no rows")
    order = np.argsort(rows["step"], kind="stable")
    rows = rows[order]
    keep_last = np.append(np.diff(rows["step"]) != 0, True)
    return rows[keep_last]


def first_reaching(steps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    """First recorded step where ``values`` is at least ``threshold``.

    ``None`` means it never got there in what was recorded. That is a result and
    the caller prints it as one (CONSTITUTION 1c).
    """
    hit = np.flatnonzero(values >= threshold)
    return int(steps[hit[0]]) if hit.size else None


def milestones(rows: np.ndarray) -> dict:
    steps = rows["step"]
    memorised = first_reaching(steps, rows["train_acc"], ACC_THRESHOLD)
    grokked = first_reaching(steps, rows["test_acc"], ACC_THRESHOLD)
    peak = int(np.argmax(rows["weight_norm"]))
    return {
        "memorised_step": memorised,
        "grokked_step": grokked,
        "ratio": (grokked / memorised) if (memorised and grokked) else None,
        "final_step": int(steps[-1]),
        "final_train_acc": float(rows["train_acc"][-1]),
        "final_test_acc": float(rows["test_acc"][-1]),
        "norm_peak_step": int(steps[peak]),
        "norm_peak": float(rows["weight_norm"][peak]),
        "norm_final": float(rows["weight_norm"][-1]),
    }


def print_milestones(m: dict) -> None:
    def fmt(step):
        return "never" if step is None else f"{step:,}"

    print(f"  train acc >= {ACC_THRESHOLD:.0%} at step  {fmt(m['memorised_step'])}")
    print(f"  test  acc >= {ACC_THRESHOLD:.0%} at step  {fmt(m['grokked_step'])}")
    if m["ratio"]:
        print(f"  grokking gap                  {m['ratio']:.1f}x in steps")
    print(f"  final (step {m['final_step']:,}): train {m['final_train_acc']:.4f}, "
          f"test {m['final_test_acc']:.4f}")
    print(f"  ||w|| peaks {m['norm_peak']:.1f} at step {m['norm_peak_step']:,}, "
          f"ends {m['norm_final']:.1f}")


# --- train --------------------------------------------------------------------


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_train(args: argparse.Namespace) -> None:
    # torch is imported here and not at module scope so that `plot` runs on a
    # machine that has only numpy and matplotlib (CONSTITUTION 7d).
    import torch

    from slt import grokking as G

    p = paths(args.tag)
    cfg = G.Config(steps=args.steps, seed=args.seed)
    device = pick_device(args.device)

    run = G.Run(cfg, device=device)
    resumed_from = 0
    if args.resume and p["last"].exists():
        run.load_state_dict(torch.load(p["last"], map_location=device, weights_only=False))
        resumed_from = run.step
        print(f"  resumed from step {resumed_from:,}")
    elif args.resume:
        print(f"  --resume given but no checkpoint at {p['last']}; starting fresh")

    p["config"].write_text(json.dumps(
        {"config": cfg.to_dict(), "device": device, "n_params": run.n_params,
         "acc_threshold": ACC_THRESHOLD, "eval_every": args.eval_every,
         "torch": torch.__version__},
        indent=2,
    ) + "\n")

    print(f"008 — grokking, ({cfg.p} x {cfg.p}) modular addition")
    print(f"  device        {device}")
    print(f"  parameters    {run.n_params:,}")
    print(f"  train / test  {cfg.n_train:,} / {cfg.n_test:,} pairs "
          f"({cfg.train_frac:.0%} train)")
    print(f"  steps         {cfg.steps:,} (full batch, AdamW lr={cfg.lr}, "
          f"wd={cfg.weight_decay})")
    print(f"  seed          {cfg.seed}")
    print(f"  writing       {p['dir'].relative_to(ROOT)}/")
    print()

    fresh = not (args.resume and p["metrics"].exists())
    handle = p["metrics"].open("w" if fresh else "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=G.RECORD_FIELDS)
    if fresh:
        writer.writeheader()

    started = time.perf_counter()
    interrupted = False
    try:
        for record in run.iterate(
            until=cfg.steps, should_record=G.dense_then_every(args.eval_every)
        ):
            writer.writerow(record.__dict__)
            handle.flush()  # a killed run must still plot (CONSTITUTION 9b)

            if record.step % args.print_every == 0 or record.step == cfg.steps:
                done = record.step - resumed_from
                rate = done / max(time.perf_counter() - started, 1e-9)
                remaining = (cfg.steps - record.step) / rate if rate else float("nan")
                print(f"  step {record.step:>7,}  "
                      f"train {record.train_loss:9.2e}/{record.train_acc:.3f}  "
                      f"test {record.test_loss:9.2e}/{record.test_acc:.3f}  "
                      f"|w| {record.weight_norm:6.1f}  "
                      f"{rate:5.1f} step/s  eta {remaining / 60:5.1f} min")

            if args.ckpt_every and record.step % args.ckpt_every == 0:
                torch.save(run.model.state_dict(),
                           p["dir"] / f"ckpt-{record.step:06d}.pt")
                torch.save(run.state_dict(), p["last"])
    except KeyboardInterrupt:
        interrupted = True
        print("\n  interrupted")
    finally:
        handle.close()
        torch.save(run.state_dict(), p["last"])

    elapsed = time.perf_counter() - started
    print(f"\n  stopped at step {run.step:,} after {elapsed / 60:.1f} min"
          f"{' (interrupted)' if interrupted else ''}")
    print_milestones(milestones(read_metrics(p["metrics"])))
    print(f"\n  now: python scripts/008_grokking.py plot"
          f"{f' --tag {args.tag}' if args.tag else ''}")
    if interrupted:
        print(f"  or resume: python scripts/008_grokking.py train --resume"
              f"{f' --tag {args.tag}' if args.tag else ''}")


# --- plot ---------------------------------------------------------------------


def figure_raw(rows: np.ndarray, cfg: dict) -> None:
    """CONSTITUTION 8a: the object, no annotation. Look at this one first."""
    step = rows["step"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.plot(step, rows["train_acc"], color=C[0], label="train")
    ax.plot(step, rows["test_acc"], color=C[1], label="test")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("optimizer step (log scale)")
    ax.set_ylabel("accuracy")
    ax.legend(loc="center left")
    V.tidy(ax, theme=THEME, grid="both")

    ax = axes[1]
    ax.plot(step, rows["train_loss"], color=C[0], label="train")
    ax.plot(step, rows["test_loss"], color=C[1], label="test")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step (log scale)")
    ax.set_ylabel("cross-entropy loss (log scale)")
    ax.legend(loc="center left")
    V.tidy(ax, theme=THEME, grid="both")

    fig.suptitle(
        f"1-layer transformer, $(a+b)\\ \\mathrm{{mod}}\\ {cfg['p']}$, "
        f"{cfg['train_frac']:.0%} of pairs used for training, full batch AdamW "
        f"(lr {cfg['lr']}, weight decay {cfg['weight_decay']}), seed {cfg['seed']}",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "008_grokking_curves.png", kind="raw")
    plt.close(fig)


def figure_diagram(rows: np.ndarray, cfg: dict, m: dict) -> None:
    """CONSTITUTION 8b: the object plus the argument. Every annotation below is
    a measurement read off these axes or a setting from config.json — nothing
    here interprets *why* the transition happens."""
    step = rows["step"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    def mark(ax):
        """The two milestone steps, on every panel, so the panels can be read
        against each other. Labels run vertically along the line: horizontal
        ones sit where the titles and annotation boxes already are."""
        for value, colour, label in (
            (m["memorised_step"], C[0], "train"),
            (m["grokked_step"], C[1], "test"),
        ):
            if value:
                ax.axvline(value, color=colour, lw=1.0, ls=(0, (4, 3)), alpha=0.7,
                           zorder=1)
                ax.text(value, 0.03, f"{label} {ACC_THRESHOLD:.0%} ",
                        transform=ax.get_xaxis_transform(), fontsize=7,
                        color=colour, va="bottom", ha="right", rotation=90)

    ax = axes[0]
    ax.plot(step, rows["train_acc"], color=C[0], label="train")
    ax.plot(step, rows["test_acc"], color=C[1], label="test")
    ax.axhline(1.0 / cfg["p"], color=THEME.ink_muted, lw=0.9, ls=":")
    ax.text(step[0], 1.0 / cfg["p"], " chance", fontsize=7,
            color=THEME.ink_muted, va="bottom", ha="left")
    mark(ax)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("optimizer step (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("A.  Train fits early, test generalises late")
    ax.legend(loc="center left")

    def reached(step: int | None, what: str) -> str:
        if step is None:
            return f"{what} never reached {ACC_THRESHOLD:.0%}"
        return f"{what} $\\geq$ {ACC_THRESHOLD:.0%} at step {step:,}"

    gap = (f"test lags train by {m['ratio']:.1f}$\\times$ in steps"
           if m["ratio"] else "no gap to quote")
    V.annotate(
        ax,
        f"{reached(m['memorised_step'], 'train acc')}\n"
        f"{reached(m['grokked_step'], 'test acc')}\n"
        f"{gap}",
        xy=(0.97, 0.45), va="top", ha="right", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="both")

    ax = axes[1]
    ax.plot(step, rows["train_loss"], color=C[0], label="train")
    ax.plot(step, rows["test_loss"], color=C[1], label="test")
    mark(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step (log scale)")
    ax.set_ylabel("cross-entropy loss (log scale)")
    ax.set_title("B.  Test loss rises before it falls")
    ax.legend(loc="lower left")
    V.annotate(
        ax,
        f"final train loss {rows['train_loss'][-1]:.2e}\n"
        f"final test loss  {rows['test_loss'][-1]:.2e}\n"
        f"both axes log; step 0 not shown",
        xy=(0.97, 0.97), va="top", ha="right", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="both")

    ax = axes[2]
    ax.plot(step, rows["weight_norm"], color=C[6])
    mark(ax)
    ax.set_xscale("log")
    ax.set_xlabel("optimizer step (log scale)")
    ax.set_ylabel(r"$\|w\|_2$, all parameters")
    ax.set_title("C.  The cheap baseline: parameter norm")
    V.annotate(
        ax,
        f"weight decay = {cfg['weight_decay']} (setting)\n"
        f"peak {m['norm_peak']:.1f} at step {m['norm_peak_step']:,}\n"
        f"final {m['norm_final']:.1f}",
        xy=(0.97, 0.97), va="top", ha="right", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="both")

    fig.suptitle(
        f"Grokking on $(a+b)\\ \\mathrm{{mod}}\\ {cfg['p']}$ — "
        f"{cfg['n_train']:,} train / {cfg['n_test']:,} test pairs, "
        f"{cfg['d_model']}-dim 1-layer transformer, seed {cfg['seed']}",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "008_grokking.png")
    plt.close(fig)


def cmd_plot(args: argparse.Namespace) -> None:
    p = paths(args.tag)
    rows = read_metrics(p["metrics"])
    if not p["config"].exists():
        sys.exit(f"no config at {p['config']} — the run directory is incomplete")
    cfg = json.loads(p["config"].read_text())["config"]

    print(f"008 — grokking, {len(rows):,} recorded steps up to {int(rows['step'][-1]):,}")
    m = milestones(rows)
    print_milestones(m)
    print()
    figure_raw(rows, cfg)
    figure_diagram(rows, cfg, m)


# --- entry point --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tag = argparse.ArgumentParser(add_help=False)
    tag.add_argument("--tag", default=None,
                     help="suffix for the run directory, to keep runs side by side")

    t = sub.add_parser("train", parents=[tag], help="run the training (hours; needs torch)")
    t.add_argument("--steps", type=int, default=40_000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    t.add_argument("--eval-every", type=int, default=10,
                   help="steps between test-set evaluations (every step below 100)")
    t.add_argument("--print-every", type=int, default=500)
    t.add_argument("--ckpt-every", type=int, default=2_000, help="0 to disable")
    t.add_argument("--resume", action="store_true",
                   help="continue from last.pt in the run directory")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("plot", parents=[tag],
                       help="draw the figures from a run directory")
    p.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
