"""008 — EXPERIMENT — Grokking on modular addition (Nanda et al. replication).

**NOT YET RUN.** The hypothesis slot in `logs/008-grokking.md` is empty and
CONSTITUTION 1 says it is filled by Alex before the run, not after. This file
exists so the run is ready to start, not so it starts.

A one-layer transformer on $(a+b) \\bmod 113$, trained full-batch with AdamW and
weight decay 1.0 on 30% of the pairs. Train accuracy saturates within ~1k steps;
test accuracy stays at chance for roughly a further decade of steps and then
rises to ~100%. Reproducing that gap is the whole deliverable.

Training and plotting are separate entry points, because the run is measured in
tens of minutes and the plot in seconds (CONSTITUTION 9b):

    python scripts/008_grokking.py train --device auto
    python scripts/008_grokking.py plot
    python scripts/008_grokking.py list                    # every run, side by side
    python scripts/008_grokking.py compare --runs A B C    # overlay them

`train` writes `runs/008-grokking/` and needs torch. Everything else reads that
directory and needs only numpy + matplotlib. A run killed halfway still plots.

`train --help` lists the model and optimizer flags. Anything left at its default
is the reference configuration; anything changed renames the run directory and
the figures after the change, so a sweep cannot overwrite itself.

`train --tensorboard` additionally streams the same scalars to `runs/<name>/tb`
for watching a run as it goes (`tensorboard --logdir runs`). That is a view, not
the record — `metrics.csv` is what the figures and the log entry come from.

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
from _bootstrap import ROOT, RUNS, run_dir, save

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
        "tb": directory / "tb",
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

    cfg = G.Config(
        p=args.p, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_mlp=args.d_mlp, train_frac=args.train_frac,
        optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, momentum=args.momentum,
        warmup_steps=args.warmup_steps, steps=args.steps, seed=args.seed,
    )
    # A non-default config lands in its own directory unless told otherwise, so
    # a sweep cannot overwrite its own earlier variants.
    tag = args.tag or G.auto_tag(cfg) or None
    p = paths(tag)
    device = pick_device(args.device)

    started_before = p["metrics"].exists() and p["metrics"].stat().st_size > 0
    if started_before and not (args.resume or args.force):
        sys.exit(
            f"{p['metrics'].relative_to(ROOT)} already holds a run.\n"
            f"  --resume to continue it, --force to overwrite it, "
            f"or --tag NAME to put this one somewhere else."
        )

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
    print(f"  model         {cfg.n_layers} block(s), d_model {cfg.d_model}, "
          f"{cfg.n_heads} heads, d_mlp {cfg.d_mlp}  ->  {run.n_params:,} parameters")
    print(f"  train / test  {cfg.n_train:,} / {cfg.n_test:,} pairs "
          f"({cfg.train_frac:.0%} train)")
    print(f"  optimizer     {cfg.optimizer} lr={cfg.lr:g} wd={cfg.weight_decay:g} "
          f"betas=({cfg.beta1:g},{cfg.beta2:g})"
          f"{f' momentum={cfg.momentum:g}' if cfg.optimizer == 'sgd' else ''}, "
          f"full batch")
    print(f"  steps         {cfg.steps:,} (warmup {cfg.warmup_steps})")
    print(f"  seed          {cfg.seed}")
    print(f"  tag           {tag or '(reference config, untagged)'}")
    print(f"  writing       {p['dir'].relative_to(ROOT)}/")
    print()

    fresh = not (args.resume and p["metrics"].exists())
    handle = p["metrics"].open("w" if fresh else "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=G.RECORD_FIELDS)
    if fresh:
        writer.writeheader()

    # TensorBoard is a *view*, not the record: metrics.csv is what the figures and
    # the log entry are built from (CONSTITUTION 9a). Nothing is written here that
    # is not also in the CSV, so deleting tb/ costs nothing.
    board = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            sys.exit("--tensorboard needs the tensorboard package: pip install tensorboard")
        board = SummaryWriter(log_dir=str(p["tb"]), flush_secs=10)
        board.add_text("config", f"```json\n{p['config'].read_text()}\n```", 0)
        print(f"  tensorboard   {p['tb'].relative_to(ROOT)}/  "
              f"(view with: tensorboard --logdir {RUNS.relative_to(ROOT)})\n")

    started = time.perf_counter()
    interrupted = False
    try:
        for record in run.iterate(
            until=cfg.steps, should_record=G.dense_then_every(args.eval_every)
        ):
            writer.writerow(record.__dict__)
            handle.flush()  # a killed run must still plot (CONSTITUTION 9b)

            if board is not None:
                board.add_scalar("accuracy/train", record.train_acc, record.step)
                board.add_scalar("accuracy/test", record.test_acc, record.step)
                board.add_scalar("loss/train", record.train_loss, record.step)
                board.add_scalar("loss/test", record.test_loss, record.step)
                board.add_scalar("weight_norm", record.weight_norm, record.step)
                board.add_scalar("lr", run.sched.get_last_lr()[0], record.step)

            if record.step % args.print_every == 0 or record.step == cfg.steps:
                done = record.step - resumed_from
                rate = done / max(time.perf_counter() - started, 1e-9)
                remaining = (cfg.steps - record.step) / rate if rate else float("nan")
                # flush: this runs for tens of minutes behind a pipe (notebook
                # cell, nohup, tee), where stdout is block-buffered and progress
                # would otherwise arrive in one lump at the end.
                print(f"  step {record.step:>7,}  "
                      f"train {record.train_loss:9.2e}/{record.train_acc:.3f}  "
                      f"test {record.test_loss:9.2e}/{record.test_acc:.3f}  "
                      f"|w| {record.weight_norm:6.1f}  "
                      f"{rate:5.1f} step/s  eta {remaining / 60:5.1f} min", flush=True)

            if args.ckpt_every and record.step % args.ckpt_every == 0:
                torch.save(run.model.state_dict(),
                           p["dir"] / f"ckpt-{record.step:06d}.pt")
                torch.save(run.state_dict(), p["last"])
    except KeyboardInterrupt:
        interrupted = True
        print("\n  interrupted")
    finally:
        handle.close()
        if board is not None:
            board.close()
        torch.save(run.state_dict(), p["last"])

    elapsed = time.perf_counter() - started
    print(f"\n  stopped at step {run.step:,} after {elapsed / 60:.1f} min"
          f"{' (interrupted)' if interrupted else ''}")
    print_milestones(milestones(read_metrics(p["metrics"])))
    flag = f" --tag {tag}" if tag else ""
    print(f"\n  now: python scripts/008_grokking.py plot{flag}")
    if interrupted:
        print("  or resume: re-run the exact same train command with --resume added")


# --- plot ---------------------------------------------------------------------


def describe(cfg: dict) -> str:
    """The configuration as a title fragment, read entirely from config.json.

    ``.get`` on the newer keys: run directories written before depth and the
    optimizer were configurable do not carry them, and an old run must still
    plot (CONSTITUTION 9).
    """
    blocks = cfg.get("n_layers", 1)
    return (f"{blocks} block{'s' if blocks != 1 else ''}, "
            f"d_model {cfg['d_model']}, {cfg['n_heads']} heads, "
            f"d_mlp {cfg['d_mlp']}, "
            f"{cfg.get('optimizer', 'adamw')} lr {cfg['lr']:g}, "
            f"weight decay {cfg['weight_decay']:g}, seed {cfg['seed']}")


def figure_raw(rows: np.ndarray, cfg: dict, suffix: str = "") -> None:
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
        f"$(a+b)\\ \\mathrm{{mod}}\\ {cfg['p']}$, "
        f"{cfg['train_frac']:.0%} of pairs used for training, full batch\n"
        f"{describe(cfg)}",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, f"008_grokking_curves{suffix}.png", kind="raw")
    plt.close(fig)


def figure_diagram(rows: np.ndarray, cfg: dict, m: dict, suffix: str = "") -> None:
    """CONSTITUTION 8b: the object plus the argument. Every annotation below is
    a measurement read off these axes or a setting from config.json — nothing
    here interprets *why* the transition happens."""
    step = rows["step"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6))
    axes = axes.ravel()

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
    ax.set_title("A.  Accuracy, and when each split is fit")
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
    ax.set_title("B.  Cross-entropy loss")
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
        f"weight decay = {cfg['weight_decay']:g} (setting)\n"
        f"peak {m['norm_peak']:.1f} at step {m['norm_peak_step']:,}\n"
        f"final {m['norm_final']:.1f}",
        xy=(0.97, 0.97), va="top", ha="right", theme=THEME,
    )
    V.tidy(ax, theme=THEME, grid="both")

    # D. The same quantity after the transition, on linear axes. Panel C's log-x
    # squeezes everything past the transition into the right-hand third, which is
    # exactly the stretch where weight decay is still working.
    ax = axes[3]
    after = step >= m["grokked_step"] if m["grokked_step"] else np.zeros_like(step, bool)
    if after.sum() >= 2:
        norm_after = rows["weight_norm"][after]
        ax.plot(step[after], norm_after, color=C[6])
        start, end = float(norm_after[0]), float(norm_after[-1])
        change = (end - start) / start
        ax.set_xlabel("optimizer step (linear scale)")
        ax.set_ylabel(r"$\|w\|_2$, all parameters")
        V.annotate(
            ax,
            f"window: step {m['grokked_step']:,} to {m['final_step']:,}\n"
            f"$\\|w\\|_2$: {start:.1f} $\\rightarrow$ {end:.1f}  ({change:+.1%})",
            xy=(0.97, 0.97), va="top", ha="right", theme=THEME,
        )
    else:
        # Never grokked, or grokked on the last recorded step: there is no window.
        # Say so on the figure rather than leave a blank panel (CONSTITUTION 1c).
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5,
                f"test accuracy never reached {ACC_THRESHOLD:.0%}\n"
                f"(best {rows['test_acc'].max():.3f})\nno post-transition window",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=THEME.ink_secondary, linespacing=1.6)
    ax.set_title(f"D.  Parameter norm after test acc $\\geq$ {ACC_THRESHOLD:.0%}, "
                 f"linear axes")
    V.tidy(ax, theme=THEME, grid="both")

    fig.suptitle(
        f"008 — $(a+b)\\ \\mathrm{{mod}}\\ {cfg['p']}$, "
        f"{cfg['n_train']:,} train / {cfg['n_test']:,} test pairs\n"
        f"{describe(cfg)}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f"008_grokking{suffix}.png")
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
    # A tagged run writes tagged figures: a 200-step smoke test must not be able
    # to overwrite the figure the log entry points at.
    suffix = f"_{args.tag}" if args.tag else ""
    figure_raw(rows, cfg, suffix)
    figure_diagram(rows, cfg, m, suffix)


# --- list / compare across runs -----------------------------------------------


def load_run(directory: Path) -> tuple[dict, np.ndarray] | None:
    """Config and metrics for one run directory, or None if it has neither."""
    config, metrics = directory / "config.json", directory / "metrics.csv"
    if not (config.exists() and metrics.exists() and metrics.stat().st_size):
        return None
    return json.loads(config.read_text()), read_metrics(metrics)


def all_runs() -> list[tuple[str, dict, np.ndarray]]:
    out = []
    for directory in sorted(RUNS.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(RUN_NAME):
            continue
        loaded = load_run(directory)
        if loaded:
            out.append((directory.name, *loaded))
    return out


def cmd_list(args: argparse.Namespace) -> None:
    runs = all_runs()
    if not runs:
        sys.exit(f"no runs in {RUNS.relative_to(ROOT)}/")

    header = (f"{'run':<34} {'opt':<6} {'lr':>7} {'wd':>5} {'L':>2} {'steps':>7} "
              f"{'memorised':>10} {'grokked':>9} {'test acc':>9}")
    print(header)
    print("-" * len(header))
    for name, meta, rows in runs:
        c, m = meta["config"], milestones(rows)
        memorised = f"{m['memorised_step']:,}" if m["memorised_step"] else "never"
        grokked = f"{m['grokked_step']:,}" if m["grokked_step"] else "never"
        print(f"{name:<34} {c['optimizer']:<6} {c['lr']:>7g} {c['weight_decay']:>5g} "
              f"{c.get('n_layers', 1):>2} {m['final_step']:>7,} "
              f"{memorised:>10} {grokked:>9} {m['final_test_acc']:>9.4f}")
    print(f"\n{len(runs)} run(s). Compare them with:")
    print(f"  python scripts/008_grokking.py compare --runs "
          f"{' '.join(n for n, _, _ in runs[:3])}")


def cmd_compare(args: argparse.Namespace) -> None:
    """One figure per sweep, so 'what if I change X' is read off a single axis
    instead of by flicking between PNGs."""
    available = {name: (meta, rows) for name, meta, rows in all_runs()}
    if not available:
        sys.exit(f"no runs in {RUNS.relative_to(ROOT)}/")

    names = args.runs or list(available)
    missing = [n for n in names if n not in available]
    if missing:
        sys.exit(f"no metrics for: {', '.join(missing)}\n"
                 f"  available: {', '.join(available)}")
    if len(names) > len(C):
        sys.exit(f"{len(names)} runs but only {len(C)} colour slots — "
                 f"pass a subset with --runs")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for i, name in enumerate(names):
        rows = available[name][1]
        m = milestones(rows)
        label = name.removeprefix(f"{RUN_NAME}-").removeprefix(RUN_NAME) or "reference"
        grok = f"grok {m['grokked_step']:,}" if m["grokked_step"] else "no grok"
        axes[0].plot(rows["step"], rows["test_acc"], color=C[i],
                     label=f"{label}  ({grok})")
        axes[0].plot(rows["step"], rows["train_acc"], color=C[i], lw=1.0,
                     ls=(0, (3, 2)), alpha=0.55)
        axes[1].plot(rows["step"], rows["weight_norm"], color=C[i], label=label)

    ax = axes[0]
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("optimizer step (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("A.  Test accuracy (solid), train accuracy (dashed)")
    ax.legend(loc="center left", fontsize=7)
    V.annotate(ax, f"'grok' = first step with test acc $\\geq$ {ACC_THRESHOLD:.0%}\n"
                   f"one seed per run unless the tag says otherwise",
               xy=(0.97, 0.03), va="bottom", ha="right", theme=THEME)
    V.tidy(ax, theme=THEME, grid="both")

    ax = axes[1]
    ax.set_xscale("log")
    ax.set_xlabel("optimizer step (log scale)")
    ax.set_ylabel(r"$\|w\|_2$, all parameters")
    ax.set_title("B.  Parameter norm")
    ax.legend(loc="best", fontsize=7)
    V.tidy(ax, theme=THEME, grid="both")

    fig.suptitle(f"008 — {len(names)} configurations compared", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, f"008_compare{f'_{args.name}' if args.name else ''}.png")
    plt.close(fig)


# --- entry point --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tag = argparse.ArgumentParser(add_help=False)
    tag.add_argument("--tag", default=None,
                     help="suffix for the run directory, to keep runs side by side")

    t = sub.add_parser(
        "train", parents=[tag], help="run the training (needs torch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Defaults are the reference configuration (Nanda et al.). Change any of "
               "them and the run gets its own directory and figures, named after what "
               "changed, unless --tag says otherwise.",
    )
    model = t.add_argument_group("model")
    model.add_argument("--n-layers", type=int, default=1,
                       help="transformer blocks; 1 is the reference")
    model.add_argument("--d-model", type=int, default=128, help="residual width")
    model.add_argument("--n-heads", type=int, default=4,
                       help="attention heads per block; must divide d-model")
    model.add_argument("--d-mlp", type=int, default=512, help="MLP hidden width")
    model.add_argument("--p", type=int, default=113, help="modulus; the task is a+b mod p")
    model.add_argument("--train-frac", type=float, default=0.3,
                       help="fraction of the p^2 pairs used for training")

    opt = t.add_argument_group("optimization")
    opt.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"],
                     help="adamw decouples the decay; adam applies it as L2 in the "
                          "gradient; sgd has no adaptive scaling")
    opt.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    opt.add_argument("--weight-decay", type=float, default=1.0,
                     help="0 removes it; the transition is not expected without it")
    opt.add_argument("--beta1", type=float, default=0.9, help="adam/adamw only")
    opt.add_argument("--beta2", type=float, default=0.98, help="adam/adamw only")
    opt.add_argument("--momentum", type=float, default=0.9, help="sgd only")
    opt.add_argument("--warmup-steps", type=int, default=10,
                     help="linear lr warmup")
    opt.add_argument("--steps", type=int, default=40_000, help="full-batch steps")
    opt.add_argument("--seed", type=int, default=0,
                     help="drives both the train/test split and the init")

    runtime = t.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto",
                         choices=["auto", "cpu", "mps", "cuda"],
                         help="auto picks cuda, then mps, then cpu")
    runtime.add_argument("--eval-every", type=int, default=10,
                         help="steps between test evaluations (every step below 100)")
    runtime.add_argument("--print-every", type=int, default=500,
                         help="steps between progress lines")
    runtime.add_argument("--ckpt-every", type=int, default=2_000,
                         help="steps between checkpoints; 0 to disable")
    runtime.add_argument("--tensorboard", action="store_true",
                         help="also stream scalars to runs/<name>/tb for live viewing; "
                              "metrics.csv stays the record either way")
    runtime.add_argument("--resume", action="store_true",
                         help="continue from last.pt in the run directory")
    runtime.add_argument("--force", action="store_true",
                         help="overwrite an existing run in that directory")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("plot", parents=[tag],
                       help="draw the figures for one run directory")
    p.set_defaults(func=cmd_plot)

    ls = sub.add_parser("list", help="every run directory, with its settings and result")
    ls.set_defaults(func=cmd_list)

    cmp_ = sub.add_parser("compare", help="overlay several runs on one figure")
    cmp_.add_argument("--runs", nargs="+", default=None,
                      help="run directory names (default: all of them)")
    cmp_.add_argument("--name", default=None,
                      help="suffix for the output figure filename")
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
