"""Grokking on modular addition: a one-layer transformer, following Nanda et al.

Reference: Nanda, Chan, Lieberum, Smith, Steinhardt, *Progress Measures for
Grokking via Mechanistic Interpretability*, ICLR 2023. The task is
:math:`(a + b) \\bmod p` presented as the token sequence ``[a, b, =]``, with the
answer read off the final position. Architecture and hyperparameters below
follow that paper's modular-addition setup; every deviation is marked ``NOTE``.

Pure module (CONSTITUTION 4a): importing this trains nothing, writes nothing and
draws nothing. ``scripts/008_grokking.py`` owns all of the I/O.

**Deliberately not re-exported from** ``slt/__init__.py``. This is the only
module in the library that needs torch, and ``import slt`` has to keep working
for someone who installed only numpy/scipy/matplotlib (CONSTITUTION 7d).
Import it explicitly::

    from slt import grokking
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields
from typing import Callable, Iterator

import torch
import torch.nn as nn

# --- Configuration ------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Every number that defines a run. Serialised verbatim into the run
    directory before the first step (CONSTITUTION 9a, 4d)."""

    p: int = 113
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    train_frac: float = 0.3

    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 10
    steps: int = 40_000

    seed: int = 0

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        return self.d_model // self.n_heads

    @property
    def d_vocab(self) -> int:
        return self.p + 1  # p digits plus the "=" token

    @property
    def equals_token(self) -> int:
        return self.p

    @property
    def n_ctx(self) -> int:
        return 3  # [a, b, =]

    @property
    def n_train(self) -> int:
        return int(self.train_frac * self.p * self.p)

    @property
    def n_test(self) -> int:
        return self.p * self.p - self.n_train

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            d_head=self.d_head, d_vocab=self.d_vocab, n_ctx=self.n_ctx,
            n_train=self.n_train, n_test=self.n_test,
        )
        return d


# --- Data ---------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    train_x: torch.Tensor  # (n_train, 3) int64
    train_y: torch.Tensor  # (n_train,)   int64
    test_x: torch.Tensor
    test_y: torch.Tensor


def make_data(cfg: Config, *, device: torch.device | str = "cpu") -> Split:
    """All p^2 pairs, split into train/test by a seeded permutation.

    The permutation is drawn on an explicit CPU generator so the split depends
    on ``cfg.seed`` alone and not on the device the run happens to use.
    """
    p = cfg.p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.full_like(a, cfg.equals_token)], dim=1)
    y = (a + b) % p

    gen = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(p * p, generator=gen)
    train_idx, test_idx = perm[: cfg.n_train], perm[cfg.n_train :]

    return Split(
        train_x=x[train_idx].to(device), train_y=y[train_idx].to(device),
        test_x=x[test_idx].to(device), test_y=y[test_idx].to(device),
    )


# --- Model --------------------------------------------------------------------


class OneLayerTransformer(nn.Module):
    """Embed -> attention -> MLP -> unembed, with residual connections.

    No LayerNorm, which is the reference setup: it makes the learned circuit
    readable later, and it is what the grokking curves in the paper were
    produced with. Attention has no biases; the MLP does, matching the
    reference implementation.

    NOTE: ``W_E`` is stored ``(d_vocab, d_model)`` rather than the reference's
    ``(d_model, d_vocab)`` so that embedding is a plain index. Mathematically
    identical, and it matters only if you load reference weights.
    """

    def __init__(self, cfg: Config, *, generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        d, h, e, m, v = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp, cfg.d_vocab

        def randn(*shape: int, scale: float) -> nn.Parameter:
            return nn.Parameter(torch.randn(*shape, generator=generator) / scale)

        root_d = math.sqrt(d)
        self.W_E = randn(v, d, scale=root_d)
        self.W_pos = randn(cfg.n_ctx, d, scale=root_d)

        self.W_Q = randn(h, e, d, scale=root_d)
        self.W_K = randn(h, e, d, scale=root_d)
        self.W_V = randn(h, e, d, scale=root_d)
        self.W_O = randn(d, h * e, scale=root_d)

        self.W_in = randn(m, d, scale=root_d)
        self.b_in = nn.Parameter(torch.zeros(m))
        self.W_out = randn(d, m, scale=root_d)
        self.b_out = nn.Parameter(torch.zeros(d))

        self.W_U = randn(d, v, scale=math.sqrt(v))

        mask = torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(batch, n_ctx)`` of token ids -> ``(batch, n_ctx, d_vocab)`` logits."""
        x = self.W_E[tokens] + self.W_pos
        x = x + self._attention(x)
        x = x + self._mlp(x)
        return x @ self.W_U

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.einsum("bpd,hed->bhpe", x, self.W_Q)
        k = torch.einsum("bpd,hed->bhpe", x, self.W_K)
        v = torch.einsum("bpd,hed->bhpe", x, self.W_V)

        scores = torch.einsum("bhpe,bhqe->bhpq", q, k) / math.sqrt(self.cfg.d_head)
        scores = scores.masked_fill(self.causal_mask, torch.finfo(scores.dtype).min)
        pattern = scores.softmax(dim=-1)

        z = torch.einsum("bhpq,bhqe->bhpe", pattern, v)
        z = z.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return z @ self.W_O.T

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_in.T + self.b_in) @ self.W_out.T + self.b_out


# --- Metrics ------------------------------------------------------------------


def loss_and_accuracy(
    logits_last: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy and accuracy at the final position. Returns tensors, so the
    loss stays differentiable; the caller casts to float.

    The cast to float64 is the reference behaviour and it earns its keep here:
    after grokking the train loss reaches ~1e-8, and a float32 log_softmax has
    already thrown away the digits that distinguish that from 1e-6.

    NOTE: MPS has no float64. On that device the computation stays float32 and
    the deep tail of the loss curve is correspondingly less trustworthy — the
    accuracy curves, which are what the grokking plot is about, are unaffected.
    """
    if logits_last.device.type != "mps":
        logits_last = logits_last.to(torch.float64)
    log_probs = logits_last.log_softmax(dim=-1)
    loss = -log_probs.gather(dim=-1, index=labels[:, None])[:, 0].mean()
    accuracy = (logits_last.argmax(dim=-1) == labels).to(log_probs.dtype).mean()
    return loss, accuracy


@dataclass(frozen=True)
class Record:
    """One row of ``metrics.csv``.

    ``step`` counts completed optimizer updates. The train metrics on a row are
    measured at the parameters *before* that update (they fall out of the same
    forward pass that produced the gradient, so they are free); the test metrics
    are measured *after* it. At the resolution of a log-x grokking plot the
    one-step offset is invisible, but it is there.
    """

    step: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    weight_norm: float
    elapsed_s: float


RECORD_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Record))


# --- Training -----------------------------------------------------------------


def dense_then_every(eval_every: int = 10, dense_until: int = 100) -> Callable[[int], bool]:
    """Record every step early, then every ``eval_every``.

    The grokking plot is read on a log-x axis, where uniform sampling puts
    almost no points in the first decade.
    """

    def should_record(step: int) -> bool:
        return step <= dense_until or step % eval_every == 0

    return should_record


class Run:
    """Model + data + optimizer for one run. Does no I/O of any kind."""

    def __init__(self, cfg: Config, *, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        # Distinct streams for the split and the init, so that changing one
        # thing later does not silently move the other.
        init_gen = torch.Generator().manual_seed(cfg.seed + 1)
        self.model = OneLayerTransformer(cfg, generator=init_gen).to(self.device)
        self.data = make_data(cfg, device=self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2),
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda i: min(1.0, (i + 1) / max(cfg.warmup_steps, 1))
        )
        self.step = 0

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def weight_norm(self) -> float:
        """L2 norm of every parameter, stacked. The cheap baseline for grokking
        (CONSTITUTION 0a): weight decay shrinking the norm is a complete
        non-SLT story about why the transition happens when it does."""
        with torch.no_grad():
            return math.sqrt(sum(float(p.pow(2).sum()) for p in self.model.parameters()))

    def train_step(self) -> tuple[float, float]:
        loss, accuracy = loss_and_accuracy(
            self.model(self.data.train_x)[:, -1], self.data.train_y
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.sched.step()
        self.step += 1
        return float(loss), float(accuracy)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        was_training = self.model.training
        self.model.eval()
        loss, accuracy = loss_and_accuracy(self.model(x)[:, -1], y)
        self.model.train(was_training)
        return float(loss), float(accuracy)

    def iterate(
        self, *, until: int, should_record: Callable[[int], bool]
    ) -> Iterator[Record]:
        """Train up to step ``until``, yielding a Record on recorded steps.

        A generator, so the caller can flush each row to disk as it arrives and
        a killed run still leaves a plottable file (CONSTITUTION 9b).
        """
        started = time.perf_counter()
        while self.step < until:
            train_loss, train_acc = self.train_step()
            if should_record(self.step) or self.step == until:
                test_loss, test_acc = self.evaluate(self.data.test_x, self.data.test_y)
                yield Record(
                    step=self.step,
                    train_loss=train_loss, train_acc=train_acc,
                    test_loss=test_loss, test_acc=test_acc,
                    weight_norm=self.weight_norm(),
                    elapsed_s=time.perf_counter() - started,
                )

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step = state["step"]
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
