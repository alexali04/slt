"""Gradient descent, curvature, and sharpness tracking.

Kept deliberately small for now: enough to overlay a GD trajectory on a
landscape figure and to read off the sharpness ``lambda_max(nabla^2 K)`` along
the way.  The progressive-sharpening / edge-of-stability experiments proper are
not in here yet and will not be run until a hypothesis is on record
(CONSTITUTION 1, the Method Rule).

Derivatives are central finite differences.  Every landscape in this project is
low dimensional and cheap, so an analytic gradient buys nothing and costs
correctness risk.

Pure module — no I/O, no plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def gradient(K, w: Array, h: float = 1e-5) -> Array:
    w = np.asarray(w, float)
    d = w.shape[-1]
    eye = np.eye(d) * h
    return np.array([(K(w + eye[i]) - K(w - eye[i])) / (2 * h) for i in range(d)])


def hessian(K, w: Array, h: float = 1e-4) -> Array:
    w = np.asarray(w, float)
    d = w.shape[-1]
    eye = np.eye(d) * h
    H = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            f_pp = K(w + eye[i] + eye[j])
            f_pm = K(w + eye[i] - eye[j])
            f_mp = K(w - eye[i] + eye[j])
            f_mm = K(w - eye[i] - eye[j])
            H[i, j] = H[j, i] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
    return H


def sharpness(K, w: Array, h: float = 1e-4) -> float:
    """``lambda_max`` of the Hessian. Note the name collision with the RLCT
    ``lambda``: in this codebase ``sharpness`` and ``rlct`` are always spelled
    out (CONSTITUTION 5)."""
    return float(np.linalg.eigvalsh(hessian(K, w, h))[-1])


@dataclass(frozen=True)
class Trajectory:
    w: Array  # (steps + 1, d)
    loss: Array  # (steps + 1,)
    sharpness: Array  # (steps + 1,) or empty if not tracked
    lr: float
    diverged: bool

    @property
    def stability_threshold(self) -> float:
        """``2 / lr`` — the descent lemma boundary that EoS hovers at."""
        return 2.0 / self.lr


def gradient_descent(
    K,
    w0,
    *,
    lr: float,
    steps: int = 500,
    track_sharpness: bool = True,
    blowup: float = 1e8,
    h: float = 1e-5,
) -> Trajectory:
    """Plain full-batch GD with a fixed learning rate."""
    w = np.array(w0, dtype=float)
    ws = [w.copy()]
    losses = [float(K(w))]
    sharps = [sharpness(K, w)] if track_sharpness else []
    diverged = False

    for _ in range(steps):
        w = w - lr * gradient(K, w, h)
        if not np.all(np.isfinite(w)) or np.max(np.abs(w)) > blowup:
            diverged = True
            break
        ws.append(w.copy())
        losses.append(float(K(w)))
        if track_sharpness:
            sharps.append(sharpness(K, w))

    return Trajectory(
        w=np.array(ws),
        loss=np.array(losses),
        sharpness=np.array(sharps),
        lr=lr,
        diverged=diverged,
    )
