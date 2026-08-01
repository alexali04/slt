"""Estimating the real log canonical threshold from volume scaling.

The whole of SLT hangs off one asymptotic.  Let

    V(eps) = Vol{ w in B : K(w) < eps }

for a compact neighbourhood ``B`` of the most degenerate point of ``W_0``.  Then

    V(eps)  ~  C * eps^lambda * (log 1/eps)^{m-1}        as eps -> 0

where ``lambda`` is the RLCT and ``m`` its multiplicity.  This is equivalent to
the statement that the zeta function ``zeta(z) = int K(w)^{-z} phi(w) dw`` has
its largest pole at ``z = -lambda`` with order ``m``, and it is what makes the
free energy expand as ``F_n = n L_n + lambda log n - (m-1) log log n + O_p(1)``.

For a **regular** model ``V(eps) ~ C eps^{d/2}``: the sub-level set is an
ellipsoid whose volume scales like the ``d``-dimensional ball of radius
``sqrt(eps)``.  So ``lambda = d/2`` is the regular case, and ``lambda <= d/2``
always.

This module measures ``V(eps)`` by Monte Carlo and fits ``lambda``.  Monte Carlo
is used rather than a grid because it generalises unchanged to the higher
dimensional models (mixtures, reduced-rank regression, HMMs) that this project
is heading towards, and because it comes with honest error bars.

Pure module — no I/O, no plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class VolumeScaling:
    """Measured ``V(eps)`` for one landscape."""

    eps: Array  # (n_eps,) thresholds, increasing
    volume: Array  # (n_eps,) estimated Vol{K < eps}
    count: Array  # (n_eps,) raw sample counts behind each estimate
    n_samples: int
    box_volume: float
    box: tuple
    seed: int

    @property
    def fraction(self) -> Array:
        return self.count / self.n_samples

    def usable(self, min_count: int = 200, max_fraction: float = 0.15) -> Array:
        """Mask of thresholds with enough samples to be trusted, and small
        enough that we are plausibly in the asymptotic regime rather than
        measuring the shape of the box."""
        return (self.count >= min_count) & (self.fraction <= max_fraction)


@dataclass(frozen=True)
class RLCTFit:
    rlct: float
    stderr: float
    intercept: float
    multiplicity_assumed: int
    n_points: int
    eps_range: tuple
    r_squared: float

    def __str__(self) -> str:
        return (
            f"lambda_hat = {self.rlct:.4f} +/- {self.stderr:.4f} "
            f"(m assumed {self.multiplicity_assumed}, "
            f"{self.n_points} pts over eps in "
            f"[{self.eps_range[0]:.2e}, {self.eps_range[1]:.2e}], "
            f"R^2 = {self.r_squared:.4f})"
        )


def volume_scaling(
    K,
    box,
    *,
    n_samples: int = 4_000_000,
    n_eps: int = 40,
    eps_lo: float | None = None,
    eps_hi: float | None = None,
    seed: int = 0,
    chunk: int = 1_000_000,
) -> VolumeScaling:
    """Monte-Carlo estimate of ``Vol{w in box : K(w) < eps}`` over an eps grid.

    ``box`` is ``((lo, hi), ...)`` and must be centred on the point of ``W_0``
    whose local RLCT you want.  Sampling is uniform over the box, so this
    measures the RLCT *relative to the uniform prior on the box* — which is the
    right thing: the RLCT is prior-dependent only through whether the prior is
    nonzero at the singularity (CONSTITUTION 6d).
    """
    box = tuple(tuple(map(float, b)) for b in box)
    dim = len(box)
    lo = np.array([b[0] for b in box])
    hi = np.array([b[1] for b in box])
    box_volume = float(np.prod(hi - lo))

    rng = np.random.default_rng(seed)
    values = np.empty(n_samples, dtype=float)
    filled = 0
    while filled < n_samples:
        k = min(chunk, n_samples - filled)
        w = rng.uniform(lo, hi, size=(k, dim))
        values[filled : filled + k] = K(w)
        filled += k

    values.sort()
    if eps_hi is None:
        # Top of the fit window: 15% of samples below threshold. Above that we
        # are measuring the box, not the singularity.
        eps_hi = float(values[int(0.15 * n_samples)])
    if eps_lo is None:
        # Bottom: keep at least ~200 samples, so the Poisson error stays ~7%.
        # High-order monomials underflow to exactly 0 near the singularity, so
        # step forward to the first strictly positive value if we land on one.
        idx = min(200, n_samples - 1)
        while idx < n_samples and values[idx] <= 0:
            idx += 1
        if idx >= n_samples:
            raise ValueError("K underflowed to zero everywhere in the box")
        eps_lo = float(values[idx])
    if not (eps_lo > 0 and eps_hi > eps_lo):
        raise ValueError(
            f"degenerate eps window [{eps_lo:.3e}, {eps_hi:.3e}]; "
            "increase n_samples or shrink the box"
        )

    eps = np.geomspace(eps_lo, eps_hi, n_eps)
    count = np.searchsorted(values, eps, side="left").astype(float)
    volume = box_volume * count / n_samples
    return VolumeScaling(
        eps=eps,
        volume=volume,
        count=count,
        n_samples=n_samples,
        box_volume=box_volume,
        box=box,
        seed=seed,
    )


def fit_rlct(
    vs: VolumeScaling,
    *,
    multiplicity: int = 1,
    min_count: int = 200,
    max_fraction: float = 0.15,
) -> RLCTFit:
    """Weighted least squares for ``lambda`` in

        log V(eps) = log C + lambda * log eps + (m-1) * log log(1/eps).

    The ``(m-1)`` term has a *known* coefficient, so it is subtracted as an
    offset rather than fitted.  Passing ``multiplicity=1`` when the truth is
    ``m=2`` biases ``lambda`` upward — that bias is itself a useful diagnostic
    and is displayed deliberately in script 003.

    Weights are the sample counts (Poisson: ``Var[log V] ~ 1/count``).
    """
    mask = vs.usable(min_count=min_count, max_fraction=max_fraction)
    if mask.sum() < 4:
        raise ValueError(
            f"only {mask.sum()} usable eps points; increase n_samples "
            f"(have {vs.n_samples})"
        )
    eps = vs.eps[mask]
    y = np.log(vs.volume[mask])
    x = np.log(eps)
    if multiplicity > 1:
        y = y - (multiplicity - 1) * np.log(np.log(1.0 / eps))

    w = vs.count[mask]
    A = np.column_stack([np.ones_like(x), x])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    intercept, slope = coef

    resid = y - A @ coef
    dof = len(x) - 2
    sigma2 = float(np.sum(w * resid**2) / dof)
    cov = sigma2 * np.linalg.inv((A * w[:, None]).T @ A)
    stderr = float(np.sqrt(cov[1, 1]))

    ybar = np.average(y, weights=w)
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    ss_res = float(np.sum(w * resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return RLCTFit(
        rlct=float(slope),
        stderr=stderr,
        intercept=float(intercept),
        multiplicity_assumed=multiplicity,
        n_points=int(mask.sum()),
        eps_range=(float(eps.min()), float(eps.max())),
        r_squared=r2,
    )


def fit_rlct_subleading(
    vs: VolumeScaling,
    *,
    multiplicity: int = 2,
    min_count: int = 200,
    max_fraction: float = 0.05,
    guess: float = 0.4,
) -> RLCTFit:
    r"""Fit ``V(eps) = A * eps^lambda * (log(1/eps) + c)^{m-1}``.

    Why this exists: over a box, ``V(eps)`` is a *sum* of contributions — the
    most degenerate point contributes ``eps^lambda (log 1/eps)^{m-1}`` and the
    branches of ``W_0`` contribute ``eps^lambda``.  Their ratio decays only like
    ``1 / log(1/eps)``, so at any reachable ``eps`` the subleading term is still
    large and :func:`fit_rlct` absorbs it into the slope.  Measured bias on the
    ``m=2`` models is +3% to +8% and it does *not* shrink usefully as ``eps``
    falls.  Carrying one subleading constant ``c`` removes it.

    For ``m=1`` this reduces to :func:`fit_rlct` and that function is used
    instead.
    """
    if multiplicity <= 1:
        return fit_rlct(vs, multiplicity=1, min_count=min_count,
                        max_fraction=max_fraction)

    from scipy.optimize import least_squares

    mask = vs.usable(min_count=min_count, max_fraction=max_fraction)
    if mask.sum() < 5:
        raise ValueError(f"only {mask.sum()} usable eps points for a 3-parameter fit")
    eps = vs.eps[mask]
    x = np.log(eps)
    y = np.log(vs.volume[mask])
    sw = np.sqrt(vs.count[mask])
    log_inv = np.log(1.0 / eps)
    p = multiplicity - 1

    def residual(theta):
        log_a, lam, c = theta
        model = log_a + lam * x + p * np.log(np.maximum(log_inv + c, 1e-9))
        return sw * (model - y)

    sol = least_squares(residual, [0.0, guess, 1.0], method="lm", max_nfev=20000)
    log_a, lam, _c = sol.x

    dof = max(len(x) - 3, 1)
    sigma2 = float(np.sum(sol.fun**2) / dof)
    try:
        cov = sigma2 * np.linalg.inv(sol.jac.T @ sol.jac)
        stderr = float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        stderr = float("nan")

    resid = sol.fun / sw
    w = vs.count[mask]
    ybar = np.average(y, weights=w)
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 1.0 - float(np.sum(w * resid**2)) / ss_tot if ss_tot > 0 else float("nan")

    return RLCTFit(
        rlct=float(lam),
        stderr=stderr,
        intercept=float(log_a),
        multiplicity_assumed=multiplicity,
        n_points=int(mask.sum()),
        eps_range=(float(eps.min()), float(eps.max())),
        r_squared=r2,
    )


def multiplicity_signature(vs: VolumeScaling, rlct: float) -> tuple:
    """``V(eps) / eps^lambda`` — flat iff ``m = 1``, growing like ``log(1/eps)``
    iff ``m = 2``.

    This is the cleanest visual test for multiplicity available without doing
    any algebraic geometry, and it is why script 003 plots it.
    """
    mask = vs.usable()
    eps = vs.eps[mask]
    return eps, vs.volume[mask] / eps**rlct


def exact_volume_1d(K, box, eps: Array, n: int = 4_000_001) -> Array:
    """High-resolution quadrature reference for 1-D landscapes.

    Used to confirm the Monte-Carlo estimator is unbiased (CONSTITUTION 3b)
    before we trust it on models with no closed form.
    """
    (lo, hi), = box
    w = np.linspace(lo, hi, n).reshape(-1, 1)
    v = np.sort(K(w))
    frac = np.searchsorted(v, eps, side="left") / n
    return (hi - lo) * frac
