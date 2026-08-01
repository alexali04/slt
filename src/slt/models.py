"""Loss landscapes for regular and singular statistical models.

Conventions (CONSTITUTION 5):

* ``w`` is the parameter, ``d = dim`` its dimension.
* ``K(w) >= 0`` is the **population** loss, normalised so ``min_w K(w) = 0``.
  For a regression model ``y = f_w(x) + N(0, 1)`` with true regression function
  ``f_0``, the KL divergence is exactly

      K(w) = (1/2) E_x [ (f_w(x) - f_0(x))^2 ]

  which is what every ``K`` below computes.  A positive constant in front of
  ``K`` changes neither ``W_0`` nor the RLCT, so the factor 1/2 is cosmetic and
  kept only for consistency with the KL.
* ``W_0 = {w : K(w) = 0}``.  Regular iff ``W_0`` is a point and the Fisher
  information (here: the Hessian of ``K``) is positive definite there.

Every ``K`` is vectorised: it accepts an array of shape ``(..., d)`` and returns
shape ``(...)``.  Pure module — no I/O, no plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Landscape:
    """A named loss landscape with its analytically known SLT invariants."""

    key: str
    title: str
    formula: str  # LaTeX for K(w), without surrounding $
    dim: int
    K: Callable[[Array], Array]
    box: tuple  # ((lo, hi), ...) sampling/plot window, centred on the singularity
    regular: bool
    zero_set: str  # human description of W_0
    rlct_theory: float | None = None  # lambda, at the most degenerate point of W_0
    multiplicity_theory: int | None = None  # m
    rlct_note: str = ""  # where the theory value comes from
    tags: tuple = field(default_factory=tuple)

    @property
    def dim_over_two(self) -> float:
        return self.dim / 2.0

    def grid(self, n: int = 400):
        """Evaluate on a regular grid. 2-D only. Returns (X, Y, Z)."""
        if self.dim != 2:
            raise ValueError(f"{self.key}: grid() is 2-D only (dim={self.dim})")
        (x0, x1), (y0, y1) = self.box
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        X, Y = np.meshgrid(xs, ys)
        W = np.stack([X, Y], axis=-1)
        return X, Y, self.K(W)

    def line(self, n: int = 2000):
        """Evaluate on a 1-D sweep. 1-D only. Returns (w, K)."""
        if self.dim != 1:
            raise ValueError(f"{self.key}: line() is 1-D only (dim={self.dim})")
        (x0, x1), = self.box
        w = np.linspace(x0, x1, n).reshape(-1, 1)
        return w[:, 0], self.K(w)


# --- Regular models -----------------------------------------------------------
# Linear regression y = w.x + N(0,1) with x ~ N(0, Sigma).  Then
#     K(w) = (1/2) (w - w*)^T Sigma (w - w*)
# so the Hessian is Sigma everywhere, W_0 = {w*} iff Sigma is invertible, and
# the RLCT is rank(Sigma)/2.


def _quadratic(sigma: Array, w_star: Array) -> Callable[[Array], Array]:
    sigma = np.asarray(sigma, float)
    w_star = np.asarray(w_star, float)

    def K(w: Array) -> Array:
        d = np.asarray(w, float) - w_star
        return 0.5 * np.einsum("...i,ij,...j->...", d, sigma, d)

    return K


def _rotation(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


LINEAR_1D = Landscape(
    key="linear_1d",
    title="1-parameter linear regression",
    formula=r"K(w) = \frac{1}{2} w^2",
    dim=1,
    K=_quadratic([[1.0]], [0.0]),
    box=((-2.0, 2.0),),
    regular=True,
    zero_set=r"the single point $w = 0$",
    rlct_theory=0.5,
    multiplicity_theory=1,
    rlct_note="regular model, lambda = d/2 = 1/2",
    tags=("regular", "linear"),
)

LINEAR_2D = Landscape(
    key="linear_2d",
    title="2-parameter linear regression, isotropic design",
    formula=r"K(w) = \frac{1}{2}\,w^\top \Sigma\, w,\ \ \Sigma = I",
    dim=2,
    K=_quadratic(np.eye(2), np.zeros(2)),
    box=((-2.0, 2.0), (-2.0, 2.0)),
    regular=True,
    zero_set=r"the single point $w = 0$",
    rlct_theory=1.0,
    multiplicity_theory=1,
    rlct_note="regular model, lambda = d/2 = 1",
    tags=("regular", "linear"),
)

_ILL = _rotation(np.pi / 6) @ np.diag([1.0, 0.01]) @ _rotation(np.pi / 6).T
LINEAR_2D_ILLCOND = Landscape(
    key="linear_2d_illcond",
    title="2-parameter linear regression, correlated design",
    formula=r"K(w) = \frac{1}{2}\,w^\top \Sigma\, w,\ \ \kappa(\Sigma) = 100",
    dim=2,
    K=_quadratic(_ILL, np.zeros(2)),
    box=((-2.0, 2.0), (-2.0, 2.0)),
    regular=True,
    zero_set=r"the single point $w = 0$",
    rlct_theory=1.0,
    multiplicity_theory=1,
    rlct_note=(
        "still regular: Sigma is positive definite, so lambda = d/2 = 1 "
        "despite the condition number. See CONSTITUTION 6a."
    ),
    tags=("regular", "linear", "ill-conditioned"),
)

_V = np.array([1.0, 1.0]) / np.sqrt(2.0)
LINEAR_2D_COLLINEAR = Landscape(
    key="linear_2d_collinear",
    title="2-parameter linear regression, perfectly collinear design",
    formula=r"K(w) = \frac{1}{2} (w_1 + w_2)^2 / 2",
    dim=2,
    K=_quadratic(np.outer(_V, _V), np.zeros(2)),
    box=((-2.0, 2.0), (-2.0, 2.0)),
    regular=False,
    zero_set=r"the line $w_1 + w_2 = 0$",
    rlct_theory=0.5,
    multiplicity_theory=1,
    rlct_note=(
        "Sigma has rank 1; lambda = rank/2 = 1/2. Degenerate but *not* a "
        "normal crossing: W_0 is a smooth 1-manifold and K is quadratic "
        "transverse to it. The bridge case between regular and singular."
    ),
    tags=("degenerate", "linear", "bridge"),
)


# --- Singular models ----------------------------------------------------------


def monomial(n: int, m: int, *, half: float = 1.0) -> Landscape:
    r"""K(w) = (1/2) (w_1^n w_2^m)^2 = (1/2) w_1^{2n} w_2^{2m}.

    A **normal crossing**: the archetype every SLT resolution of singularities
    reduces to.  For K = |w_1|^{2k_1} |w_2|^{2k_2} with a smooth measure that is
    nonzero at the origin, the zeta function has its first pole at

        lambda = min_i 1 / (2 k_i) = 1 / (2 max(n, m)),

    with multiplicity m = #{i achieving the min} = 2 if n == m else 1.
    """
    if n < 1 or m < 1:
        raise ValueError("monomial exponents must be >= 1")

    def K(w: Array) -> Array:
        w = np.asarray(w, float)
        return 0.5 * (w[..., 0] ** n * w[..., 1] ** m) ** 2

    both = "" if n == m else " (unequal exponents break the tie)"
    pow_n = "" if n == 1 else f"^{{{n}}}"
    pow_m = "" if m == 1 else f"^{{{m}}}"
    return Landscape(
        key=f"monomial_{n}_{m}",
        title=rf"Monomial model  $f(w) = w_1{pow_n} w_2{pow_m}$",
        formula=rf"K(w) = \frac{{1}}{{2}}\, w_1^{{{2 * n}}} w_2^{{{2 * m}}}",
        dim=2,
        K=K,
        box=((-1.5, 1.5), (-1.5, 1.5)),
        regular=False,
        zero_set=r"the cross $\{w_1 = 0\} \cup \{w_2 = 0\}$",
        rlct_theory=1.0 / (2 * max(n, m)),
        multiplicity_theory=2 if n == m else 1,
        rlct_note=(
            f"normal crossing, lambda = 1/(2*max({n},{m})) = {1 / (2 * max(n, m))}, "
            f"m = {2 if n == m else 1}{both}"
        ),
        tags=("singular", "normal-crossing", "monomial"),
    )


PRODUCT = monomial(1, 1)  # the canonical w1 * w2


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def tanh_g(a: Array) -> Array:
    r"""``g(a) = E_{x~N(0,1)}[tanh^2(a x)]``, accurate for every ``a``.

    Gauss-Hermite is the obvious rule here and it is **wrong** for ``|a| >~ 2``.
    As ``a`` grows, ``tanh^2(a x)`` approaches a step function of ``x`` and no
    polynomial rule tracks it: a 64-node Hermite rule returns ``g(10) = 0.976``
    against a true ``0.921`` and ``g(100) = 1`` exactly against a true ``0.992``.
    It manufactures a saturation that does not exist. See ``logs/006``.

    Integrate the complement instead, where the structure actually is::

        1 - g(a) = E[sech^2(a x)] = (2/a) * int_0^U sech^2(u) phi(u/a) du

    substituting ``u = a x``, with ``U = min(20, 8a)``.  ``sech^2`` decays like
    ``4 e^{-2u}`` so ``U = 20`` truncates below ``1e-17``, and the integrand is
    smooth on ``[0, U]`` for every ``a``.  48-node Gauss-Legendre then holds
    ~1e-14 from ``a = 0.01`` to ``a = 1e4``.

    Asymptotically ``g(a) = 1 - sqrt(2/pi)/|a| + O(a^-2)``: the approach to 1 is
    algebraic, not exponential, so ``K`` never stops depending on ``w_1``.
    """
    a = np.abs(np.asarray(a, float))
    out = np.zeros(a.shape, dtype=float)
    nz = a > 0
    if not np.any(nz):
        return out
    av = a[nz][..., None]
    U = np.minimum(20.0, 8.0 * av)
    u = 0.5 * U * (_GL_X + 1.0)
    wt = 0.5 * U * _GL_W
    integrand = np.exp(-((u / av) ** 2) / 2.0) * _INV_SQRT_2PI / np.cosh(u) ** 2
    out[nz] = 1.0 - (2.0 / av[..., 0]) * np.sum(wt * integrand, axis=-1)
    return out


def _tanh_K(w: Array) -> Array:
    # K is *exactly* quadratic in w_2: K = (1/2) w_2^2 g(w_1). No approximation.
    w = np.asarray(w, float)
    return 0.5 * w[..., 1] ** 2 * tanh_g(w[..., 0])


TANH_1D = Landscape(
    key="tanh_1d",
    title=r"One-unit tanh network  $f_w(x) = w_2\tanh(w_1 x)$",
    formula=r"K(w) = \frac{1}{2}\,\mathbb{E}_{x\sim N(0,1)}[w_2\tanh(w_1 x)]^2",
    dim=2,
    K=_tanh_K,
    box=((-2.5, 2.5), (-2.5, 2.5)),
    regular=False,
    zero_set=r"the cross $\{w_1 = 0\} \cup \{w_2 = 0\}$",
    rlct_theory=0.5,
    multiplicity_theory=2,
    rlct_note=(
        "true function is f_0 = 0. Expanding tanh(ax) = ax - (ax)^3/3 + ... gives "
        "K(w) = (1/2) w_1^2 w_2^2 E[x^2] + O(|w|^6) = (1/2) w_1^2 w_2^2 + ..., "
        "i.e. the same normal crossing as monomial(1,1): lambda = 1/2, m = 2. "
        "Note lambda = 1/2 < d/2 = 1."
    ),
    tags=("singular", "neural-network", "normal-crossing"),
)


def _square_K(w: Array) -> Array:
    w = np.asarray(w, float)
    return 0.5 * w[..., 0] ** 4


SQUARE_1D = Landscape(
    key="square_1d",
    title=r"Degenerate 1-parameter model  $f_w = w^2$",
    formula=r"K(w) = \frac{1}{2} w^4",
    dim=1,
    K=_square_K,
    box=((-1.5, 1.5),),
    regular=False,
    zero_set=r"the single point $w = 0$",
    rlct_theory=0.25,
    multiplicity_theory=1,
    rlct_note=(
        "W_0 is a single point yet the model is singular: K''(0) = 0, so the "
        "Fisher information vanishes. Zeta pole of int |w|^{-4z} dw at z = 1/4, "
        "so lambda = 1/4 < d/2 = 1/2. Degeneracy without a flat direction."
    ),
    tags=("singular", "cusp"),
)


# --- Registry -----------------------------------------------------------------

REGULAR = [LINEAR_1D, LINEAR_2D, LINEAR_2D_ILLCOND]
BRIDGE = [LINEAR_2D_COLLINEAR]
SINGULAR = [PRODUCT, monomial(2, 1), monomial(2, 2), monomial(3, 1), TANH_1D, SQUARE_1D]

ALL = {ls.key: ls for ls in REGULAR + BRIDGE + SINGULAR}


def get(key: str) -> Landscape:
    try:
        return ALL[key]
    except KeyError:
        raise KeyError(f"unknown landscape {key!r}; have {sorted(ALL)}") from None
