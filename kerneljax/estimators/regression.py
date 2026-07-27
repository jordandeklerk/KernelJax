"""Local polynomial regression for mixed-type data."""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth, broadcast_h
from kerneljax.basis import LocalPolyBasis
from kerneljax.data import MixedData
from kerneljax.kernels import KernelSet
from kerneljax.ksum import _pad_index, _pad_rows, kweights
from kerneljax.linalg import wls
from kerneljax.typing import Array, FloatArray

__all__ = ["LocalPolyFit", "local_poly"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["mean", "grad", "coef", "rcond", "bandwidth"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True)
class LocalPolyFit:
    r"""Result of a local polynomial regression fit.

    The basis is centered at each evaluation point :math:`x`, so the fitted coefficients
    read off directly,

    .. math::

        \hat m(x) = \beta_0, \qquad
        \frac{\partial \hat m}{\partial x_j}(x) = \frac{\beta_j}{h_j}

    with :math:`\beta_j` the coefficient of the first order term in column :math:`j` and
    :math:`h_j` its bandwidth.

    Parameters
    ----------
    mean : Float[Array, " n_eval"]
        The fitted regression value at every evaluation point.
    grad : Float[Array, "n_eval p_con"] or None
        The gradient of the fitted value with respect to every
        continuous column at every evaluation point, or ``None`` when
        the fit did not request one.
    coef : Float[Array, "n_eval k"]
        The full coefficient vector at every evaluation point, in
        bandwidth units.
    rcond : Float[Array, " n_eval"]
        The reciprocal condition number of the weighted moment system
        at every evaluation point, from :func:`~kerneljax.linalg.wls`.
    bandwidth : Bandwidth
        The bandwidth used to produce the fit.
    """

    mean: Float[Array, " n_eval"]
    grad: Float[Array, "n_eval p_con"] | None
    coef: Float[Array, "n_eval k"]
    rcond: Float[Array, " n_eval"]
    bandwidth: Bandwidth


def local_poly(
    train: MixedData,
    y: Float[Array, " n"],
    bw: Bandwidth,
    *,
    at: MixedData | None = None,
    kernels: KernelSet | None = None,
    degree: int = 1,
    gradient: bool = False,
    fold: Array | None = None,
    chunk: int | tuple[int, int] | None = None,
    penalty: FloatArray | float = 0.0,
) -> LocalPolyFit:
    r"""Fit a local polynomial regression of mixed-type data.

    At every evaluation point :math:`x`, this solves

    .. math::

        \min_{\beta} \sum_{i=1}^{n} K_h(x, X_i) \,
        \bigl(y_i - \beta^\top b(X_i, x)\bigr)^2

    where :math:`b` is the polynomial basis of [1]_ in bandwidth units and
    :math:`K_h` is the generalized product kernel of [2]_ over every column
    kind. Degree 0 gives the Nadaraya-Watson estimator, degree 1 gives local
    linear regression, and degree 2 gives local quadratic regression.

    Only continuous columns enter the polynomial basis. Categorical columns
    still shape the fit entirely through the kernel weights.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the rows of the weighted design at
        every evaluation point.
    y : Float[Array, " n"]
        Response values, one per row of ``train``.
    bw : Bandwidth
        Bandwidths for every column.
    at : MixedData, optional
        Evaluation points. Defaults to ``train``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to
        ``KernelSet()``.
    degree : int, optional
        Total degree of the polynomial basis. Supports 0, 1 and 2.
    gradient : bool, optional
        Whether to also return the gradient of the fitted value with
        respect to every continuous column. Requires ``degree`` of at
        least 1, since a constant fit carries no slope to report.
        Passing ``gradient=True`` with ``degree=0`` raises
        ``ValueError``.

        Also requires ``bw.h_axis`` to be ``"shared"`` or ``"eval"``.
        A training indexed bandwidth attaches no bandwidth to the
        evaluation point, so no per point gradient is defined, and
        ``gradient=True`` with ``bw.h_axis="train"`` raises
        ``ValueError``.
    fold : Array, optional
        Fold label of every training point, shape ``(n,)``. Training
        points sharing a fold with an evaluation row are dropped from
        its weighted moments, giving a leave-one-out or k-fold fit.
        ``at`` must then be ``None`` or match ``train`` in length.
    chunk : int or tuple of int, optional
        Chunk sizes as ``(eval, train)``. A bare int chunks only the
        evaluation axis. Bounds the peak memory of the fit at the cost
        of additional compute.
    penalty : FloatArray or float, optional
        The ridge penalty passed through to
        :func:`~kerneljax.linalg.wls`. Defaults to no penalty.

    Returns
    -------
    LocalPolyFit
        The fitted value, the optional gradient, the full coefficient
        vector, the per point conditioning estimate, and the bandwidth
        used to produce the fit.

    Examples
    --------
    Fit a local linear regression to an exact line.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...: from kerneljax.estimators import local_poly
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: y = 3.0 + 2.0 * x[:, 0]
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: fit = local_poly(train, y, bw, degree=1)
           ...: print(fit.mean[:5])

    References
    ----------
    .. [1] Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and
           Its Applications. Chapman and Hall.
    .. [2] Li, Q., & Racine, J. S. (2007). Nonparametric Econometrics.
           Princeton University Press.
    """
    if gradient and degree == 0:
        raise ValueError("gradient requires degree >= 1, a constant fit carries no slope information")
    if gradient and bw.h_axis == "train":
        raise ValueError(
            "gradient requires bw.h_axis to be 'shared' or 'eval', "
            "a training indexed bandwidth attaches no bandwidth to the evaluation point"
        )

    evaluate = train if at is None else at
    kernels = KernelSet() if kernels is None else kernels
    basis = LocalPolyBasis(degree=degree)
    p_con = train.spec.p_con

    if evaluate.spec.kinds != train.spec.kinds or evaluate.spec.n_levels != train.spec.n_levels:
        raise ValueError("train and at must share the same column kinds and level counts")

    if chunk is None:
        chunk_eval, chunk_train = None, None
    elif isinstance(chunk, tuple):
        chunk_eval, chunk_train = chunk
    else:
        chunk_eval, chunk_train = chunk, None

    if chunk_eval is None:
        eval_idx = jnp.arange(evaluate.n)
        mean, coef, rcond, grad = _fit_block(
            train, bw, y, evaluate, eval_idx, basis, kernels, fold, gradient, penalty, chunk_train, p_con
        )
    else:
        mean, coef, rcond, grad = _fit_eval_chunks(
            train, bw, y, evaluate, basis, kernels, fold, gradient, penalty, chunk_eval, chunk_train, p_con
        )

    return LocalPolyFit(mean=mean, grad=grad, coef=coef, rcond=rcond, bandwidth=bw)


def _trim(leaf: Array, size: int) -> Array:
    """Collapse the block axis into the evaluation axis and drop the padded tail."""
    return leaf.reshape(-1, *leaf.shape[2:])[:size]


def _trim_optional(leaf: Array | None, size: int) -> Array | None:
    """Apply ``_trim`` unless the leaf is absent because no gradient was requested."""
    return None if leaf is None else _trim(leaf, size)


def _moments(
    train: MixedData,
    bw: Bandwidth,
    y: Array,
    at_row: MixedData,
    basis: LocalPolyBasis,
    kernels: KernelSet,
    fold: Array | None,
    row_index: Array,
    chunk_train: int | None,
) -> tuple[Array, Array]:
    """Form the weighted design moments for a single evaluation point, chunking the training axis when asked."""
    if chunk_train is None:
        design = basis.design(train, at_row, bw)
        weights = kweights(train, bw, at=at_row, kernels=kernels)[0]
        if fold is not None:
            weights = jnp.where(fold[row_index] != fold, weights, 0.0)
        weighted = design * weights[:, None]
        return weighted.T @ design, weighted.T @ y

    train_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_train), train)
    y_blocks = _pad_rows(y, chunk_train)
    idx_blocks = _pad_index(jnp.arange(train.n), train.n, chunk_train)
    h_blocks = _pad_rows(bw.h, chunk_train) if bw.h_axis == "train" else None

    dim = basis.dim(train.spec, basis.degree)
    initial = (jnp.zeros((dim, dim), dtype=train.con.dtype), jnp.zeros((dim,), dtype=train.con.dtype))

    def step(
        carry: tuple[Array, Array], block: tuple[MixedData, Array, Array, Array | None]
    ) -> tuple[tuple[Array, Array], None]:
        train_block, y_block, idx_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)

        weights = kweights(train_block, bw_block, at=at_row, kernels=kernels)[0]
        weights = jnp.where(idx_block < train.n, weights, 0.0)
        if fold is not None:
            weights = jnp.where(fold[row_index] != fold[idx_block], weights, 0.0)

        design = basis.design(train_block, at_row, bw_block)
        weighted = design * weights[:, None]

        xtwx, xtwy = carry
        return (xtwx + weighted.T @ design, xtwy + weighted.T @ y_block), None

    (xtwx, xtwy), _ = jax.lax.scan(jax.checkpoint(step), initial, (train_blocks, y_blocks, idx_blocks, h_blocks))
    return xtwx, xtwy


def _fit_point(
    train: MixedData,
    bw: Bandwidth,
    y: Array,
    con_row: Array,
    uno_row: Array,
    orde_row: Array,
    row_index: Array,
    h_row: Array | None,
    basis: LocalPolyBasis,
    kernels: KernelSet,
    fold: Array | None,
    gradient: bool,
    penalty: FloatArray | float,
    chunk_train: int | None,
    p_con: int,
) -> tuple[Array, Array, Array, Array | None]:
    """Solve the weighted least squares moments for a single evaluation point."""
    at_row = MixedData(con=con_row[None, :], uno=uno_row[None, :], orde=orde_row[None, :], spec=train.spec)
    bw_row = bw if h_row is None else bw.replace(h=h_row, h_axis="shared")

    xtwx, xtwy = _moments(train, bw_row, y, at_row, basis, kernels, fold, row_index, chunk_train)
    fit = wls(xtwx, xtwy, penalty=penalty)

    grad = None
    if gradient:
        h = broadcast_h(bw_row, p_con)[0, 0]
        grad = fit.coef[1 : 1 + p_con] / h

    return fit.coef[0], fit.coef, fit.rcond, grad


def _fit_block(
    train: MixedData,
    bw: Bandwidth,
    y: Array,
    eval_block: MixedData,
    idx_block: Array,
    basis: LocalPolyBasis,
    kernels: KernelSet,
    fold: Array | None,
    gradient: bool,
    penalty: FloatArray | float,
    chunk_train: int | None,
    p_con: int,
) -> tuple[Array, Array, Array, Array | None]:
    """Fit every point of one evaluation block by vmapping the per-point weighted solve."""
    h_rows = bw.h if bw.h_axis == "eval" else None

    def step(
        con_row: Array, uno_row: Array, orde_row: Array, row_index: Array, h_row: Array | None
    ) -> tuple[Array, Array, Array, Array | None]:
        return _fit_point(
            train,
            bw,
            y,
            con_row,
            uno_row,
            orde_row,
            row_index,
            h_row,
            basis,
            kernels,
            fold,
            gradient,
            penalty,
            chunk_train,
            p_con,
        )

    in_axes = (0, 0, 0, 0, None if h_rows is None else 0)
    return jax.vmap(step, in_axes=in_axes)(eval_block.con, eval_block.uno, eval_block.orde, idx_block, h_rows)


def _fit_eval_chunks(
    train: MixedData,
    bw: Bandwidth,
    y: Array,
    evaluate: MixedData,
    basis: LocalPolyBasis,
    kernels: KernelSet,
    fold: Array | None,
    gradient: bool,
    penalty: FloatArray | float,
    chunk_eval: int,
    chunk_train: int | None,
    p_con: int,
) -> tuple[Array, Array, Array, Array | None]:
    """Chunk the evaluation axis, fitting each block before dropping the padded tail."""
    eval_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_eval), evaluate)
    idx_blocks = _pad_index(jnp.arange(evaluate.n), evaluate.n, chunk_eval)
    h_blocks = _pad_rows(bw.h, chunk_eval) if bw.h_axis == "eval" else None

    def step(
        carry: None, block: tuple[MixedData, Array, Array | None]
    ) -> tuple[None, tuple[Array, Array, Array, Array | None]]:
        eval_block, idx_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)
        result = _fit_block(
            train, bw_block, y, eval_block, idx_block, basis, kernels, fold, gradient, penalty, chunk_train, p_con
        )
        return carry, result

    _, stacked = jax.lax.scan(jax.checkpoint(step), None, (eval_blocks, idx_blocks, h_blocks))

    mean, coef, rcond, grad = stacked
    return _trim(mean, evaluate.n), _trim(coef, evaluate.n), _trim(rcond, evaluate.n), _trim_optional(grad, evaluate.n)
