"""Local polynomial regression for mixed-type data."""

from __future__ import annotations

from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth, SelectionResult, _require_usable, broadcast_h, normal_reference
from kerneljax.basis import LocalPolyBasis
from kerneljax.data import MixedData, _as_points
from kerneljax.estimators.fit import LocalPolyFit
from kerneljax.kernels import KernelSet
from kerneljax.kernels._checks import _check_conv_at_zero, _check_grad_diagonal
from kerneljax.kernels._numerics import safe_div
from kerneljax.kernels.sets import _resolve_kernels
from kerneljax.ksum import _pad_index, _pad_rows, kweights
from kerneljax.linalg import wls
from kerneljax.selection.criteria import RegressionCriterion
from kerneljax.selection.optimize import select_bandwidth
from kerneljax.typing import Array, FloatArray

__all__ = ["LocalPolyFit", "local_poly"]


def local_poly(
    train: MixedData | Array,
    y: Float[Array, " n"],
    bw: Bandwidth | SelectionResult | LocalPolyFit | str,
    *,
    at: MixedData | Array | None = None,
    kernels: KernelSet | None = None,
    degree: int | None = None,
    gradient: bool = False,
    se: bool = False,
    fold: Array | None = None,
    n_starts: int = 3,
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

    When ``se=True``, the fit also returns the standard error of the fitted mean,
    built from the constant basis row alone whatever ``degree`` is fit. Writing
    :math:`w_i` for the product kernel weight of training point :math:`i` at the
    evaluation point,

    .. math::

        \hat\sigma^2(x) = \frac{\sum_i w_i y_i^2}{\sum_i w_i}
                        - \left(\frac{\sum_i w_i y_i}{\sum_i w_i}\right)^{2}, \qquad
        \widehat{\mathrm{se}}(\hat m(x))
          = \sqrt{\frac{\hat\sigma^2(x)\, R(K)}{\sum_i w_i}}

    with :math:`R(K) = \int K^2(u)\,du` the roughness of the continuous kernel and
    :math:`\hat\sigma^2` clamped at zero before the square root.

    Parameters
    ----------
    train : MixedData or Array
        Training sample, supplying the rows of the weighted design at
        every evaluation point. A raw array is read as continuous columns.
    y : Float[Array, " n"]
        Response values, one per row of ``train``.
    bw : Bandwidth or SelectionResult or LocalPolyFit or str
        Bandwidths for every column, or a way of arriving at them. A
        :class:`~kerneljax.Bandwidth` is used as given. ``"cv_ls"`` and ``"aic"``
        select one by minimizing that criterion, and ``"normal_reference"``
        applies :func:`~kerneljax.normal_reference` without a search. A previous
        selection or fit reuses its bandwidth along with the degree it was
        chosen under. To select with a criterion built by hand,
        minimize it with :func:`~kerneljax.select_bandwidth` and pass the
        result here.
    at : MixedData or Array, optional
        Evaluation points. Defaults to ``train``. A raw array is accepted
        only when the training sample is purely continuous.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to
        ``KernelSet()``.
    degree : int, optional
        Total degree of the polynomial basis. Supports 0, 1 and 2. Defaults
        to the degree carried by ``bw`` when it carries one, and to 0
        otherwise, giving a local constant fit. Passing a degree that
        contradicts the one ``bw`` carries raises ``ValueError``.
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
    se : bool, optional
        Whether to also return the standard error of the fitted mean at every
        evaluation point. The variance is estimated from the constant basis row
        whatever the ``degree``, so for ``degree >= 1`` it is not the residual
        variance of the polynomial actually fitted. The standard error covers
        the fitted mean only, not the gradient.
    fold : Array, optional
        Fold label of every training point, shape ``(n,)``. Training
        points sharing a fold with an evaluation row are dropped from
        its weighted moments, giving a leave-one-out or k-fold fit.
        ``at`` must then be ``None`` or match ``train`` in length.
    n_starts : int, optional
        Number of perturbed starting points screened when ``bw`` names a
        criterion to minimize.
    chunk : int or tuple of int, optional
        Chunk sizes as ``(eval, train)``. A bare int chunks only the
        evaluation axis. Bounds the peak memory of the fit at the cost
        of additional compute.
    penalty : FloatArray or float, optional
        The ridge penalty passed through to
        :func:`~kerneljax.wls`. Defaults to no penalty.

    Returns
    -------
    LocalPolyFit
        Object containing the local polynomial fit:

        - **mean**: Fitted regression value at every evaluation point
        - **grad**: Gradient of the fitted value with respect to every continuous column, or None
        - **coef**: Full coefficient vector at every evaluation point, in bandwidth units
        - **rcond**: Reciprocal condition number of the weighted moment system at every point
        - **bandwidth**: Bandwidth used to produce the fit
        - **se**: Standard error of the fitted mean at every evaluation point, or None
        - **selection**: Selection that produced the bandwidth, or None when it was supplied directly
        - **degree**: Total degree of the local polynomial basis
        - **kernels**: Kernel families the fit was produced with
        - **spec**: Column metadata of the training sample
        - **n_train**: Number of training points
        - **r_squared**: Squared correlation with the response, or None when ``at`` was given
        - **residual_se**: Root mean squared residual, on the same terms

    Examples
    --------
    Fit a local linear regression, letting least squares cross validation choose the
    bandwidth. The fit carries the selected bandwidth and the goodness of fit
    alongside the fitted values.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import kerneljax as kj
           ...:
           ...: rng = np.random.default_rng(0)
           ...: x = rng.uniform(0.0, 1.0, 150)
           ...: y = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.2, 150)
           ...: fit = kj.local_poly(x, y, "cv_ls", degree=1)
           ...: print(f"bandwidth {float(fit.bandwidth.h[0]):.4f}   R^2 {float(fit.r_squared):.4f}")

    A categorical column joins through :meth:`~kerneljax.MixedData.from_blocks`, and
    its smoothing parameter is selected jointly with the continuous bandwidth. A value
    near zero says the criterion is declining to pool across categories, so the
    variable matters.

    .. ipython::
        :okwarning:

        In [2]: region = rng.integers(0, 3, 150)
           ...: data = kj.MixedData.from_blocks(continuous=x, unordered=region)
           ...: mixed = kj.local_poly(data, y + 0.5 * region, "cv_ls", degree=1)
           ...: print(f"h {float(mixed.bandwidth.h[0]):.4f}   lambda {float(mixed.bandwidth.lam_uno[0]):.4f}")

    See Also
    --------
    density : Estimate a mixed-type probability density.
    cdf : Estimate a mixed-type cumulative distribution.
    summary : Measure how well a fitted estimator describes the sample it was fit on.
    select_bandwidth : Select a bandwidth by minimizing a cross-validation criterion.

    References
    ----------
    .. [1] Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and Its
           Applications. Chapman and Hall.
    .. [2] Li, Q., & Racine, J. S. (2007). Nonparametric Econometrics: Theory
           and Practice. Princeton University Press.
    """
    kernels = _resolve_kernels(kernels, getattr(bw, "kernels", None))
    train = _as_points(train)
    if train.spec.p_con:
        if isinstance(bw, str) and bw != "normal_reference":
            _check_grad_diagonal(kernels.continuous, "value")
        if se:
            _check_conv_at_zero(kernels.continuous)

    bandwidth, selection, degree = _resolve_bandwidth(train, y, bw, degree, kernels, n_starts, chunk)

    if gradient and degree == 0:
        raise ValueError("gradient requires degree >= 1, a constant fit carries no slope information")
    if gradient and bandwidth.h_axis == "train":
        raise ValueError(
            "gradient requires bw.h_axis to be 'shared' or 'eval', "
            "a training indexed bandwidth attaches no bandwidth to the evaluation point"
        )

    evaluate = train if at is None else _as_points(at, train.spec)
    if fold is not None:
        if fold.shape[0] != train.n:
            raise ValueError(f"fold carries {fold.shape[0]} labels for {train.n} training rows, one label per row")
        if evaluate.n != train.n:
            raise ValueError(
                "fold aligns evaluation rows with training rows, so at must be "
                f"dropped or match train in length, got {evaluate.n} evaluation rows for {train.n} training rows"
            )
    if at is not None and isinstance(bw, SelectionResult | LocalPolyFit) and bandwidth.h_axis != "shared":
        raise ValueError(
            "reusing a bandwidth at new evaluation points requires h_axis 'shared', "
            f"got {bandwidth.h_axis!r} which is tied to the rows it was built for"
        )

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

    mean, coef, rcond, grad, se_value, r_squared, residual_se = _fit_values(
        train,
        bandwidth,
        y,
        evaluate,
        fold,
        penalty,
        basis=basis,
        kernels=kernels,
        gradient=gradient,
        se=se,
        chunk_eval=chunk_eval,
        chunk_train=chunk_train,
        p_con=p_con,
        compare_to_y=at is None,
    )

    _require_usable(bandwidth)
    return LocalPolyFit(
        mean=mean,
        grad=grad,
        coef=coef,
        rcond=rcond,
        bandwidth=bandwidth,
        se=se_value,
        selection=selection,
        degree=degree,
        kernels=kernels,
        spec=train.spec,
        n_train=train.n,
        r_squared=r_squared,
        residual_se=residual_se,
    )


def _resolve_degree(explicit: int | None, carried: int | None) -> int:
    """Settle on one degree, refusing an explicit value that contradicts a carried one."""
    if explicit is None:
        return 0 if carried is None else carried

    if carried is not None and explicit != carried:
        raise ValueError(f"degree={explicit} contradicts the degree {carried} that bw was selected under")

    return explicit


def _resolve_bandwidth(
    train: MixedData,
    y: Float[Array, " n"],
    bw: Bandwidth | SelectionResult | LocalPolyFit | str,
    degree: int | None,
    kernels: KernelSet,
    n_starts: int,
    chunk: int | tuple[int, int] | None,
) -> tuple[Bandwidth, SelectionResult | None, int]:
    """Turn any accepted ``bw`` into a bandwidth, the selection behind it, and a degree."""
    if isinstance(bw, Bandwidth):
        return bw, None, _resolve_degree(degree, None)

    if isinstance(bw, LocalPolyFit):
        return bw.bandwidth, bw.selection, _resolve_degree(degree, bw.degree)

    if isinstance(bw, SelectionResult):
        return cast(Bandwidth, bw.bandwidth), bw, _resolve_degree(degree, getattr(bw.criterion, "degree", None))

    if not isinstance(bw, str):
        raise TypeError(
            f"bw must be a Bandwidth, a SelectionResult, a LocalPolyFit or a method name, got {type(bw).__name__}"
        )

    if bw == "normal_reference":
        return normal_reference(train, kernels), None, _resolve_degree(degree, None)

    if bw not in ("cv_ls", "aic"):
        raise ValueError(f"bw must be 'cv_ls', 'aic' or 'normal_reference', got {bw!r}")

    method = cast(Literal["cv_ls", "aic"], bw)
    criterion = RegressionCriterion(method=method, degree=_resolve_degree(degree, None))
    selection = select_bandwidth(train, criterion, y=y, kernels=kernels, n_starts=n_starts, chunk=chunk)
    return selection.bandwidth, selection, criterion.degree


def _augmented_moments(weights: Array, design: Array, y: Array) -> tuple[Array, Array, Array]:
    r"""Contract the weighted augmented design against itself in a single reduction.

    Appending the response to the design as one more column gives :math:`Z = [X, y]`, whose
    weighted cross product holds all three moments at once,

    .. math::

        Z^\top W Z = \begin{pmatrix} X^\top W X & X^\top W y \\
                                     y^\top W X & y^\top W y \end{pmatrix}

    so the Gram block, the moment vector, and the weighted sum of squares are the blocks of
    one symmetric matrix rather than three separate contractions.

    Only the upper triangle is summed, and every entry is accumulated in one pass over the
    training points. Reading the weights and each design column exactly once is what keeps
    the full ``(n_train, k)`` design from having to be held for every evaluation point at
    once, which is what makes the memory grow with the sample rather than with its square.

    Parameters
    ----------
    weights : Array
        Kernel weight of every training point, shape ``(n_train,)``,
        already masked.
    design : Array
        Design matrix at the evaluation point, shape ``(n_train, k)``.
    y : Array
        Response values, shape ``(n_train,)``.

    Returns
    -------
    tuple of Array
        The Gram matrix ``(k, k)``, the moment vector ``(k,)`` and the
        weighted sum of squared responses.
    """
    dim = design.shape[-1]
    columns = [design[:, index] for index in range(dim)] + [y]

    pairs = [(row, column) for row in range(dim + 1) for column in range(row, dim + 1)]
    terms = tuple(weights * columns[row] * columns[column] for row, column in pairs)
    zeros = tuple(jnp.zeros((), dtype=term.dtype) for term in terms)

    totals = jax.lax.reduce(terms, zeros, lambda left, right: tuple(map(jnp.add, left, right)), (0,))

    gram: list[list[Array]] = [[zeros[0]] * (dim + 1) for _ in range(dim + 1)]
    for (row, column), total in zip(pairs, totals, strict=True):
        gram[row][column] = total
        gram[column][row] = total

    xtwx = jnp.stack([jnp.stack(gram[row][:dim]) for row in range(dim)])
    xtwy = jnp.stack([gram[row][dim] for row in range(dim)])
    return xtwx, xtwy, gram[dim][dim]


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
) -> tuple[Array, Array, Array]:
    """Form the weighted design moments for a single evaluation point, chunking the training axis when asked."""
    if chunk_train is None:
        weights = kweights(train, bw, at=at_row, kernels=kernels)[0]
        if fold is not None:
            weights = jnp.where(fold[row_index] != fold, weights, 0.0)
        return _augmented_moments(weights, basis.design(train, at_row, bw), y)

    train_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_train), train)
    y_blocks = _pad_rows(y, chunk_train)
    idx_blocks = _pad_index(jnp.arange(train.n), train.n, chunk_train)
    h_blocks = _pad_rows(bw.h, chunk_train) if bw.h_axis == "train" else None

    dim = basis.dim(train.spec, basis.degree)
    initial = (
        jnp.zeros((dim, dim), dtype=train.con.dtype),
        jnp.zeros((dim,), dtype=train.con.dtype),
        jnp.zeros((), dtype=train.con.dtype),
    )

    def step(
        carry: tuple[Array, Array, Array], block: tuple[MixedData, Array, Array, Array | None]
    ) -> tuple[tuple[Array, Array, Array], None]:
        train_block, y_block, idx_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)

        weights = kweights(train_block, bw_block, at=at_row, kernels=kernels)[0]
        weights = jnp.where(idx_block < train.n, weights, 0.0)
        if fold is not None:
            weights = jnp.where(fold[row_index] != fold[idx_block], weights, 0.0)

        xtwx, xtwy, weight_y2 = carry
        block_xtwx, block_xtwy, block_y2 = _augmented_moments(
            weights, basis.design(train_block, at_row, bw_block), y_block
        )
        return (xtwx + block_xtwx, xtwy + block_xtwy, weight_y2 + block_y2), None

    (xtwx, xtwy, weight_y2), _ = jax.lax.scan(
        jax.checkpoint(step), initial, (train_blocks, y_blocks, idx_blocks, h_blocks)
    )
    return xtwx, xtwy, weight_y2


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
    se: bool,
    penalty: FloatArray | float,
    chunk_train: int | None,
    p_con: int,
) -> tuple[Array, Array, Array, Array | None, Array | None]:
    """Solve the weighted least squares moments for a single evaluation point."""
    at_row = MixedData(con=con_row[None, :], uno=uno_row[None, :], orde=orde_row[None, :], spec=train.spec)
    bw_row = bw if h_row is None else bw.replace(h=h_row, h_axis="shared")

    xtwx, xtwy, weight_y2 = _moments(train, bw_row, y, at_row, basis, kernels, fold, row_index, chunk_train)
    fit = wls(xtwx, xtwy, penalty=penalty)

    grad = None
    if gradient:
        h = broadcast_h(bw_row, p_con)[0, 0]
        grad = fit.coef[1 : 1 + p_con] / h

    se_value = None
    if se:
        weight_total = xtwx[0, 0]
        mean_y = safe_div(xtwy[0], weight_total)
        mean_y2 = safe_div(weight_y2, weight_total)
        sigma2 = jnp.clip(mean_y2 - mean_y * mean_y, 0.0, None)
        # The continuous kernels here carry no 1/h factor, so a self
        # convolution at zero difference is the same for every h, and the
        # weight sum already carries the bandwidth product through the
        # unnormalized kernel values that built it.
        roughness = jnp.prod(kernels.continuous.conv(con_row, con_row, jnp.ones_like(con_row)))
        # The sqrt adjoint is infinite at zero, so the argument is swapped for a
        # harmless one wherever the variance term is zero and that branch discarded.
        variance_term = safe_div(sigma2 * roughness, weight_total)
        positive = variance_term > 0.0
        se_value = jnp.where(positive, jnp.sqrt(jnp.where(positive, variance_term, 1.0)), 0.0)

    return fit.coef[0], fit.coef, fit.rcond, grad, se_value


@partial(
    jax.jit,
    static_argnames=("basis", "kernels", "gradient", "se", "chunk_eval", "chunk_train", "p_con", "compare_to_y"),
)
def _fit_values(
    train: MixedData,
    bandwidth: Bandwidth,
    y: Array,
    evaluate: MixedData,
    fold: Array | None,
    penalty: FloatArray | float,
    *,
    basis: LocalPolyBasis,
    kernels: KernelSet,
    gradient: bool,
    se: bool,
    chunk_eval: int | None,
    chunk_train: int | None,
    p_con: int,
    compare_to_y: bool,
) -> tuple[Array, Array, Array, Array | None, Array | None, Array | None, Array | None]:
    """Fit every evaluation point.

    Parameters
    ----------
    train : MixedData
        Training sample the fit is built from.
    bandwidth : Bandwidth
        Bandwidths for every column.
    y : Array
        Response values, one per training row.
    evaluate : MixedData
        Evaluation points.
    fold : Array, optional
        Fold label of every point, dropping any pair that shares one.
    penalty : FloatArray or float
        Ridge penalty added to the moment system.
    basis : LocalPolyBasis
        Polynomial basis, fixing the degree. Static.
    kernels : KernelSet
        Kernel families, one per column kind. Static.
    gradient : bool
        Whether to return the derivative rows. Static.
    se : bool
        Whether to return standard errors. Static.
    chunk_eval : int, optional
        Evaluation axis chunk size. Static.
    chunk_train : int, optional
        Training axis chunk size. Static.
    p_con : int
        Number of continuous columns. Static.
    compare_to_y : bool
        Whether the fit sits on its own training points, so fitted and
        observed values line up and can be compared. Static.

    Returns
    -------
    tuple
        The fitted mean, the coefficients, the reciprocal condition
        estimate, the derivative rows, the standard errors, the squared
        correlation with the response and the root mean squared residual.
    """
    if chunk_eval is None:
        eval_idx = jnp.arange(evaluate.n)
        mean, coef, rcond, grad, se_value = _fit_block(
            train, bandwidth, y, evaluate, eval_idx, basis, kernels, fold, gradient, se, penalty, chunk_train, p_con
        )
    else:
        mean, coef, rcond, grad, se_value = _fit_eval_chunks(
            train, bandwidth, y, evaluate, basis, kernels, fold, gradient, se, penalty, chunk_eval, chunk_train, p_con
        )

    if not compare_to_y:
        return mean, coef, rcond, grad, se_value, None, None

    centered = y - jnp.mean(y)
    predicted = mean - jnp.mean(y)
    covariance = jnp.sum(centered * predicted)
    r_squared = covariance * covariance / (jnp.sum(centered**2) * jnp.sum(predicted**2))

    return mean, coef, rcond, grad, se_value, r_squared, jnp.sqrt(jnp.mean((y - mean) ** 2))


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
    se: bool,
    penalty: FloatArray | float,
    chunk_train: int | None,
    p_con: int,
) -> tuple[Array, Array, Array, Array | None, Array | None]:
    """Fit every point of one evaluation block by vmapping the per-point weighted solve."""
    h_rows = bw.h if bw.h_axis == "eval" else None

    def step(
        con_row: Array, uno_row: Array, orde_row: Array, row_index: Array, h_row: Array | None
    ) -> tuple[Array, Array, Array, Array | None, Array | None]:
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
            se,
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
    se: bool,
    penalty: FloatArray | float,
    chunk_eval: int,
    chunk_train: int | None,
    p_con: int,
) -> tuple[Array, Array, Array, Array | None, Array | None]:
    """Chunk the evaluation axis, fitting each block before dropping the padded tail."""
    eval_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_eval), evaluate)
    idx_blocks = _pad_index(jnp.arange(evaluate.n), evaluate.n, chunk_eval)
    h_blocks = _pad_rows(bw.h, chunk_eval) if bw.h_axis == "eval" else None

    def step(
        carry: None, block: tuple[MixedData, Array, Array | None]
    ) -> tuple[None, tuple[Array, Array, Array, Array | None, Array | None]]:
        eval_block, idx_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)
        result = _fit_block(
            train, bw_block, y, eval_block, idx_block, basis, kernels, fold, gradient, se, penalty, chunk_train, p_con
        )
        return carry, result

    _, stacked = jax.lax.scan(jax.checkpoint(step), None, (eval_blocks, idx_blocks, h_blocks))

    trimmed = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:])[: evaluate.n], stacked)
    mean, coef, rcond, grad, se_value = trimmed
    return mean, coef, rcond, grad, se_value
