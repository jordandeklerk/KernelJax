"""Cross-validation criteria for bandwidth selection."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth
from kerneljax.basis import LocalPolyBasis
from kerneljax.data import MixedData, quantile_grid
from kerneljax.kernels import KernelSet, Op
from kerneljax.ksum import _pad_index, _pad_rows, ksum, kweights
from kerneljax.linalg import hat_diagonal, wls
from kerneljax.typing import Array, ScalarFloat

__all__ = [
    "aic_c_regression",
    "cv_cdf_distribution",
    "cv_ls_density",
    "cv_ls_regression",
    "cv_ml_density",
]


def cv_ml_density(
    train: MixedData,
    bw: Bandwidth,
    *,
    kernels: KernelSet | None = None,
    chunk: int | tuple[int, int] | None = None,
) -> ScalarFloat:
    r"""Likelihood cross-validation criterion for a density.

    The likelihood cross validation criterion of [1]_ is

    .. math::

        \mathrm{CV}_{ml}(h, \lambda) = -\sum_{i=1}^{n} \log \hat f_{-i}(X_i)

    where :math:`\hat f_{-i}` is the leave-one-out density, normalized by
    :math:`(n - 1) \prod h`.

    Minimizing this criterion over :math:`(h, \lambda)`, as
    :func:`~kerneljax.select_bandwidth` does, gives the likelihood cross
    validated bandwidth.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the leave-one-out sum over :math:`i`.
    bw : Bandwidth
        Bandwidths for every column.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`~kerneljax.ksum`.

    Returns
    -------
    ScalarFloat
        The criterion value, minimized over ``bw`` to select a bandwidth.

    Examples
    --------
    Evaluate the likelihood cross validation criterion at a bandwidth.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: print(kj.cv_ml_density(train, bw))

    References
    ----------
    .. [1] Hall, P., Racine, J., & Li, Q. (2004). "Cross-validation and the
           estimation of conditional probability densities." Journal of the
           American Statistical Association, 99, 1015-1026.
    """
    kernels = KernelSet() if kernels is None else kernels

    f_loo = _loo_density(train, bw, kernels, chunk)
    return -jnp.sum(jnp.log(f_loo))


def cv_ls_density(
    train: MixedData,
    bw: Bandwidth,
    *,
    kernels: KernelSet | None = None,
    chunk: int | tuple[int, int] | None = None,
) -> ScalarFloat:
    r"""Least squares cross-validation criterion for a density.

    The least squares cross validation criterion of [1]_ is

    .. math::

        \mathrm{CV}_{ls}(h, \lambda) = \int \hat f^2
        - \frac{2}{n} \sum_{i=1}^{n} \hat f_{-i}(X_i)

    with the integrated square term given explicitly by

    .. math::

        \int \hat f^2 = \frac{1}{n^2 \prod h} \sum_{i} \sum_{j} \bar K(X_i, X_j)

    where :math:`\bar K` is the convolution product across every column kind, continuous and
    categorical alike. The double sum runs over the full matrix including its diagonal.

    Here :math:`\hat f_{-i}` is the same leave-one-out density used by
    :func:`~kerneljax.cv_ml_density`.

    Minimizing this criterion over :math:`(h, \lambda)`, as
    :func:`~kerneljax.select_bandwidth` does, gives the least squares cross
    validated bandwidth.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying both sums.
    bw : Bandwidth
        Bandwidths for every column.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`~kerneljax.ksum`.

    Returns
    -------
    ScalarFloat
        The criterion value, minimized over ``bw`` to select a bandwidth.

    Examples
    --------
    Evaluate the least squares cross validation criterion at a bandwidth.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: print(kj.cv_ls_density(train, bw))

    References
    ----------
    .. [1] Hall, P., Racine, J., & Li, Q. (2004). "Cross-validation and the
           estimation of conditional probability densities." Journal of the
           American Statistical Association, 99, 1015-1026.
    """
    kernels = KernelSet() if kernels is None else kernels
    n = train.n
    scale: Literal["per_train", "per_eval"] = "per_train" if bw.h_axis == "train" else "per_eval"

    convolved = ksum(train, bw, kernels=kernels, op=Op.CONV, weight_scale=scale, chunk=chunk)
    integral_f_squared = jnp.sum(convolved) / (float(n) * n)

    f_loo = _loo_density(train, bw, kernels, chunk)
    return integral_f_squared - 2.0 * jnp.sum(f_loo) / n


def cv_ls_regression(
    train: MixedData,
    bw: Bandwidth,
    *,
    y: Float[Array, " n"],
    kernels: KernelSet | None = None,
    degree: int = 0,
    chunk: int | tuple[int, int] | None = None,
) -> ScalarFloat:
    r"""Least squares cross-validation criterion for a local polynomial regression.

    The leave-one-out mean squared residual of [1]_ is

    .. math::

        \mathrm{CV}_{ls}(h, \lambda) = \frac{1}{n} \sum_{i=1}^{n}
            \bigl(y_i - \hat m_{-i}(X_i)\bigr)^2

    where :math:`\hat m_{-i}` is the local polynomial fit of :math:`m(x) = E[y \mid X = x]`
    with training point :math:`i` held out of its own weighted design.

    Minimizing this criterion over :math:`(h, \lambda)`, as
    :func:`~kerneljax.select_bandwidth` does, gives the least squares cross
    validated bandwidth for the regression.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the leave-one-out sum over :math:`i`.
    bw : Bandwidth
        Bandwidths for every column.
    y : Float[Array, " n"]
        Response values, one per row of ``train``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    degree : int, optional
        Total degree of the local polynomial basis. Supports 0, 1 and 2. Static.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to the underlying fit.

    Returns
    -------
    ScalarFloat
        The criterion value, minimized over ``bw`` to select a bandwidth.

    Examples
    --------
    Evaluate the least squares cross validation criterion at a bandwidth.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: y = 3.0 + 2.0 * x[:, 0]
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: print(kj.cv_ls_regression(train, bw, y=y))

    Minimizing the same criterion over the bandwidth selects it.

    .. ipython::
        :okwarning:

        In [2]: result = kj.select_bandwidth(train, kj.cv_ls_regression, y=y, n_starts=1)
           ...: print(result.bandwidth.h)

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2007). Nonparametric Econometrics: Theory
           and Practice. Princeton University Press.
    """
    from kerneljax.estimators.regression import local_poly

    kernels = KernelSet() if kernels is None else kernels
    fold = jnp.arange(train.n)

    fit = local_poly(train, y, bw, kernels=kernels, degree=degree, fold=fold, chunk=chunk)
    residual = y - fit.mean
    return jnp.mean(residual * residual)


def aic_c_regression(
    train: MixedData,
    bw: Bandwidth,
    *,
    y: Float[Array, " n"],
    kernels: KernelSet | None = None,
    degree: int = 0,
    chunk: int | tuple[int, int] | None = None,
) -> ScalarFloat:
    r"""Corrected Akaike information criterion for a local polynomial regression.

    The corrected AIC of [1]_ weighs the fit of the full local polynomial regression against
    the effective degrees of freedom it spends,

    .. math::

        \mathrm{AIC}_c(h, \lambda) = \log \hat\sigma^2
            + \frac{1 + \operatorname{tr}(H) / n}{1 - (\operatorname{tr}(H) + 2) / n}

    where :math:`\hat\sigma^2` is the mean squared residual of the full fit, using every
    training point, and :math:`H` is the smoother matrix carrying the observed responses to
    the fitted values.

    Writing :math:`w_i` for the kernel weight of training point :math:`i` on itself and
    centering the local polynomial basis at that point, its own row in the design reduces to
    a single one in the constant position, so the diagonal entries of the smoother matrix
    follow [2]_ as

    .. math::

        H_{ii} = w_i \bigl[(X^\top W X)^{-1}\bigr]_{11}, \qquad
        \operatorname{tr}(H) = \sum_{i=1}^{n} H_{ii}

    with the inverse taken at that same training point. Past a pole at
    :math:`\operatorname{tr}(H) = n - 2`, where the denominator above turns negative, the
    penalty term is replaced by a smooth barrier instead.

    The barrier agrees with the formula above wherever the denominator is comfortably
    positive, stays finite everywhere else, and grows as the trace increases past the pole,
    pushing a gradient based search back toward the valid region rather than leaving it to
    wander.

    Minimizing this criterion over :math:`(h, \lambda)`, as
    :func:`~kerneljax.select_bandwidth` does, gives the corrected AIC
    bandwidth for the regression.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the residual sum and the smoother matrix.
    bw : Bandwidth
        Bandwidths for every column.
    y : Float[Array, " n"]
        Response values, one per row of ``train``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    degree : int, optional
        Total degree of the local polynomial basis. Supports 0, 1 and 2. Static.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to the underlying fit and the smoother matrix trace.

    Returns
    -------
    ScalarFloat
        The criterion value, minimized over ``bw`` to select a bandwidth.

    Examples
    --------
    Evaluate the corrected AIC criterion at a bandwidth.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: y = 3.0 + 2.0 * x[:, 0]
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: print(kj.aic_c_regression(train, bw, y=y))

    Minimizing the same criterion over the bandwidth selects it.

    .. ipython::
        :okwarning:

        In [2]: result = kj.select_bandwidth(train, kj.aic_c_regression, y=y, n_starts=1)
           ...: print(result.bandwidth.h)

    References
    ----------
    .. [1] Hurvich, C. M., Simonoff, J. S., & Tsai, C. L. (1998). "Smoothing
           parameter selection in nonparametric regression using an improved
           Akaike information criterion." Journal of the Royal Statistical
           Society B, 60, 271-293.
    .. [2] Li, Q., & Racine, J. S. (2007). Nonparametric Econometrics: Theory
           and Practice. Princeton University Press.
    """
    from kerneljax.estimators.regression import local_poly

    kernels = KernelSet() if kernels is None else kernels
    n = train.n

    fit = local_poly(train, y, bw, kernels=kernels, degree=degree, chunk=chunk)
    residual = y - fit.mean
    sigma_squared = jnp.mean(residual * residual)

    trace = _hat_trace(train, bw, kernels, degree, chunk)
    return jnp.log(sigma_squared) + _aic_c_penalty(trace, n)


def _loo_density(
    train: MixedData,
    bw: Bandwidth,
    kernels: KernelSet,
    chunk: int | tuple[int, int] | None,
) -> Array:
    r"""Compute the leave-one-out density at every training point, divided by :math:`(n - 1)`."""
    fold = jnp.arange(train.n)
    scale: Literal["per_train", "per_eval"] = "per_train" if bw.h_axis == "train" else "per_eval"
    total = ksum(train, bw, kernels=kernels, fold=fold, weight_scale=scale, chunk=chunk)
    return total.reshape(-1) / (train.n - 1)


def _hat_trace(
    train: MixedData,
    bw: Bandwidth,
    kernels: KernelSet,
    degree: int,
    chunk: int | tuple[int, int] | None,
) -> ScalarFloat:
    """Sum the leverage of every training point under the full weighted least squares design."""
    from kerneljax.estimators.regression import _moments

    basis = LocalPolyBasis(degree=degree)
    dim = basis.dim(train.spec, degree)
    onehot = jnp.zeros((dim,), dtype=train.con.dtype).at[0].set(1.0)
    dummy_y = jnp.zeros(train.n, dtype=train.con.dtype)
    dummy_index = jnp.asarray(0)

    if chunk is None:
        chunk_eval, chunk_train = None, None
    elif isinstance(chunk, tuple):
        chunk_eval, chunk_train = chunk
    else:
        chunk_eval, chunk_train = chunk, None

    def leverage(con_row: Array, uno_row: Array, orde_row: Array, h_row: Array | None) -> Array:
        at_row = MixedData(con=con_row[None, :], uno=uno_row[None, :], orde=orde_row[None, :], spec=train.spec)
        self_bw = bw if h_row is None else bw.replace(h=h_row, h_axis="shared")
        moments_bw = self_bw if bw.h_axis == "eval" else bw

        xtwx, _, _ = _moments(train, moments_bw, dummy_y, at_row, basis, kernels, None, dummy_index, chunk_train)
        cho = wls(xtwx, onehot[:, None]).cho
        weight_self = kweights(at_row, self_bw, kernels=kernels)[0, 0]
        return hat_diagonal(cho, onehot, weight_self)

    h_rows = bw.h if bw.h_axis != "shared" else None

    if chunk_eval is None:
        in_axes = (0, 0, 0, None if h_rows is None else 0)
        leverages = jax.vmap(leverage, in_axes=in_axes)(train.con, train.uno, train.orde, h_rows)
        return jnp.sum(leverages)

    train_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_eval), train)
    idx_blocks = _pad_index(jnp.arange(train.n), train.n, chunk_eval)
    h_blocks = _pad_rows(bw.h, chunk_eval) if h_rows is not None else None

    def step(carry: Array, block: tuple[MixedData, Array, Array | None]) -> tuple[Array, None]:
        train_block, idx_block, h_block = block
        in_axes = (0, 0, 0, None if h_block is None else 0)
        block_leverages = jax.vmap(leverage, in_axes=in_axes)(
            train_block.con, train_block.uno, train_block.orde, h_block
        )
        valid = idx_block < train.n
        return carry + jnp.sum(jnp.where(valid, block_leverages, 0.0)), None

    total, _ = jax.lax.scan(step, jnp.zeros((), dtype=train.con.dtype), (train_blocks, idx_blocks, h_blocks))
    return total


def _aic_c_penalty(trace: ScalarFloat, n: int) -> ScalarFloat:
    """Return the corrected AIC penalty, replacing its pole with a smooth increasing barrier."""
    denominator = 1.0 - (trace + 2.0) / n
    margin = 0.1

    floor = margin * jnp.exp((denominator - margin) / margin)
    safe_denominator = jnp.where(denominator > margin, denominator, floor)

    return (1.0 + trace / n) / safe_denominator


def cv_cdf_distribution(
    train: MixedData,
    bw: Bandwidth,
    *,
    at: MixedData | None = None,
    full_integral: bool = False,
    n_grid: int = 100,
    kernels: KernelSet | None = None,
    chunk: int | tuple[int, int] | None = None,
) -> ScalarFloat:
    r"""Least squares cross-validation criterion for a cumulative distribution.

    The criterion of [1]_ compares the leave-one-out estimate against the
    indicator it is trying to reproduce, averaged over evaluation points,

    .. math::

        \mathrm{CV}_{cdf}(h, \lambda) = \frac{1}{n N} \sum_{j=1}^{N} \sum_{i=1}^{n}
            \Bigl( \mathbf{1}\{X_i \le x_j\} - \hat F_{-i}(x_j) \Bigr)^2

    where the leave-one-out estimate removes the pair weight from the row total
    rather than dropping a fold,

    .. math::

        \hat F_{-i}(x_j) = \frac{1}{n - 1}
            \Bigl( \sum_{k=1}^{n} G(x_j, X_k) - G(x_j, X_i) \Bigr).

    The indicator compares every continuous and ordered column at once, so a
    training point counts only when it lies at or below the evaluation point in
    all of them.

    Evaluation points come from one of three places. Supplying ``at`` uses those
    points, ``full_integral`` evaluates on the training sample itself and drops
    the self term from each row, and otherwise the points come from
    :func:`~kerneljax.quantile_grid`. Evaluating on the training sample without
    dropping the self term is a different criterion, so passing the sample as
    ``at`` does not reproduce ``full_integral``.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the sum over :math:`i`.
    bw : Bandwidth
        Bandwidths for every column.
    at : MixedData, optional
        Evaluation points. Takes priority over ``full_integral``.
    full_integral : bool, optional
        Whether to evaluate on the training sample and drop the self term from
        every row. Static.
    n_grid : int, optional
        Number of evaluation points when neither ``at`` nor ``full_integral``
        is given. Static.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    chunk : int or tuple of int, optional
        Chunk size along the evaluation axis, bounding peak memory at the cost
        of additional compute. A tuple gives its first entry, since the leave
        one out reads a whole weight row and the training axis cannot be split.
        Static.

    Returns
    -------
    ScalarFloat
        The criterion value, minimized over ``bw`` to select a bandwidth.

    Examples
    --------
    Evaluate the criterion at a bandwidth.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: print(kj.cv_cdf_distribution(train, bw))

    See Also
    --------
    cv_ml_density : Leave-one-out log likelihood for a density.
    cv_ls_density : Integrated squared error for a density.

    References
    ----------
    .. [1] Li, Q., Li, J., & Racine, J. S. (2017). "Optimal bandwidth selection
           for nonparametric conditional distribution and quantile functions."
           Journal of Business and Economic Statistics, 35, 57-65.
    """
    kernels = KernelSet() if kernels is None else kernels

    if train.spec.p_uno:
        raise ValueError(
            "cv_cdf_distribution supports continuous and ordered columns only, "
            f"got {train.spec.p_uno} unordered columns which carry no order to accumulate along"
        )

    on_train = at is None and full_integral
    evaluate = train if on_train else (at if at is not None else quantile_grid(train, n=n_grid))
    chunk_eval = chunk[0] if isinstance(chunk, tuple) else chunk

    if chunk_eval is None:
        total = _cv_cdf_block(train, bw, kernels, evaluate, jnp.arange(evaluate.n), evaluate.n, on_train)
    else:
        blocks = (
            _pad_rows(evaluate.con, chunk_eval),
            _pad_rows(evaluate.orde, chunk_eval),
            _pad_index(jnp.arange(evaluate.n), evaluate.n, chunk_eval),
        )

        def accumulate(running: ScalarFloat, block: tuple[Array, Array, Array]) -> tuple[ScalarFloat, None]:
            con, orde, rows = block
            piece = MixedData(
                con=con,
                uno=jnp.zeros((chunk_eval, 0), dtype=evaluate.uno.dtype),
                orde=orde,
                spec=evaluate.spec,
            )
            return running + _cv_cdf_block(train, bw, kernels, piece, rows, evaluate.n, on_train), None

        total, _ = jax.lax.scan(accumulate, jnp.zeros((), dtype=train.con.dtype), blocks)

    return total / (train.n * evaluate.n)


def _cv_cdf_block(
    train: MixedData,
    bw: Bandwidth,
    kernels: KernelSet,
    evaluate: MixedData,
    rows: Array,
    n_eval: int,
    on_train: bool,
) -> ScalarFloat:
    """Sum the squared leave-one-out error over one block of evaluation points."""
    weights = kweights(train, bw, at=evaluate, kernels=kernels, op=Op.CDF)
    loo = (jnp.sum(weights, axis=1, keepdims=True) - weights) / (train.n - 1.0)

    below = jnp.ones_like(weights, dtype=bool)
    if train.spec.p_con:
        below = below & jnp.all(train.con[None, :, :] <= evaluate.con[:, None, :], axis=-1)
    if train.spec.p_ord:
        below = below & jnp.all(train.orde[None, :, :] <= evaluate.orde[:, None, :], axis=-1)

    residual = below.astype(weights.dtype) - loo
    squared = residual * residual

    # A padded block repeats its last row, so those rows carry an index past the end
    # of the evaluation points and must not reach the total.
    squared = jnp.where(rows[:, None] < n_eval, squared, 0.0)
    if on_train:
        squared = jnp.where(rows[:, None] == jnp.arange(train.n)[None, :], 0.0, squared)

    return jnp.sum(squared)
