"""Conditional density and conditional distribution over mixed-type data."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp

from kerneljax.bandwidth import (
    BandwidthTransform,
    ConditionalBandwidth,
    ConditionalTransform,
    SelectionResult,
    _require_usable,
    _search_start,
    normal_reference,
)
from kerneljax.data import MixedData, _as_points
from kerneljax.estimators.fit import ConditionalFit, ModeFit, QuantileFit
from kerneljax.kernels import KernelSet, Op
from kerneljax.kernels._numerics import safe_div
from kerneljax.kernels.sets import _resolve_kernels
from kerneljax.ksum import kweights
from kerneljax.typing import Array, ScalarFloat

__all__ = ["ConditionalFit", "ModeFit", "QuantileFit", "cdensity", "cdist", "cmode", "cquantile"]


def cdensity(
    x: MixedData | Array,
    y: MixedData | Array,
    bw: ConditionalBandwidth | SelectionResult | ConditionalFit | str,
    *,
    at_x: MixedData | Array | None = None,
    at_y: MixedData | Array | None = None,
    kernels: KernelSet | None = None,
    n_starts: int = 3,
) -> ConditionalFit:
    r"""Estimate the conditional density of ``y`` given ``x``.

    Implements the conditional density estimator of [1]_ as the ratio of a
    joint density to a marginal one, which reduces to the response kernel
    averaged under weights that the conditioning sample normalizes,

    .. math::

        \hat f(y \mid x) = \frac{\sum_i K_x(x, X_i)\, K_y(y, Y_i)}
                                {\prod_j h_{y,j} \sum_i K_x(x, X_i)} .

    Parameters
    ----------
    x : MixedData or Array
        Conditioning sample.
    y : MixedData or Array
        Response sample, with the same number of rows as ``x``.
    bw : ConditionalBandwidth, SelectionResult, ConditionalFit or str
        The bandwidth, a selection to reuse, an earlier fit, or the name of
        a selection rule, ``"cv_ml"``, ``"cv_ls"`` or ``"normal_reference"``.
    at_x : MixedData or Array, optional
        Conditioning evaluation points. Defaults to ``x``.
    at_y : MixedData or Array, optional
        Response evaluation points. Defaults to ``y``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Static.
    n_starts : int
        Number of restarts when a rule name asks for a search. Static.

    Returns
    -------
    ConditionalFit
        The estimate, the bandwidth behind it, and the selection if one ran.

    Examples
    --------
    Estimate the density of a response that tracks its covariate, with the
    bandwidth chosen by likelihood cross validation. The density is high at
    response values the covariate makes likely and vanishes elsewhere.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import kerneljax as kj
           ...:
           ...: rng = np.random.default_rng(0)
           ...: x = rng.normal(0.0, 1.0, 200)
           ...: y = x + rng.normal(0.0, 0.4, 200)
           ...: fit = kj.cdensity(x, y, "cv_ml", at_x=np.array([-1.0, -1.0, 1.0]), at_y=np.array([-1.0, 1.0, 1.0]))
           ...: print(np.asarray(fit.value))

    See Also
    --------
    cdist : Estimate a conditional cumulative distribution.
    density : Estimate a mixed-type probability density.
    summary : Measure how well a fitted estimator describes the sample it was fit on.

    References
    ----------
    .. [1] Hall, P., Racine, J. S., & Li, Q. (2004). "Cross-validation and
           the estimation of conditional probability densities." Journal of
           the American Statistical Association, 99, 1015-1026.
    """
    return _conditional(x, y, bw, at_x, at_y, kernels, n_starts, "density")


def cdist(
    x: MixedData | Array,
    y: MixedData | Array,
    bw: ConditionalBandwidth | SelectionResult | ConditionalFit | str,
    *,
    at_x: MixedData | Array | None = None,
    at_y: MixedData | Array | None = None,
    kernels: KernelSet | None = None,
    n_starts: int = 3,
) -> ConditionalFit:
    r"""Estimate the conditional distribution of ``y`` given ``x``.

    The response kernel is integrated from below rather than evaluated, and
    the bandwidth product drops out because a distribution function carries
    no density scale,

    .. math::

        \hat F(y \mid x) = \frac{\sum_i K_x(x, X_i)\, G_y(y, Y_i)}
                                {\sum_i K_x(x, X_i)} .

    Parameters
    ----------
    x : MixedData or Array
        Conditioning sample.
    y : MixedData or Array
        Response sample, with the same number of rows as ``x``.
    bw : ConditionalBandwidth, SelectionResult, ConditionalFit or str
        The bandwidth, a selection to reuse, an earlier fit, or the name of
        a selection rule, either ``"cv_ls"`` or ``"normal_reference"``.
        Likelihood selection is refused here because the likelihood of a CDF
        value rewards oversmoothing without bound.
    at_x : MixedData or Array, optional
        Conditioning evaluation points. Defaults to ``x``.
    at_y : MixedData or Array, optional
        Response evaluation points. Defaults to ``y``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Static.
    n_starts : int
        Number of restarts when a rule name asks for a search. Static.

    Returns
    -------
    ConditionalFit
        The estimate, the bandwidth behind it, and the selection if one ran.

    Examples
    --------
    Select a bandwidth under the conditional density, then reuse it to read
    the distribution of the response at a cut. Nearly all of the mass sits
    below zero when the covariate is negative and almost none when it is
    positive.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import kerneljax as kj
           ...:
           ...: rng = np.random.default_rng(0)
           ...: x = rng.normal(0.0, 1.0, 200)
           ...: y = x + rng.normal(0.0, 0.4, 200)
           ...: fit = kj.cdensity(x, y, "cv_ml")
           ...: F = kj.cdist(x, y, fit, at_x=np.array([-1.0, 1.0]), at_y=np.zeros(2))
           ...: print(np.asarray(F.value))

    See Also
    --------
    cdensity : Estimate a conditional probability density.
    cdf : Estimate a mixed-type cumulative distribution.
    summary : Measure how well a fitted estimator describes the sample it was fit on.
    """
    return _conditional(x, y, bw, at_x, at_y, kernels, n_starts, "distribution")


def cquantile(
    x: MixedData | Array,
    y: MixedData | Array,
    bw: ConditionalBandwidth | SelectionResult | ConditionalFit | str,
    *,
    tau: float = 0.5,
    at_x: MixedData | Array | None = None,
    kernels: KernelSet | None = None,
    n_starts: int = 3,
    n_iter: int = 64,
) -> QuantileFit:
    r"""Estimate the conditional quantile of ``y`` given ``x``.

    The quantile inverts the conditional distribution by bisection over the
    observed response range,

    .. math::

        \hat q_\tau(x) = \inf \{\, y : \hat F(y \mid x) \ge \tau \,\},

    with evaluation points whose distribution never reaches ``tau`` clamped
    to the nearest end of that range.

    Parameters
    ----------
    x : MixedData or Array
        Conditioning sample.
    y : MixedData or Array
        Response sample, a single continuous column.
    bw : ConditionalBandwidth, SelectionResult, ConditionalFit or str
        The bandwidth, a selection to reuse, an earlier fit, or the name of
        a selection rule, either ``"cv_ls"`` or ``"normal_reference"``. The
        bandwidth is the one a conditional distribution uses.
    tau : float
        The quantile level in the open unit interval. Static.
    at_x : MixedData or Array, optional
        Conditioning evaluation points. Defaults to ``x``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Static.
    n_starts : int
        Number of restarts when a rule name asks for a search. Static.
    n_iter : int
        Bisection steps. The default halves the response range past any
        useful floating point resolution. Static.

    Returns
    -------
    QuantileFit
        The quantile estimates, the bandwidth behind them, and the selection
        if one ran.

    Examples
    --------
    Estimate a conditional median that tracks its covariate. The fitted
    median follows the response upward as the covariate grows.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import kerneljax as kj
           ...:
           ...: rng = np.random.default_rng(0)
           ...: x = rng.uniform(0.0, 1.0, 200)
           ...: y = 2.0 * x + rng.normal(0.0, 0.3, 200)
           ...: fit = kj.cquantile(x, y, "normal_reference", at_x=np.array([0.2, 0.5, 0.8]))
           ...: print(np.asarray(fit.value))

    See Also
    --------
    cdist : Estimate a conditional cumulative distribution.
    cdensity : Estimate a conditional probability density.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must lie strictly between 0 and 1, got {tau}")

    kernels = _resolve_kernels(kernels, getattr(bw, "kernels", None))
    x_train, y_train = _as_points(x), _as_points(y)

    if x_train.n != y_train.n:
        raise ValueError(
            f"x and y must describe the same sample, got {x_train.n} conditioning rows "
            f"against {y_train.n} response rows"
        )
    if y_train.spec.p != 1 or y_train.spec.p_con != 1:
        raise ValueError("cquantile inverts a scalar distribution, so y must be a single continuous column")

    bandwidth, selection = _resolve_conditional(x_train, y_train, bw, kernels, n_starts, "distribution")
    _require_usable(bandwidth.x)
    _require_usable(bandwidth.y)

    x_eval = x_train if at_x is None else _as_points(at_x, x_train.spec)
    weights_x = kweights(x_train, bandwidth.x, at=x_eval, kernels=kernels)
    weight_sum = jnp.sum(weights_x, axis=1)

    def distribution(candidates: Array) -> Array:
        accumulated = kweights(y_train, bandwidth.y, at=MixedData.continuous(candidates), kernels=kernels, op=Op.CDF)
        return safe_div(jnp.sum(weights_x * accumulated, axis=1), weight_sum)

    y_min = jnp.min(y_train.con[:, 0])
    y_max = jnp.max(y_train.con[:, 0])
    low = jnp.full(x_eval.n, y_min)
    high = jnp.full(x_eval.n, y_max)

    at_low = distribution(low)
    at_high = distribution(high)
    clamp_low = at_low >= tau
    clamp_high = at_high < tau

    def halve(_: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
        low, high = bracket
        mid = 0.5 * (low + high)
        upper = distribution(mid) >= tau
        return jnp.where(upper, low, mid), jnp.where(upper, mid, high)

    low, high = jax.lax.fori_loop(0, n_iter, halve, (low, high))
    value = 0.5 * (low + high)
    value = jnp.where(clamp_low, y_min, jnp.where(clamp_high, y_max, value))

    return QuantileFit(
        value=value,
        tau=tau,
        bandwidth=bandwidth,
        selection=selection,
        kernels=kernels,
        x_spec=x_train.spec,
        y_spec=y_train.spec,
        n_train=x_train.n,
    )


def cmode(
    x: MixedData | Array,
    y: MixedData | Array,
    bw: ConditionalBandwidth | SelectionResult | ConditionalFit | str,
    *,
    at_x: MixedData | Array | None = None,
    kernels: KernelSet | None = None,
    n_starts: int = 3,
) -> ModeFit:
    r"""Estimate the conditional mode of a categorical response.

    The conditional density is evaluated at every response level and the
    level with the largest density wins,

    .. math::

        \hat m(x) = \arg\max_{\ell} \hat f(\ell \mid x),

    with ties resolved toward the lowest level.

    Parameters
    ----------
    x : MixedData or Array
        Conditioning sample.
    y : MixedData or Array
        Response sample, a single unordered or ordered column.
    bw : ConditionalBandwidth, SelectionResult, ConditionalFit or str
        The bandwidth, a selection to reuse, an earlier fit, or the name of
        a selection rule, ``"cv_ml"``, ``"cv_ls"`` or ``"normal_reference"``.
        The bandwidth is the one a conditional density uses.
    at_x : MixedData or Array, optional
        Conditioning evaluation points. Defaults to ``x``, in which case the
        fit also reports the share of observations it classifies correctly.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Static.
    n_starts : int
        Number of restarts when a rule name asks for a search. Static.

    Returns
    -------
    ModeFit
        The modal levels, the density behind each, and the selection if one
        ran.

    Examples
    --------
    Recover the dominant group along a covariate. The modal level switches
    as the covariate crosses the group boundary.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import kerneljax as kj
           ...:
           ...: rng = np.random.default_rng(0)
           ...: x = rng.uniform(-2.0, 2.0, 300)
           ...: y = (x + rng.normal(0.0, 0.5, 300) > 0.0).astype(int)
           ...: data = kj.MixedData.continuous(x)
           ...: labels = kj.MixedData.from_blocks(unordered=y, unordered_levels=2)
           ...: fit = kj.cmode(data, labels, "normal_reference", at_x=np.array([-1.5, 0.0, 1.5]))
           ...: print(np.asarray(fit.value))

    See Also
    --------
    cdensity : Estimate a conditional probability density.
    cquantile : Estimate a conditional quantile.
    """
    kernels = _resolve_kernels(kernels, getattr(bw, "kernels", None))
    x_train, y_train = _as_points(x), _as_points(y)

    if x_train.n != y_train.n:
        raise ValueError(
            f"x and y must describe the same sample, got {x_train.n} conditioning rows "
            f"against {y_train.n} response rows"
        )
    if y_train.spec.p != 1 or y_train.spec.p_con:
        raise ValueError("cmode ranks response levels, so y must be a single unordered or ordered column")

    bandwidth, selection = _resolve_conditional(x_train, y_train, bw, kernels, n_starts, "density")
    _require_usable(bandwidth.x)
    _require_usable(bandwidth.y)

    x_eval = x_train if at_x is None else _as_points(at_x, x_train.spec)
    weights_x = kweights(x_train, bandwidth.x, at=x_eval, kernels=kernels)

    n_levels = (y_train.spec.uno_levels + y_train.spec.ord_levels)[0]
    codes = jnp.arange(n_levels)
    if y_train.spec.p_uno:
        candidates = MixedData.from_blocks(unordered=codes, unordered_levels=n_levels)
    else:
        candidates = MixedData.from_blocks(ordered=codes, ordered_levels=n_levels)

    per_level = kweights(y_train, bandwidth.y, at=candidates, kernels=kernels)
    probabilities = (weights_x @ per_level.T) / jnp.sum(weights_x, axis=1, keepdims=True)

    value = jnp.argmax(probabilities, axis=1)
    density = jnp.max(probabilities, axis=1)

    accuracy = None
    if at_x is None:
        observed = (y_train.uno if y_train.spec.p_uno else y_train.orde)[:, 0]
        accuracy = jnp.mean((value == observed).astype(density.dtype))

    return ModeFit(
        value=value,
        density=density,
        accuracy=accuracy,
        bandwidth=bandwidth,
        selection=selection,
        kernels=kernels,
        x_spec=x_train.spec,
        y_spec=y_train.spec,
        n_train=x_train.n,
    )


def cv_ml_conditional(
    x_train: MixedData,
    y_train: MixedData,
    bandwidth: ConditionalBandwidth,
    *,
    kernels: KernelSet | None = None,
) -> ScalarFloat:
    r"""Score a conditional bandwidth by leave-one-out likelihood.

    The criterion is the negated mean log density of each response under a
    fit that excluded it,

    .. math::

        \mathrm{CV}(h, \lambda) = -\frac{1}{n}
            \sum_{i=1}^{n} \log \hat f_{-i}(Y_i \mid X_i) .

    Parameters
    ----------
    x_train : MixedData
        Conditioning sample.
    y_train : MixedData
        Response sample.
    bandwidth : ConditionalBandwidth
        The bandwidth to score.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Static.

    Returns
    -------
    ScalarFloat
        The criterion value, smaller being better.
    """
    kernels = KernelSet() if kernels is None else kernels
    fold = jnp.arange(x_train.n)
    held_out = _evaluate(x_train, y_train, x_train, y_train, bandwidth, kernels, "density", fold=fold)
    return -jnp.mean(jnp.log(held_out))


def cv_ls_conditional_density(
    x_train: MixedData,
    y_train: MixedData,
    bandwidth: ConditionalBandwidth,
    *,
    kernels: KernelSet | None = None,
) -> ScalarFloat:
    r"""Score a conditional bandwidth by least squares on the density.

    The criterion estimates the integrated squared error of the conditional
    density through two terms,

    .. math::

        \mathrm{CV}(h, \lambda) = \frac{1}{n} \sum_{i=1}^{n} \Biggl(
            \frac{\sum_{j,k} K_x(X_i, X_j) K_x(X_i, X_k)
                  \, (K_y \ast K_y)(Y_j, Y_k)}
                 {\bigl( \sum_j K_x(X_i, X_j) \bigr)^2}
            - 2 \hat f_{-i}(Y_i \mid X_i) \Biggr),

    the first integrating the squared conditional density over the response
    with every observation kept, the second a leave-one-out fit at the
    observed pair. The response kernel enters the first term through its
    self-convolution, so the response kernels must implement ``conv``.

    Parameters
    ----------
    x_train : MixedData
        Conditioning sample.
    y_train : MixedData
        Response sample.
    bandwidth : ConditionalBandwidth
        The bandwidth to score.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Static.

    Returns
    -------
    ScalarFloat
        The criterion value, smaller being better.

    References
    ----------
    .. [1] Hall, P., Racine, J. S., & Li, Q. (2004). "Cross-validation and
           the estimation of conditional probability densities." Journal of
           the American Statistical Association, 99, 1015-1026.
    """
    kernels = KernelSet() if kernels is None else kernels

    weights_x = kweights(x_train, bandwidth.x, kernels=kernels)
    convolved = kweights(y_train, bandwidth.y, kernels=kernels, op=Op.CONV)
    values = kweights(y_train, bandwidth.y, kernels=kernels)
    scale = jnp.prod(bandwidth.y.h) if y_train.spec.p_con else 1.0

    full_sum = jnp.sum(weights_x, axis=1)
    integrated = jnp.sum((weights_x @ convolved) * weights_x, axis=1) / (full_sum**2 * scale)

    masked = weights_x * (1.0 - jnp.eye(x_train.n))
    cross = jnp.sum(masked * values, axis=1) / (jnp.sum(masked, axis=1) * scale)

    return jnp.mean(integrated - 2.0 * cross)


def cv_ls_conditional_distribution(
    x_train: MixedData,
    y_train: MixedData,
    bandwidth: ConditionalBandwidth,
    *,
    kernels: KernelSet | None = None,
    y_grid: MixedData | Array | None = None,
    n_grid: int = 100,
) -> ScalarFloat:
    r"""Score a conditional bandwidth by least squares on the indicator.

    The held-out conditional distribution is compared with the indicator of
    the observed response across a grid of response values,

    .. math::

        \mathrm{CV}(h, \lambda) = \frac{1}{n G} \sum_{i=1}^{n} \sum_{g=1}^{G}
            \bigl( \mathbf{1}(Y_i \le y_g)
            - \hat F_{-i}(y_g \mid X_i) \bigr)^2 .

    Unlike a likelihood, this loss is proper for a distribution function, so
    it is the criterion behind ``cdist(x, y, "cv_ls")``.

    Parameters
    ----------
    x_train : MixedData
        Conditioning sample.
    y_train : MixedData
        Response sample.
    bandwidth : ConditionalBandwidth
        The bandwidth to score.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Static.
    y_grid : MixedData or Array, optional
        Response values the indicator is compared on. Defaults to ``n_grid``
        per column quantiles of the response, which requires a fully
        continuous response.
    n_grid : int
        Size of the default quantile grid. Static.

    Returns
    -------
    ScalarFloat
        The criterion value, smaller being better.

    References
    ----------
    .. [1] Li, Q., Lin, J., & Racine, J. S. (2013). "Optimal bandwidth
           selection for nonparametric conditional distribution and quantile
           functions." Journal of Business & Economic Statistics, 31, 57-65.
    """
    kernels = KernelSet() if kernels is None else kernels
    grid = _response_grid(y_train, y_grid, n_grid)

    weights_x = kweights(x_train, bandwidth.x, kernels=kernels)
    accumulated = kweights(y_train, bandwidth.y, at=grid, kernels=kernels, op=Op.CDF)

    keep = 1.0 - jnp.eye(x_train.n)
    masked = weights_x * keep
    held_out = (masked @ accumulated.T) / jnp.sum(masked, axis=1, keepdims=True)

    return jnp.mean((_indicator(y_train, grid) - held_out) ** 2)


def select_conditional_bandwidth(
    x: MixedData | Array,
    y: MixedData | Array,
    *,
    kernels: KernelSet | None = None,
    solver: Callable[..., tuple[Array, ScalarFloat, Array, Array]] | None = None,
    n_starts: int = 3,
    method: Literal["cv_ml", "cv_ls"] = "cv_ml",
    target: Literal["density", "distribution"] = "density",
) -> SelectionResult:
    """Select a conditional bandwidth by held-out likelihood or least squares."""
    from kerneljax.selection.optimize import _multistart, lbfgs

    kernels = KernelSet() if kernels is None else kernels
    solver = lbfgs if solver is None else solver
    x_train, y_train = _as_points(x), _as_points(y)
    if method == "cv_ml":
        if target == "distribution":
            raise ValueError("cv_ml cannot select for a distribution, use cv_ls")
        criterion = cv_ml_conditional
    else:
        criterion = cv_ls_conditional_density if target == "density" else cv_ls_conditional_distribution

    transform = _conditional_transform(x_train, y_train, kernels)
    start = _conditional_reference(x_train, y_train, kernels, search=True)
    z0 = transform.to_unconstrained(start)

    def objective(z: Array) -> ScalarFloat:
        bandwidth = transform.from_unconstrained(z)
        return criterion(x_train, y_train, bandwidth, kernels=kernels)

    return _multistart(objective, transform, z0, solver, n_starts, criterion, kernels)


def _conditional(
    x: MixedData | Array,
    y: MixedData | Array,
    bw: ConditionalBandwidth | SelectionResult | ConditionalFit | str,
    at_x: MixedData | Array | None,
    at_y: MixedData | Array | None,
    kernels: KernelSet | None,
    n_starts: int,
    target: Literal["density", "distribution"],
) -> ConditionalFit:
    """Estimate a conditional density or distribution, which differ only in the response operator."""
    kernels = _resolve_kernels(kernels, getattr(bw, "kernels", None))
    x_train, y_train = _as_points(x), _as_points(y)

    if x_train.n != y_train.n:
        raise ValueError(
            f"x and y must describe the same sample, got {x_train.n} conditioning rows "
            f"against {y_train.n} response rows"
        )

    bandwidth, selection = _resolve_conditional(x_train, y_train, bw, kernels, n_starts, target)
    _require_usable(bandwidth.x)
    _require_usable(bandwidth.y)

    x_eval = x_train if at_x is None else _as_points(at_x, x_train.spec)
    y_eval = y_train if at_y is None else _as_points(at_y, y_train.spec)

    value = _evaluate(x_train, y_train, x_eval, y_eval, bandwidth, kernels, target)

    return ConditionalFit(
        value=value,
        bandwidth=bandwidth,
        selection=selection,
        kernels=kernels,
        x_spec=x_train.spec,
        y_spec=y_train.spec,
        n_train=x_train.n,
        target=target,
    )


def _evaluate(
    x_train: MixedData,
    y_train: MixedData,
    x_eval: MixedData,
    y_eval: MixedData,
    bandwidth: ConditionalBandwidth,
    kernels: KernelSet,
    target: Literal["density", "distribution"],
    fold: Array | None = None,
) -> Array:
    """Contract the conditioning weights against the response kernel and normalize."""
    weights_x = kweights(x_train, bandwidth.x, at=x_eval, kernels=kernels)
    response_op = Op.VALUE if target == "density" else Op.CDF
    weights_y = kweights(y_train, bandwidth.y, at=y_eval, kernels=kernels, op=response_op)

    if fold is not None:
        keep = fold[:, None] != fold[None, :]
        weights_x = jnp.where(keep, weights_x, 0.0)

    numerator = jnp.sum(weights_x * weights_y, axis=1)
    denominator = jnp.sum(weights_x, axis=1)

    if target == "density" and y_train.spec.p_con:
        numerator = numerator / jnp.prod(bandwidth.y.h)

    return safe_div(numerator, denominator)


def _resolve_conditional(
    x_train: MixedData,
    y_train: MixedData,
    bw: ConditionalBandwidth | SelectionResult | ConditionalFit | str,
    kernels: KernelSet,
    n_starts: int,
    target: Literal["density", "distribution"],
) -> tuple[ConditionalBandwidth, SelectionResult | None]:
    """Turn any accepted ``bw`` into a conditional bandwidth and the selection behind it."""
    if isinstance(bw, ConditionalBandwidth):
        return bw, None

    if isinstance(bw, ConditionalFit):
        return bw.bandwidth, bw.selection

    if isinstance(bw, SelectionResult):
        if not isinstance(bw.bandwidth, ConditionalBandwidth):
            raise TypeError(
                "the selection carries a single block bandwidth, so it came from an "
                "unconditional estimator and cannot be reused here"
            )
        return bw.bandwidth, bw

    if not isinstance(bw, str):
        raise TypeError(
            "bw must be a ConditionalBandwidth, a SelectionResult, a ConditionalFit or a "
            f"method name, got {type(bw).__name__}"
        )

    if bw == "normal_reference":
        return _conditional_reference(x_train, y_train, kernels, search=False), None

    if bw not in ("cv_ml", "cv_ls"):
        raise ValueError(f"bw must be 'cv_ml', 'cv_ls' or 'normal_reference', got {bw!r}")

    if bw == "cv_ml" and target == "distribution":
        raise ValueError(
            "cv_ml cannot select a bandwidth for a conditional distribution, since the "
            "likelihood of a CDF value rewards oversmoothing without bound. Select with "
            "cv_ls, reuse a cdensity fit, or supply a ConditionalBandwidth"
        )

    method = cast(Literal["cv_ml", "cv_ls"], bw)
    selection = select_conditional_bandwidth(
        x_train, y_train, kernels=kernels, n_starts=n_starts, method=method, target=target
    )
    return cast(ConditionalBandwidth, selection.bandwidth), selection


def _conditional_reference(
    x_train: MixedData, y_train: MixedData, kernels: KernelSet, *, search: bool
) -> ConditionalBandwidth:
    """Build a conditional rule of thumb, or the interior point a search starts from."""
    rule = _search_start if search else normal_reference
    return ConditionalBandwidth(x=rule(x_train, kernels), y=rule(y_train, kernels))


def _conditional_transform(x_train: MixedData, y_train: MixedData, kernels: KernelSet) -> ConditionalTransform:
    """Pair the two block transforms the selector optimizes through."""
    return ConditionalTransform(
        x=BandwidthTransform(spec=x_train.spec, kernels=kernels),
        y=BandwidthTransform(spec=y_train.spec, kernels=kernels),
    )


def _response_grid(y_train: MixedData, y_grid: MixedData | Array | None, n_grid: int) -> MixedData:
    """Build the response values the indicator is compared on."""
    if y_grid is not None:
        return _as_points(y_grid, y_train.spec)
    if y_train.spec.p_uno or y_train.spec.p_ord:
        raise ValueError(
            "the default quantile grid needs a fully continuous response, so pass the "
            "response values to compare on through y_grid"
        )
    probs = jnp.linspace(0.0, 1.0, n_grid)
    return MixedData.continuous(jnp.quantile(y_train.con, probs, axis=0))


def _indicator(y_train: MixedData, grid: MixedData) -> Array:
    """Compare every response with every grid point."""
    if y_train.spec.p_uno:
        raise ValueError("an unordered response has no ordering for the indicator to use")
    below = jnp.ones((y_train.n, grid.n), dtype=y_train.con.dtype if y_train.spec.p_con else None)
    if y_train.spec.p_con:
        below = below * jnp.all(y_train.con[:, None, :] <= grid.con[None, :, :], axis=-1)
    if y_train.spec.p_ord:
        below = below * jnp.all(y_train.orde[:, None, :] <= grid.orde[None, :, :], axis=-1)
    return below
