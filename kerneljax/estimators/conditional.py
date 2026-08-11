"""Conditional density and conditional distribution over mixed-type data."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import (
    BandwidthTransform,
    ConditionalBandwidth,
    ConditionalTransform,
    SelectionResult,
    _require_usable,
    _search_start,
    normal_reference,
)
from kerneljax.data import ColumnSpec, MixedData, _as_points
from kerneljax.kernels import KernelSet, Op
from kerneljax.kernels.sets import _resolve_kernels
from kerneljax.ksum import kweights
from kerneljax.typing import Array, ScalarFloat

__all__ = ["ConditionalFit", "cdensity", "cdist"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "bandwidth", "selection"],
    meta_fields=["kernels", "x_spec", "y_spec", "n_train", "target"],
)
@dataclasses.dataclass(frozen=True)
class ConditionalFit:
    """Result of a conditional density or conditional distribution estimate.

    Attributes
    ----------
    value : Float[Array, " n_eval"]
        The estimate at each evaluation pair.
    bandwidth : ConditionalBandwidth
        The bandwidth used to produce ``value``, one block per sample.
    selection : SelectionResult, optional
        The selection that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    kernels : KernelSet
        Kernel families the estimate was produced with. Static.
    x_spec : ColumnSpec, optional
        Column metadata of the conditioning sample. Static.
    y_spec : ColumnSpec, optional
        Column metadata of the response sample. Static.
    n_train : int
        Number of training points. Static.
    target : {"density", "distribution"}
        Which estimate ``value`` holds. Static.
    """

    value: Float[Array, " n_eval"]
    bandwidth: ConditionalBandwidth
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    x_spec: ColumnSpec | None = None
    y_spec: ColumnSpec | None = None
    n_train: int = 0
    target: Literal["density", "distribution"] = "density"


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
        a selection rule, either ``"cv_ml"`` or ``"normal_reference"``.
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
        value rewards oversmoothing without bound, which is why np offers no
        such method either.
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


def cv_ls_conditional(
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
        Size of the default quantile grid, matching np's ``ngrid``. Static.

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
) -> SelectionResult:
    """Select a conditional bandwidth by held-out likelihood or least squares."""
    from kerneljax.selection.optimize import _multistart, lbfgs

    kernels = KernelSet() if kernels is None else kernels
    solver = lbfgs if solver is None else solver
    x_train, y_train = _as_points(x), _as_points(y)
    criterion = cv_ml_conditional if method == "cv_ml" else cv_ls_conditional

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

    return numerator / denominator


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

    if bw == "cv_ls" and target == "density":
        raise ValueError(
            "cv_ls scores the conditional distribution against the response indicator, "
            "so it selects for cdist. Select a conditional density with cv_ml"
        )

    method = cast(Literal["cv_ml", "cv_ls"], bw)
    selection = select_conditional_bandwidth(x_train, y_train, kernels=kernels, n_starts=n_starts, method=method)
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
