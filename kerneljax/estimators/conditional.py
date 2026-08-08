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

    The estimate is the ratio of a joint density to a marginal one, which
    reduces to the response kernel averaged under weights that the
    conditioning sample normalizes,

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
    """
    return _conditional(x, y, bw, at_x, at_y, kernels, n_starts, "distribution")


def cv_ml_conditional(
    x_train: MixedData,
    y_train: MixedData,
    bandwidth: ConditionalBandwidth,
    *,
    kernels: KernelSet | None = None,
    target: Literal["density", "distribution"] = "density",
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
    target : {"density", "distribution"}
        Which estimate to score. Static.

    Returns
    -------
    ScalarFloat
        The criterion value, smaller being better.
    """
    kernels = KernelSet() if kernels is None else kernels
    fold = jnp.arange(x_train.n)
    held_out = _evaluate(x_train, y_train, x_train, y_train, bandwidth, kernels, target, fold=fold)
    return -jnp.mean(jnp.log(held_out))


def select_conditional_bandwidth(
    x: MixedData | Array,
    y: MixedData | Array,
    *,
    kernels: KernelSet | None = None,
    solver: Callable[..., tuple[Array, ScalarFloat, Array, Array]] | None = None,
    n_starts: int = 3,
    target: Literal["density", "distribution"] = "density",
) -> SelectionResult:
    """Select a conditional bandwidth by minimizing the leave-one-out likelihood."""
    from kerneljax.selection.optimize import _multistart, lbfgs

    kernels = KernelSet() if kernels is None else kernels
    solver = lbfgs if solver is None else solver
    x_train, y_train = _as_points(x), _as_points(y)

    transform = _conditional_transform(x_train, y_train, kernels)
    start = _conditional_reference(x_train, y_train, kernels, search=True)
    z0 = transform.to_unconstrained(start)

    def objective(z: Array) -> ScalarFloat:
        bandwidth = transform.from_unconstrained(z)
        return cv_ml_conditional(x_train, y_train, bandwidth, kernels=kernels, target=target)

    return _multistart(objective, transform, z0, solver, n_starts, cv_ml_conditional, kernels)


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

    if bw != "cv_ml":
        raise ValueError(f"bw must be 'cv_ml' or 'normal_reference', got {bw!r}")

    selection = select_conditional_bandwidth(x_train, y_train, kernels=kernels, n_starts=n_starts, target=target)
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
