"""Unconditional mixed-type cumulative distribution estimation."""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth, SelectionResult, normal_reference
from kerneljax.data import ColumnSpec, MixedData, _as_points
from kerneljax.kernels import KernelSet, Op
from kerneljax.ksum import ksum
from kerneljax.tuning.criteria import DistributionCriterion
from kerneljax.tuning.optimize import select_bandwidth
from kerneljax.typing import Array

__all__ = ["DistributionFit", "cdf"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "se", "bandwidth", "selection"],
    meta_fields=["kernels", "spec", "n_train"],
)
@dataclasses.dataclass(frozen=True)
class DistributionFit:
    """Result of a mixed-type cumulative distribution estimate.

    Attributes
    ----------
    value : Float[Array, " n_eval"]
        The distribution estimate at each evaluation point.
    se : Float[Array, " n_eval"]
        The standard error of the estimate at each evaluation point.
    bandwidth : Bandwidth
        The bandwidth used to produce ``value``.
    selection : SelectionResult, optional
        The selection that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    kernels : KernelSet
        Kernel families the estimate was produced with. Static.
    spec : ColumnSpec, optional
        Column metadata of the training sample. Static.
    n_train : int
        Number of training points. Static.
    """

    value: Float[Array, " n_eval"]
    se: Float[Array, " n_eval"]
    bandwidth: Bandwidth
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    spec: ColumnSpec | None = None
    n_train: int = 0


def cdf(
    train: MixedData | Array,
    bw: Bandwidth | SelectionResult | DistributionFit | str,
    *,
    at: MixedData | Array | None = None,
    kernels: KernelSet | None = None,
    n_starts: int = 3,
    chunk: int | tuple[int, int] | None = None,
) -> DistributionFit:
    r"""Estimate a mixed-type cumulative distribution.

    Averaging the integrated product kernel over the training sample gives

    .. math::

        \hat F(x) = \frac{1}{n} \sum_{i=1}^{n} \prod_{d} G_d(x_d, X_{id})

    where :math:`G_d` integrates the kernel of column :math:`d` from below up to
    the evaluation point. Unlike a density there is no :math:`\prod_d h_d`
    divisor, since an integrated kernel already carries the bandwidth.

    Treating the estimate as a proportion gives its standard error as

    .. math::

        \widehat{\mathrm{se}}(\hat F(x))
        = \sqrt{\frac{\hat F(x)\bigl(1 - \hat F(x)\bigr)}{n}}

    with :math:`n` the training sample size whatever the evaluation points are.

    Only continuous and ordered columns are supported, since an unordered column
    carries no order to accumulate along. Bandwidth selection for this estimator
    follows [1]_.

    Parameters
    ----------
    train : MixedData or Array
        Training sample, supplying the sum over :math:`i`. A raw array is read
        as continuous columns.
    bw : Bandwidth or SelectionResult or DistributionFit or str
        Bandwidths for every column, or a way of arriving at them. A
        :class:`~kerneljax.Bandwidth` is used as given, ``"cv_cdf"`` selects one
        by minimizing that criterion, and ``"normal_reference"`` applies
        :func:`~kerneljax.normal_reference` without a search. A previous
        selection or estimate reuses its bandwidth. To select with a criterion
        built by hand, minimize it with :func:`~kerneljax.select_bandwidth` and
        pass the result here.
    at : MixedData or Array, optional
        Evaluation points. Defaults to ``train``. A raw array is accepted only
        when the training sample is purely continuous.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    n_starts : int, optional
        Number of perturbed starting points screened when ``bw`` names a
        criterion to minimize. Ignored otherwise. Static.
    chunk : int or tuple of int, optional
        Chunk sizes as ``(eval, train)``. A bare int chunks only the evaluation
        axis. Bounds the peak memory of the sum at the cost of additional
        compute. Static.

    Returns
    -------
    DistributionFit
        Object containing the distribution estimate:

        - **value**: Distribution estimate at every evaluation point
        - **se**: Standard error of the estimate at every evaluation point
        - **bandwidth**: Bandwidth used to produce the estimate
        - **selection**: Selection that produced the bandwidth, or None when it was supplied directly
        - **kernels**: Kernel families the estimate was produced with
        - **spec**: Column metadata of the training sample
        - **n_train**: Number of training points

    Examples
    --------
    Estimate a distribution with the bandwidth chosen by cross validation and read it
    at three points. The estimate increases with the evaluation point, and the
    standard error is largest near the median where the binomial variance peaks.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import kerneljax as kj
           ...:
           ...: rng = np.random.default_rng(0)
           ...: z = np.concatenate([rng.normal(-2.0, 0.7, 100), rng.normal(2.0, 1.0, 100)])
           ...: fit = kj.cdf(z, "cv_cdf", at=np.array([-2.0, 0.0, 2.0]))
           ...: print(np.asarray(fit.value))
           ...: print(np.asarray(fit.se))

    See Also
    --------
    density : Estimate a mixed-type probability density.
    local_poly : Fit a local polynomial regression of mixed-type data.
    select_bandwidth : Select a bandwidth by minimizing a cross-validation criterion.

    References
    ----------
    .. [1] Li, Q., Li, J., & Racine, J. S. (2017). "Optimal bandwidth selection
           for nonparametric conditional distribution and quantile functions."
           Journal of Business and Economic Statistics, 35, 57-65.
    """
    kernels = KernelSet() if kernels is None else kernels
    train = _as_points(train)

    if train.spec.p_uno:
        raise ValueError(
            "cdf supports continuous and ordered columns only, "
            f"got {train.spec.p_uno} unordered columns which carry no order to accumulate along"
        )

    bandwidth, selection = _resolve_bandwidth(train, bw, kernels, n_starts, chunk)

    evaluate = None if at is None else _as_points(at, train.spec)
    if at is not None and isinstance(bw, SelectionResult | DistributionFit) and bandwidth.h_axis != "shared":
        raise ValueError(
            "reusing a bandwidth at new evaluation points requires h_axis 'shared', "
            f"got {bandwidth.h_axis!r} which is tied to the rows it was built for"
        )

    value, se = _cdf_values(train, bandwidth, evaluate, kernels=kernels, chunk=chunk)
    return DistributionFit(
        value=value,
        se=se,
        bandwidth=bandwidth,
        selection=selection,
        kernels=kernels,
        spec=train.spec,
        n_train=train.n,
    )


@partial(jax.jit, static_argnames=("kernels", "chunk"))
def _cdf_values(
    train: MixedData,
    bandwidth: Bandwidth,
    evaluate: MixedData | None,
    *,
    kernels: KernelSet,
    chunk: int | tuple[int, int] | None,
) -> tuple[Array, Array]:
    """Average the integrated product kernel over the training sample.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the sum.
    bandwidth : Bandwidth
        Bandwidths for every column.
    evaluate : MixedData, optional
        Evaluation points. ``None`` evaluates at the training sample.
    kernels : KernelSet
        Kernel families, one per column kind. Static.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`~kerneljax.ksum`. Static.

    Returns
    -------
    tuple of Array
        The distribution estimate and its standard error at every
        evaluation point.
    """
    total = ksum(train, bandwidth, at=evaluate, kernels=kernels, op=Op.CDF, chunk=chunk)
    value = (total / train.n).reshape(-1)
    return value, jnp.sqrt(value * (1.0 - value) / train.n)


def _resolve_bandwidth(
    train: MixedData,
    bw: Bandwidth | SelectionResult | DistributionFit | str,
    kernels: KernelSet,
    n_starts: int,
    chunk: int | tuple[int, int] | None,
) -> tuple[Bandwidth, SelectionResult | None]:
    """Turn any accepted ``bw`` into a bandwidth and the selection behind it."""
    if isinstance(bw, Bandwidth):
        return bw, None

    if isinstance(bw, DistributionFit):
        return bw.bandwidth, bw.selection

    if isinstance(bw, SelectionResult):
        return bw.bandwidth, bw

    if not isinstance(bw, str):
        raise TypeError(
            f"bw must be a Bandwidth, a SelectionResult, a DistributionFit or a method name, got {type(bw).__name__}"
        )

    if bw == "normal_reference":
        return normal_reference(train, kernels, target="distribution"), None

    if bw != "cv_cdf":
        raise ValueError(f"bw must be 'cv_cdf' or 'normal_reference', got {bw!r}")

    selection = select_bandwidth(train, DistributionCriterion(), kernels=kernels, n_starts=n_starts, chunk=chunk)
    return selection.bandwidth, selection
