"""Unconditional mixed-type density estimation."""

from __future__ import annotations

from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp

from kerneljax.bandwidth import Bandwidth, SelectionResult, _require_usable, normal_reference
from kerneljax.data import MixedData, _as_points
from kerneljax.estimators.fit import DensityFit
from kerneljax.kernels import KernelSet
from kerneljax.kernels._checks import _check_conv_matches, _check_grad_diagonal, _check_value_mass
from kerneljax.kernels.sets import _resolve_kernels
from kerneljax.ksum import ksum
from kerneljax.selection.criteria import DensityCriterion
from kerneljax.selection.optimize import select_bandwidth
from kerneljax.typing import Array

__all__ = ["DensityFit", "density"]


@partial(jax.jit, static_argnames=("kernels", "weight_scale", "chunk"))
def _density_values(
    train: MixedData,
    bandwidth: Bandwidth,
    evaluate: MixedData | None,
    fold: Array | None,
    *,
    kernels: KernelSet,
    weight_scale: Literal["per_train", "per_eval"],
    chunk: int | tuple[int, int] | None,
) -> Array:
    """Contract the kernel weights and normalize them into a density."""
    total = ksum(train, bandwidth, at=evaluate, kernels=kernels, fold=fold, weight_scale=weight_scale, chunk=chunk)

    if fold is None:
        denom = jnp.asarray(train.n, dtype=total.dtype)
    else:
        _, codes = jnp.unique(fold, return_inverse=True, size=fold.shape[0])
        kept = train.n - jnp.bincount(codes, length=fold.shape[0])[codes]
        denom = kept.astype(total.dtype)[:, None]

    return (total / denom).reshape(-1)


def density(
    train: MixedData | Array,
    bw: Bandwidth | SelectionResult | DensityFit | str,
    *,
    at: MixedData | Array | None = None,
    kernels: KernelSet | None = None,
    fold: Array | None = None,
    n_starts: int = 3,
    chunk: int | tuple[int, int] | None = None,
) -> DensityFit:
    r"""Estimate a mixed-type probability density.

    Implements the generalized product kernel estimator of [1]_ for data with continuous,
    unordered categorical and ordered categorical columns.

    .. math::

        \hat f(x) = \frac{1}{n \prod_d h_d} \sum_{i=1}^{n} \prod_d K_d(x_d, X_{id})

    The product over :math:`d` runs across the continuous, unordered and ordered columns,
    each with the kernel :math:`K_d` appropriate to its kind. Kernels supplied through
    ``kernels`` must return values with no bandwidth factor of their own, since this applies
    :math:`1 / \prod_d h_d` exactly once.

    Parameters
    ----------
    train : MixedData or Array
        Training sample, supplying the sum over :math:`i`. A raw array is
        read as continuous columns.
    bw : Bandwidth or SelectionResult or DensityFit or str
        Bandwidths for every column, or a way of arriving at them. A
        :class:`~kerneljax.Bandwidth` is used as given. ``"cv_ml"`` and ``"cv_ls"``
        select one by minimizing that criterion, and ``"normal_reference"``
        applies :func:`~kerneljax.normal_reference` without a search. A previous
        selection or estimate reuses its bandwidth. To select
        with a criterion built by hand, minimize it with
        :func:`~kerneljax.select_bandwidth` and pass the result here.
    at : MixedData or Array, optional
        Evaluation points. Defaults to ``train``. A raw array is accepted
        only when the training sample is purely continuous.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    fold : Array, optional
        Fold label of every training point, shape ``(n,)``. Training
        points sharing a fold with evaluation row ``j`` are dropped from
        its sum, and the divisor becomes the number retained for that
        row. ``at`` must then be ``None`` or match ``train`` in length.
    n_starts : int, optional
        Number of perturbed starting points screened when ``bw`` names a
        criterion to minimize. Ignored otherwise. Static.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`~kerneljax.ksum`.

    Returns
    -------
    DensityFit
        Object containing the density estimate:

        - **value**: Density estimate at every evaluation point
        - **bandwidth**: Bandwidth used to produce the estimate
        - **selection**: Selection that produced the bandwidth, or None when it was supplied directly
        - **kernels**: Kernel families the estimate was produced with
        - **spec**: Column metadata of the training sample
        - **n_train**: Number of training points

    Examples
    --------
    Estimate a bimodal density with the bandwidth chosen by likelihood cross
    validation, then evaluate it at the two modes and at the trough between them.
    The estimate is high at each mode and low in between.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import kerneljax as kj
           ...:
           ...: rng = np.random.default_rng(0)
           ...: z = np.concatenate([rng.normal(-2.0, 0.7, 100), rng.normal(2.0, 1.0, 100)])
           ...: fit = kj.density(z, "cv_ml", at=np.array([-2.0, 0.0, 2.0]))
           ...: print(np.asarray(fit.value))

    See Also
    --------
    local_poly : Fit a local polynomial regression of mixed-type data.
    cdf : Estimate a mixed-type cumulative distribution.
    cdensity : Estimate a conditional probability density.
    summary : Measure how well a fitted estimator describes the sample it was fit on.
    select_bandwidth : Select a bandwidth by minimizing a cross-validation criterion.

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2003). "Nonparametric estimation of
           distributions with categorical and continuous data." Journal of
           Multivariate Analysis, 86, 266-292.
    """
    kernels = _resolve_kernels(kernels, getattr(bw, "kernels", None))
    train = _as_points(train)

    if train.spec.p_con:
        _check_value_mass(kernels.continuous)
        if isinstance(bw, str) and bw != "normal_reference":
            _check_grad_diagonal(kernels.continuous, "value")
        if bw == "cv_ls":
            _check_conv_matches(kernels.continuous)

    bandwidth, selection = _resolve_bandwidth(train, bw, kernels, n_starts, chunk)
    _require_usable(bandwidth)

    evaluate = None if at is None else _as_points(at, train.spec)
    if at is not None and isinstance(bw, SelectionResult | DensityFit) and bandwidth.h_axis != "shared":
        raise ValueError(
            "reusing a bandwidth at new evaluation points requires h_axis 'shared', "
            f"got {bandwidth.h_axis!r} which is tied to the rows it was built for"
        )

    scale: Literal["per_train", "per_eval"] = "per_train" if bandwidth.h_axis == "train" else "per_eval"
    value = _density_values(train, bandwidth, evaluate, fold, kernels=kernels, weight_scale=scale, chunk=chunk)

    return DensityFit(
        value=value,
        bandwidth=bandwidth,
        selection=selection,
        kernels=kernels,
        spec=train.spec,
        n_train=train.n,
    )


def _resolve_bandwidth(
    train: MixedData,
    bw: Bandwidth | SelectionResult | DensityFit | str,
    kernels: KernelSet,
    n_starts: int,
    chunk: int | tuple[int, int] | None,
) -> tuple[Bandwidth, SelectionResult | None]:
    """Turn any accepted ``bw`` into a bandwidth and the selection behind it."""
    if isinstance(bw, Bandwidth):
        return bw, None

    if isinstance(bw, DensityFit):
        return bw.bandwidth, bw.selection

    if isinstance(bw, SelectionResult):
        return cast(Bandwidth, bw.bandwidth), bw

    if not isinstance(bw, str):
        raise TypeError(
            f"bw must be a Bandwidth, a SelectionResult, a DensityFit or a method name, got {type(bw).__name__}"
        )

    if bw == "normal_reference":
        return normal_reference(train, kernels), None

    if bw not in ("cv_ml", "cv_ls"):
        raise ValueError(f"bw must be 'cv_ml', 'cv_ls' or 'normal_reference', got {bw!r}")

    criterion = DensityCriterion(method=cast(Literal["cv_ml", "cv_ls"], bw))
    selection = select_bandwidth(train, criterion, kernels=kernels, n_starts=n_starts, chunk=chunk)
    return selection.bandwidth, selection
