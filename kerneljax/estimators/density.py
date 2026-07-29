"""Unconditional mixed-type density estimation."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth, SelectionResult, normal_reference
from kerneljax.data import ColumnSpec, MixedData, _as_points
from kerneljax.kernels import KernelSet
from kerneljax.ksum import ksum
from kerneljax.tuning.criteria import DensityCriterion
from kerneljax.tuning.optimize import select_bandwidth
from kerneljax.typing import Array

__all__ = ["DensityFit", "density"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "bandwidth", "selection"],
    meta_fields=["kernels", "spec", "n_train"],
)
@dataclasses.dataclass(frozen=True)
class DensityFit:
    """Result of a mixed-type density estimate.

    Parameters
    ----------
    value : Float[Array, " n_eval"]
        The density estimate at each evaluation point.
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
    bandwidth: Bandwidth
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    spec: ColumnSpec | None = None
    n_train: int = 0


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
    Estimate a density from a sample of continuous data.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: fit = kj.density(train, bw)
           ...: print(fit.value[:5])

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2003). "Nonparametric estimation of
           distributions with categorical and continuous data." Journal of
           Multivariate Analysis, 86, 266-292.
    """
    kernels = KernelSet() if kernels is None else kernels
    train = _as_points(train)
    bandwidth, selection = _resolve_bandwidth(train, bw, kernels, n_starts, chunk)

    evaluate = None if at is None else _as_points(at, train.spec)
    if at is not None and isinstance(bw, SelectionResult | DensityFit) and bandwidth.h_axis != "shared":
        raise ValueError(
            "reusing a bandwidth at new evaluation points requires h_axis 'shared', "
            f"got {bandwidth.h_axis!r} which is tied to the rows it was built for"
        )

    scale: Literal["per_train", "per_eval"] = "per_train" if bandwidth.h_axis == "train" else "per_eval"
    total = ksum(train, bandwidth, at=evaluate, kernels=kernels, fold=fold, weight_scale=scale, chunk=chunk)

    if fold is None:
        denom = jnp.asarray(train.n, dtype=total.dtype)
    else:
        kept = train.n - jnp.bincount(fold, length=fold.shape[0])[fold]
        denom = kept.astype(total.dtype)[:, None]

    return DensityFit(
        value=(total / denom).reshape(-1),
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
        return bw.bandwidth, bw

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
