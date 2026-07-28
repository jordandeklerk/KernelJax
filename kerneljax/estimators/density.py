"""Unconditional mixed-type density estimation."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth, SelectionResult
from kerneljax.data import ColumnSpec, MixedData
from kerneljax.kernels import KernelSet
from kerneljax.ksum import ksum
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
    train: MixedData,
    bw: Bandwidth,
    *,
    at: MixedData | None = None,
    kernels: KernelSet | None = None,
    fold: Array | None = None,
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
    train : MixedData
        Training sample, supplying the sum over :math:`i`.
    bw : Bandwidth
        Bandwidths for every column.
    at : MixedData, optional
        Evaluation points. Defaults to ``train``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    fold : Array, optional
        Fold label of every training point, shape ``(n,)``. Training
        points sharing a fold with evaluation row ``j`` are dropped from
        its sum, and the divisor becomes the number retained for that
        row. ``at`` must then be ``None`` or match ``train`` in length.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`~kerneljax.ksum`.

    Returns
    -------
    DensityFit
        The density estimate and the bandwidth used to produce it.

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
    scale: Literal["per_train", "per_eval"] = "per_train" if bw.h_axis == "train" else "per_eval"
    total = ksum(train, bw, at=at, kernels=kernels, fold=fold, weight_scale=scale, chunk=chunk)

    if fold is None:
        denom = jnp.asarray(train.n, dtype=total.dtype)
    else:
        kept = train.n - jnp.bincount(fold, length=fold.shape[0])[fold]
        denom = kept.astype(total.dtype)[:, None]

    return DensityFit(
        value=(total / denom).reshape(-1),
        bandwidth=bw,
        kernels=kernels,
        spec=train.spec,
        n_train=train.n,
    )
