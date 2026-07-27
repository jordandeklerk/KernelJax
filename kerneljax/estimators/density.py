"""Unconditional mixed-type density estimation."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import MixedData
from kerneljax.kernels import KernelSet
from kerneljax.ksum import ksum
from kerneljax.typing import Array

__all__ = ["DensityFit", "density"]


@partial(jax.tree_util.register_dataclass, data_fields=["value", "bandwidth"], meta_fields=[])
@dataclasses.dataclass(frozen=True)
class DensityFit:
    """Result of a mixed-type density estimate.

    Parameters
    ----------
    value : Float[Array, " n_eval"]
        The density estimate at each evaluation point.
    bandwidth : Bandwidth
        The bandwidth used to produce ``value``.
    """

    value: Float[Array, " n_eval"]
    bandwidth: Bandwidth


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

    .. math::

        \hat f(x) = \frac{1}{n \prod_d h_d} \sum_{i=1}^{n} \prod_d K_d(x_d, X_{id})

    Kernels supplied through ``kernels`` must return values with no
    bandwidth factor of their own, since this applies
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
        Fold label of every training point, shape ``(n,)``. When given,
        the training points sharing a fold with evaluation row ``j`` are
        dropped from its sum, and the divisor for row ``j`` is the number
        of training points retained for that row. ``at`` must be ``None``
        or a sample of the same length as ``train`` whenever ``fold`` is
        given.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`kerneljax.ksum.ksum`.

    Returns
    -------
    DensityFit
        The density estimate and the bandwidth used to produce it.
    """
    kernels = KernelSet() if kernels is None else kernels
    scale: Literal["per_train", "per_eval"] = "per_train" if bw.h_axis == "train" else "per_eval"
    total = ksum(train, bw, at=at, kernels=kernels, fold=fold, weight_scale=scale, chunk=chunk)

    if fold is None:
        denom = jnp.asarray(train.n, dtype=total.dtype)
    else:
        kept = train.n - jnp.bincount(fold, length=fold.shape[0])[fold]
        denom = kept.astype(total.dtype)[:, None]

    return DensityFit(value=(total / denom).reshape(-1), bandwidth=bw)
