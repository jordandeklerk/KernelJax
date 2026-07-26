"""Cross-validation criteria for bandwidth selection, written to be minimized."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import MixedData
from kerneljax.kernels import KernelSet, Op
from kerneljax.ksum import ksum
from kerneljax.typing import Array, ScalarFloat

__all__ = ["cv_ls_density", "cv_ml_density"]


def cv_ml_density(
    train: MixedData,
    bw: Bandwidth,
    *,
    kernels: KernelSet | None = None,
    chunk: int | tuple[int, int] | None = None,
) -> ScalarFloat:
    r"""Likelihood cross-validation criterion for a density.

    .. math::

        \mathrm{CV}_{ml} = -\sum_{i=1}^{n} \log \hat f_{-i}(X_i)

    where :math:`\hat f_{-i}` is the leave-one-out density, normalized by
    :math:`(n - 1) \prod h`. np reports the negative of this value.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the leave-one-out sum over :math:`i`.
    bw : Bandwidth
        Bandwidths for every column.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`kerneljax.ksum.ksum`.

    Returns
    -------
    ScalarFloat
        The criterion value.
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

    .. math::

        \mathrm{CV}_{ls} = \int \hat f^2 - \frac{2}{n} \sum_{i=1}^{n} \hat f_{-i}(X_i)

    with

    .. math::

        \int \hat f^2 = \frac{1}{n^2 \prod h} \sum_{i} \sum_{j} \bar K(X_i, X_j)

    where :math:`\bar K` is the convolution product across every column
    kind, continuous and categorical alike, and :math:`\hat f_{-i}` is the
    same leave-one-out density used by :func:`cv_ml_density`. The double
    sum runs over the full matrix including its diagonal. np reports the
    negative of this value.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying both sums.
    bw : Bandwidth
        Bandwidths for every column.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to :func:`kerneljax.ksum.ksum`.

    Returns
    -------
    ScalarFloat
        The criterion value.
    """
    kernels = KernelSet() if kernels is None else kernels
    n = train.n
    scale: Literal["per_train", "per_eval"] = "per_train" if bw.h_axis == "train" else "per_eval"

    convolved = ksum(train, bw, kernels=kernels, op=Op.CONV, weight_scale=scale, chunk=chunk)
    integral_f_squared = jnp.sum(convolved) / (float(n) * n)

    f_loo = _loo_density(train, bw, kernels, chunk)
    return integral_f_squared - 2.0 * jnp.sum(f_loo) / n


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
