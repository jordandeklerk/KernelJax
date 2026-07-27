"""The generalized product kernel weight matrix, its derivative, and their sum."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float

from kerneljax.bandwidth import Bandwidth, broadcast_h
from kerneljax.data import ColumnSpec, Kind, MixedData
from kerneljax.kernels import KernelSet, Op
from kerneljax.typing import Array

__all__ = ["ksum", "kweights", "kweights_grad"]

OpSpec = str | Mapping[Kind, str] | tuple[str, ...]
WeightScale = Literal["none", "per_eval", "per_train"]


def kweights(
    train: MixedData,
    bw: Bandwidth,
    *,
    at: MixedData | None = None,
    kernels: KernelSet | None = None,
    op: OpSpec = Op.VALUE,
    mask: Bool[Array, "n_eval n_train"] | None = None,
    power: int = 1,
) -> Float[Array, "n_eval n_train"]:
    r"""Compute the generalized product kernel weight matrix.

    Every entry multiplies one kernel factor per column across the
    continuous, unordered and ordered blocks, following the generalized
    product kernel of [1]_,

    .. math::

        W_{ji} = \prod_{d} K_d(\mathrm{at}_{jd}, \mathrm{train}_{id}).

    The default kernel families are the Gaussian kernel for continuous
    columns, the Aitchison and Aitken (1976) kernel for unordered columns
    [2]_, and the Wang and van Ryzin (1981) kernel for ordered columns
    [3]_.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the second weight matrix axis.
    bw : Bandwidth
        Bandwidths for every column. ``bw.h_axis`` controls whether the
        continuous bandwidth is shared, evaluation indexed or training
        indexed.
    at : MixedData, optional
        Evaluation points, supplying the first weight matrix axis. Defaults
        to ``train``, giving the square training weight matrix.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
        Static.
    op : str or Mapping[Kind, str] or tuple of str
        Kernel operator, naming a method on the kernel. A string
        applies to every column, a mapping keyed by ``Kind`` applies one
        operator per kind, and a tuple gives one operator per column in
        original column order. Static.
    mask : Bool[Array, "n_eval n_train"], optional
        Multiplicative mask applied after the product and the power.
        ``True`` keeps a weight, ``False`` zeroes it.
    power : int
        Power applied to the product before masking. Static.

    Returns
    -------
    Float[Array, "n_eval n_train"]
        The weight matrix.

    Examples
    --------
    Compute the product kernel weight matrix for a continuous sample.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 5).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: weights = kj.kweights(train, bw)
           ...: print(weights.shape)

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2003). "Nonparametric estimation of
           distributions with categorical and continuous data." Journal of
           Multivariate Analysis, 86(2), 266-292.
    .. [2] Aitchison, J., & Aitken, C. G. G. (1976). "Multivariate binary
           discrimination by the kernel method." Biometrika, 63(3), 413-420.
    .. [3] Wang, M. C., & van Ryzin, J. (1981). "A class of smooth
           estimators for discrete distributions." Biometrika, 68(1), 301-309.
    """
    evaluate = train if at is None else at
    kernels = KernelSet() if kernels is None else kernels
    spec = train.spec

    if evaluate.spec.kinds != spec.kinds or evaluate.spec.n_levels != spec.n_levels:
        raise ValueError("train and at must share the same column kinds and level counts")

    op_con, uno_ops, ord_ops = _resolve_ops(spec, op)
    weights = jnp.ones((evaluate.n, train.n), dtype=train.con.dtype)

    if spec.p_con:
        bandwidth = broadcast_h(bw, spec.p_con)
        factors_con = jnp.asarray(
            getattr(kernels.continuous, op_con)(evaluate.con[:, None, :], train.con[None, :, :], bandwidth)
        )
        weights = weights * jnp.prod(factors_con, axis=-1)

    for column, levels in enumerate(spec.uno_levels):
        factor_uno = jnp.asarray(
            getattr(kernels.unordered, uno_ops[column])(
                evaluate.uno[:, None, column], train.uno[None, :, column], bw.lam_uno[column], levels
            )
        )
        weights = weights * factor_uno

    for column, levels in enumerate(spec.ord_levels):
        factor_ord = jnp.asarray(
            getattr(kernels.ordered, ord_ops[column])(
                evaluate.orde[:, None, column], train.orde[None, :, column], bw.lam_ord[column], levels
            )
        )
        weights = weights * factor_ord

    if power != 1:
        weights = weights**power

    if mask is not None:
        weights = jnp.where(mask, weights, 0.0)

    return weights


def kweights_grad(
    train: MixedData,
    bw: Bandwidth,
    *,
    at: MixedData | None = None,
    kernels: KernelSet | None = None,
    mask: Bool[Array, "n_eval n_train"] | None = None,
    chunk: int | tuple[int, int] | None = None,
) -> Float[Array, "p_con n_eval n_train"]:
    r"""Compute the derivative weight tensor, one continuous factor differentiated at a time.

    Every entry multiplies one kernel factor per column exactly as
    :func:`kweights` does with ``op=Op.VALUE``, except that the
    :math:`l`-th continuous factor is replaced by its derivative with
    respect to the evaluation coordinate,

    .. math::

        G_{l j i} = \frac{\partial K_l}{\partial x_l}
                    \!\left(\mathrm{at}_{jl}, \mathrm{train}_{il}\right)
                    \prod_{d \neq l} K_d(\mathrm{at}_{jd}, \mathrm{train}_{id}).

    Only continuous columns are differentiated, so the leading axis has
    length :math:`p_{\mathrm{con}}` rather than :math:`p`. Categorical
    factors always enter through their value.

    The product over :math:`d \neq l` is formed from prefix and suffix
    cumulative products over the continuous factor axis, so no factor
    is ever divided out. A zero factor in one continuous column then
    never turns into a division by zero when differentiating another.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the third tensor axis.
    bw : Bandwidth
        Bandwidths for every column, used exactly as in :func:`kweights`.
    at : MixedData, optional
        Evaluation points, supplying the second tensor axis. Defaults
        to ``train``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to
        ``KernelSet()``. Only the continuous family needs a ``deriv``
        method. Static.
    mask : Bool[Array, "n_eval n_train"], optional
        Multiplicative mask, broadcast across the leading axis and
        applied after the product. ``True`` keeps an entry, ``False``
        zeroes it.
    chunk : int or tuple of int, optional
        Chunk sizes as ``(eval, train)``. A bare int chunks only the
        evaluation axis. Bounds the peak memory of the computation at
        the cost of additional compute. Static.

    Returns
    -------
    Float[Array, "p_con n_eval n_train"]
        The derivative weight tensor.

    Examples
    --------
    Compute the derivative weight tensor for a continuous sample.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...: from kerneljax.ksum import kweights_grad
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 5).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: grad = kweights_grad(train, bw)
           ...: print(grad.shape)

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2007). Nonparametric Econometrics:
           Theory and Practice. Princeton University Press.
    """
    evaluate = train if at is None else at
    kernels = KernelSet() if kernels is None else kernels
    spec = train.spec

    if evaluate.spec.kinds != spec.kinds or evaluate.spec.n_levels != spec.n_levels:
        raise ValueError("train and at must share the same column kinds and level counts")

    if chunk is None:
        chunk_eval, chunk_train = None, None
    elif isinstance(chunk, tuple):
        chunk_eval, chunk_train = chunk
    else:
        chunk_eval, chunk_train = chunk, None

    if not spec.p_con:
        result = jnp.zeros((0, evaluate.n, train.n), dtype=train.con.dtype)
    elif chunk_eval is None:
        result = _grad_over_train(train, bw, evaluate, kernels, chunk_train)
    else:
        result = _grad_over_eval_chunks(train, bw, evaluate, kernels, chunk_eval, chunk_train)

    if mask is not None:
        result = jnp.where(mask[None, :, :], result, 0.0)

    return result


def ksum(
    train: MixedData,
    bw: Bandwidth,
    v: Float[Array, "n_train m"] | None = None,
    *,
    at: MixedData | None = None,
    kernels: KernelSet | None = None,
    op: OpSpec = Op.VALUE,
    fold: Array | None = None,
    power: int = 1,
    sample_weight: Float[Array, " n_train"] | None = None,
    weight_scale: WeightScale = "none",
    chunk: int | tuple[int, int] | None = None,
) -> Float[Array, "n_eval m"]:
    r"""Contract the product kernel weight matrix against ``v``.

    Computes the weighted kernel sum

    .. math::

        \mathrm{out}_{jk} = \sum_{i} W_{ji} \, v_{ik}

    with :math:`W` the generalized product kernel matrix [1]_ that
    :func:`kweights` returns for the same ``train``, ``bw``, ``at``,
    ``kernels``, ``op`` and ``power``, after any pair sharing a fold is
    dropped.

    This contraction is the primitive from which the density and
    cross-validation estimators in this package are built.

    Passing ``chunk`` never changes the result, only how much memory
    computing it needs.

    Parameters
    ----------
    train : MixedData
        Training sample, supplying the summation index.
    bw : Bandwidth
        Bandwidths for every column, passed through to :func:`kweights`.
    v : Float[Array, "n_train m"], optional
        Values to contract against. Defaults to a column of ones, so
        ``ksum`` then returns the row sums of the weight matrix.
    at : MixedData, optional
        Evaluation points, supplying the first output axis. Defaults to
        ``train``.
    kernels : KernelSet, optional
        Kernel families, passed through to :func:`kweights`. Defaults to
        ``KernelSet()``. Static.
    op : str or Mapping[Kind, str] or tuple of str
        Kernel operator, passed through to :func:`kweights`. Static.
    fold : Array, optional
        Fold label of every point, shape ``(n,)``. A pair is dropped
        wherever the evaluation and training fold agree. ``jnp.arange(n)``
        gives leave-one-out.
    power : int
        Power applied to the weights before the contraction. Static.
    sample_weight : Float[Array, " n_train"], optional
        Per training point scale, folded into ``v`` before the
        contraction.
    weight_scale : {"none", "per_eval", "per_train"}
        Placement of the :math:`1 / \prod h` divisor. ``"per_train"``
        folds it into ``v`` before the contraction, needed when
        ``bw.h_axis`` is ``"train"`` since the divisor varies with the
        sum index. ``"per_eval"`` divides the contracted result. Static.
    chunk : int or tuple of int, optional
        Chunk sizes as ``(eval, train)``. A bare int chunks only the
        evaluation axis. Bounds the peak memory of the contraction at the
        cost of additional compute. Static.

    Returns
    -------
    Float[Array, "n_eval m"]
        The contracted result.

    Examples
    --------
    Sum the product kernel weights across the training axis.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 20).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: bw = kj.Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: total = kj.ksum(train, bw)
           ...: print(total.shape)

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2003). "Nonparametric estimation of
           distributions with categorical and continuous data." Journal of
           Multivariate Analysis, 86(2), 266-292.
    """
    evaluate = train if at is None else at
    kernels = KernelSet() if kernels is None else kernels
    p_con = train.spec.p_con

    if v is None:
        v = jnp.ones((train.n, 1), dtype=train.con.dtype)
    if sample_weight is not None:
        v = v * sample_weight[:, None]

    if weight_scale == "per_train" and p_con:
        divisor = _h_divisor(bw, p_con)
        v = v / divisor[:, None] if divisor.ndim else v / divisor

    if chunk is None:
        chunk_eval, chunk_train = None, None
    elif isinstance(chunk, tuple):
        chunk_eval, chunk_train = chunk
    else:
        chunk_eval, chunk_train = chunk, None

    if chunk_eval is None:
        eval_idx = jnp.arange(evaluate.n)
        out = _sum_over_train(train, bw, v, evaluate, eval_idx, fold, kernels, op, power, chunk_train)
    else:
        out = _sum_over_eval_chunks(train, bw, v, evaluate, fold, kernels, op, power, chunk_eval, chunk_train)

    if weight_scale == "per_eval" and p_con:
        divisor = _h_divisor(bw, p_con)
        out = out / divisor[:, None] if divisor.ndim else out / divisor

    return out


def _resolve_ops(spec: ColumnSpec, op: OpSpec) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Normalize an operator specification to a method name per column."""
    if isinstance(op, str):
        return op, (op,) * spec.p_uno, (op,) * spec.p_ord

    if isinstance(op, Mapping):
        return (
            op.get(Kind.CONTINUOUS, Op.VALUE),
            (op.get(Kind.UNORDERED, Op.VALUE),) * spec.p_uno,
            (op.get(Kind.ORDERED, Op.VALUE),) * spec.p_ord,
        )

    if not isinstance(op, tuple):
        raise TypeError(f"op must be a string, a mapping or a tuple, got {type(op).__name__}")
    if len(op) != spec.p:
        raise ValueError(f"op has {len(op)} entries for {spec.p} columns")

    con_ops = tuple(entry for entry, kind in zip(op, spec.kinds, strict=True) if kind is Kind.CONTINUOUS)
    uno_ops = tuple(entry for entry, kind in zip(op, spec.kinds, strict=True) if kind is Kind.UNORDERED)
    ord_ops = tuple(entry for entry, kind in zip(op, spec.kinds, strict=True) if kind is Kind.ORDERED)

    return con_ops[0] if con_ops else Op.VALUE, uno_ops, ord_ops


def _h_divisor(bw: Bandwidth, p_con: int) -> Array:
    """Return the product of the continuous bandwidths, per row when train indexed."""
    if p_con == 0:
        return jnp.asarray(1.0)
    if bw.h_axis == "shared":
        return jnp.prod(bw.h)
    return jnp.prod(bw.h, axis=-1)


def _pad_rows(x: Array, chunk: int) -> Array:
    """Pad an array's leading axis to a multiple of chunk by repeating its last row, reshaped into blocks."""
    n_blocks = -(-x.shape[0] // chunk)
    pad = n_blocks * chunk - x.shape[0]
    if pad:
        x = jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0)
    return x.reshape(n_blocks, chunk, *x.shape[1:])


def _pad_index(idx: Array, size: int, chunk: int) -> Array:
    """Pad an index vector with an out of range fill value, reshaped into blocks."""
    n_blocks = -(-size // chunk)
    pad = n_blocks * chunk - size
    if pad:
        idx = jnp.concatenate([idx, jnp.full((pad,), size, dtype=idx.dtype)])
    return idx.reshape(n_blocks, chunk)


def _sum_over_train(
    train: MixedData,
    bw: Bandwidth,
    v: Array,
    eval_block: MixedData,
    eval_idx: Array,
    fold: Array | None,
    kernels: KernelSet,
    op: OpSpec,
    power: int,
    chunk_train: int | None,
) -> Array:
    """Contract the weight matrix for one evaluation block against v, chunking the training axis when asked."""
    if chunk_train is None:
        weights = kweights(train, bw, at=eval_block, kernels=kernels, op=op, power=power)
        if fold is not None:
            weights = weights * (fold[eval_idx][:, None] != fold[None, :])
        return weights @ v

    train_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_train), train)
    v_blocks = _pad_rows(v, chunk_train)
    idx_blocks = _pad_index(jnp.arange(train.n), train.n, chunk_train)
    h_blocks = _pad_rows(bw.h, chunk_train) if bw.h_axis == "train" else None

    def step(accumulated: Array, block: tuple[MixedData, Array, Array, Array | None]) -> tuple[Array, None]:
        train_block, v_block, idx_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)

        valid = (idx_block < train.n)[None, :]
        mask = valid if fold is None else valid & (fold[eval_idx][:, None] != fold[idx_block][None, :])

        weights = kweights(train_block, bw_block, at=eval_block, kernels=kernels, op=op, power=power)
        return accumulated + (weights * mask) @ v_block, None

    initial = jnp.zeros((eval_block.n, v.shape[1]), dtype=v.dtype)
    total, _ = jax.lax.scan(jax.checkpoint(step), initial, (train_blocks, v_blocks, idx_blocks, h_blocks))
    return total


def _sum_over_eval_chunks(
    train: MixedData,
    bw: Bandwidth,
    v: Array,
    evaluate: MixedData,
    fold: Array | None,
    kernels: KernelSet,
    op: OpSpec,
    power: int,
    chunk_eval: int,
    chunk_train: int | None,
) -> Array:
    """Chunk the evaluation axis, contracting each block against the training axis."""
    eval_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_eval), evaluate)
    idx_blocks = _pad_index(jnp.arange(evaluate.n), evaluate.n, chunk_eval)
    h_blocks = _pad_rows(bw.h, chunk_eval) if bw.h_axis == "eval" else None

    def step(carry: None, block: tuple[MixedData, Array, Array | None]) -> tuple[None, Array]:
        eval_block, eval_idx, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)
        result = _sum_over_train(train, bw_block, v, eval_block, eval_idx, fold, kernels, op, power, chunk_train)
        return carry, result

    _, stacked = jax.lax.scan(jax.checkpoint(step), None, (eval_blocks, idx_blocks, h_blocks))

    n_blocks = -(-evaluate.n // chunk_eval)
    return jnp.asarray(stacked.reshape(n_blocks * chunk_eval, -1)[: evaluate.n])


def _grad_block(train: MixedData, bw: Bandwidth, evaluate: MixedData, kernels: KernelSet) -> Array:
    """Compute the derivative weight tensor for one train and eval block with no chunking, assuming p_con > 0."""
    spec = train.spec
    p_con = spec.p_con

    categorical = jnp.ones((evaluate.n, train.n), dtype=train.con.dtype)

    for column, levels in enumerate(spec.uno_levels):
        factor_uno = jnp.asarray(
            kernels.unordered.value(
                evaluate.uno[:, None, column], train.uno[None, :, column], bw.lam_uno[column], levels
            )
        )
        categorical = categorical * factor_uno

    for column, levels in enumerate(spec.ord_levels):
        factor_ord = jnp.asarray(
            kernels.ordered.value(
                evaluate.orde[:, None, column], train.orde[None, :, column], bw.lam_ord[column], levels
            )
        )
        categorical = categorical * factor_ord

    bandwidth = broadcast_h(bw, p_con)
    values = jnp.asarray(kernels.continuous.value(evaluate.con[:, None, :], train.con[None, :, :], bandwidth))
    derivatives = jnp.asarray(kernels.continuous.deriv(evaluate.con[:, None, :], train.con[None, :, :], bandwidth))

    prefix_inclusive = jnp.cumprod(values, axis=-1)
    prefix = jnp.concatenate([jnp.ones_like(values[..., :1]), prefix_inclusive[..., :-1]], axis=-1)

    suffix_inclusive = jnp.cumprod(values[..., ::-1], axis=-1)[..., ::-1]
    suffix = jnp.concatenate([suffix_inclusive[..., 1:], jnp.ones_like(values[..., :1])], axis=-1)

    continuous = jnp.moveaxis(derivatives * prefix * suffix, -1, 0)
    return continuous * categorical[None, :, :]


def _grad_over_train(
    train: MixedData,
    bw: Bandwidth,
    eval_block: MixedData,
    kernels: KernelSet,
    chunk_train: int | None,
) -> Array:
    """Compute the derivative weight tensor for one evaluation block, chunking the training axis when asked."""
    if chunk_train is None:
        return _grad_block(train, bw, eval_block, kernels)

    train_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_train), train)
    h_blocks = _pad_rows(bw.h, chunk_train) if bw.h_axis == "train" else None

    def step(carry: None, block: tuple[MixedData, Array | None]) -> tuple[None, Array]:
        train_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)
        return carry, _grad_block(train_block, bw_block, eval_block, kernels)

    _, stacked = jax.lax.scan(jax.checkpoint(step), None, (train_blocks, h_blocks))

    p_con = train.spec.p_con
    combined = jnp.moveaxis(stacked, 0, 2).reshape(p_con, eval_block.n, -1)
    return combined[:, :, : train.n]


def _grad_over_eval_chunks(
    train: MixedData,
    bw: Bandwidth,
    evaluate: MixedData,
    kernels: KernelSet,
    chunk_eval: int,
    chunk_train: int | None,
) -> Array:
    """Chunk the evaluation axis, computing the derivative weight tensor for each block."""
    eval_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_eval), evaluate)
    h_blocks = _pad_rows(bw.h, chunk_eval) if bw.h_axis == "eval" else None

    def step(carry: None, block: tuple[MixedData, Array | None]) -> tuple[None, Array]:
        eval_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)
        return carry, _grad_over_train(train, bw_block, eval_block, kernels, chunk_train)

    _, stacked = jax.lax.scan(jax.checkpoint(step), None, (eval_blocks, h_blocks))

    p_con = train.spec.p_con
    combined = jnp.moveaxis(stacked, 0, 1).reshape(p_con, -1, train.n)
    return combined[:, : evaluate.n, :]
