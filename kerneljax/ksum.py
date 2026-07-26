"""The generalized product kernel sum and its memory-efficient contraction."""

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

__all__ = ["fold_mask", "ksum", "kweights"]

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
    continuous, unordered and ordered blocks.

    .. math::

        W_{ji} = \prod_{d} K_d(\mathrm{at}_{jd}, \mathrm{train}_{id})

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
        Kernel operator, resolved to a method name at trace time. A string
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


def fold_mask(fold_eval: Array, fold_train: Array) -> Bool[Array, "n_eval n_train"]:
    r"""Flag every evaluation and training pair that does not share a fold. ``True`` keeps the weight.

    A single fold vector reused for both axes subsumes leave-one-out,
    k-fold, blocked and clustered cross validation at the same shape.
    Leave-one-out is ``fold_eval = fold_train = jnp.arange(n)``.

    Parameters
    ----------
    fold_eval : Array
        Fold label of every evaluation point, shape ``(n_eval,)``.
    fold_train : Array
        Fold label of every training point, shape ``(n_train,)``.

    Returns
    -------
    Bool[Array, "n_eval n_train"]
        ``True`` where the pair is kept, ``False`` where the fold agrees
        and the pair is held out.
    """
    return fold_eval[:, None] != fold_train[None, :]


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

    .. math::

        \mathrm{out}_{jk} = \sum_{i} W_{ji} \, v_{ik}

    with :math:`W` the matrix :func:`kweights` returns for the same
    ``train``, ``bw``, ``at``, ``kernels``, ``op`` and ``power``, after any
    pair sharing a fold is dropped. Passing ``chunk`` never changes the
    result, only how much memory computing it needs.

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
        ``bw.h_axis`` is ``"train"`` since the divisor then varies with the
        summation index. ``"per_eval"`` divides the contracted result.
        Static.
    chunk : int or tuple of int, optional
        Chunk sizes as ``(eval, train)``. A bare int chunks only the
        evaluation axis. Bounds the peak memory of the contraction at the
        cost of additional compute. Static.

    Returns
    -------
    Float[Array, "n_eval m"]
        The contracted result.
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
            weights = weights * fold_mask(fold[eval_idx], fold)
        return weights @ v

    train_blocks = jax.tree.map(lambda column: _pad_rows(column, chunk_train), train)
    v_blocks = _pad_rows(v, chunk_train)
    idx_blocks = _pad_index(jnp.arange(train.n), train.n, chunk_train)
    h_blocks = _pad_rows(bw.h, chunk_train) if bw.h_axis == "train" else None

    def step(accumulated: Array, block: tuple[MixedData, Array, Array, Array | None]) -> tuple[Array, None]:
        train_block, v_block, idx_block, h_block = block
        bw_block = bw if h_block is None else bw.replace(h=h_block)

        valid = (idx_block < train.n)[None, :]
        mask = valid if fold is None else valid & fold_mask(fold[eval_idx], fold[idx_block])

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
    """Chunk the evaluation axis with lax.scan, contracting each block against the training axis."""
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
