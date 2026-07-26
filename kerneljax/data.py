"""A mixed-type design matrix stored as three dense blocks plus static column metadata."""

from __future__ import annotations

import dataclasses
import enum
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from kerneljax.typing import Array

__all__ = ["ColumnSpec", "Kind", "MixedData"]


class Kind(enum.Enum):
    r"""The kind of a design matrix column.

    ``ColumnSpec`` and ``MixedData`` use this enum to tag each column of a
    mixed-type design matrix as continuous, unordered categorical, or ordered
    categorical. The tag determines which of the three dense blocks of a
    :class:`MixedData` instance the column's data lives in, and which kernel
    arithmetic applies to it.

    Attributes
    ----------
    CONTINUOUS : Kind
        A real-valued column, smoothed with a continuous kernel such as the
        Gaussian or Epanechnikov kernel.
    UNORDERED : Kind
        An unordered categorical column with no ordering among its levels,
        smoothed with a kernel that only distinguishes equal from unequal
        categories.
    ORDERED : Kind
        An ordered categorical column whose integer levels have a natural
        ordering, smoothed with a kernel that decays with level distance.
    """

    CONTINUOUS = "continuous"
    UNORDERED = "unordered"
    ORDERED = "ordered"


@dataclasses.dataclass(frozen=True)
class ColumnSpec:
    r"""Static, hashable, array-free description of a mixed-type design matrix.

    A ``ColumnSpec`` records the kind and level count of every column using
    only tuples of enums, ints and strings, with no arrays. That makes it
    safe to use as the static metadata field of a :class:`MixedData` pytree,
    since :func:`jax.jit` hashes static fields to build its cache key and
    would reject anything containing an array.

    Parameters
    ----------
    kinds : tuple of Kind
        The kind of every column, in original column order.
    n_levels : tuple of int
        The number of levels of every column, in original column order. Zero
        for continuous columns.
    names : tuple of str, optional
        Optional column names, in original column order. Included in
        equality and hashing on purpose, since excluding them would let two
        differently-named ``MixedData`` instances share a jit cache entry
        and silently return the wrong metadata.
    """

    kinds: tuple[Kind, ...]
    n_levels: tuple[int, ...]
    names: tuple[str, ...] | None = None

    @property
    def p(self) -> int:
        """Total number of columns."""
        return len(self.kinds)

    @property
    def p_con(self) -> int:
        """Number of continuous columns."""
        return sum(k is Kind.CONTINUOUS for k in self.kinds)

    @property
    def p_uno(self) -> int:
        """Number of unordered categorical columns."""
        return sum(k is Kind.UNORDERED for k in self.kinds)

    @property
    def p_ord(self) -> int:
        """Number of ordered categorical columns."""
        return sum(k is Kind.ORDERED for k in self.kinds)

    @property
    def uno_levels(self) -> tuple[int, ...]:
        """Level counts of the unordered columns, in block order."""
        return tuple(c for k, c in zip(self.kinds, self.n_levels, strict=True) if k is Kind.UNORDERED)

    @property
    def ord_levels(self) -> tuple[int, ...]:
        """Level counts of the ordered columns, in block order."""
        return tuple(c for k, c in zip(self.kinds, self.n_levels, strict=True) if k is Kind.ORDERED)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["con", "uno", "orde"],
    meta_fields=["spec"],
)
@dataclasses.dataclass(frozen=True)
class MixedData:
    r"""A mixed-type design matrix held as three dense blocks.

    Continuous columns use :math:`(x - y) / h`, unordered columns use
    :math:`x = y`, and ordered columns use :math:`|i - j|`. Registered as a
    JAX pytree with ``spec`` static. Build with :meth:`from_blocks` or
    :meth:`continuous`, which are where validation runs.

    Parameters
    ----------
    con : Float[Array, "n p_con"]
        Continuous columns.
    uno : Int[Array, "n p_uno"]
        Unordered categorical columns, as signed int32 codes in ``[0, c)``.
    orde : Int[Array, "n p_ord"]
        Ordered categorical columns, as signed int32 levels on the integer
        lattice. Named ``orde`` rather than ``ord`` to avoid shadowing the
        builtin.
    spec : ColumnSpec
        Static column metadata.
    """

    # Validation lives in `from_blocks`, not here. JAX rebuilds pytrees by
    # calling this constructor, and under `vmap` it passes placeholder objects
    # instead of arrays, so anything that inspects the values would blow up.
    con: Float[Array, "n p_con"]
    uno: Int[Array, "n p_uno"]
    orde: Int[Array, "n p_ord"]
    spec: ColumnSpec

    @property
    def n(self) -> int:
        """Number of observations."""
        return self.con.shape[0]

    @classmethod
    def from_blocks(
        cls,
        con: Array | None = None,
        uno: Array | None = None,
        orde: Array | None = None,
        *,
        uno_levels: tuple[int, ...] = (),
        ord_levels: tuple[int, ...] = (),
        names: tuple[str, ...] | None = None,
    ) -> MixedData:
        r"""Build a validated ``MixedData`` from per-kind blocks.

        Shapes are checked, category codes are cast to signed int32 and
        range-checked against their level counts, and the resulting
        :class:`ColumnSpec` is assembled from the block widths, in block
        order (continuous, then unordered, then ordered).

        Parameters
        ----------
        con : Array, optional
            Continuous columns, shape ``(n, p_con)``.
        uno : Array, optional
            Unordered categorical codes, shape ``(n, p_uno)``, codes in
            ``[0, uno_levels[j])`` for column ``j``.
        orde : Array, optional
            Ordered categorical levels, shape ``(n, p_ord)``, codes in
            ``[0, ord_levels[j])`` for column ``j``.
        uno_levels : tuple of int, optional
            Level count of each unordered column, in block order, one entry
            per column of ``uno``.
        ord_levels : tuple of int, optional
            Level count of each ordered column, in block order, one entry
            per column of ``orde``.
        names : tuple of str, optional
            Column names, in block order (continuous, unordered, ordered).

        Returns
        -------
        MixedData
            The validated design matrix.
        """
        blocks = [b for b in (con, uno, orde) if b is not None]
        if not blocks:
            raise ValueError("at least one of con, uno or orde must be given")
        n = jnp.asarray(blocks[0]).shape[0]

        con_a = jnp.zeros((n, 0)) if con is None else jnp.asarray(con)
        # Force a concrete dtype. An array built from a bare Python scalar
        # (e.g. `jnp.full(shape, 3.0)`) is "weakly typed" in JAX, and two
        # otherwise identical MixedData instances differing only in that
        # flag make jax.jit retrace instead of hitting the cache.
        con_a = con_a.astype(con_a.dtype)
        uno_a = jnp.zeros((n, 0), jnp.int32) if uno is None else jnp.asarray(uno).astype(jnp.int32)
        ord_a = jnp.zeros((n, 0), jnp.int32) if orde is None else jnp.asarray(orde).astype(jnp.int32)

        if con_a.ndim < 2 or uno_a.ndim < 2 or ord_a.ndim < 2:
            raise ValueError("all blocks must have at least two dimensions, shape (n, p_kind)")
        if len(uno_levels) != uno_a.shape[1]:
            raise ValueError(f"uno_levels has {len(uno_levels)} entries for {uno_a.shape[1]} unordered columns")
        if len(ord_levels) != ord_a.shape[1]:
            raise ValueError(f"ord_levels has {len(ord_levels)} entries for {ord_a.shape[1]} ordered columns")
        for levels, block, label in ((uno_levels, uno_a, "unordered"), (ord_levels, ord_a, "ordered")):
            for j, c in enumerate(levels):
                if c < 2:
                    raise ValueError(f"{label} column {j} has {c} levels, need at least 2")
                col = block[:, j]
                if col.size and (int(col.min()) < 0 or int(col.max()) >= c):
                    raise ValueError(f"{label} column {j} has codes outside [0, {c})")

        kinds = (
            (Kind.CONTINUOUS,) * con_a.shape[1] + (Kind.UNORDERED,) * uno_a.shape[1] + (Kind.ORDERED,) * ord_a.shape[1]
        )
        n_levels = (0,) * con_a.shape[1] + tuple(uno_levels) + tuple(ord_levels)
        spec = ColumnSpec(kinds=kinds, n_levels=n_levels, names=names)
        return cls(con=con_a, uno=uno_a, orde=ord_a, spec=spec)

    @classmethod
    def continuous(cls, x: Array) -> MixedData:
        r"""Build a purely continuous design matrix.

        Parameters
        ----------
        x : Array
            Continuous data, shape ``(n, p)``. A 1-D array of shape ``(n,)``
            is promoted to a single column of shape ``(n, 1)``.

        Returns
        -------
        MixedData
            A design matrix with an empty unordered block and an empty
            ordered block.
        """
        x_a = jnp.asarray(x)
        if x_a.ndim == 1:
            x_a = x_a[:, None]
        return cls.from_blocks(con=x_a)

    def replace(self, **changes: Any) -> MixedData:
        r"""Return a copy with the given fields replaced.

        Parameters
        ----------
        **changes : Array or ColumnSpec
            New values for ``con``, ``uno``, ``orde`` or ``spec``, by
            keyword. Fields not named are copied unchanged.

        Returns
        -------
        MixedData
            The updated copy. ``self`` is left unmodified, since
            ``MixedData`` is frozen.
        """
        return dataclasses.replace(self, **changes)
