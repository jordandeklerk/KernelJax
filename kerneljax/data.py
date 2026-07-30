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

__all__ = ["ColumnSpec", "Kind", "MixedData", "grid", "quantile_grid"]


class Kind(enum.Enum):
    r"""The kind of a design matrix column.

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
    r"""Static description of a mixed-type design matrix.

    Parameters
    ----------
    kinds : tuple of Kind
        The kind of every column, in original column order.
    n_levels : tuple of int
        The number of levels of every column, in original column order. Zero
        for continuous columns.
    names : tuple of str, optional
        Optional column names, in original column order. Names take part in
        equality, so two specs differing only in their names are not equal.
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
        :class:`~kerneljax.ColumnSpec` is assembled from the block widths, in block
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


def grid(
    data: MixedData | Array,
    *,
    vary: int | str = 0,
    n: int = 50,
    quantile: float = 0.5,
    trim: float = 0.0,
) -> MixedData:
    r"""Build evaluation points that sweep one column and hold the rest fixed.

    The swept column runs across its observed range for a continuous column, or
    across every level for a categorical one. Each remaining column is pinned at
    a representative value, its quantile when continuous, its most common level
    when unordered, and the first level reaching ``quantile`` of the sample when
    ordered.

    Parameters
    ----------
    data : MixedData or Array
        Sample the grid is built from. A raw array is read as continuous columns.
    vary : int or str
        Column to sweep, either its position in original column order or its
        name. Static.
    n : int
        Number of points across a continuous column. Ignored for a categorical
        column, which uses one point per level. Static.
    quantile : float
        Quantile that pins every column other than the swept one. Static.
    trim : float
        Shrinks the swept range toward the middle of the sample, so ``0.1``
        runs from the tenth to the ninetieth percentile. A negative value
        instead reflects the inner quantiles about the extremes to reach past
        the observed range, where the estimate is extrapolating. Static.

    Returns
    -------
    MixedData
        Evaluation points sharing the column metadata of ``data``, with ``n``
        rows for a continuous sweep and one row per level otherwise.

    Examples
    --------
    Sweep the first column of a two column sample.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.stack([jnp.linspace(-2.0, 2.0, 20), jnp.linspace(0.0, 1.0, 20)], axis=1)
           ...: points = kj.grid(kj.MixedData.continuous(x), n=5)
           ...: print(points.con)

    See Also
    --------
    local_poly : Fit a local polynomial regression.
    density : Estimate a mixed-type probability density.
    """
    data = _as_points(data)
    spec = data.spec
    swept = _swept_column(spec, vary)
    rows = n if spec.kinds[swept] is Kind.CONTINUOUS else spec.n_levels[swept]

    positions = {kind: [i for i, k in enumerate(spec.kinds) if k is kind] for kind in Kind}

    con = [
        jnp.linspace(*_sweep_range(data.con[:, pos], trim), rows)
        if original == swept
        else jnp.full((rows,), jnp.quantile(data.con[:, pos], quantile))
        for pos, original in enumerate(positions[Kind.CONTINUOUS])
    ]

    uno = []
    for pos, original in enumerate(positions[Kind.UNORDERED]):
        levels = spec.uno_levels[pos]
        column = data.uno[:, pos]
        if original == swept:
            uno.append(jnp.arange(levels, dtype=column.dtype))
        else:
            uno.append(jnp.full((rows,), jnp.argmax(jnp.bincount(column, length=levels)), dtype=column.dtype))

    orde = []
    for pos, original in enumerate(positions[Kind.ORDERED]):
        levels = spec.ord_levels[pos]
        column = data.orde[:, pos]
        if original == swept:
            orde.append(jnp.arange(levels, dtype=column.dtype))
        else:
            reached = jnp.searchsorted(jnp.cumsum(jnp.bincount(column, length=levels)), quantile * column.shape[0])
            orde.append(jnp.full((rows,), reached, dtype=column.dtype))

    return MixedData(
        con=jnp.stack(con, axis=1) if con else jnp.zeros((rows, 0), dtype=data.con.dtype),
        uno=jnp.stack(uno, axis=1) if uno else jnp.zeros((rows, 0), dtype=data.uno.dtype),
        orde=jnp.stack(orde, axis=1) if orde else jnp.zeros((rows, 0), dtype=data.orde.dtype),
        spec=spec,
    )


def quantile_grid(data: MixedData | Array, *, n: int = 100) -> MixedData:
    r"""Build evaluation points at evenly spaced probabilities, one point per probability.

    Every column is evaluated at the same vector of probabilities
    :math:`p_1, \ldots, p_n` spanning the unit interval, so point :math:`j` is
    the tuple of column quantiles at :math:`p_j`. The result has ``n`` rows
    whatever the number of columns, unlike a product grid.

    A continuous column takes its own quantile, an unordered column takes its
    most common level at every probability, and an ordered column takes the
    first level whose cumulative share reaches the probability.

    Parameters
    ----------
    data : MixedData or Array
        Sample the quantiles are taken from. A raw array is read as continuous
        columns.
    n : int
        Number of probabilities, and so the number of rows returned. Static.

    Returns
    -------
    MixedData
        Evaluation points sharing the column metadata of ``data``, with ``n`` rows.

    Examples
    --------
    Build a hundred points spanning a two column sample.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.stack([jnp.linspace(-2.0, 2.0, 20), jnp.linspace(0.0, 1.0, 20)], axis=1)
           ...: points = kj.quantile_grid(kj.MixedData.continuous(x), n=5)
           ...: print(points.con)

    See Also
    --------
    grid : Sweep one column and hold the rest fixed.
    """
    data = _as_points(data)
    spec = data.spec
    probs = jnp.linspace(0.0, 1.0, n, dtype=data.con.dtype if spec.p_con else jnp.float32)

    con = [jnp.quantile(data.con[:, pos], probs) for pos in range(spec.p_con)]

    uno = []
    for pos, levels in enumerate(spec.uno_levels):
        column = data.uno[:, pos]
        mode = jnp.argmax(jnp.bincount(column, length=levels))
        uno.append(jnp.full((n,), mode, dtype=column.dtype))

    orde = []
    for pos, levels in enumerate(spec.ord_levels):
        column = data.orde[:, pos]
        cumulative = jnp.cumsum(jnp.bincount(column, length=levels))
        reached = jnp.searchsorted(cumulative, probs * column.shape[0])
        orde.append(jnp.clip(reached, 0, levels - 1).astype(column.dtype))

    return MixedData(
        con=jnp.stack(con, axis=1) if con else jnp.zeros((n, 0), dtype=data.con.dtype),
        uno=jnp.stack(uno, axis=1) if uno else jnp.zeros((n, 0), dtype=data.uno.dtype),
        orde=jnp.stack(orde, axis=1) if orde else jnp.zeros((n, 0), dtype=data.orde.dtype),
        spec=spec,
    )


def _as_points(data: MixedData | Array, spec: ColumnSpec | None = None) -> MixedData:
    """Promote a raw array to a purely continuous sample, leaving a ``MixedData`` untouched."""
    if isinstance(data, MixedData):
        return data

    if spec is not None and (spec.p_uno or spec.p_ord):
        raise TypeError("evaluation points must be a MixedData when the training sample has categorical columns")

    return MixedData.continuous(jnp.asarray(data))


def _swept_column(spec: ColumnSpec, vary: int | str) -> int:
    """Resolve a column position or name to a position in original column order."""
    if isinstance(vary, str):
        if spec.names is None or vary not in spec.names:
            raise ValueError(f"no column named {vary!r}, the sample has names {spec.names}")
        return spec.names.index(vary)

    if not -spec.p <= vary < spec.p:
        raise ValueError(f"vary={vary} is out of range for {spec.p} columns")

    return vary % spec.p


def _sweep_range(column: Array, trim: float) -> tuple[Array, Array]:
    """Bounds of a swept continuous column, pulled inward or pushed past the sample."""
    if trim < 0.0:
        edge = abs(trim)
        lowest, inner_low, inner_high, highest = jnp.quantile(
            column, jnp.asarray([0.0, edge, 1.0 - edge, 1.0], dtype=column.dtype)
        )
        return 2.0 * lowest - inner_low, 2.0 * highest - inner_high

    lower, upper = jnp.quantile(column, jnp.asarray([trim, 1.0 - trim], dtype=column.dtype))
    return lower, upper
