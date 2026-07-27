"""The polynomial design basis used by local polynomial regression."""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth, broadcast_h
from kerneljax.data import ColumnSpec, MixedData
from kerneljax.typing import Array

__all__ = ["Basis", "LocalPolyBasis"]


class Basis(Protocol):
    r"""Interface for a design basis shared by local and global smoothers.

    ``design`` is called once per evaluation point for a local method
    such as local polynomial regression, with ``at`` a single row of
    :class:`~kerneljax.data.MixedData` compared against every row of
    ``train``.

    A future series basis instead builds one design over every
    evaluation point at once, so it would call ``design`` a single time
    globally, with ``at`` holding the whole evaluation sample.

    ``dim`` takes only a :class:`~kerneljax.data.ColumnSpec` and a
    degree, both static, so the column count of the design is known
    before any array is built.
    """

    def dim(self, spec: ColumnSpec, degree: int) -> int:
        """Return the number of design columns for a spec and a degree.

        Parameters
        ----------
        spec : ColumnSpec
            Static column metadata of the training sample.
        degree : int
            Degree of the basis.

        Returns
        -------
        int
            The number of design columns.
        """
        ...

    def design(self, train: MixedData, at: MixedData, bw: Bandwidth) -> Float[Array, "n k"]:
        """Return the design matrix of ``train`` relative to the point ``at``.

        Parameters
        ----------
        train : MixedData
            Training sample, supplying the row axis of the design.
        at : MixedData
            The evaluation point, a single row.
        bw : Bandwidth
            Bandwidths used to scale the design.

        Returns
        -------
        Float[Array, "n k"]
            The design matrix, one row per training point.
        """
        ...

    def deriv(self, train: MixedData, at: MixedData, bw: Bandwidth, var: int, order: int) -> Float[Array, "n k"]:
        """Return the derivative of the design matrix with respect to one evaluation coordinate.

        Parameters
        ----------
        train : MixedData
            Training sample, supplying the row axis of the design.
        at : MixedData
            The evaluation point, a single row.
        bw : Bandwidth
            Bandwidths used to scale the design.
        var : int
            Index of the continuous column to differentiate. Static.
        order : int
            Order of the derivative. Static.

        Returns
        -------
        Float[Array, "n k"]
            The derivative of the design matrix, one row per training point.
        """
        ...


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LocalPolyBasis:
    r"""Multivariate monomial basis of the bandwidth-scaled evaluation coordinate.

    For :math:`p` continuous columns and a total degree :math:`d`, the
    basis holds every monomial of :math:`u = (X_i - x) / h` with total
    degree at most :math:`d`, giving

    .. math::

        k = \binom{p + d}{d}

    columns, following the local polynomial convention of [1]_.

    Degree 0 gives the constant column alone, the Nadaraya-Watson
    estimator. Degree 1 adds each :math:`u_j`, giving local linear
    regression. Degree 2 further adds every square and cross product of
    the :math:`u_j`, each appearing exactly once.

    Only continuous columns enter the basis. Categorical columns are
    smoothed entirely through the kernel weights, so ``design`` and
    ``deriv`` ignore ``uno`` and ``orde``.

    Fitted coefficients come back in bandwidth units. For a degree 1
    fit, the constant column recovers the fitted value, while the
    coefficient of :math:`u_j` recovers :math:`h_j` times the
    derivative in :math:`x_j`, not the derivative itself.

    Parameters
    ----------
    degree : int
        Total degree of the polynomial. Supports 0, 1 and 2.

    Examples
    --------
    Evaluate the local linear design at a single point.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...: from kerneljax.basis import LocalPolyBasis
           ...:
           ...: train = kj.MixedData.continuous(jnp.linspace(0.0, 1.0, 5).reshape(-1, 1))
           ...: at = kj.MixedData.continuous(jnp.array([[0.5]]))
           ...: bw = kj.Bandwidth(h=jnp.array([0.2]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
           ...: print(LocalPolyBasis(degree=1).design(train, at, bw))

    References
    ----------
    .. [1] Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and
           Its Applications. Chapman and Hall.
    """

    degree: int

    def dim(self, spec: ColumnSpec, degree: int) -> int:
        r"""Return the binomial coefficient counting monomials up to the given degree.

        Parameters
        ----------
        spec : ColumnSpec
            Static column metadata of the training sample.
        degree : int
            Degree of the basis.

        Returns
        -------
        int
            The number of design columns, :math:`\binom{p + d}{d}` for
            :math:`p` continuous columns and degree :math:`d`.
        """
        return math.comb(spec.p_con + degree, degree)

    def design(self, train: MixedData, at: MixedData, bw: Bandwidth) -> Float[Array, "n k"]:
        r"""Evaluate every monomial of :math:`u` at the training points relative to ``at``.

        Parameters
        ----------
        train : MixedData
            Training sample, supplying the row axis of the design.
        at : MixedData
            The evaluation point, a single row.
        bw : Bandwidth
            Bandwidths scaling the continuous columns.

        Returns
        -------
        Float[Array, "n k"]
            The design matrix, one row per training point and one
            column per monomial of :math:`u = (X_i - x) / h`.
        """
        p_con = train.spec.p_con
        h = broadcast_h(bw, p_con)[0]
        u = (train.con - at.con) / h

        columns = [_monomial_column(u, exponent) for exponent in _monomial_exponents(p_con, self.degree)]
        return jnp.stack(columns, axis=-1)

    def deriv(self, train: MixedData, at: MixedData, bw: Bandwidth, var: int, order: int) -> Float[Array, "n k"]:
        r"""Differentiate every design column with respect to one evaluation coordinate.

        Since :math:`u_j = (X_{ij} - x_j) / h_j` and only :math:`u_j`
        depends on :math:`x_j`, its derivative is
        :math:`\partial u_j / \partial x_j = -1 / h_j`, so the first
        derivative of a monomial is

        .. math::

            \frac{\partial}{\partial x_j} \prod_d u_d^{e_d}
                = -e_j \, u_j^{e_j - 1} \prod_{d \neq j} u_d^{e_d} \, \frac{1}{h_j}

        Differentiating once more with respect to the same coordinate
        picks up another factor of :math:`-1 / h_j` and reduces the
        power of :math:`u_j` by one more,

        .. math::

            \frac{\partial^2}{\partial x_j^2} \prod_d u_d^{e_d}
                = e_j (e_j - 1) \, u_j^{e_j - 2} \prod_{d \neq j} u_d^{e_d} \, \frac{1}{h_j^2}

        Every order of differentiation contributes a factor of
        :math:`-1 / h_j`, so odd orders carry an overall minus sign
        and even orders do not.

        Supports order 1 and order 2.

        Parameters
        ----------
        train : MixedData
            Training sample, supplying the row axis of the design.
        at : MixedData
            The evaluation point, a single row.
        bw : Bandwidth
            Bandwidths scaling the continuous columns.
        var : int
            Index of the continuous column to differentiate, among the
            continuous columns in original column order. Static.
        order : int
            Order of the derivative, 1 or 2. Static.

        Returns
        -------
        Float[Array, "n k"]
            The derivative of the design matrix, one row per training
            point and one column per monomial.
        """
        p_con = train.spec.p_con
        h = broadcast_h(bw, p_con)[0]
        u = (train.con - at.con) / h
        scale = h[..., var] ** order
        sign = (-1) ** order

        columns = []
        for exponent in _monomial_exponents(p_con, self.degree):
            coefficient, reduced = _derivative_exponent(exponent, var, order)
            columns.append(sign * coefficient * _monomial_column(u, reduced) / scale)

        return jnp.stack(columns, axis=-1)


def _monomial_exponents(p_con: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate the exponent tuples of every monomial in u up to the given total degree."""
    exponents = []

    for total in range(degree + 1):
        for combination in itertools.combinations_with_replacement(range(p_con), total):
            exponent = [0] * p_con
            for index in combination:
                exponent[index] += 1
            exponents.append(tuple(exponent))

    return tuple(exponents)


def _monomial_column(u: Array, exponent: tuple[int, ...]) -> Array:
    """Evaluate one monomial of u for a static tuple of per-column exponents."""
    column = jnp.ones(u.shape[:-1], dtype=u.dtype)

    for index, power in enumerate(exponent):
        if power:
            column = column * u[..., index] ** power

    return column


def _derivative_exponent(exponent: tuple[int, ...], var: int, order: int) -> tuple[int, tuple[int, ...]]:
    """Return the falling factorial coefficient and reduced exponent tuple for one derivative."""
    value = exponent[var]
    if value < order:
        return 0, exponent

    coefficient = 1
    for step in range(order):
        coefficient *= value - step

    reduced = tuple(entry - order if index == var else entry for index, entry in enumerate(exponent))
    return coefficient, reduced
