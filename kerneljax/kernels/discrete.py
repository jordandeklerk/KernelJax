"""Discrete kernel families for unordered and ordered categorical data."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from kerneljax.kernels._numerics import safe_pow
from kerneljax.kernels.base import OrderedKernel, UnorderedKernel
from kerneljax.typing import FloatArray, IntArray

__all__ = ["AitchisonAitken", "LiRacine", "WangVanRyzin"]


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class AitchisonAitken(UnorderedKernel):
    """Aitchison and Aitken unordered categorical kernel."""

    def value(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Evaluate the kernel.

        With :math:`c` the level count,

        .. math::

            L(x, y) = \begin{cases} 1 - \lambda & x = y \\ \lambda / (c - 1) & x \neq y \end{cases}.

        Parameters
        ----------
        x : IntArray
            Evaluation codes.
        y : IntArray
            Data codes, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, upper_bound(levels)]``.
        levels : int
            Number of levels of the column, static under ``jit``.

        Returns
        -------
        FloatArray
            The kernel value.
        """
        return jnp.where(x == y, 1.0 - lam, lam / (levels - 1.0))

    def conv(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Convolve the kernel with itself by summing the product over the support.

        Writing :math:`c` for the level count, the closed form is

        .. math::

            \sum_{s} L(x, s) L(y, s) = \begin{cases}
            (1 - \lambda)^2 + \dfrac{\lambda^2}{c - 1} & x = y \\
            \dfrac{2 (1 - \lambda) \lambda}{c - 1} + \dfrac{(c - 2) \lambda^2}{(c - 1)^2} & x \neq y
            \end{cases}.

        Parameters
        ----------
        x : IntArray
            Evaluation codes.
        y : IntArray
            Data codes, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, upper_bound(levels)]``.
        levels : int
            Number of levels of the column, static under ``jit``.

        Returns
        -------
        FloatArray
            The self-convolution of the kernel.
        """
        c = float(levels)
        off = lam / (c - 1.0)
        equal = (1.0 - lam) ** 2 + lam * lam / (c - 1.0)
        unequal = 2.0 * (1.0 - lam) * off + (c - 2.0) * off * off
        return jnp.where(x == y, equal, unequal)

    def upper_bound(self, levels: int) -> float:
        r""":math:`(c - 1) / c` as the largest admissible :math:`\lambda`.

        Parameters
        ----------
        levels : int
            Number of levels of the column.

        Returns
        -------
        float
            The upper bound on ``lam``.
        """
        return (levels - 1.0) / levels


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class WangVanRyzin(OrderedKernel):
    """Wang and van Ryzin ordered categorical kernel."""

    def value(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Evaluate the kernel.

        With :math:`d = |x - y|`,

        .. math::

            L(x, y) = \begin{cases} 1 - \lambda & d = 0 \\ \dfrac{1 - \lambda}{2} \lambda^{d} & d > 0 \end{cases}.

        Parameters
        ----------
        x : IntArray
            Evaluation levels.
        y : IntArray
            Data levels, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, upper_bound(levels))``.
        levels : int
            Number of levels of the column, static under ``jit``.

        Returns
        -------
        FloatArray
            The kernel value.
        """
        d = jnp.abs(x - y)
        return jnp.where(d == 0, 1.0 - lam, 0.5 * (1.0 - lam) * safe_pow(lam, d))

    def cdf(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Sum the kernel over the integer lattice at or below ``x``.

        With :math:`a = (1 - \lambda) / 2`, accumulating over every integer at
        or below :math:`x` gives

        .. math::

            \sum_{s \le x} \ell(s, y) =
            \begin{cases}
                \lambda^{\,y - x} / 2, & x < y \\
                1 - \lambda^{\,x - y + 1} / 2, & x \ge y
            \end{cases}

        The sum runs over all integers, not just the observed levels, so the
        weight reaches one above the support rather than stopping short of it.

        Parameters
        ----------
        x : IntArray
            Evaluation levels.
        y : IntArray
            Data levels, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, 1]``.
        levels : int
            Unused, accepted for interface uniformity with the other kernels.

        Returns
        -------
        FloatArray
            The cumulative kernel weight at or below ``x``.
        """
        del levels
        distance = jnp.abs(x - y)
        below = safe_pow(lam, distance) / 2.0
        above = 1.0 - safe_pow(lam, distance + 1) / 2.0
        return jnp.where(x < y, below, above)

    def conv(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Convolve the kernel with itself over the entire integer lattice.

        Writing :math:`\ell(x, s) = a \lambda^{|x - s|} + a [s = x]` with
        :math:`a = (1 - \lambda) / 2` and :math:`d = |x - y|`, summing over every
        integer :math:`s` gives

        .. math::

            \sum_{s \in \mathbb{Z}} \ell(x, s) \ell(y, s)
            = a^2 \left( \lambda^{d} \left(d + 3 + \dfrac{2 \lambda^2}{1 - \lambda^2}\right)
            + [d = 0] \right).

        The sum runs over all integers, not just the observed levels.

        Parameters
        ----------
        x : IntArray
            Evaluation levels.
        y : IntArray
            Data levels, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, 1]``.
        levels : int
            Unused, accepted for interface uniformity with the other kernels.

        Returns
        -------
        FloatArray
            The self-convolution of the kernel.
        """
        del levels
        d = jnp.abs(x - y)
        a = 0.5 * (1.0 - lam)
        # One (1 - lam) of the prefactor cancels the pole in 2 lam^2 / (1 - lam^2),
        # so the bound lam = 1 evaluates and differentiates finitely.
        cross = 0.5 * lam * lam * (1.0 - lam) / (1.0 + lam)
        return a * a * (safe_pow(lam, d) * (d + 3.0) + jnp.where(d == 0, 1.0, 0.0)) + safe_pow(lam, d) * cross

    def upper_bound(self, levels: int) -> float:
        r"""``1`` as the largest admissible :math:`\lambda`.

        Parameters
        ----------
        levels : int
            Number of levels of the column, unused since the bound does not depend on it.

        Returns
        -------
        float
            The upper bound on ``lam``.
        """
        del levels
        return 1.0


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LiRacine(OrderedKernel):
    """Li and Racine ordered categorical kernel."""

    def value(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Evaluate the kernel.

        With :math:`d = |x - y|`,

        .. math::

            L(x, y) = \frac{1 - \lambda}{1 + \lambda} \lambda^{d}.

        The leading constant makes the kernel sum to one over the integer lattice.
        It is common to every pair in a column, so it cancels from a regression fit
        and matters only where the weights are used unnormalized, as in a density.

        Parameters
        ----------
        x : IntArray
            Evaluation levels.
        y : IntArray
            Data levels, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, 1)``.
        levels : int
            Number of levels of the column, static under ``jit``.

        Returns
        -------
        FloatArray
            The kernel value.
        """
        del levels
        return (1.0 - lam) / (1.0 + lam) * safe_pow(lam, jnp.abs(x - y))

    def cdf(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Sum the kernel over the integer lattice at or below ``x``.

        Accumulating over every integer at or below :math:`x` gives

        .. math::

            \sum_{s \le x} L(s, y) =
            \begin{cases}
                \dfrac{\lambda^{\,y - x}}{1 + \lambda}, & x < y \\[2mm]
                \dfrac{1 + \lambda - \lambda^{\,x - y + 1}}{1 + \lambda}, & x \ge y
            \end{cases}

        The sum runs over all integers, not just the observed levels, so the
        weight reaches one above the support rather than stopping short of it.

        Parameters
        ----------
        x : IntArray
            Evaluation levels.
        y : IntArray
            Data levels, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, 1]``.
        levels : int
            Unused, accepted for interface uniformity with the other kernels.

        Returns
        -------
        FloatArray
            The cumulative kernel weight at or below ``x``.
        """
        del levels
        distance = jnp.abs(x - y)
        below = safe_pow(lam, distance) / (1.0 + lam)
        above = (1.0 + lam - safe_pow(lam, distance + 1)) / (1.0 + lam)
        return jnp.where(x < y, below, above)

    def conv(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        r"""Convolve the kernel with itself over the entire integer lattice.

        Writing :math:`c = (1 - \lambda) / (1 + \lambda)` and :math:`d = |x - y|`,
        summing over every integer :math:`s` gives

        .. math::

            \sum_{s \in \mathbb{Z}} L(x, s) L(y, s)
            = c^{2} \lambda^{d} \left(d + 1 + \frac{2 \lambda^{2}}{1 - \lambda^{2}}\right).

        The sum runs over all integers, not just the observed levels.

        Parameters
        ----------
        x : IntArray
            Evaluation levels.
        y : IntArray
            Data levels, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, 1]``.
        levels : int
            Unused, accepted for interface uniformity with the other kernels.

        Returns
        -------
        FloatArray
            The self-convolution of the kernel.
        """
        del levels
        d = jnp.abs(x - y)
        c = (1.0 - lam) / (1.0 + lam)
        # One (1 - lam) of the squared prefactor cancels the pole in the tail,
        # so the bound lam = 1 evaluates and differentiates finitely.
        cross = 2.0 * lam * lam * (1.0 - lam) / ((1.0 + lam) ** 3)
        return safe_pow(lam, d) * (c * c * (d + 1.0) + cross)

    def upper_bound(self, levels: int) -> float:
        r"""``1`` as the largest admissible :math:`\lambda`.

        Parameters
        ----------
        levels : int
            Number of levels of the column, unused since the bound does not depend on it.

        Returns
        -------
        float
            The upper bound.
        """
        del levels
        return 1.0
