"""Continuous kernel families."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kerneljax.kernels.base import ContinuousKernel
from kerneljax.typing import FloatArray

__all__ = ["Gaussian"]


class Gaussian(ContinuousKernel):
    """Second order Gaussian kernel with no :math:`1/h` factor.

    Parameters
    ----------
    order : int
        Kernel order. Only order 2 is supported here, orders 4, 6 and 8
        arrive later through a Hermite construction and are signed.
    """

    order: int = 2

    def __post_init__(self) -> None:
        """Reject any order other than 2."""
        # Gaussian is not a pytree, so this runs only on Python construction
        # and never again during tree_unflatten inside jit.
        if self.order != 2:
            raise ValueError(f"order must be 2, got order={self.order}")

    def value(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        r"""Evaluate the standard normal density of the scaled difference.

        With :math:`u = (x - y) / h`, this returns

        .. math::

            \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{u^2}{2}\right).

        Parameters
        ----------
        x : FloatArray
            Evaluation points.
        y : FloatArray
            Data points, broadcastable against ``x``.
        h : FloatArray
            Bandwidth, broadcastable against ``x`` and ``y``.

        Returns
        -------
        FloatArray
            The kernel value, with no :math:`1/h` factor applied.
        """
        u = (x - y) / h
        return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)

    def deriv(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        r"""Differentiate the kernel with respect to ``x``.

        With :math:`u = (x - y) / h`, this returns

        .. math::

            \frac{\partial}{\partial x} K\!\left(\frac{x - y}{h}\right)
            = -\frac{u}{h}\, \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{u^2}{2}\right).

        Unlike ``value`` and ``conv``, this carries a :math:`1/h` factor from the chain
        rule, which is exactly what a derivative in ``x`` requires.

        Parameters
        ----------
        x : FloatArray
            Evaluation points.
        y : FloatArray
            Data points, broadcastable against ``x``.
        h : FloatArray
            Bandwidth, broadcastable against ``x`` and ``y``.

        Returns
        -------
        FloatArray
            The derivative of the kernel value with respect to ``x``.
        """
        u = (x - y) / h
        return -u / h * jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)

    def cdf(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        r"""Integrate the kernel from below up to ``x``.

        With :math:`u = (x - y) / h`, this returns the standard normal
        distribution function

        .. math::

            \int_{-\infty}^{u} K(t)\,dt = \Phi(u).

        Parameters
        ----------
        x : FloatArray
            Evaluation points.
        y : FloatArray
            Data points, broadcastable against ``x``.
        h : FloatArray
            Bandwidth, broadcastable against ``x`` and ``y``.

        Returns
        -------
        FloatArray
            The integrated kernel, lying in the unit interval.
        """
        return jax.scipy.special.ndtr((x - y) / h)

    def conv(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        r"""Evaluate the unnormalized self-convolution of the kernel.

        With :math:`u = (x - y) / h`, this returns

        .. math::

            \frac{1}{2 \sqrt{\pi}} \exp\left(-\frac{u^2}{4}\right).
        """
        u = (x - y) / h
        return jnp.exp(-0.25 * u * u) / (2.0 * jnp.sqrt(jnp.pi))
