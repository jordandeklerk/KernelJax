"""Kernel abstract base classes and the operator namespace."""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp

from kerneljax.typing import FloatArray, IntArray

__all__ = ["ContinuousKernel", "Op", "OrderedKernel", "UnorderedKernel"]


class _Leaf:
    """Make every subclass a frozen, value-hashed, registered static object that checks its invariants."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True)(cls)
        for name in ("__setattr__", "__delattr__"):
            if name in cls.__dict__:
                delattr(cls, name)
        generated = cls.__init__

        @functools.wraps(generated)
        def __init__(self: object, *args: object, **kwargs: object) -> None:
            generated(self, *args, **kwargs)
            for klass in type(self).__mro__:
                check = klass.__dict__.get("__check_init__")
                if check is not None:
                    check(self)

        cls.__init__ = __init__
        with contextlib.suppress(ValueError):
            jax.tree_util.register_static(cls)

    def __setattr__(self, name: str, value: object) -> None:
        raise dataclasses.FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise dataclasses.FrozenInstanceError(f"cannot delete field {name!r}")


class Op:
    """Operator names resolved on a kernel instance by ``getattr``."""

    VALUE = "value"
    DERIV = "deriv"
    CDF = "cdf"
    CONV = "conv"


class ContinuousKernel(_Leaf, abc.ABC):
    r"""Base class for continuous kernels.

    Write the kernel as a function of the scaled difference in :meth:`k` and
    the other operators follow from it,

    .. math::

        u = \frac{x - y}{h}, \qquad
        \mathrm{value} = k(u), \qquad
        \mathrm{deriv} = \frac{k'(u)}{h}, \qquad
        \mathrm{cdf} = \int_{-\infty}^{u} k(t)\,dt, \qquad
        \mathrm{conv} = (k * k)(u).

    Only ``deriv`` carries the :math:`1/h` factor, since the estimator divides
    by the bandwidth product once. ``cdf`` and ``conv`` are tabulated on
    :math:`|u| \le 256`, so a heavy-tailed kernel should implement ``cdf``
    itself. Any operator can be overridden with a closed form, and annotated
    class attributes are configuration that takes part in equality.
    """

    def k(self, u: FloatArray) -> FloatArray:
        """Evaluate the kernel at the scaled difference ``u``.

        Parameters
        ----------
        u : FloatArray
            Scaled differences ``(x - y) / h``.

        Returns
        -------
        FloatArray
            The kernel value in ``u`` units, with unit mass and no ``1/h`` factor.
        """
        raise NotImplementedError(f"{type(self).__name__} implements neither k nor value")

    def __check_init__(self) -> None:
        """Refuse a kernel that implements neither ``k`` nor ``value``."""
        cls = type(self)
        if cls.k is ContinuousKernel.k and cls.value is ContinuousKernel.value:
            raise TypeError(f"{cls.__name__} implements neither k nor value; write k(u) or value(x, y, h)")

    def value(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        """Evaluate the kernel at ``x`` against data point ``y`` with bandwidth ``h``.

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
            The kernel value, with no ``1/h`` factor applied.
        """
        return self.k((x - y) / h)

    def deriv(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        r"""Differentiate the kernel with respect to ``x``.

        Unlike ``value``, ``cdf`` and ``conv``, the result carries a :math:`1/h`
        factor from the chain rule.

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
            :math:`\partial\, \mathrm{value}(x, y, h) / \partial x`.
        """
        dtype = jnp.result_type(x, y, h)
        x = jnp.asarray(x, dtype)
        y = jnp.asarray(y, dtype)
        h = jnp.asarray(h, dtype)
        return jax.jvp(lambda z: self.value(z, y, h), (x,), (jnp.ones_like(x),))[1]

    def cdf(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        """Integrate the kernel from below up to ``x``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement cdf")

    def conv(self, x: FloatArray, y: FloatArray, h: FloatArray) -> FloatArray:
        """Evaluate the self-convolution of the kernel at ``x`` and ``y``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement conv")


class UnorderedKernel(_Leaf, abc.ABC):
    """Base class for unordered categorical kernels.

    A concrete kernel implements :meth:`value` on the integer codes
    ``(x, y)`` given the smoothing parameter ``lam`` and the level count
    ``levels``.
    """

    @abc.abstractmethod
    def value(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        """Evaluate the kernel at code ``x`` against data code ``y``.

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

    def cdf(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        """Sum the kernel cumulatively over the levels at or below ``x``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement cdf")

    def conv(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        """Convolve the kernel with itself by summing the product over the support."""
        raise NotImplementedError(f"{type(self).__name__} does not implement conv")

    @abc.abstractmethod
    def upper_bound(self, levels: int) -> float:
        """Largest admissible ``lam`` for this kernel at the given level count.

        Parameters
        ----------
        levels : int
            Number of levels of the column.

        Returns
        -------
        float
            The upper bound on ``lam``.
        """


class OrderedKernel(_Leaf, abc.ABC):
    """Base class for ordered categorical kernels.

    A concrete kernel implements :meth:`value` on the integer levels
    ``(x, y)`` given the smoothing parameter ``lam`` and the level count
    ``levels``, where the levels carry a natural order.
    """

    @abc.abstractmethod
    def value(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        """Evaluate the kernel at level ``x`` against data level ``y``.

        Parameters
        ----------
        x : IntArray
            Evaluation levels.
        y : IntArray
            Data levels, broadcastable against ``x``.
        lam : FloatArray
            Smoothing parameter, in ``[0, upper_bound(levels)]``.
        levels : int
            Number of levels of the column, static under ``jit``.

        Returns
        -------
        FloatArray
            The kernel value.
        """

    def cdf(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        """Sum the kernel over the integer lattice at or below ``x``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement cdf")

    def conv(self, x: IntArray, y: IntArray, lam: FloatArray, levels: int) -> FloatArray:
        """Convolve the kernel with itself over the integer lattice."""
        raise NotImplementedError(f"{type(self).__name__} does not implement conv")

    @abc.abstractmethod
    def upper_bound(self, levels: int) -> float:
        """Largest admissible ``lam`` for this kernel at the given level count.

        Parameters
        ----------
        levels : int
            Number of levels of the column.

        Returns
        -------
        float
            The upper bound on ``lam``.
        """
