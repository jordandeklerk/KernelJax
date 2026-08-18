"""Numerically safe primitives shared by the kernel families."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kerneljax.typing import Array

__all__ = ["safe_div", "safe_pow"]


@jax.custom_jvp
def safe_pow(base: Array, exponent: Array) -> Array:
    r"""Raise ``base`` to ``exponent`` with the one-sided derivative at ``base == 0``.

    Plain exponentiation gives a NaN gradient when the base is zero, since

    .. math::

        \frac{\partial}{\partial \lambda} \lambda^{d} = d \, \lambda^{d - 1}

    is :math:`0 \cdot \infty` at :math:`\lambda = 0` with :math:`d = 0`. The ordered
    categorical kernels hit this, because :math:`\lambda = 0` is a real optimum meaning no
    smoothing.

    A custom JVP applies the power rule verbatim, so derivatives at zero base follow the
    limit from inside the domain rather than the zero that a masked ``jnp.where`` would
    report. The rule reuses ``safe_pow`` for :math:`\lambda^{d - 1}`, which makes every
    higher-order derivative inherit the same convention. The exponent is treated as a
    constant, matching the kernels, which only ever differentiate with respect to the base.

    Parameters
    ----------
    base : Array
        Base values. May be exactly zero.
    exponent : Array
        Exponents, broadcastable against ``base``.

    Returns
    -------
    Array
        ``base ** exponent`` elementwise, with ``safe_pow(0, 0) == 1`` and
        ``safe_pow(0, d) == 0`` for ``d > 0``.
    """
    positive = base > 0.0

    # Swap in a harmless base before calling jnp.power. jnp.where still runs
    # both sides, so passing zero through would give a NaN gradient even though
    # we throw that value away on the next line.
    safe_base = jnp.where(positive, base, jnp.ones_like(base))
    powered = jnp.power(safe_base, exponent)

    masked = jnp.where(positive, powered, jnp.zeros_like(powered))
    return jnp.where(exponent == 0, jnp.ones_like(masked), masked)


safe_pow.defjvps(
    lambda base_dot, ans, base, exponent: exponent * safe_pow(base, exponent - 1.0) * base_dot,
    None,
)


def safe_div(numerator: Array, denominator: Array) -> Array:
    r"""Divide ``numerator`` by ``denominator`` with a finite gradient at ``denominator == 0``.

    Plain division blows up as the denominator approaches zero, since

    .. math::

        \frac{\partial}{\partial y} \frac{x}{y} = -\frac{x}{y^{2}}.

    Kernel weight ratios such as a Nadaraya-Watson denominator can be exactly zero at an
    isolated evaluation point.

    Parameters
    ----------
    numerator : Array
        Numerator values.
    denominator : Array
        Denominator values, broadcastable against ``numerator``. May be
        exactly zero.

    Returns
    -------
    Array
        ``numerator / denominator`` elementwise, equal to ordinary division
        away from zero, and zero wherever ``denominator == 0``.
    """
    nonzero = denominator != 0.0
    # Same trick as safe_pow. Swap in a harmless denominator before dividing,
    # then throw the result away where the real one was zero.
    safe_denominator = jnp.where(nonzero, denominator, jnp.ones_like(denominator))
    return jnp.where(nonzero, numerator / safe_denominator, jnp.zeros_like(numerator))
