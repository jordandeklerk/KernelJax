"""Numeric verification of the continuous kernel."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from kerneljax.kernels.base import ContinuousKernel
from kerneljax.typing import Array

_passed: set[tuple[str, object]] = set()


def _u_grid(span: float = 256.0, step: float = 1.0 / 128.0) -> np.ndarray:
    """Build the probe grid in u units, wide enough that a Cauchy tail clears every tolerance."""
    return np.linspace(-span, span, round(2.0 * span / step) + 1)


def _conv_offsets(reach: float = 8.0, spacing: float = 0.25) -> np.ndarray:
    """Build the offsets conv is compared at, dense enough to sit near any truncation edge."""
    return np.arange(-reach, reach + spacing, spacing)


def _mass(method: Callable[..., Array], h: float) -> float:
    """Integrate a kernel operator in u units, through the substitution the estimator performs."""
    grid = _u_grid()
    values = method(jnp.asarray(0.0), jnp.asarray(-h * grid), jnp.asarray(h))
    return float(np.trapezoid(np.asarray(values, dtype=np.float64), grid))


def _in_u_units(method: Callable[..., Array], offsets: np.ndarray, h: float) -> np.ndarray:
    """Evaluate a kernel operator at the given u offsets."""
    values = method(jnp.asarray(0.0), jnp.asarray(-h * offsets), jnp.asarray(h))
    return np.asarray(values, dtype=np.float64)


def _numeric_conv(kernel: ContinuousKernel, offsets: np.ndarray) -> np.ndarray:
    """Compute the self-convolution of value from value itself."""
    grid = _u_grid()
    base = _in_u_units(kernel.value, grid, 1.0)
    shifted = kernel.value(jnp.asarray(0.0), jnp.asarray(-(offsets[:, None] - grid[None, :])), jnp.asarray(1.0))
    return np.asarray(np.trapezoid(base[None, :] * np.asarray(shifted, dtype=np.float64), grid, axis=1))


def _check_value_mass(kernel: ContinuousKernel, *, tol: float = 1.0 / 48.0) -> None:
    """Refuse a value that does not integrate to one in u units."""
    key = ("value_mass", kernel)
    if key in _passed:
        return

    name = type(kernel).__name__
    with jax.ensure_compile_time_eval():
        at_one = _mass(kernel.value, 1.0)
        at_two = _mass(kernel.value, 2.0)

    if abs(at_two - 1.0) > tol:
        if abs(at_one - 1.0) <= tol:
            raise ValueError(
                f"{name}.value integrates to one at h=1 but to {at_two:.4f} at h=2, the signature "
                "of a kernel carrying its own 1/h factor. Return the kernel in u units with no "
                "normalization by h, since the estimator divides by h exactly once itself."
            )
        raise ValueError(
            f"{name}.value integrates to {at_two:.4f} in u units rather than one, so every "
            "density it produces is scaled by that factor. A regression fit is a ratio and "
            "cancels the constant, which is why this only fires from a density."
        )

    _passed.add(key)


def _check_conv_at_zero(kernel: ContinuousKernel, *, tol: float = 1.0 / 48.0) -> None:
    """Refuse a conv whose value at zero is not R(k), which is all a standard error consumes."""
    key = ("conv_at_zero", kernel)
    if key in _passed:
        return

    grid = _u_grid()
    with jax.ensure_compile_time_eval():
        values = _in_u_units(kernel.value, grid, 1.0)
        roughness = float(np.trapezoid(values**2, grid))
        at_zero = float(kernel.conv(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0)))

    if abs(at_zero - roughness) > tol:
        name = type(kernel).__name__
        raise ValueError(
            f"{name}.conv(0) is {at_zero:.4f} but the self-convolution of {name}.value at zero "
            f"is R(k) = {roughness:.4f}. conv must be the self-convolution of value, in u units."
        )

    _passed.add(key)


def _check_conv_matches(kernel: ContinuousKernel, *, tol: float = 1e-3) -> None:
    """Refuse a conv that is not the self-convolution of value, compared pointwise."""
    key = ("conv_matches", kernel)
    if key in _passed:
        return

    name = type(kernel).__name__
    offsets = _conv_offsets()
    with jax.ensure_compile_time_eval():
        numeric = _numeric_conv(kernel, offsets)
        at_one = _in_u_units(kernel.conv, offsets, 1.0)
        at_two = _in_u_units(kernel.conv, offsets, 2.0)

    if np.max(np.abs(at_one - numeric)) <= tol < np.max(np.abs(at_two - numeric)):
        raise ValueError(
            f"{name}.conv matches the self-convolution of value at h=1 but not at h=2, the "
            "signature of a 1/h factor. conv is written in u units like value, with no h "
            "normalization."
        )

    truncated = (np.abs(at_one) <= tol) & (numeric > 10.0 * tol)
    if bool(np.any(truncated)):
        edge = float(offsets[np.argmax(truncated)])
        raise ValueError(
            f"{name}.conv is zero at u = {edge:g} where the self-convolution of {name}.value is "
            f"not. A self-convolution doubles the support, so for a kernel on |u| <= 1 conv "
            "reaches |u| <= 2, and truncating it to the kernel's own support is the usual cause."
        )

    worst = int(np.argmax(np.abs(at_one - numeric)))
    if abs(at_one[worst] - numeric[worst]) > tol:
        raise ValueError(
            f"{name}.conv is {at_one[worst]:.4f} at u = {offsets[worst]:g} where the "
            f"self-convolution of {name}.value is {numeric[worst]:.4f}. conv must be the "
            "self-convolution of value, in u units."
        )

    _passed.add(key)


def _check_cdf_limits(kernel: ContinuousKernel, *, tol: float = 1.0 / 200.0) -> None:
    """Refuse a cdf that does not run from zero to one at every bandwidth."""
    key = ("cdf_limits", kernel)
    if key in _passed:
        return

    edge = float(_u_grid()[-1])
    for h in (1.0, 2.0):
        with jax.ensure_compile_time_eval():
            lower = float(kernel.cdf(jnp.asarray(-edge * h), jnp.asarray(0.0), jnp.asarray(h)))
            upper = float(kernel.cdf(jnp.asarray(edge * h), jnp.asarray(0.0), jnp.asarray(h)))
        if abs(lower) > tol or abs(upper - 1.0) > tol:
            name = type(kernel).__name__
            raise ValueError(
                f"{name}.cdf runs from {lower:.4f} to {upper:.4f} at h={h:g} rather than from "
                "zero to one. cdf is the kernel integrated from below, in u units with no 1/h "
                "factor, so it is a distribution function at every bandwidth."
            )

    _passed.add(key)


def _check_grad_diagonal(kernel: ContinuousKernel, op: str, *, offsets: tuple[float, ...] = (0.0, 0.73, 2.31)) -> None:
    """Refuse an operator whose bandwidth gradient is not finite at separations a fit contains."""
    key = (f"grad_{op}", kernel)
    if key in _passed:
        return

    method = getattr(kernel, op)
    for offset in offsets:
        with jax.ensure_compile_time_eval():
            point = jnp.asarray(offset)
            gradient = float(jax.grad(lambda h, x=point: jnp.sum(method(x, jnp.asarray(0.0), h)))(jnp.asarray(1.0)))
        if not jnp.isfinite(gradient):
            name = type(kernel).__name__
            raise ValueError(
                f"{name}.{op} has a non-finite bandwidth gradient at |x - y| = {offset:g}, a "
                "separation any sample can contain. jnp.where differentiates both branches, so "
                "guard the argument inside the untaken branch, not just the branch."
            )

    _passed.add(key)
