"""Tables that derive a continuous cdf and self-convolution from the kernel alone."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from kerneljax.kernels._checks import _u_grid
from kerneljax.typing import Array

_tables: dict[object, Tables] = {}


@dataclasses.dataclass(frozen=True)
class Tables:
    """Sampled cdf and self-convolution of one kernel, in u units."""

    grid: np.ndarray
    cdf: np.ndarray
    offsets: np.ndarray
    conv: np.ndarray


def tables(kernel: object) -> Tables:
    """Build or fetch the tables for a hashable kernel instance."""
    if kernel in _tables:
        return _tables[kernel]

    grid = _u_grid()
    step = float(grid[1] - grid[0])

    with jax.ensure_compile_time_eval():
        sample = np.asarray(kernel.value(jnp.asarray(0.0), jnp.asarray(-grid), jnp.asarray(1.0)), dtype=np.float64)

    grid, sample = _trim(grid, sample)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (sample[1:] + sample[:-1]) * step)])

    length = 2 * sample.size - 1
    convolved = np.fft.irfft(np.fft.rfft(sample, length) ** 2, length) * step

    if sample[0] == 0.0:
        convolved[0] = 0.0
    if sample[-1] == 0.0:
        convolved[-1] = 0.0
    offsets = 2.0 * grid[0] + step * np.arange(length)

    built = Tables(grid=grid, cdf=cumulative, offsets=offsets, conv=convolved)
    _tables[kernel] = built
    return built


def lookup(kernel: object, x: Array, y: Array, h: Array, op: str) -> Array:
    """Interpolate a derived operator at the scaled difference."""
    built = tables(kernel)
    u = (jnp.asarray(x) - jnp.asarray(y)) / jnp.asarray(h)
    if op == "cdf":
        return jnp.interp(u, jnp.asarray(built.grid, dtype=u.dtype), jnp.asarray(built.cdf, dtype=u.dtype))
    return jnp.interp(u, jnp.asarray(built.offsets, dtype=u.dtype), jnp.asarray(built.conv, dtype=u.dtype))


def _trim(grid: np.ndarray, sample: np.ndarray, *, guard: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Cut the sample to its nonzero span plus a guard cell on each side."""
    nonzero = np.flatnonzero(sample != 0.0)
    if nonzero.size == 0:
        return grid, sample
    lo = max(int(nonzero[0]) - guard, 0)
    hi = min(int(nonzero[-1]) + guard, sample.size - 1)
    return grid[lo : hi + 1], sample[lo : hi + 1]
