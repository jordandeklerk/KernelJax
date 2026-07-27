"""Tests for numerically safe primitives."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.kernels._numerics import safe_div, safe_pow


def test_safe_pow_matches_power_away_from_zero():
    base = jnp.array([0.5, 0.9])
    exponent = jnp.array([2, 3])
    assert jnp.allclose(safe_pow(base, exponent), jnp.power(base, exponent))


@pytest.mark.parametrize(("exponent", "expected"), [(0, 1.0), (3, 0.0), (0.0, 1.0), (3.0, 0.0)])
def test_safe_pow_zero_base(exponent, expected):
    assert safe_pow(jnp.array(0.0), jnp.array(exponent)) == expected


@pytest.mark.parametrize(
    ("exponent", "dtype"),
    [
        (0, jnp.int32),
        (1, jnp.int32),
        (2, jnp.int32),
        (0.0, jnp.float32),
        (1.0, jnp.float32),
        (2.0, jnp.float32),
    ],
)
def test_safe_pow_gradient_is_finite_at_zero_base(exponent, dtype):
    g = jax.grad(lambda base: safe_pow(base, jnp.array(exponent, dtype=dtype)))(jnp.array(0.0))
    assert jnp.isfinite(g)


def test_naive_power_nans_but_safe_pow_is_finite():
    naive_grad = jax.grad(lambda base: jnp.power(base, jnp.array(0.0)))(jnp.array(0.0))
    safe_grad = jax.grad(lambda base: safe_pow(base, jnp.array(0.0)))(jnp.array(0.0))
    assert not jnp.isfinite(naive_grad)
    assert jnp.isfinite(safe_grad)


def test_safe_div_gradient_finite_at_zero_denom():
    g = jax.grad(lambda denominator: safe_div(jnp.array(1.0), denominator))(jnp.array(0.0))
    assert jnp.isfinite(g)
