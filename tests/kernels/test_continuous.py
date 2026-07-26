"""Tests for continuous kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.kernels.continuous import Gaussian


@pytest.mark.parametrize(
    ("x", "y", "h"),
    [(1.0, 0.5, 0.7), (-2.0, 3.0, 1.5), (0.0, 0.0, 2.0), (5.0, 5.0, 0.3)],
)
def test_value_matches_standard_normal_density_of_u(gaussian, x, y, h):
    u = (x - y) / h
    expected = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    got = gaussian.value(jnp.array(x), jnp.array(y), jnp.array(h))
    assert float(got) == pytest.approx(expected, rel=1e-6)


def test_value_carries_no_inverse_bandwidth_factor(gaussian):
    at_h1 = gaussian.value(jnp.array(0.3), jnp.array(0.0), jnp.array(1.0))
    at_h2 = gaussian.value(jnp.array(0.6), jnp.array(0.0), jnp.array(2.0))
    assert float(at_h1) == pytest.approx(float(at_h2), rel=1e-12)


def test_conv_matches_numeric_self_convolution(gaussian):
    h = 0.8
    delta = 0.37
    grid = np.linspace(-40.0, 40.0, 400_001)
    step = grid[1] - grid[0]
    kh = np.exp(-0.5 * (grid / h) ** 2) / (h * np.sqrt(2 * np.pi))
    kh_shifted = np.exp(-0.5 * ((delta - grid) / h) ** 2) / (h * np.sqrt(2 * np.pi))
    numeric = float(np.sum(kh * kh_shifted) * step) * h
    got = gaussian.conv(jnp.array(delta), jnp.array(0.0), jnp.array(h))
    assert float(got) == pytest.approx(numeric, rel=1e-6)


def test_conv_at_coincident_points_equals_closed_form_constant(gaussian):
    got = gaussian.conv(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0))
    assert float(got) == pytest.approx(1.0 / (2.0 * np.sqrt(np.pi)), rel=1e-6)


def test_value_integrates_to_one_once_bandwidth_is_supplied(gaussian):
    h = 1.3
    grid = np.linspace(-60.0 * h, 60.0 * h, 400_001)
    step = grid[1] - grid[0]
    density = np.array(gaussian.value(jnp.array(grid), jnp.array(0.0), jnp.array(h))) / h
    total = float(np.sum(density) * step)
    assert total == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(("x", "y", "h"), [(1.0, -2.0, 0.5), (3.3, 3.3, 1.0), (-1.0, 4.0, 2.0)])
def test_value_is_symmetric_in_x_and_y(gaussian, x, y, h):
    forward = gaussian.value(jnp.array(x), jnp.array(y), jnp.array(h))
    backward = gaussian.value(jnp.array(y), jnp.array(x), jnp.array(h))
    assert float(forward) == pytest.approx(float(backward), rel=1e-12)


def test_value_is_strictly_decreasing_in_absolute_difference(gaussian):
    diffs = jnp.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0])
    values = gaussian.value(diffs, jnp.array(0.0), jnp.array(1.0))
    assert jnp.all(jnp.diff(values) < 0.0)


@pytest.mark.parametrize("op", ["value", "conv"])
def test_is_jittable_grad_and_vmap_compatible(gaussian, op):
    fn = getattr(gaussian, op)
    f = jax.jit(lambda h: fn(jnp.array(0.3), jnp.array(0.0), h))
    assert jnp.isfinite(f(jnp.array(0.5)))
    assert jnp.isfinite(jax.grad(f)(jnp.array(0.5)))
    assert jax.vmap(f)(jnp.array([0.3, 0.5, 0.9])).shape == (3,)


def test_default_gaussian_equals_and_hashes_equal_to_another():
    assert Gaussian() == Gaussian()
    assert hash(Gaussian()) == hash(Gaussian())


@pytest.mark.parametrize("order", [0, 1, 3, 4, 6, 8])
def test_order_other_than_two_is_rejected(order):
    with pytest.raises(ValueError, match="order"):
        Gaussian(order=order)
