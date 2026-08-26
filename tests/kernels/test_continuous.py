"""Tests for continuous kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.kernels.base import ContinuousKernel as kj_ContinuousKernel
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


def test_conv_at_coincident_points_has_closed_form(gaussian):
    got = gaussian.conv(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0))
    assert float(got) == pytest.approx(1.0 / (2.0 * np.sqrt(np.pi)), rel=1e-6)


def test_value_integrates_to_one(gaussian):
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


def test_value_decreases_in_abs_difference(gaussian):
    diffs = jnp.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0])
    values = gaussian.value(diffs, jnp.array(0.0), jnp.array(1.0))
    assert jnp.all(jnp.diff(values) < 0.0)


@pytest.mark.parametrize("op", ["value", "conv", "deriv"])
def test_is_jittable_grad_and_vmap_compatible(gaussian, op):
    fn = getattr(gaussian, op)
    f = jax.jit(lambda h: fn(jnp.array(0.3), jnp.array(0.0), h))
    assert jnp.isfinite(f(jnp.array(0.5)))
    assert jnp.isfinite(jax.grad(f)(jnp.array(0.5)))
    assert jax.vmap(f)(jnp.array([0.3, 0.5, 0.9])).shape == (3,)


@pytest.mark.parametrize(
    ("x", "y", "h"),
    [(1.0, 0.5, 0.7), (-2.0, 3.0, 1.5), (0.0, 0.0, 2.0), (5.0, 5.2, 0.3), (-0.4, -1.9, 1.1)],
)
def test_deriv_matches_grad_of_value(gaussian, x, y, h):
    grad_of_value = jax.grad(gaussian.value, argnums=0)
    expected = grad_of_value(jnp.array(x), jnp.array(y), jnp.array(h))
    got = gaussian.deriv(jnp.array(x), jnp.array(y), jnp.array(h))
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.parametrize("point", [0.0, 1.7, -3.2])
@pytest.mark.parametrize("h", [0.3, 1.0, 2.5])
def test_deriv_is_zero_at_coincident_points(gaussian, point, h):
    got = gaussian.deriv(jnp.array(point), jnp.array(point), jnp.array(h))
    assert float(got) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("x", "y", "h"), [(1.0, -2.0, 0.5), (3.3, 0.1, 1.0), (-1.0, 4.0, 2.0)])
def test_deriv_is_antisymmetric_in_x_and_y(gaussian, x, y, h):
    forward = gaussian.deriv(jnp.array(x), jnp.array(y), jnp.array(h))
    backward = gaussian.deriv(jnp.array(y), jnp.array(x), jnp.array(h))
    assert float(forward) == pytest.approx(-float(backward), rel=1e-12)


@pytest.mark.parametrize("u", [0.5, 1.3, -0.8])
@pytest.mark.parametrize(("h_first", "h_second"), [(0.5, 1.0), (0.3, 2.0), (1.0, 3.0)])
def test_deriv_scales_as_inverse_bandwidth_at_fixed_u(gaussian, u, h_first, h_second):
    y = 0.0
    x_first = y + u * h_first
    x_second = y + u * h_second
    deriv_first = gaussian.deriv(jnp.array(x_first), jnp.array(y), jnp.array(h_first))
    deriv_second = gaussian.deriv(jnp.array(x_second), jnp.array(y), jnp.array(h_second))
    assert float(deriv_first) * h_first == pytest.approx(float(deriv_second) * h_second, rel=1e-6)


def test_default_instances_compare_and_hash_equal():
    assert Gaussian() == Gaussian()
    assert hash(Gaussian()) == hash(Gaussian())


@pytest.mark.parametrize("order", [0, 1, 3, 4, 6, 8])
def test_order_other_than_two_is_rejected(order):
    with pytest.raises(ValueError, match="order"):
        Gaussian(order=order)


@pytest.mark.parametrize("h", [0.3, 1.0, 2.5])
def test_cdf_matches_the_normal_integral(h):
    x = jnp.linspace(-4.0, 4.0, 17)
    got = Gaussian().cdf(x, jnp.asarray(0.0), jnp.asarray(h))
    want = jax.scipy.stats.norm.cdf(np.asarray(x) / h)
    assert jnp.allclose(got, want, rtol=1e-6)


def test_cdf_is_monotone_and_spans_the_unit():
    x = jnp.linspace(-40.0, 40.0, 201)
    got = Gaussian().cdf(x, jnp.asarray(0.0), jnp.asarray(1.0))
    assert jnp.all(jnp.diff(got) >= 0.0)
    assert float(got[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(got[-1]) == pytest.approx(1.0, abs=1e-12)


def test_cdf_integrates_the_value_kernel():
    kernel = Gaussian()
    grid = jnp.linspace(-12.0, 1.5, 4001)
    step = float(grid[1] - grid[0])
    midpoints = 0.5 * (grid[1:] + grid[:-1])
    want = float(jnp.sum(kernel.value(midpoints, jnp.asarray(0.0), jnp.asarray(1.0))) * step)
    got = float(kernel.cdf(jnp.asarray(1.5), jnp.asarray(0.0), jnp.asarray(1.0)))
    assert got == pytest.approx(want, rel=1e-4)


class Epanechnikov(kj_ContinuousKernel):
    def k(self, u):
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)


class GaussianLeaf(kj_ContinuousKernel):
    def k(self, u):
        return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)


@pytest.mark.parametrize("u", [-0.9, -0.3, 0.0, 0.4, 0.99, 1.5])
def test_value_is_the_leaf_at_the_scaled_difference(u):
    got = Epanechnikov().value(jnp.array(0.3 + 0.2 * u), jnp.array(0.3), jnp.array(0.2))
    want = 0.75 * (1.0 - u * u) if abs(u) <= 1.0 else 0.0
    assert float(got) == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("u", [-0.9, -0.3, 0.4, 0.99])
def test_derived_deriv_matches_the_analytic_slope(u):
    h = 0.2
    got = Epanechnikov().deriv(jnp.array(0.3 + h * u), jnp.array(0.3), jnp.array(h))
    assert float(got) == pytest.approx(-1.5 * u / h, rel=1e-5)


def test_derived_deriv_matches_the_shipped_gaussian(gaussian):
    x = jnp.linspace(-2.0, 2.0, 9)[:, None, None]
    y = jnp.linspace(-1.0, 1.0, 7)[None, :, None]
    h = jnp.array([0.5, 1.5])
    np.testing.assert_allclose(GaussianLeaf().deriv(x, y, h), gaussian.deriv(x, y, h), rtol=1e-5, atol=1e-7)


def test_derived_deriv_keeps_the_broadcast_shape():
    x = jnp.zeros((5, 1, 2))
    y = jnp.linspace(-1.0, 1.0, 7).reshape(1, 7, 1) * jnp.ones((1, 1, 2))
    h = jnp.array([0.5, 0.7])
    assert Epanechnikov().deriv(x, y, h).shape == (5, 7, 2)


def test_derived_deriv_has_a_finite_bandwidth_gradient():
    grad = jax.grad(lambda h: jnp.sum(Epanechnikov().deriv(jnp.array(0.4), jnp.array(0.0), h)))(jnp.array(0.5))
    assert jnp.isfinite(grad)


def test_a_kernel_with_neither_k_nor_value_is_refused():
    class Empty(kj_ContinuousKernel):
        pass

    with pytest.raises(TypeError, match=r"k.*value|value.*k"):
        Empty()


def test_value_can_still_be_written_directly():
    class Direct(kj_ContinuousKernel):
        def value(self, x, y, h):
            return jnp.ones_like((x - y) / h)

    assert float(Direct().value(jnp.array(1.0), jnp.array(0.0), jnp.array(2.0))) == 1.0
