"""Tests for the derived continuous cdf and conv."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtr

from kerneljax.estimators.density import density
from kerneljax.estimators.distribution import cdf
from kerneljax.estimators.regression import local_poly
from kerneljax.kernels import KernelSet
from kerneljax.kernels._checks import (
    _check_cdf_limits,
    _check_conv_at_zero,
    _check_conv_matches,
    _check_value_mass,
    _passed,
)
from kerneljax.kernels._derived import _tables, tables
from kerneljax.kernels.base import ContinuousKernel


class Epanechnikov(ContinuousKernel):
    def k(self, u):
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)


class Triangular(ContinuousKernel):
    def k(self, u):
        return jnp.maximum(1.0 - jnp.abs(u), 0.0)


class Uniform(ContinuousKernel):
    def k(self, u):
        return jnp.where(jnp.abs(u) <= 1.0, 0.5, 0.0)


class GaussianLeaf(ContinuousKernel):
    def k(self, u):
        return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)


def _epan_cdf(u):
    z = np.clip(u, -1.0, 1.0)
    return 0.5 + 0.75 * z - 0.25 * z**3


def _epan_conv(u):
    z = np.minimum(np.abs(u), 2.0)
    return 3.0 / 160.0 * (2.0 - z) ** 3 * (z * z + 6.0 * z + 4.0)


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
def test_derived_cdf_matches_the_analytic_epanechnikov(h):
    u = np.linspace(-1.5, 1.5, 13)
    got = Epanechnikov().cdf(jnp.asarray(h * u), jnp.array(0.0), jnp.array(h))
    np.testing.assert_allclose(got, _epan_cdf(u), atol=1e-3)


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
def test_derived_conv_matches_the_analytic_epanechnikov(h):
    u = np.linspace(-2.5, 2.5, 21)
    got = Epanechnikov().conv(jnp.asarray(h * u), jnp.array(0.0), jnp.array(h))
    np.testing.assert_allclose(got, _epan_conv(u), atol=1e-3)


def test_derived_operators_match_the_shipped_gaussian(gaussian):
    u = jnp.linspace(-4.0, 4.0, 17)
    np.testing.assert_allclose(GaussianLeaf().cdf(u, 0.0, 1.0), gaussian.cdf(u, 0.0, 1.0), atol=1e-3)
    np.testing.assert_allclose(GaussianLeaf().conv(u, 0.0, 1.0), gaussian.conv(u, 0.0, 1.0), atol=1e-3)
    np.testing.assert_allclose(GaussianLeaf().cdf(u, 0.0, 1.0), ndtr(u), atol=1e-3)


def test_derived_operators_are_exact_zero_outside_the_doubled_support():
    kernel = Epanechnikov()
    assert float(kernel.cdf(jnp.array(-1.5), jnp.array(0.0), jnp.array(1.0))) == 0.0
    assert float(kernel.cdf(jnp.array(1.5), jnp.array(0.0), jnp.array(1.0))) == pytest.approx(1.0, abs=1e-4)
    assert float(kernel.conv(jnp.array(2.5), jnp.array(0.0), jnp.array(1.0))) == 0.0


@pytest.mark.parametrize("kernel", [Epanechnikov(), Triangular(), Uniform(), GaussianLeaf()])
def test_derived_operators_pass_every_shipped_check(kernel):
    _passed.clear()
    _check_value_mass(kernel)
    _check_conv_at_zero(kernel)
    _check_conv_matches(kernel)
    _check_cdf_limits(kernel)


def test_derived_operators_keep_the_broadcast_shape():
    x = jnp.zeros((5, 1, 2))
    y = jnp.linspace(-1.0, 1.0, 7).reshape(1, 7, 1) * jnp.ones((1, 1, 2))
    h = jnp.array([0.5, 0.7])
    assert Epanechnikov().cdf(x, y, h).shape == (5, 7, 2)
    assert Epanechnikov().conv(x, y, h).shape == (5, 7, 2)


def test_derived_operators_have_finite_bandwidth_gradients():
    for op in ("cdf", "conv"):
        fn = getattr(Epanechnikov(), op)
        grad = jax.grad(lambda h, f=fn: jnp.sum(f(jnp.array(0.4), jnp.array(0.0), h)))(jnp.array(0.5))
        assert jnp.isfinite(grad)


def test_derived_operators_run_under_jit():
    kernel = Epanechnikov()
    got = jax.jit(lambda u: kernel.cdf(u, jnp.array(0.0), jnp.array(1.0)))(jnp.array(0.5))
    assert float(got) == pytest.approx(_epan_cdf(0.5), abs=1e-3)


def test_a_k_only_kernel_serves_every_estimator(probe_x, probe_y, probe_bw):
    kernels = KernelSet(continuous=Epanechnikov())

    fit = local_poly(probe_x, probe_y, "cv_ls", degree=1, kernels=kernels, gradient=True, se=True, n_starts=1)
    assert fit.grad.shape == (40, 1)
    assert fit.se.shape == (40,)

    assert density(probe_x, "cv_ls", kernels=kernels, n_starts=1).value.shape == (40,)
    assert density(probe_x, "cv_ml", kernels=kernels, n_starts=1).value.shape == (40,)
    assert cdf(probe_x, "cv_cdf", kernels=kernels, n_starts=1).value.shape == (40,)


def test_tables_are_built_once_per_instance():
    _tables.clear()
    tables(Epanechnikov())
    tables(Epanechnikov())
    assert len(_tables) == 1
