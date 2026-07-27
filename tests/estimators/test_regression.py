"""Tests for local polynomial regression."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.estimators.regression import LocalPolyFit, local_poly
from kerneljax.ksum import kweights


def test_degree_zero_matches_nadaraya_watson(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=0)

    weights = np.asarray(kweights(poly_train, poly_bandwidth, at=poly_at))
    response = np.asarray(poly_response)
    want = (weights * response[None, :]).sum(axis=1) / weights.sum(axis=1)

    assert np.max(np.abs(np.asarray(fit.mean) - want) / np.abs(want)) < 1e-6


@pytest.mark.parametrize("intercept, slope", [(3.0, 2.0), (-1.5, -4.0), (0.0, 7.5)])
def test_degree_one_reproduces_exact_line(poly_train, poly_at, poly_bandwidth, intercept, slope):
    x = np.asarray(poly_train.con)[:, 0]
    y = jnp.asarray(intercept + slope * x)
    fit = local_poly(poly_train, y, poly_bandwidth, at=poly_at, degree=1)

    at_x = np.asarray(poly_at.con)[:, 0]
    want = intercept + slope * at_x
    assert np.max(np.abs(np.asarray(fit.mean) - want) / np.abs(want)) < 1e-6


@pytest.mark.parametrize("a, b, c", [(1.0, -2.0, 0.5), (-0.5, 3.0, 1.0)])
def test_degree_two_reproduces_exact_quadratic(poly_train, poly_at, poly_bandwidth, a, b, c):
    x = np.asarray(poly_train.con)[:, 0]
    y = jnp.asarray(a * x**2 + b * x + c)
    fit = local_poly(poly_train, y, poly_bandwidth, at=poly_at, degree=2)

    at_x = np.asarray(poly_at.con)[:, 0]
    want = a * at_x**2 + b * at_x + c
    assert np.max(np.abs(np.asarray(fit.mean) - want) / np.abs(want)) < 1e-6


def test_degree_one_misses_a_quadratic(poly_train, poly_at, poly_bandwidth):
    x = np.asarray(poly_train.con)[:, 0]
    y = jnp.asarray(x**2)
    fit = local_poly(poly_train, y, poly_bandwidth, at=poly_at, degree=1)

    at_x = np.asarray(poly_at.con)[:, 0]
    want = at_x**2
    assert not np.allclose(np.asarray(fit.mean), want, rtol=1e-3)


@pytest.mark.parametrize("slope", [2.0, -3.5, 0.1, -10.0])
def test_gradient_of_a_line_matches_slope(poly_train, poly_at, poly_bandwidth, slope):
    x = np.asarray(poly_train.con)[:, 0]
    y = jnp.asarray(5.0 + slope * x)
    fit = local_poly(poly_train, y, poly_bandwidth, at=poly_at, degree=1, gradient=True)

    got = np.asarray(fit.grad)[:, 0]
    assert np.allclose(got, slope, rtol=1e-5, atol=1e-4)


def test_eval_axis_gradient_matches_slope(poly_train, poly_at, poly_eval_indexed_bandwidth):
    x = np.asarray(poly_train.con)[:, 0]
    y = jnp.asarray(3.0 + 2.0 * x)
    fit = local_poly(poly_train, y, poly_eval_indexed_bandwidth, at=poly_at, degree=1, gradient=True)

    got = np.asarray(fit.grad)[:, 0]
    assert np.allclose(got, 2.0, rtol=1e-5, atol=1e-4)


def test_train_axis_gradient_raises(poly_train, poly_at, poly_train_indexed_bandwidth, poly_response):
    with pytest.raises(ValueError):
        local_poly(poly_train, poly_response, poly_train_indexed_bandwidth, at=poly_at, degree=1, gradient=True)


@pytest.mark.parametrize("a, b", [(1.0, -2.0), (-0.5, 3.0)])
def test_gradient_of_quadratic_matches_derivative(poly_train, poly_at, poly_bandwidth, a, b):
    x = np.asarray(poly_train.con)[:, 0]
    y = jnp.asarray(a * x**2 + b * x + 0.5)
    fit = local_poly(poly_train, y, poly_bandwidth, at=poly_at, degree=2, gradient=True)

    at_x = np.asarray(poly_at.con)[:, 0]
    want = 2.0 * a * at_x + b
    assert np.allclose(np.asarray(fit.grad)[:, 0], want, rtol=1e-5)


def test_mixed_data_fits_without_error(poly_mixed_train, poly_mixed_bandwidth, poly_mixed_response):
    fit = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=1)
    assert jnp.all(jnp.isfinite(fit.mean))


def test_categorical_bandwidth_changes_the_fit(poly_mixed_train, poly_mixed_bandwidth, poly_mixed_response):
    base = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=1).mean
    changed_bandwidth = poly_mixed_bandwidth.replace(lam_uno=jnp.array([0.9]), lam_ord=jnp.array([0.9]))
    changed = local_poly(poly_mixed_train, poly_mixed_response, changed_bandwidth, degree=1).mean
    assert not jnp.allclose(base, changed)


def test_gradient_needs_degree_at_least_one(poly_train, poly_bandwidth, poly_response):
    with pytest.raises(ValueError):
        local_poly(poly_train, poly_response, poly_bandwidth, degree=0, gradient=True)


def test_grad_is_none_without_gradient_request(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1)
    assert fit.grad is None


@pytest.mark.parametrize("chunk", [1, 3, 5, (2, 7), (3, 40)])
def test_chunked_matches_unchunked(poly_train, poly_at, poly_bandwidth, poly_response, chunk):
    ref = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, gradient=True)
    got = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, gradient=True, chunk=chunk)

    assert jnp.allclose(got.mean, ref.mean, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(got.grad, ref.grad, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(got.coef, ref.coef, rtol=1e-6, atol=1e-8)


def test_fold_gives_a_different_loo_fit(poly_train, poly_bandwidth, poly_response):
    fold = jnp.arange(poly_train.n)
    loo = local_poly(poly_train, poly_response, poly_bandwidth, fold=fold, degree=1)
    full = local_poly(poly_train, poly_response, poly_bandwidth, degree=1)
    assert not jnp.allclose(loo.mean, full.mean)


def test_returns_a_record_with_expected_shapes(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=2, gradient=True)
    assert isinstance(fit, LocalPolyFit)
    assert fit.mean.shape == (poly_at.n,)
    assert fit.grad.shape == (poly_at.n, 1)
    assert fit.coef.shape == (poly_at.n, 3)
    assert fit.rcond.shape == (poly_at.n,)
    assert fit.bandwidth is poly_bandwidth


def test_pytree_survives_vmap_over_bandwidths(poly_train, poly_at, poly_bandwidth, poly_response):
    def run(h):
        return local_poly(poly_train, poly_response, poly_bandwidth.replace(h=h), at=poly_at, degree=1, gradient=True)

    fits = jax.vmap(run)(jnp.array([[0.3], [0.5], [0.8]]))

    assert isinstance(fits, LocalPolyFit)
    assert fits.mean.shape == (3, poly_at.n)
    assert fits.grad.shape == (3, poly_at.n, 1)
    assert fits.coef.shape == (3, poly_at.n, 2)
    assert fits.rcond.shape == (3, poly_at.n)
    assert fits.bandwidth.h.shape == (3, 1)


def test_rcond_finite_in_unit_interval(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1)
    assert jnp.all(jnp.isfinite(fit.rcond))
    assert jnp.all(fit.rcond >= 0.0)
    assert jnp.all(fit.rcond <= 1.0 + 1e-6)


def test_jit_grad_vmap(poly_train, poly_at, poly_bandwidth, poly_response):
    eager = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, gradient=True)
    assert jnp.all(jnp.isfinite(eager.mean))

    jitted = jax.jit(lambda bw: local_poly(poly_train, poly_response, bw, at=poly_at, degree=1, gradient=True))
    assert jnp.allclose(jitted(poly_bandwidth).mean, eager.mean, rtol=1e-6)

    def total(bw):
        return local_poly(poly_train, poly_response, bw, at=poly_at, degree=1).mean.sum()

    gradient = jax.grad(total)(poly_bandwidth)
    assert jnp.all(jnp.isfinite(gradient.h))

    out = jax.vmap(lambda h: total(poly_bandwidth.replace(h=h)))(jnp.array([[0.4], [0.6]]))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))
