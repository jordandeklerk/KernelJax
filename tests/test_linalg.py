"""Tests for the weighted least squares moment solve."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.linalg import WLS, hat_diagonal, wls


def _moments(design, weights, response):
    weighted = design * weights[:, None]
    return weighted.T @ design, weighted.T @ response


def _singular_moments(design, weights):
    xtwx, xtwy = _moments(design, weights, jnp.ones((design.shape[0], 1)))
    return xtwx - 1e-2 * jnp.eye(design.shape[1]), xtwy


def test_matches_independent_numpy_solve(wls_design):
    design, weights, true_coef = wls_design
    rng = np.random.default_rng(21)
    noise = jnp.asarray(rng.normal(scale=0.05, size=(design.shape[0], true_coef.shape[1])))
    xtwx, xtwy = _moments(design, weights, design @ true_coef + noise)

    got = wls(xtwx, xtwy).coef
    want = np.linalg.solve(np.asarray(xtwx, dtype=np.float64), np.asarray(xtwy, dtype=np.float64))
    assert np.allclose(np.asarray(got), want, rtol=1e-6)


def test_exact_linear_response_has_no_error(wls_design):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)

    got = wls(xtwx, xtwy).coef
    assert jnp.allclose(got, true_coef, rtol=1e-6)


def test_ok_is_true_for_a_well_posed_system(wls_design):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)
    assert bool(wls(xtwx, xtwy).ok)


def test_rcond_near_one_for_scaled_identity():
    xtwx = 3.0 * jnp.eye(4)
    xtwy = jnp.ones((4, 1))

    fit = wls(xtwx, xtwy)
    assert bool(fit.ok)
    assert float(fit.rcond) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0])
def test_rcond_tracks_condition_number(scale):
    rng = np.random.default_rng(5)
    points = rng.uniform(-scale, scale, size=40)
    design = np.stack([np.ones_like(points), points, points**2], axis=1)
    gram = design.T @ design

    true_rcond = 1.0 / np.linalg.cond(gram)
    fit = wls(jnp.asarray(gram, dtype=jnp.float32), jnp.zeros((3, 1), dtype=jnp.float32))

    assert bool(fit.ok)
    assert true_rcond / 10.0 <= float(fit.rcond) <= true_rcond * 10.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_rcond_in_unit_interval_for_spd(dim, seed):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(dim, dim))
    spd = factor @ factor.T + dim * np.eye(dim)

    fit = wls(jnp.asarray(spd, dtype=jnp.float32), jnp.zeros((dim, 1), dtype=jnp.float32))
    assert 0.0 <= float(fit.rcond) <= 1.0 + 1e-6


@pytest.mark.parametrize("penalty", [0.0, 0.5, 3.0])
def test_cho_reproduces_the_regularized_gram_matrix(wls_design, penalty):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)

    cho = wls(xtwx, xtwy, penalty=penalty).cho
    want = xtwx + penalty * jnp.eye(xtwx.shape[0])
    assert jnp.allclose(cho @ cho.T, want, rtol=1e-6)


def test_penalty_shrinks_coef_monotonically(wls_design):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)

    norms = [float(jnp.linalg.norm(wls(xtwx, xtwy, penalty=penalty).coef)) for penalty in (0.0, 0.1, 1.0, 10.0)]
    assert all(earlier >= later for earlier, later in itertools.pairwise(norms))
    assert norms[0] > norms[-1]


def test_matrix_penalty_matches_equivalent_scalar(wls_design):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)
    dim = xtwx.shape[0]

    scalar_fit = wls(xtwx, xtwy, penalty=2.0)
    matrix_fit = wls(xtwx, xtwy, penalty=2.0 * jnp.eye(dim))
    assert jnp.allclose(scalar_fit.coef, matrix_fit.coef, rtol=1e-12)


def test_singular_gram_sets_ok_false_coef_finite(singular_design):
    design, weights = singular_design
    xtwx, xtwy = _singular_moments(design, weights)

    fit = wls(xtwx, xtwy)
    assert not bool(fit.ok)
    assert jnp.all(jnp.isfinite(fit.coef))


def test_singular_gram_sets_rcond_zero_and_finite(singular_design):
    design, weights = singular_design
    xtwx, xtwy = _singular_moments(design, weights)

    fit = wls(xtwx, xtwy)
    assert float(fit.rcond) == 0.0
    assert jnp.isfinite(fit.rcond)


def test_singular_gram_keeps_gradients_finite(singular_design):
    design, weights = singular_design
    xtwx, xtwy = _singular_moments(design, weights)

    gradient = jax.grad(lambda matrix: wls(matrix, xtwy).coef.sum())(xtwx)
    assert jnp.all(jnp.isfinite(gradient))


def test_rcond_gradient_finite_for_both_systems(wls_design, singular_design):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)
    gradient = jax.grad(lambda matrix: wls(matrix, xtwy).rcond)(xtwx)
    assert jnp.isfinite(gradient).all()

    design_s, weights_s = singular_design
    xtwx_s, xtwy_s = _singular_moments(design_s, weights_s)
    gradient_s = jax.grad(lambda matrix: wls(matrix, xtwy_s).rcond)(xtwx_s)
    assert jnp.isfinite(gradient_s).all()


def test_hat_diagonal_matches_a_numpy_hat_matrix(wls_design):
    design, _, _ = wls_design
    xtwx = design.T @ design
    cho = wls(xtwx, jnp.zeros((design.shape[1], 1))).cho

    got = jax.vmap(lambda row: hat_diagonal(cho, row, 1.0))(design)

    numpy_design = np.asarray(design, dtype=np.float64)
    hat = numpy_design @ np.linalg.solve(numpy_design.T @ numpy_design, numpy_design.T)
    want = np.diag(hat)
    assert np.allclose(np.asarray(got), want, rtol=1e-6)


def test_leverage_in_unit_interval_for_weighted(wls_design):
    design, weights, _ = wls_design
    xtwx, xtwy = _moments(design, weights, jnp.zeros((design.shape[0], 1)))
    cho = wls(xtwx, xtwy).cho

    got = jax.vmap(lambda row, weight: hat_diagonal(cho, row, weight))(design, weights)
    assert jnp.all(got >= 0.0)
    assert jnp.all(got <= 1.0 + 1e-6)


def test_hat_diagonal_jit_and_vmap(wls_design):
    design, weights, _ = wls_design
    xtwx, xtwy = _moments(design, weights, jnp.zeros((design.shape[0], 1)))
    cho = wls(xtwx, xtwy).cho

    leverage = jax.jit(jax.vmap(lambda row, weight: hat_diagonal(cho, row, weight)))(design, weights)
    assert jnp.all(jnp.isfinite(leverage))


def test_wls_is_a_pytree_that_survives_vmap(wls_design):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)
    batch_xtwx = jnp.stack([xtwx, xtwx])
    batch_xtwy = jnp.stack([xtwy, xtwy])

    fit = jax.vmap(wls)(batch_xtwx, batch_xtwy)

    assert isinstance(fit, WLS)
    assert fit.coef.shape == (2, *xtwy.shape)
    assert fit.cho.shape == (2, *xtwx.shape)
    assert fit.ok.shape == (2,)
    assert fit.rcond.shape == (2,)


def test_jit_grad_vmap(wls_design, singular_design):
    design, weights, true_coef = wls_design
    xtwx, xtwy = _moments(design, weights, design @ true_coef)

    total = jax.jit(lambda matrix, moment: wls(matrix, moment).coef.sum())
    assert jnp.isfinite(total(xtwx, xtwy))
    assert jnp.all(jnp.isfinite(jax.grad(total)(xtwx, xtwy)))

    batch = jax.vmap(wls)(jnp.stack([xtwx, xtwx]), jnp.stack([xtwy, xtwy]))
    assert jnp.all(jnp.isfinite(batch.coef))

    design_s, weights_s = singular_design
    xtwx_s, xtwy_s = _singular_moments(design_s, weights_s)
    assert jnp.isfinite(total(xtwx_s, xtwy_s))
    assert jnp.all(jnp.isfinite(jax.grad(total)(xtwx_s, xtwy_s)))


def test_rcond_flags_ill_conditioned_design():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1000.0, 1000.0, size=30)
    design = jnp.asarray(np.stack([np.ones_like(points), points, points**2], axis=1), dtype=jnp.float32)
    xtwx = design.T @ design

    fit = wls(xtwx, jnp.zeros((3, 1), dtype=jnp.float32))
    assert bool(fit.ok)
    assert float(fit.rcond) < 1e-6
