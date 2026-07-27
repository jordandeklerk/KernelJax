"""Tests for the cross-validation criteria."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import MixedData
from kerneljax.estimators.density import density
from kerneljax.kernels import Op
from kerneljax.ksum import kweights
from kerneljax.tuning.objectives import cv_ls_density, cv_ml_density


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_finite_on_a_mixed_design(cv_mixed_data, cv_mixed_bandwidth, criterion):
    assert jnp.isfinite(criterion(cv_mixed_data, cv_mixed_bandwidth))


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_finite_on_a_continuous_only_design(cv_continuous_data, cv_continuous_bandwidth, criterion):
    assert jnp.isfinite(criterion(cv_continuous_data, cv_continuous_bandwidth))


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_finite_on_a_discrete_only_design(cv_discrete_data, cv_discrete_bandwidth, criterion):
    assert jnp.isfinite(criterion(cv_discrete_data, cv_discrete_bandwidth))


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
@pytest.mark.parametrize("lam_ord", [0.0, 0.999])
@pytest.mark.parametrize("lam_uno", [0.0, 2.0 / 3.0])
def test_finite_at_the_categorical_boundary(cv_mixed_data, cv_mixed_bandwidth, criterion, lam_uno, lam_ord):
    bandwidth = cv_mixed_bandwidth.replace(lam_uno=jnp.array([lam_uno]), lam_ord=jnp.array([lam_ord]))
    assert jnp.isfinite(criterion(cv_mixed_data, bandwidth))


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
@pytest.mark.parametrize("h", [0.5, 0.15])
@pytest.mark.parametrize("lam_ord", [0.0, 0.999])
@pytest.mark.parametrize("lam_uno", [0.0, 2.0 / 3.0])
def test_gradients_are_finite_at_the_boundary_and_as_h_shrinks(
    cv_mixed_data, cv_mixed_bandwidth, criterion, lam_uno, lam_ord, h
):
    bandwidth = cv_mixed_bandwidth.replace(h=jnp.array([h]), lam_uno=jnp.array([lam_uno]), lam_ord=jnp.array([lam_ord]))
    grads = jax.grad(criterion, argnums=1)(cv_mixed_data, bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))
    assert jnp.all(jnp.isfinite(grads.lam_uno))
    assert jnp.all(jnp.isfinite(grads.lam_ord))


def test_cv_ml_matches_a_hand_built_leave_one_out_expression(cv_mixed_data, cv_mixed_bandwidth):
    got = cv_ml_density(cv_mixed_data, cv_mixed_bandwidth)

    n = cv_mixed_data.n
    h_prod = float(jnp.prod(cv_mixed_bandwidth.h))
    weights = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth)).copy()
    np.fill_diagonal(weights, 0.0)
    f_loo = weights.sum(axis=1) / ((n - 1) * h_prod)
    want = -np.sum(np.log(f_loo))

    assert float(got) == pytest.approx(want, rel=1e-6)


def test_cv_ls_matches_a_hand_built_expression_with_the_full_conv_matrix(cv_mixed_data, cv_mixed_bandwidth):
    got = cv_ls_density(cv_mixed_data, cv_mixed_bandwidth)

    n = cv_mixed_data.n
    h_prod = float(jnp.prod(cv_mixed_bandwidth.h))
    conv_matrix = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth, op=Op.CONV))
    integral_f_squared = conv_matrix.sum() / (n * n * h_prod)

    weights = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth)).copy()
    np.fill_diagonal(weights, 0.0)
    f_loo = weights.sum(axis=1) / ((n - 1) * h_prod)
    want = integral_f_squared - 2.0 * f_loo.sum() / n

    assert float(got) == pytest.approx(want, rel=1e-6)


def test_removing_the_diagonal_from_the_conv_matrix_changes_the_answer(cv_mixed_data, cv_mixed_bandwidth):
    n = cv_mixed_data.n
    conv_matrix = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth, op=Op.CONV))

    with_diagonal = conv_matrix.sum() / (n * n)
    without_diagonal = conv_matrix.copy()
    np.fill_diagonal(without_diagonal, 0.0)
    without_diagonal = without_diagonal.sum() / (n * n)

    assert with_diagonal != pytest.approx(without_diagonal)


def test_continuous_only_integral_term_matches_a_gaussian_widened_by_sqrt_two(
    cv_continuous_data, cv_continuous_bandwidth
):
    n = cv_continuous_data.n
    h = float(cv_continuous_bandwidth.h[0])

    conv_matrix = np.asarray(kweights(cv_continuous_data, cv_continuous_bandwidth, op=Op.CONV))
    integral_f_squared = conv_matrix.sum() / (n * n * h)

    x = np.asarray(cv_continuous_data.con)[:, 0]
    difference = x[:, None] - x[None, :]
    sigma = h * np.sqrt(2.0)
    widened = np.exp(-0.5 * (difference / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    want = widened.sum() / (n * n)

    assert integral_f_squared == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
@pytest.mark.parametrize("chunk", [4, 5, (3, 4)])
def test_chunked_matches_unchunked(cv_mixed_data, cv_mixed_bandwidth, criterion, chunk):
    ref = criterion(cv_mixed_data, cv_mixed_bandwidth)
    got = criterion(cv_mixed_data, cv_mixed_bandwidth, chunk=chunk)
    assert float(got) == pytest.approx(float(ref), rel=1e-6)


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_jit_grad_and_vmap(cv_mixed_data, cv_mixed_bandwidth, criterion):
    fitted = jax.jit(criterion)
    assert jnp.isfinite(fitted(cv_mixed_data, cv_mixed_bandwidth))

    grads = jax.grad(criterion, argnums=1)(cv_mixed_data, cv_mixed_bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))

    out = jax.vmap(lambda h: criterion(cv_mixed_data, cv_mixed_bandwidth.replace(h=h)))(
        jnp.array([[0.3], [0.5], [0.7]])
    )
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_cv_ml_density_matches_the_r_np_parity_value(cv_mixed_data, cv_mixed_bandwidth):
    assert float(cv_ml_density(cv_mixed_data, cv_mixed_bandwidth)) == 77.91303253173828


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_agrees_across_h_axis_settings_at_a_uniform_bandwidth(cv_mixed_data, cv_mixed_bandwidth, criterion):
    n = cv_mixed_data.n
    h_value = float(cv_mixed_bandwidth.h[0])
    shared = criterion(cv_mixed_data, cv_mixed_bandwidth)

    train_indexed = cv_mixed_bandwidth.replace(h=jnp.full((n, 1), h_value), h_axis="train")
    eval_indexed = cv_mixed_bandwidth.replace(h=jnp.full((n, 1), h_value), h_axis="eval")

    assert float(criterion(cv_mixed_data, train_indexed)) == pytest.approx(float(shared), rel=1e-12)
    assert float(criterion(cv_mixed_data, eval_indexed)) == pytest.approx(float(shared), rel=1e-12)


def test_cv_ls_density_traces_without_overflow_at_a_large_n():
    def make():
        data = MixedData.continuous(jnp.zeros((50000, 1)))
        bw = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
        return cv_ls_density(data, bw)

    out = jax.eval_shape(make)
    assert out.shape == ()


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_criterion_quartet_eager_jit_grad_vmap(public_api_data, public_api_bandwidth, criterion):
    eager = criterion(public_api_data, public_api_bandwidth)
    assert jnp.isfinite(eager)

    jitted = jax.jit(criterion)(public_api_data, public_api_bandwidth)
    assert float(jitted) == pytest.approx(float(eager), rel=1e-12)

    grad = jax.grad(criterion, argnums=1)(public_api_data, public_api_bandwidth)
    assert jnp.all(jnp.isfinite(grad.h))

    out = jax.vmap(lambda h: criterion(public_api_data, public_api_bandwidth.replace(h=h)))(jnp.array([[0.5], [0.6]]))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


def test_jit_compiles_once_across_bandwidth_values(public_api_data, public_api_bandwidth):
    calls = {"n": 0}

    @jax.jit
    def wrapped(data, bw):
        calls["n"] += 1
        return density(data, bw).value.sum()

    for h in (0.4, 0.5, 0.6, 0.7):
        wrapped(public_api_data, public_api_bandwidth.replace(h=jnp.array([h])))

    assert calls["n"] == 1
