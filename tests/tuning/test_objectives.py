"""Tests for the cross-validation criteria."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth, normal_reference
from kerneljax.data import MixedData
from kerneljax.estimators.density import density
from kerneljax.kernels import KernelSet, Op
from kerneljax.ksum import kweights
from kerneljax.tuning.objectives import _aic_c_penalty, aic_c_regression, cv_ls_density, cv_ls_regression, cv_ml_density
from kerneljax.tuning.optimize import select_bandwidth


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
def test_grads_finite_at_boundary_and_shrinking_h(cv_mixed_data, cv_mixed_bandwidth, criterion, lam_uno, lam_ord, h):
    bandwidth = cv_mixed_bandwidth.replace(h=jnp.array([h]), lam_uno=jnp.array([lam_uno]), lam_ord=jnp.array([lam_ord]))
    grads = jax.grad(criterion, argnums=1)(cv_mixed_data, bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))
    assert jnp.all(jnp.isfinite(grads.lam_uno))
    assert jnp.all(jnp.isfinite(grads.lam_ord))


def test_cv_ml_matches_hand_built_leave_one_out(cv_mixed_data, cv_mixed_bandwidth):
    got = cv_ml_density(cv_mixed_data, cv_mixed_bandwidth)

    n = cv_mixed_data.n
    h_prod = float(jnp.prod(cv_mixed_bandwidth.h))
    weights = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth)).copy()
    np.fill_diagonal(weights, 0.0)
    f_loo = weights.sum(axis=1) / ((n - 1) * h_prod)
    want = -np.sum(np.log(f_loo))

    assert float(got) == pytest.approx(want, rel=1e-6)


def test_cv_ls_matches_hand_built_conv_matrix(cv_mixed_data, cv_mixed_bandwidth):
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


def test_removing_diagonal_changes_the_sum(cv_mixed_data, cv_mixed_bandwidth):
    n = cv_mixed_data.n
    conv_matrix = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth, op=Op.CONV))

    with_diagonal = conv_matrix.sum() / (n * n)
    without_diagonal = conv_matrix.copy()
    np.fill_diagonal(without_diagonal, 0.0)
    without_diagonal = without_diagonal.sum() / (n * n)

    assert with_diagonal != pytest.approx(without_diagonal)


def test_integral_term_matches_widened_gaussian(cv_continuous_data, cv_continuous_bandwidth):
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


def test_cv_ml_value_is_stable(cv_mixed_data, cv_mixed_bandwidth):
    assert float(cv_ml_density(cv_mixed_data, cv_mixed_bandwidth)) == pytest.approx(77.913033, rel=1e-6)


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_agrees_across_h_axis_at_uniform_bandwidth(cv_mixed_data, cv_mixed_bandwidth, criterion):
    n = cv_mixed_data.n
    h_value = float(cv_mixed_bandwidth.h[0])
    shared = criterion(cv_mixed_data, cv_mixed_bandwidth)

    train_indexed = cv_mixed_bandwidth.replace(h=jnp.full((n, 1), h_value), h_axis="train")
    eval_indexed = cv_mixed_bandwidth.replace(h=jnp.full((n, 1), h_value), h_axis="eval")

    assert float(criterion(cv_mixed_data, train_indexed)) == pytest.approx(float(shared), rel=1e-12)
    assert float(criterion(cv_mixed_data, eval_indexed)) == pytest.approx(float(shared), rel=1e-12)


def test_cv_ls_traces_without_overflow_at_large_n():
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


@pytest.mark.parametrize("criterion", [cv_ls_regression, aic_c_regression])
@pytest.mark.parametrize("degree", [0, 1])
def test_regression_finite_on_a_mixed_design(
    cv_mixed_data, cv_mixed_bandwidth, regression_mixed_response, criterion, degree
):
    value = criterion(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response, degree=degree)
    assert jnp.isfinite(value)


@pytest.mark.parametrize("criterion", [cv_ls_regression, aic_c_regression])
@pytest.mark.parametrize("degree", [0, 1])
def test_regression_finite_on_continuous_only(
    cv_continuous_data, cv_continuous_bandwidth, regression_continuous_response, criterion, degree
):
    value = criterion(cv_continuous_data, cv_continuous_bandwidth, y=regression_continuous_response, degree=degree)
    assert jnp.isfinite(value)


@pytest.mark.parametrize("criterion", [cv_ls_regression, aic_c_regression])
@pytest.mark.parametrize("degree", [0, 1])
def test_regression_finite_on_discrete_only(
    cv_discrete_data, cv_discrete_bandwidth, regression_discrete_response, criterion, degree
):
    value = criterion(cv_discrete_data, cv_discrete_bandwidth, y=regression_discrete_response, degree=degree)
    assert jnp.isfinite(value)


@pytest.mark.parametrize("criterion", [cv_ls_regression, aic_c_regression])
@pytest.mark.parametrize("lam_ord", [0.0, 0.999])
@pytest.mark.parametrize("lam_uno", [0.0, 2.0 / 3.0])
def test_grad_finite_at_categorical_boundary(
    cv_mixed_data, cv_mixed_bandwidth, regression_mixed_response, criterion, lam_uno, lam_ord
):
    bandwidth = cv_mixed_bandwidth.replace(lam_uno=jnp.array([lam_uno]), lam_ord=jnp.array([lam_ord]))
    grads = jax.grad(criterion, argnums=1)(cv_mixed_data, bandwidth, y=regression_mixed_response)
    assert jnp.all(jnp.isfinite(grads.h))
    assert jnp.all(jnp.isfinite(grads.lam_uno))
    assert jnp.all(jnp.isfinite(grads.lam_ord))


@pytest.mark.parametrize(
    "criterion, degree, h, lam, expected",
    [
        (cv_ls_regression, 0, 0.20415158048189577, 0.00809926966485721, 0.824703130598134),
        (cv_ls_regression, 1, 0.38138784655683594, 0.00401153878039302, 0.114188303008324),
        (aic_c_regression, 0, 0.528705477633985, 0.130136081395895, 0.648913106180714),
        (aic_c_regression, 1, 0.950293916838712, 0.047652641664437, -0.593551239967511),
    ],
)
def test_matches_the_reference_optimum(
    regression_reference_train, regression_reference_y, criterion, degree, h, lam, expected
):
    bandwidth = Bandwidth(h=jnp.array([h]), lam_uno=jnp.array([lam]), lam_ord=jnp.zeros(0))
    got = criterion(regression_reference_train, bandwidth, y=regression_reference_y, degree=degree)
    assert float(got) == pytest.approx(expected, rel=2e-6)


def test_cv_ls_matches_hand_built_leave_one_out(cv_mixed_data, cv_mixed_bandwidth, regression_mixed_response):
    weights = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth)).copy()
    np.fill_diagonal(weights, 0.0)
    row_sum = weights.sum(axis=1)

    response = np.asarray(regression_mixed_response)
    mean_loo = (weights @ response) / row_sum
    want = np.mean((response - mean_loo) ** 2)

    got = cv_ls_regression(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response, degree=0)
    assert float(got) == pytest.approx(float(want), rel=1e-6)


def test_aic_c_matches_hand_built_shortcut_trace(cv_mixed_data, cv_mixed_bandwidth, regression_mixed_response):
    n = cv_mixed_data.n
    weights = np.asarray(kweights(cv_mixed_data, cv_mixed_bandwidth))
    row_sum = weights.sum(axis=1)

    response = np.asarray(regression_mixed_response)
    mean_full = (weights @ response) / row_sum
    sigma_squared = np.mean((response - mean_full) ** 2)

    trace = np.sum(np.diagonal(weights) / row_sum)
    denominator = 1.0 - (trace + 2.0) / n
    penalty = (1.0 + trace / n) / denominator
    want = np.log(sigma_squared) + penalty

    got = aic_c_regression(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response, degree=0)
    assert float(got) == pytest.approx(float(want), rel=1e-6)


def test_penalty_matches_formula_when_valid():
    n = 15
    trace = jnp.asarray(7.0)
    denominator = 1.0 - (trace + 2.0) / n
    want = (1.0 + trace / n) / denominator

    got = _aic_c_penalty(trace, n)
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_penalty_finite_and_increasing_past_pole():
    n = 15
    pole = n - 2
    traces = jnp.linspace(pole - 5.0, pole + 25.0, 40)

    values = jax.vmap(lambda trace: _aic_c_penalty(trace, n))(traces)
    grads = jax.vmap(jax.grad(lambda trace: _aic_c_penalty(trace, n)))(traces)

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(grads))
    assert jnp.all(jnp.diff(values[traces >= pole]) > 0)


def test_penalty_gradient_finite_at_the_pole():
    n = 15
    trace = jnp.asarray(float(n - 2))

    assert jnp.isfinite(_aic_c_penalty(trace, n))
    assert jnp.isfinite(jax.grad(_aic_c_penalty)(trace, n))


@pytest.mark.parametrize("criterion", [cv_ls_regression, aic_c_regression])
@pytest.mark.parametrize("chunk", [4, 5, (3, 4)])
def test_chunked_matches_unchunked_regression(
    cv_mixed_data, cv_mixed_bandwidth, regression_mixed_response, criterion, chunk
):
    ref = criterion(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response)
    got = criterion(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response, chunk=chunk)
    assert float(got) == pytest.approx(float(ref), rel=1e-6)


@pytest.mark.parametrize("criterion", [cv_ls_regression, aic_c_regression])
def test_jit_grad_and_vmap_for_regression(cv_mixed_data, cv_mixed_bandwidth, regression_mixed_response, criterion):
    fitted = jax.jit(criterion)
    assert jnp.isfinite(fitted(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response))

    grads = jax.grad(criterion, argnums=1)(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response)
    assert jnp.all(jnp.isfinite(grads.h))

    out = jax.vmap(lambda h: criterion(cv_mixed_data, cv_mixed_bandwidth.replace(h=h), y=regression_mixed_response))(
        jnp.array([[0.3], [0.5], [0.7]])
    )
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_named_response_improves_selection(poly_mixed_train, poly_mixed_response):
    start = normal_reference(poly_mixed_train, KernelSet())
    starting_value = cv_ls_regression(poly_mixed_train, start, y=poly_mixed_response)

    result = select_bandwidth(poly_mixed_train, cv_ls_regression, y=poly_mixed_response, n_starts=1)
    assert float(result.value) <= float(starting_value)


def test_varying_response_reuses_compilation(poly_mixed_train, poly_mixed_response):
    select_bandwidth(poly_mixed_train, cv_ls_regression, y=poly_mixed_response, n_starts=1)
    compiled = select_bandwidth._cache_size()

    select_bandwidth(poly_mixed_train, cv_ls_regression, y=poly_mixed_response + 1.0, n_starts=1)
    assert select_bandwidth._cache_size() == compiled
