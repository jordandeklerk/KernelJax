"""Tests for bandwidth selection."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.bandwidth import BandwidthTransform, normal_reference
from kerneljax.kernels import KernelSet
from kerneljax.selection.criteria import DensityCriterion, RegressionCriterion
from kerneljax.selection.objectives import cv_ls_density, cv_ls_regression, cv_ml_density
from kerneljax.selection.optimize import SelectionResult, lbfgs, select_bandwidth


def test_lbfgs_minimizes_a_quadratic():
    def fun(z):
        return jnp.sum((z - jnp.array([1.0, -2.0])) ** 2)

    z, value, _, converged = lbfgs(fun, jnp.zeros(2))
    assert bool(converged)
    assert jnp.allclose(z, jnp.array([1.0, -2.0]), atol=1e-5)
    assert float(value) == pytest.approx(0.0, abs=1e-9)


def test_lbfgs_minimizes_rosenbrock():
    def fun(z):
        return (1 - z[0]) ** 2 + 100.0 * (z[1] - z[0] ** 2) ** 2

    z, _, _, converged = lbfgs(fun, jnp.array([-1.2, 1.0]), max_iter=2000)
    assert bool(converged)
    assert jnp.allclose(z, jnp.ones(2), atol=1e-3)


def test_lbfgs_is_jittable():
    def fun(z):
        return jnp.sum(z**2)

    out = jax.jit(lambda z0: lbfgs(fun, z0)[0])(jnp.ones(3))
    assert jnp.allclose(out, jnp.zeros(3), atol=1e-5)


def test_lbfgs_is_vmappable_over_starting_points():
    def fun(z):
        return (1 - z[0]) ** 2 + 100.0 * (z[1] - z[0] ** 2) ** 2

    starts = jnp.array([[-1.2, 1.0], [0.8, 1.5], [2.0, -1.0]])
    z, _, _, converged = jax.vmap(lambda z0: lbfgs(fun, z0, max_iter=2000))(starts)
    assert z.shape == (3, 2)
    assert jnp.all(converged)
    assert jnp.allclose(z, jnp.ones((3, 2)), atol=1e-2)


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_selected_bandwidth_is_inside_the_box(cv_mixed_data, criterion):
    result = select_bandwidth(cv_mixed_data, criterion, n_starts=3)
    transform = BandwidthTransform(spec=cv_mixed_data.spec, kernels=KernelSet())
    _, upper = transform.bounds()
    flat = jnp.concatenate([result.bandwidth.h, result.bandwidth.lam_uno, result.bandwidth.lam_ord])

    assert jnp.all(result.bandwidth.h > 0)
    assert jnp.all(flat >= 0.0)
    assert jnp.all(flat <= upper + 1e-4)


@pytest.mark.parametrize("n_starts", [1, 2, 3])
@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_selected_value_no_worse_than_start(cv_mixed_data, criterion, n_starts):
    start = normal_reference(cv_mixed_data, KernelSet())
    start_value = criterion(cv_mixed_data, start)

    result = select_bandwidth(cv_mixed_data, criterion, n_starts=n_starts)
    assert float(result.value) <= float(start_value) + 1e-6


def test_selection_vmaps_over_bootstrap_replicates(cv_mixed_data):
    key = jax.random.key(0)
    idx = jax.random.randint(key, (8, cv_mixed_data.n), 0, cv_mixed_data.n)

    def one(indices):
        resampled = cv_mixed_data.replace(
            con=cv_mixed_data.con[indices], uno=cv_mixed_data.uno[indices], orde=cv_mixed_data.orde[indices]
        )
        return select_bandwidth(resampled, cv_ml_density, n_starts=1).bandwidth.h[0]

    out = jax.vmap(one)(idx)
    assert out.shape == (8,)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out > 0)


def test_custom_solver_is_accepted_and_called(cv_mixed_data):
    calls = {"count": 0}

    def custom_solver(fun, z0, **kwargs):
        calls["count"] += 1
        return z0, fun(z0), jnp.asarray(0), jnp.asarray(True)

    select_bandwidth(cv_mixed_data, cv_ml_density, solver=custom_solver, n_starts=1)
    assert calls["count"] >= 1


def test_selection_result_is_a_pytree(cv_mixed_data):
    result = select_bandwidth(cv_mixed_data, cv_ml_density, n_starts=2)
    assert isinstance(result, SelectionResult)
    leaves = jax.tree_util.tree_leaves(result)
    assert all(hasattr(leaf, "shape") for leaf in leaves)


def test_named_y_matches_the_mapping(criteria_train, criteria_response):
    named = select_bandwidth(criteria_train, cv_ls_regression, y=criteria_response, n_starts=1)
    mapped = select_bandwidth(criteria_train, cv_ls_regression, criterion_kwargs={"y": criteria_response}, n_starts=1)
    assert float(named.value) == float(mapped.value)
    assert jnp.array_equal(named.bandwidth.h, mapped.bandwidth.h)


def test_criterion_without_a_response_gets_none(criteria_train):
    result = select_bandwidth(criteria_train, cv_ml_density, n_starts=1)
    assert jnp.all(jnp.isfinite(result.bandwidth.h))


def test_result_records_the_criterion(criteria_train, criteria_response):
    criterion = RegressionCriterion(method="aic", degree=2)
    result = select_bandwidth(criteria_train, criterion, y=criteria_response, n_starts=1)
    assert result.criterion == criterion
    assert result.criterion.degree == 2


def test_recorded_degree_is_concrete_in_a_trace(criteria_train, criteria_response):
    result = select_bandwidth(
        criteria_train, RegressionCriterion(method="cv_ls", degree=2), y=criteria_response, n_starts=1
    )

    @jax.jit
    def basis_width(selection):
        return jnp.zeros(selection.criterion.degree + 1)

    assert basis_width(result).shape == (3,)


def test_positional_construction_still_works(bandwidth):
    result = SelectionResult(bandwidth, jnp.asarray(1.0), jnp.asarray(3), jnp.asarray(True))
    assert result.criterion is None


def test_result_round_trips_through_jit(criteria_train, criteria_response):
    result = select_bandwidth(criteria_train, DensityCriterion(method="cv_ml"), n_starts=1)
    same = jax.jit(lambda r: r)(result)
    assert same.criterion == result.criterion
    assert jnp.array_equal(same.bandwidth.h, result.bandwidth.h)


@pytest.mark.parametrize("n_starts", [3, 5, 9])
def test_more_starts_does_not_blow_up(noiseless_train, noiseless_response, n_starts):
    one = select_bandwidth(noiseless_train, cv_ls_regression, y=noiseless_response, n_starts=1)
    many = select_bandwidth(noiseless_train, cv_ls_regression, y=noiseless_response, n_starts=n_starts)
    assert float(many.value) <= float(one.value) * 2.0
    assert bool(many.converged)


def test_a_stalled_start_does_not_win(noiseless_train, noiseless_response):
    criterion = RegressionCriterion(method="cv_ls", degree=1)
    result = select_bandwidth(noiseless_train, criterion, y=noiseless_response, n_starts=3)
    assert bool(result.converged)
    assert float(result.value) < 1e-5


def test_selection_matches_the_reference(selection_reference_train, selection_reference_y):
    criterion = RegressionCriterion(method="cv_ls", degree=1)

    result = select_bandwidth(selection_reference_train, criterion, y=selection_reference_y)

    assert float(result.bandwidth.h[0]) == pytest.approx(0.4391394172, rel=1e-4)
    assert float(result.bandwidth.lam_uno[0]) == pytest.approx(0.0467427679, rel=1e-4)


def test_select_bandwidth_takes_a_raw_array():
    x = jnp.linspace(-2.0, 2.0, 40)
    y = jnp.sin(x)

    result = select_bandwidth(x, RegressionCriterion(method="cv_ls", degree=1), y=y)

    assert result.bandwidth.h.shape == (1,)
    assert float(result.bandwidth.h[0]) > 0.0
