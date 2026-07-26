"""Tests for bandwidth selection."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.bandwidth import BandwidthTransform, normal_reference
from kerneljax.kernels import KernelSet
from kerneljax.tuning.objectives import cv_ls_density, cv_ml_density
from kerneljax.tuning.optimize import SelectionResult, lbfgs, select_bandwidth


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
def test_selected_value_is_no_worse_than_the_starting_value(cv_mixed_data, criterion, n_starts):
    start = normal_reference(cv_mixed_data, KernelSet())
    start_value = criterion(cv_mixed_data, start)

    result = select_bandwidth(cv_mixed_data, criterion, n_starts=n_starts)
    assert float(result.value) <= float(start_value) + 1e-6


def test_selection_is_vmappable_over_bootstrap_replicates(cv_mixed_data):
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


def test_a_user_supplied_solver_is_accepted_and_called(cv_mixed_data):
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
