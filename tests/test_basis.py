"""Tests for the polynomial design basis."""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth
from kerneljax.basis import LocalPolyBasis
from kerneljax.data import ColumnSpec, Kind, MixedData
from kerneljax.ksum import kweights


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("p_con", [0, 1, 2, 3, 5])
def test_dim_matches_the_binomial_formula(p_con, degree):
    spec = ColumnSpec(kinds=(Kind.CONTINUOUS,) * p_con, n_levels=(0,) * p_con)
    assert LocalPolyBasis(degree=degree).dim(spec, degree) == math.comb(p_con + degree, degree)


def test_dim_ignores_categorical_columns(column_spec):
    assert LocalPolyBasis(degree=2).dim(column_spec, 2) == math.comb(column_spec.p_con + 2, 2)


def test_degree_zero_gives_a_single_column_of_ones(basis_train, basis_at, basis_bandwidth):
    design = LocalPolyBasis(degree=0).design(basis_train, basis_at, basis_bandwidth)

    assert design.shape == (basis_train.n, 1)
    assert jnp.allclose(design, jnp.ones_like(design))


def test_degree_one_matches_an_explicit_numpy_construction(basis_train, basis_at, basis_bandwidth):
    train = np.asarray(basis_train.con, dtype=np.float64)
    at = np.asarray(basis_at.con, dtype=np.float64)
    h = np.asarray(basis_bandwidth.h, dtype=np.float64)
    u = (at - train) / h

    expected = np.concatenate([np.ones((train.shape[0], 1)), u], axis=1)
    got = LocalPolyBasis(degree=1).design(basis_train, basis_at, basis_bandwidth)
    assert np.allclose(np.asarray(got), expected, rtol=1e-6)


def test_degree_two_includes_every_square_and_cross_product_exactly_once(basis_train, basis_at, basis_bandwidth):
    train = np.asarray(basis_train.con, dtype=np.float64)
    at = np.asarray(basis_at.con, dtype=np.float64)
    h = np.asarray(basis_bandwidth.h, dtype=np.float64)
    u = (at - train) / h

    expected = np.stack(
        [np.ones(train.shape[0]), u[:, 0], u[:, 1], u[:, 0] ** 2, u[:, 0] * u[:, 1], u[:, 1] ** 2],
        axis=1,
    )

    basis = LocalPolyBasis(degree=2)
    got = np.asarray(basis.design(basis_train, basis_at, basis_bandwidth))

    assert got.shape[1] == basis.dim(basis_train.spec, 2)
    assert np.allclose(got, expected, rtol=1e-6)

    for first, second in itertools.combinations(range(got.shape[1]), 2):
        assert not np.allclose(got[:, first], got[:, second])


@pytest.mark.parametrize("degree", [1, 2])
def test_design_at_the_evaluation_point_itself_has_a_constant_first_column(basis_train, basis_bandwidth, degree):
    at = MixedData.continuous(basis_train.con[:1])
    design = LocalPolyBasis(degree=degree).design(basis_train, at, basis_bandwidth)

    assert design[0, 0] == pytest.approx(1.0)
    assert jnp.allclose(design[0, 1:], 0.0)


@pytest.mark.parametrize("scale", [1.0, 100.0, 1000.0])
def test_condition_number_of_the_weighted_gram_stays_bounded_across_data_scales(scale):
    rng = np.random.default_rng(4)
    points = rng.uniform(-scale, scale, size=40).reshape(-1, 1)
    train = MixedData.continuous(jnp.asarray(points))
    at = MixedData.continuous(jnp.asarray(points[:1]))
    bw = Bandwidth(h=jnp.array([0.3 * scale]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

    design = np.asarray(LocalPolyBasis(degree=2).design(train, at, bw), dtype=np.float64)
    weights = np.asarray(kweights(train, bw, at=at), dtype=np.float64)[0]
    gram = design.T @ (weights[:, None] * design)

    assert np.linalg.cond(gram) < 100.0


@pytest.mark.parametrize("var", [0, 1])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_deriv_order_one_matches_jacfwd_of_design(basis_train, basis_at, basis_bandwidth, degree, var):
    basis = LocalPolyBasis(degree=degree)
    coordinate = basis_at.con[0]

    def design_at(value):
        at = MixedData.continuous(coordinate.at[var].set(value)[None, :])
        return basis.design(basis_train, at, basis_bandwidth)

    jacobian = jax.jacfwd(design_at)(coordinate[var])
    analytic = basis.deriv(basis_train, basis_at, basis_bandwidth, var, 1)

    assert jnp.allclose(jacobian, analytic, rtol=1e-6)


@pytest.mark.parametrize("var", [0, 1])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_deriv_order_two_matches_the_second_jacfwd_of_design(basis_train, basis_at, basis_bandwidth, degree, var):
    basis = LocalPolyBasis(degree=degree)
    coordinate = basis_at.con[0]

    def first_order_at(value):
        at = MixedData.continuous(coordinate.at[var].set(value)[None, :])
        return basis.deriv(basis_train, at, basis_bandwidth, var, 1)

    jacobian = jax.jacfwd(first_order_at)(coordinate[var])
    analytic = basis.deriv(basis_train, basis_at, basis_bandwidth, var, 2)

    assert jnp.allclose(jacobian, analytic, rtol=1e-6)


@pytest.mark.parametrize("var", [0, 1])
@pytest.mark.parametrize("degree", [1, 2])
def test_deriv_stays_finite_when_the_evaluation_point_coincides_with_a_training_point(
    basis_train, basis_bandwidth, degree, var
):
    at = MixedData.continuous(basis_train.con[:1])
    basis = LocalPolyBasis(degree=degree)

    first_order = basis.deriv(basis_train, at, basis_bandwidth, var, 1)
    second_order = basis.deriv(basis_train, at, basis_bandwidth, var, 2)

    assert jnp.all(jnp.isfinite(first_order))
    assert jnp.all(jnp.isfinite(second_order))


def test_local_poly_basis_carries_no_pytree_leaves():
    leaves = jax.tree_util.tree_leaves(LocalPolyBasis(degree=1))
    assert leaves == []


def test_two_instances_with_the_same_degree_compare_and_hash_equal():
    first = LocalPolyBasis(degree=2)
    second = LocalPolyBasis(degree=2)

    assert first == second
    assert hash(first) == hash(second)


def test_jit_accepts_the_basis_without_static_argnames(basis_train, basis_at, basis_bandwidth):
    def total(basis, train, at, bw):
        return basis.design(train, at, bw).sum()

    result = jax.jit(total)(LocalPolyBasis(degree=1), basis_train, basis_at, basis_bandwidth)
    assert jnp.isfinite(result)


def test_jit_does_not_retrace_for_a_new_instance_with_the_same_degree(basis_train, basis_at, basis_bandwidth):
    calls = {"n": 0}

    @jax.jit
    def total(basis, train, at, bw):
        calls["n"] += 1
        return basis.design(train, at, bw).sum()

    total(LocalPolyBasis(degree=1), basis_train, basis_at, basis_bandwidth)
    total(LocalPolyBasis(degree=1), basis_train, basis_at, basis_bandwidth)
    assert calls["n"] == 1


def test_jit_accepts_deriv_with_var_and_order_marked_static(basis_train, basis_at, basis_bandwidth):
    basis = LocalPolyBasis(degree=2)

    total = jax.jit(
        lambda train, at, bw, var, order: basis.deriv(train, at, bw, var, order).sum(),
        static_argnums=(3, 4),
    )
    result = total(basis_train, basis_at, basis_bandwidth, 0, 1)
    assert jnp.isfinite(result)


def test_grad_over_the_bandwidth_is_finite(basis_train, basis_at, basis_bandwidth):
    def total(h):
        return LocalPolyBasis(degree=2).design(basis_train, basis_at, basis_bandwidth.replace(h=h)).sum()

    gradient = jax.grad(total)(basis_bandwidth.h)
    assert jnp.all(jnp.isfinite(gradient))


def test_vmap_over_the_bandwidth_is_finite(basis_train, basis_at, basis_bandwidth):
    def total(h):
        return LocalPolyBasis(degree=2).design(basis_train, basis_at, basis_bandwidth.replace(h=h)).sum()

    out = jax.vmap(total)(jnp.array([[0.4, 0.6], [0.5, 0.7], [0.6, 0.9]]))
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_design_under_vmap_over_evaluation_points_gives_a_leading_batch_axis(basis_train, basis_bandwidth):
    basis = LocalPolyBasis(degree=1)
    eval_points = jnp.array([[0.1, -0.2], [0.3, 0.4], [0.0, 0.0]])

    def design_at_one(point):
        at = MixedData.continuous(point[None, :])
        return basis.design(basis_train, at, basis_bandwidth)

    batched = jax.vmap(design_at_one)(eval_points)
    assert batched.shape == (3, basis_train.n, basis.dim(basis_train.spec, 1))
