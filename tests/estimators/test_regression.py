"""Tests for local polynomial regression."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth, normal_reference
from kerneljax.data import MixedData
from kerneljax.estimators.regression import LocalPolyFit, local_poly
from kerneljax.kernels import KernelSet
from kerneljax.ksum import kweights
from kerneljax.tuning.criteria import DensityCriterion, RegressionCriterion
from kerneljax.tuning.objectives import cv_ls_regression
from kerneljax.tuning.optimize import select_bandwidth


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
    assert np.max(np.abs(np.asarray(fit.mean) - want) / np.abs(want)) < 1e-5


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


def test_se_matches_manual_formula(poly_mixed_train, poly_mixed_bandwidth, poly_mixed_response):
    fit = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=1, se=True)

    weights = np.asarray(kweights(poly_mixed_train, poly_mixed_bandwidth))
    response = np.asarray(poly_mixed_response)
    weight_total = weights.sum(axis=1)
    mean_response = (weights * response[None, :]).sum(axis=1) / weight_total
    mean_square = (weights * (response**2)[None, :]).sum(axis=1) / weight_total
    variance = np.clip(mean_square - mean_response**2, 0.0, None)

    roughness = 1.0 / (2.0 * np.sqrt(np.pi))
    want = np.sqrt(variance * roughness / weight_total)

    assert np.allclose(np.asarray(fit.se), want, rtol=1e-6, atol=1e-5)


def test_se_nonnegative_and_finite(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, se=True)
    assert jnp.all(jnp.isfinite(fit.se))
    assert jnp.all(fit.se >= 0.0)


def test_se_zero_for_constant_response(poly_train, poly_at, poly_bandwidth):
    y = jnp.full((poly_train.n,), 1.0)
    fit = local_poly(poly_train, y, poly_bandwidth, at=poly_at, degree=1, se=True)
    assert jnp.allclose(fit.se, 0.0, atol=1e-3)


def test_se_shrinks_with_bandwidth(poly_flat_train, poly_at, poly_bandwidth, poly_flat_response):
    se_by_bandwidth = []
    for h in (0.3, 0.6, 1.2, 2.4):
        bandwidth = poly_bandwidth.replace(h=jnp.array([h]))
        fit = local_poly(poly_flat_train, poly_flat_response, bandwidth, at=poly_at, degree=1, se=True)
        se_by_bandwidth.append(np.asarray(fit.se))

    for smaller, larger in itertools.pairwise(se_by_bandwidth):
        assert np.all(larger < smaller)


def test_se_shrinks_with_sample_size(poly_growing_sample):
    trains, responses, at, bandwidth = poly_growing_sample
    se_by_size = [
        float(local_poly(train, y, bandwidth, at=at, degree=1, se=True).se[0])
        for train, y in zip(trains, responses, strict=True)
    ]

    assert se_by_size[0] > se_by_size[1] > se_by_size[2]


def test_se_none_without_request(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1)
    assert fit.se is None


def test_se_train_axis_gives_finite_values(poly_train, poly_at, poly_train_indexed_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_train_indexed_bandwidth, at=poly_at, degree=1, se=True)
    assert jnp.all(jnp.isfinite(fit.se))
    assert jnp.all(fit.se >= 0.0)


def test_roughness_matches_gaussian_constant(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, se=True)

    weights = np.asarray(kweights(poly_train, poly_bandwidth, at=poly_at))
    response = np.asarray(poly_response)
    weight_total = weights.sum(axis=1)
    mean_response = (weights * response[None, :]).sum(axis=1) / weight_total
    mean_square = (weights * (response**2)[None, :]).sum(axis=1) / weight_total
    variance = np.clip(mean_square - mean_response**2, 0.0, None)

    implied_roughness = np.asarray(fit.se) ** 2 * weight_total / variance
    want = 1.0 / (2.0 * np.sqrt(np.pi))
    assert np.allclose(implied_roughness, want, rtol=1e-4, atol=1e-4)


def test_roughness_is_product_over_columns(
    kweights_grad_purely_continuous_train, kweights_grad_purely_continuous_bandwidth
):
    train = kweights_grad_purely_continuous_train
    bandwidth = kweights_grad_purely_continuous_bandwidth
    rng = np.random.default_rng(36)
    y = jnp.asarray(rng.normal(size=train.n))

    fit = local_poly(train, y, bandwidth, degree=1, se=True)

    weights = np.asarray(kweights(train, bandwidth))
    response = np.asarray(y)
    weight_total = weights.sum(axis=1)
    mean_response = (weights * response[None, :]).sum(axis=1) / weight_total
    mean_square = (weights * (response**2)[None, :]).sum(axis=1) / weight_total
    variance = np.clip(mean_square - mean_response**2, 0.0, None)

    implied_roughness = np.asarray(fit.se) ** 2 * weight_total / variance
    want = (1.0 / (2.0 * np.sqrt(np.pi))) ** 2
    assert np.allclose(implied_roughness, want, rtol=1e-4, atol=1e-4)


def test_se_composes_with_gradient(poly_train, poly_at, poly_bandwidth, poly_response):
    fit = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, gradient=True, se=True)
    assert jnp.all(jnp.isfinite(fit.grad))
    assert jnp.all(jnp.isfinite(fit.se))


def test_se_composes_with_fold(poly_train, poly_bandwidth, poly_response):
    fold = jnp.arange(poly_train.n)
    fit = local_poly(poly_train, poly_response, poly_bandwidth, fold=fold, degree=1, se=True)
    assert jnp.all(jnp.isfinite(fit.se))
    assert jnp.all(fit.se >= 0.0)


@pytest.mark.parametrize("chunk", [1, 3, 5, (2, 7), (3, 40)])
def test_se_chunked_matches_unchunked(poly_train, poly_at, poly_bandwidth, poly_response, chunk):
    ref = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, se=True)
    got = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, se=True, chunk=chunk)

    assert jnp.allclose(got.se, ref.se, rtol=1e-6, atol=1e-8)


def test_se_pytree_survives_vmap(poly_train, poly_at, poly_bandwidth, poly_response):
    def run(h):
        return local_poly(poly_train, poly_response, poly_bandwidth.replace(h=h), at=poly_at, degree=1, se=True)

    fits = jax.vmap(run)(jnp.array([[0.3], [0.5], [0.8]]))

    assert isinstance(fits, LocalPolyFit)
    assert fits.se.shape == (3, poly_at.n)


def test_se_jit_grad_vmap(poly_train, poly_at, poly_bandwidth, poly_response):
    eager = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, se=True)
    assert jnp.all(jnp.isfinite(eager.se))

    jitted = jax.jit(lambda bw: local_poly(poly_train, poly_response, bw, at=poly_at, degree=1, se=True))
    assert jnp.allclose(jitted(poly_bandwidth).se, eager.se, rtol=1e-6)

    def total(bw):
        return local_poly(poly_train, poly_response, bw, at=poly_at, degree=1, se=True).se.sum()

    gradient = jax.grad(total)(poly_bandwidth)
    assert jnp.all(jnp.isfinite(gradient.h))

    out = jax.vmap(lambda h: total(poly_bandwidth.replace(h=h)))(jnp.array([[0.4], [0.6]]))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


def test_se_matches_equal_weight_closed_form(
    poly_degenerate_train, poly_degenerate_at, poly_degenerate_bandwidth, poly_degenerate_response
):
    fit = local_poly(
        poly_degenerate_train,
        poly_degenerate_response,
        poly_degenerate_bandwidth,
        at=poly_degenerate_at,
        degree=0,
        se=True,
    )

    response = np.asarray(poly_degenerate_response)
    variance = np.mean((response - response.mean()) ** 2)
    want = np.sqrt(variance / (response.shape[0] * np.sqrt(2.0)))

    assert np.allclose(np.asarray(fit.se), want, rtol=1e-6)


def test_se_scale_invariant_in_response(poly_train, poly_at, poly_bandwidth, poly_response):
    base = local_poly(poly_train, poly_response, poly_bandwidth, at=poly_at, degree=1, se=True)
    scaled = local_poly(poly_train, 3.0 * poly_response, poly_bandwidth, at=poly_at, degree=1, se=True)

    assert np.allclose(np.asarray(scaled.se), 3.0 * np.asarray(base.se), rtol=1e-5)


def test_positional_construction_still_works(bandwidth):
    fit = LocalPolyFit(jnp.zeros(3), None, jnp.zeros((3, 1)), jnp.ones(3), bandwidth, None)
    assert fit.selection is None
    assert fit.degree == 1
    assert fit.spec is None
    assert fit.n_train == 0


def test_fit_records_what_produced_it(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth):
    fit = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=2)
    assert fit.degree == 2
    assert fit.n_train == poly_mixed_train.n
    assert fit.spec == poly_mixed_train.spec


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_degree_changes_the_treedef(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree):
    fit = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=degree)
    other = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=degree + 1)
    assert jax.tree_util.tree_structure(fit) != jax.tree_util.tree_structure(other)


def test_fit_round_trips_through_jit(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth):
    fit = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=1, se=True)
    same = jax.jit(lambda f: f)(fit)
    assert same.degree == fit.degree
    assert same.n_train == fit.n_train
    assert same.spec == fit.spec
    assert jnp.array_equal(same.mean, fit.mean)


def test_static_fields_hold_no_leaves(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth):
    fit = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=1)
    leaves = jax.tree_util.tree_leaves(fit)
    assert len(leaves) == 6
    assert all(jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) for leaf in leaves)


def test_stacked_fits_vmap(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth):
    widths = jnp.array([[0.3], [0.5], [0.7]])

    def fit_at(h):
        return local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth.replace(h=h), degree=1)

    stacked = jax.vmap(fit_at)(widths)
    assert stacked.mean.shape == (3, poly_mixed_train.n)
    assert stacked.degree == 1
    assert stacked.n_train == poly_mixed_train.n


def test_attached_selection_blocks_grad(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth):
    fit = local_poly(poly_mixed_train, poly_mixed_response, poly_mixed_bandwidth, degree=1)
    assert jnp.all(jnp.isfinite(jax.grad(lambda f: f.mean.sum())(fit).mean))

    selection = select_bandwidth(poly_mixed_train, cv_ls_regression, y=poly_mixed_response, n_starts=1)
    with pytest.raises(TypeError, match="real- or complex-valued"):
        jax.grad(lambda f: f.mean.sum())(dataclasses.replace(fit, selection=selection))


@pytest.mark.parametrize("method", ["cv_ls", "aic"])
def test_string_bw_matches_the_two_step(criteria_train, criteria_response, method):
    criterion = RegressionCriterion(method=method)
    selection = select_bandwidth(criteria_train, criterion, y=criteria_response)
    want = local_poly(criteria_train, criteria_response, selection.bandwidth)

    got = local_poly(criteria_train, criteria_response, method)
    assert jnp.allclose(got.mean, want.mean)
    assert got.selection is not None


def test_normal_reference_matches_the_rule(criteria_train, criteria_response):
    want = normal_reference(criteria_train, KernelSet())
    got = local_poly(criteria_train, criteria_response, "normal_reference")
    assert jnp.allclose(got.bandwidth.h, want.h)
    assert got.selection is None


def test_reusing_a_fit_matches_its_bandwidth(criteria_train, criteria_response):
    first = local_poly(criteria_train, criteria_response, "cv_ls")
    again = local_poly(criteria_train, criteria_response, first)
    assert jnp.array_equal(again.bandwidth.h, first.bandwidth.h)
    assert again.degree == first.degree


def test_array_train_matches_mixed_data(criteria_response):
    raw = jnp.linspace(-2.0, 2.0, criteria_response.shape[0])
    bw = Bandwidth(h=jnp.array([0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    want = local_poly(MixedData.continuous(raw), criteria_response, bw)
    got = local_poly(raw, criteria_response, bw)
    assert jnp.allclose(got.mean, want.mean)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_agreeing_degree_is_accepted(criteria_train, criteria_response, degree):
    first = local_poly(criteria_train, criteria_response, "cv_ls", degree=degree)
    again = local_poly(criteria_train, criteria_response, first, degree=degree)
    assert again.degree == degree


def test_contradicting_degree_raises(criteria_train, criteria_response):
    first = local_poly(criteria_train, criteria_response, "cv_ls", degree=0)
    with pytest.raises(ValueError, match="contradicts"):
        local_poly(criteria_train, criteria_response, first, degree=2)


@pytest.mark.parametrize("bad", [0.4, None, ("cv_ls",), jnp.array([0.4]), np.array([0.4, 0.5])])
def test_unsupported_bw_type_raises(criteria_train, criteria_response, bad):
    with pytest.raises(TypeError, match="bw must be"):
        local_poly(criteria_train, criteria_response, bad)


def test_wrong_family_criterion_is_rejected(criteria_train, criteria_response):
    with pytest.raises(TypeError, match="bw must be"):
        local_poly(criteria_train, criteria_response, DensityCriterion(method="cv_ml"))


def test_unknown_string_bw_raises(criteria_train, criteria_response):
    with pytest.raises(ValueError, match="normal_reference"):
        local_poly(criteria_train, criteria_response, "cv.ls")


def test_array_at_needs_a_continuous_spec(cv_mixed_data, regression_mixed_response, cv_mixed_bandwidth):
    with pytest.raises(TypeError, match="categorical"):
        local_poly(cv_mixed_data, regression_mixed_response, cv_mixed_bandwidth, at=jnp.zeros((3, 1)))


def test_reusing_a_non_shared_bandwidth_raises(criteria_train, criteria_response):
    per_row = Bandwidth(
        h=jnp.full((criteria_train.n, 1), 0.4),
        lam_uno=jnp.zeros(0),
        lam_ord=jnp.zeros(0),
        h_axis="eval",
    )
    fit = local_poly(criteria_train, criteria_response, per_row)
    with pytest.raises(ValueError, match="h_axis 'shared'"):
        local_poly(criteria_train, criteria_response, fit, at=jnp.zeros((3, 1)))
