"""Tests for conditional density and conditional distribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import ConditionalBandwidth
from kerneljax.data import MixedData
from kerneljax.estimators.conditional import (
    ConditionalFit,
    QuantileFit,
    cdensity,
    cdist,
    cquantile,
    cv_ls_conditional,
    cv_ml_conditional,
)
from kerneljax.ksum import kweights


def test_conditional_density_matches_the_ratio_it_is_defined_as(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    got = cdensity(x, y, conditional_bandwidth).value

    weights_x = kweights(x, conditional_bandwidth.x)
    weights_y = kweights(y, conditional_bandwidth.y)
    want = jnp.sum(weights_x * weights_y, axis=1) / jnp.prod(conditional_bandwidth.y.h) / jnp.sum(weights_x, axis=1)

    assert jnp.allclose(got, want, rtol=1e-6)


def test_conditional_distribution_integrates_the_response_kernel(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    got = cdist(x, y, conditional_bandwidth).value

    weights_x = kweights(x, conditional_bandwidth.x)
    accumulated = kweights(y, conditional_bandwidth.y, op="cdf")
    want = jnp.sum(weights_x * accumulated, axis=1) / jnp.sum(weights_x, axis=1)

    assert jnp.allclose(got, want, rtol=1e-6)


def test_a_conditional_density_is_positive(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample
    assert jnp.all(cdensity(x, y, conditional_bandwidth).value > 0.0)


def test_a_conditional_distribution_stays_in_the_unit_interval(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    value = cdist(x, y, conditional_bandwidth).value

    assert jnp.all(value >= 0.0)
    assert jnp.all(value <= 1.0)


def test_the_distribution_is_the_running_total_of_the_density(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample
    grid = jnp.linspace(-8.0, 8.0, 8001)
    pinned = _pin_first(x, grid.size)
    one_row = _pin_first(x, 1)

    density = cdensity(x, y, conditional_bandwidth, at_x=pinned, at_y=grid[:, None]).value
    far = cdist(x, y, conditional_bandwidth, at_x=one_row, at_y=jnp.array([[8.0]])).value

    assert float(jnp.trapezoid(density, grid)) == pytest.approx(1.0, abs=1e-3)
    assert float(far[0]) == pytest.approx(1.0, abs=1e-3)


def test_a_reference_rule_and_a_search_both_produce_a_fit(conditional_sample):
    x, y = conditional_sample

    reference = cdensity(x, y, "normal_reference")
    searched = cdensity(x, y, "cv_ml", n_starts=1)

    assert reference.selection is None
    assert searched.selection is not None
    assert bool(jnp.all(jnp.isfinite(searched.value)))


def test_a_search_beats_its_own_starting_point(conditional_sample):
    x, y = conditional_sample

    searched = cdensity(x, y, "cv_ml", n_starts=1)
    start = cdensity(x, y, "normal_reference").bandwidth

    assert float(searched.selection.value) <= float(cv_ml_conditional(x, y, start))


def test_a_selection_carries_its_bandwidth_back_into_a_fit(conditional_sample):
    x, y = conditional_sample

    searched = cdensity(x, y, "cv_ml", n_starts=1)
    reused = cdensity(x, y, searched.selection)

    assert jnp.allclose(reused.value, searched.value)


def test_a_fit_can_be_handed_back_in_place_of_its_bandwidth(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    first = cdensity(x, y, conditional_bandwidth)
    again = cdensity(x, y, first)

    assert jnp.allclose(again.value, first.value)


def test_mismatched_sample_sizes_are_refused(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    shorter = MixedData.continuous(y.con[:-1])

    with pytest.raises(ValueError, match="same sample"):
        cdensity(x, shorter, conditional_bandwidth)


def test_an_unknown_rule_name_is_refused(conditional_sample):
    x, y = conditional_sample

    with pytest.raises(ValueError, match="cv_ml"):
        cdensity(x, y, "cv_ls")


def test_likelihood_selection_is_refused_for_a_distribution(conditional_sample):
    x, y = conditional_sample

    with pytest.raises(ValueError, match="oversmoothing"):
        cdist(x, y, "cv_ml")


def test_a_density_selection_carries_into_a_distribution(conditional_sample):
    x, y = conditional_sample

    fit = cdensity(x, y, "cv_ml", n_starts=1)
    carried = cdist(x, y, fit)

    assert carried.bandwidth is fit.bandwidth
    assert bool(jnp.all(carried.value >= 0.0))
    assert bool(jnp.all(carried.value <= 1.0))


def test_the_fit_is_a_pytree_carrying_static_metadata(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    fit = cdensity(x, y, conditional_bandwidth)
    leaves = jax.tree_util.tree_leaves(fit)

    assert isinstance(fit, ConditionalFit)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert fit.n_train == x.n
    assert fit.target == "density"


def test_selection_differentiates_through_both_blocks(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    grad = jax.grad(lambda bw: cv_ml_conditional(x, y, bw))(conditional_bandwidth)

    assert isinstance(grad, ConditionalBandwidth)
    assert bool(jnp.all(jnp.isfinite(grad.x.h)))
    assert bool(jnp.all(jnp.isfinite(grad.y.h)))
    assert bool(jnp.all(jnp.isfinite(grad.x.lam_uno)))


def test_least_squares_matches_a_dense_reimplementation(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample
    n = x.n

    got = float(cv_ls_conditional(x, y, conditional_bandwidth))

    grid = np.quantile(np.asarray(y.con[:, 0]), np.linspace(0.0, 1.0, 100))
    weights_x = np.asarray(kweights(x, conditional_bandwidth.x))
    accumulated = np.asarray(kweights(y, conditional_bandwidth.y, at=MixedData.continuous(grid), op="cdf"))
    total = 0.0
    for i in range(n):
        w = weights_x[i].copy()
        w[i] = 0.0
        for g in range(100):
            f = float(np.dot(w, accumulated[g]) / w.sum())
            total += (float(y.con[i, 0] <= grid[g]) - f) ** 2

    assert got == pytest.approx(total / (n * 100), rel=1e-5)


def test_a_distribution_selects_under_least_squares(conditional_sample):
    x, y = conditional_sample

    fit = cdist(x, y, "cv_ls", n_starts=1)

    assert fit.selection is not None
    assert bool(jnp.isfinite(fit.selection.value))
    assert bool(jnp.all(jnp.isfinite(fit.value)))


def test_least_squares_beats_its_own_starting_point(conditional_sample):
    x, y = conditional_sample

    fit = cdist(x, y, "cv_ls", n_starts=1)
    start = cdist(x, y, "normal_reference").bandwidth

    assert float(fit.selection.value) <= float(cv_ls_conditional(x, y, start))


def test_an_explicit_grid_of_the_quantiles_matches_the_default(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    grid = jnp.quantile(y.con, jnp.linspace(0.0, 1.0, 100), axis=0)
    explicit = cv_ls_conditional(x, y, conditional_bandwidth, y_grid=grid)
    default = cv_ls_conditional(x, y, conditional_bandwidth)

    assert float(explicit) == pytest.approx(float(default))


def test_least_squares_is_refused_for_a_density(conditional_sample):
    x, y = conditional_sample

    with pytest.raises(ValueError, match="selects for cdist"):
        cdensity(x, y, "cv_ls")


def test_a_categorical_response_needs_an_explicit_grid(conditional_sample, conditional_bandwidth):
    x, _ = conditional_sample
    codes = np.asarray(x.uno[:, 0])
    y = MixedData.from_blocks(ordered=codes, ordered_levels=3)

    with pytest.raises(ValueError, match="y_grid"):
        cv_ls_conditional(x, y, conditional_bandwidth)


def test_least_squares_differentiates_through_both_blocks(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    grad = jax.grad(lambda bw: cv_ls_conditional(x, y, bw))(conditional_bandwidth)

    assert bool(jnp.all(jnp.isfinite(grad.x.h)))
    assert bool(jnp.all(jnp.isfinite(grad.y.h)))
    assert bool(jnp.all(jnp.isfinite(grad.x.lam_uno)))


def test_the_quantile_inverts_the_distribution(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    fit = cquantile(x, y, conditional_bandwidth, tau=0.5)
    at_quantile = cdist(x, y, conditional_bandwidth, at_x=x, at_y=fit.value[:, None]).value

    assert bool(jnp.all(jnp.abs(at_quantile - 0.5) < 1e-3))


def test_quantiles_increase_with_tau(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    lower = cquantile(x, y, conditional_bandwidth, tau=0.25).value
    middle = cquantile(x, y, conditional_bandwidth, tau=0.5).value
    upper = cquantile(x, y, conditional_bandwidth, tau=0.75).value

    assert bool(jnp.all(lower <= middle))
    assert bool(jnp.all(middle <= upper))


def test_extreme_levels_clamp_to_the_response_range(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    bottom = cquantile(x, y, conditional_bandwidth, tau=0.001).value
    top = cquantile(x, y, conditional_bandwidth, tau=0.999).value

    assert bool(jnp.all(bottom >= jnp.min(y.con)))
    assert bool(jnp.all(top <= jnp.max(y.con)))


def test_a_distribution_fit_hands_its_bandwidth_to_the_quantile(conditional_sample):
    x, y = conditional_sample

    selected = cdist(x, y, "cv_ls", n_starts=1)
    fit = cquantile(x, y, selected, tau=0.5)

    assert fit.bandwidth is selected.bandwidth
    assert fit.selection is selected.selection


def test_likelihood_selection_is_refused_for_quantiles(conditional_sample):
    x, y = conditional_sample

    with pytest.raises(ValueError, match="oversmoothing"):
        cquantile(x, y, "cv_ml")


def test_a_multicolumn_response_is_refused_by_the_quantile(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample
    wide = MixedData.continuous(jnp.column_stack([y.con[:, 0], y.con[:, 0]]))

    with pytest.raises(ValueError, match="single continuous column"):
        cquantile(x, wide, conditional_bandwidth)


def test_a_level_outside_the_unit_interval_is_refused(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    with pytest.raises(ValueError, match="strictly between"):
        cquantile(x, y, conditional_bandwidth, tau=1.0)


def test_the_quantile_fit_is_a_pytree(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    fit = cquantile(x, y, conditional_bandwidth)
    leaves = jax.tree_util.tree_leaves(fit)

    assert isinstance(fit, QuantileFit)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert fit.tau == 0.5


def _pin_first(x, rows):
    """Repeat the first conditioning row so a response grid can vary against it."""
    return MixedData.from_blocks(
        continuous=jnp.full((rows, 1), x.con[0, 0]),
        unordered=jnp.full((rows, 1), x.uno[0, 0]),
        unordered_levels=3,
    )
