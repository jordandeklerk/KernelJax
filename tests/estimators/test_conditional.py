"""Tests for conditional density and conditional distribution."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.bandwidth import ConditionalBandwidth
from kerneljax.data import MixedData
from kerneljax.estimators.conditional import ConditionalFit, cdensity, cdist, cv_ml_conditional
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


def _pin_first(x, rows):
    """Repeat the first conditioning row so a response grid can vary against it."""
    return MixedData.from_blocks(
        continuous=jnp.full((rows, 1), x.con[0, 0]),
        unordered=jnp.full((rows, 1), x.uno[0, 0]),
        unordered_levels=3,
    )
