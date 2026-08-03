"""Tests for mixed-type cumulative distribution estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth, normal_reference
from kerneljax.data import MixedData
from kerneljax.estimators.density import density
from kerneljax.estimators.distribution import DistributionFit, _cdf_values, cdf
from kerneljax.kernels import KernelSet
from kerneljax.tuning.criteria import DistributionCriterion
from kerneljax.tuning.optimize import select_bandwidth


def test_estimate_is_monotone_and_in_the_unit(criteria_train, criteria_bandwidth):
    grid = MixedData.continuous(jnp.linspace(-3.0, 3.0, 40).reshape(-1, 1))
    fit = cdf(criteria_train, criteria_bandwidth, at=grid)
    assert jnp.all(jnp.diff(fit.value) >= -1e-6)
    assert jnp.all(fit.value >= 0.0)
    assert jnp.all(fit.value <= 1.0)


def test_estimate_spans_the_unit_far_out(criteria_train, criteria_bandwidth):
    far = MixedData.continuous(jnp.array([[-60.0], [60.0]]))
    fit = cdf(criteria_train, criteria_bandwidth, at=far)
    assert float(fit.value[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(fit.value[1]) == pytest.approx(1.0, abs=1e-6)


def test_se_is_the_binomial_form(criteria_train, criteria_bandwidth):
    fit = cdf(criteria_train, criteria_bandwidth)
    want = jnp.sqrt(fit.value * (1.0 - fit.value) / criteria_train.n)
    assert jnp.allclose(fit.se, want, rtol=1e-6)


def test_no_bandwidth_divisor_is_applied(criteria_train):
    narrow = Bandwidth(h=jnp.array([0.05]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    wide = Bandwidth(h=jnp.array([2.0]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    for bw in (narrow, wide):
        assert jnp.all(cdf(criteria_train, bw).value <= 1.0)


def test_converges_to_the_empirical_cdf(criteria_train):
    tiny = Bandwidth(h=jnp.array([1e-4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    got = cdf(criteria_train, tiny).value

    column = np.asarray(criteria_train.con)[:, 0]
    want = np.array([np.mean(column < x) + 0.5 * np.mean(column == x) for x in column])
    assert np.max(np.abs(np.asarray(got) - want)) < 1e-3


def test_unordered_columns_are_rejected(cv_mixed_data, cv_mixed_bandwidth):
    with pytest.raises(ValueError, match="unordered"):
        cdf(cv_mixed_data, cv_mixed_bandwidth)


def test_ordered_columns_are_accepted(grid_sample):
    data = MixedData.from_blocks(continuous=grid_sample.con, ordered=grid_sample.orde, ordered_levels=(4,))
    bw = Bandwidth(h=jnp.array([0.4, 0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.array([0.3]))
    fit = cdf(data, bw)
    assert jnp.all(jnp.isfinite(fit.value))
    assert jnp.all(fit.value <= 1.0)


def test_fit_records_what_produced_it(criteria_train, criteria_bandwidth):
    fit = cdf(criteria_train, criteria_bandwidth)
    assert fit.n_train == criteria_train.n
    assert fit.spec == criteria_train.spec
    assert fit.selection is None


def test_accepts_a_raw_array(criteria_train, criteria_bandwidth):
    raw = criteria_train.con[:, 0]
    assert jnp.allclose(
        cdf(raw, criteria_bandwidth).value,
        cdf(criteria_train, criteria_bandwidth).value,
    )


@pytest.mark.parametrize("chunk", [4, 7, (3, 5)])
def test_chunked_matches_unchunked(criteria_train, criteria_bandwidth, chunk):
    ref = cdf(criteria_train, criteria_bandwidth).value
    got = cdf(criteria_train, criteria_bandwidth, chunk=chunk).value
    assert jnp.allclose(got, ref, rtol=1e-6)


def test_jit_grad_and_vmap(criteria_train, criteria_bandwidth):
    fitted = jax.jit(cdf)
    assert jnp.all(jnp.isfinite(fitted(criteria_train, criteria_bandwidth).value))

    grads = jax.grad(lambda b: cdf(criteria_train, b).value.sum())(criteria_bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))

    out = jax.vmap(lambda h: cdf(criteria_train, criteria_bandwidth.replace(h=h)).value)(
        jnp.array([[0.3], [0.5], [0.7]])
    )
    assert out.shape == (3, criteria_train.n)


def test_round_trips_through_jit(criteria_train, criteria_bandwidth):
    fit = cdf(criteria_train, criteria_bandwidth)
    same = jax.jit(lambda f: f)(fit)
    assert isinstance(same, DistributionFit)
    assert same.n_train == fit.n_train
    assert jnp.array_equal(same.value, fit.value)


def test_cv_cdf_matches_the_two_step(cdf_train):
    selection = select_bandwidth(cdf_train, DistributionCriterion())
    want = cdf(cdf_train, selection.bandwidth)

    got = cdf(cdf_train, "cv_cdf")
    assert jnp.allclose(got.value, want.value)
    assert got.selection is not None


def test_normal_reference_uses_the_distribution_target(cdf_train):
    want = normal_reference(cdf_train, KernelSet(), target="distribution")
    got = cdf(cdf_train, "normal_reference")
    assert jnp.allclose(got.bandwidth.h, want.h)
    assert got.selection is None


def test_normal_reference_is_not_the_density_rule(cdf_train):
    density_rule = normal_reference(cdf_train, KernelSet())
    got = cdf(cdf_train, "normal_reference")
    assert not jnp.allclose(got.bandwidth.h, density_rule.h)


def test_reusing_a_fit_matches_its_bandwidth(cdf_train):
    first = cdf(cdf_train, "cv_cdf")
    again = cdf(cdf_train, first)
    assert jnp.array_equal(again.bandwidth.h, first.bandwidth.h)


def test_reusing_a_selection_matches_its_bandwidth(cdf_train):
    first = cdf(cdf_train, "cv_cdf")
    again = cdf(cdf_train, first.selection)
    assert jnp.array_equal(again.bandwidth.h, first.bandwidth.h)


@pytest.mark.parametrize("bad", [0.4, None, ("cv_cdf",), jnp.array([0.4])])
def test_unsupported_bw_type_raises(cdf_train, bad):
    with pytest.raises(TypeError, match="bw must be"):
        cdf(cdf_train, bad)


@pytest.mark.parametrize("method", ["cv_ls", "cv_ml", "aic", ""])
def test_unknown_method_raises(cdf_train, method):
    with pytest.raises(ValueError, match="normal_reference"):
        cdf(cdf_train, method)


def test_wrong_family_fit_is_rejected(cdf_train, cdf_bandwidth):
    other = density(cdf_train, cdf_bandwidth)
    with pytest.raises(TypeError, match="bw must be"):
        cdf(cdf_train, other)


def test_reusing_a_non_shared_bandwidth_raises(criteria_train, criteria_response):
    per_row = Bandwidth(
        h=jnp.full((criteria_train.n, 1), 0.4),
        lam_uno=jnp.zeros(0),
        lam_ord=jnp.zeros(0),
        h_axis="eval",
    )
    fit = cdf(criteria_train, per_row)
    with pytest.raises(ValueError, match="h_axis 'shared'"):
        cdf(criteria_train, fit, at=jnp.zeros((3, 1)))


def test_cdf_reuses_one_compile(criteria_train, criteria_bandwidth):
    _cdf_values.clear_cache()

    for h in (0.4, 0.5, 0.6, 0.7):
        cdf(criteria_train, criteria_bandwidth.replace(h=jnp.array([h])))

    assert _cdf_values._cache_size() == 1
