"""Tests for mixed-type cumulative distribution estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import MixedData
from kerneljax.estimators.distribution import DistributionFit, distribution


def test_estimate_is_monotone_and_in_the_unit(criteria_train, criteria_bandwidth):
    grid = MixedData.continuous(jnp.linspace(-3.0, 3.0, 40).reshape(-1, 1))
    fit = distribution(criteria_train, criteria_bandwidth, at=grid)
    assert jnp.all(jnp.diff(fit.value) >= -1e-6)
    assert jnp.all(fit.value >= 0.0)
    assert jnp.all(fit.value <= 1.0)


def test_estimate_spans_the_unit_far_out(criteria_train, criteria_bandwidth):
    far = MixedData.continuous(jnp.array([[-60.0], [60.0]]))
    fit = distribution(criteria_train, criteria_bandwidth, at=far)
    assert float(fit.value[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(fit.value[1]) == pytest.approx(1.0, abs=1e-6)


def test_se_is_the_binomial_form(criteria_train, criteria_bandwidth):
    fit = distribution(criteria_train, criteria_bandwidth)
    want = jnp.sqrt(fit.value * (1.0 - fit.value) / criteria_train.n)
    assert jnp.allclose(fit.se, want, rtol=1e-6)


def test_no_bandwidth_divisor_is_applied(criteria_train):
    narrow = Bandwidth(h=jnp.array([0.05]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    wide = Bandwidth(h=jnp.array([2.0]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    for bw in (narrow, wide):
        assert jnp.all(distribution(criteria_train, bw).value <= 1.0)


def test_converges_to_the_empirical_cdf(criteria_train):
    tiny = Bandwidth(h=jnp.array([1e-4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    got = distribution(criteria_train, tiny).value

    column = np.asarray(criteria_train.con)[:, 0]
    want = np.array([np.mean(column < x) + 0.5 * np.mean(column == x) for x in column])
    assert np.max(np.abs(np.asarray(got) - want)) < 1e-3


def test_unordered_columns_are_rejected(cv_mixed_data, cv_mixed_bandwidth):
    with pytest.raises(ValueError, match="unordered"):
        distribution(cv_mixed_data, cv_mixed_bandwidth)


def test_ordered_columns_are_accepted(grid_sample):
    data = MixedData.from_blocks(con=grid_sample.con, orde=grid_sample.orde, ord_levels=(4,))
    bw = Bandwidth(h=jnp.array([0.4, 0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.array([0.3]))
    fit = distribution(data, bw)
    assert jnp.all(jnp.isfinite(fit.value))
    assert jnp.all(fit.value <= 1.0)


def test_fit_records_what_produced_it(criteria_train, criteria_bandwidth):
    fit = distribution(criteria_train, criteria_bandwidth)
    assert fit.n_train == criteria_train.n
    assert fit.spec == criteria_train.spec
    assert fit.selection is None


def test_accepts_a_raw_array(criteria_train, criteria_bandwidth):
    raw = criteria_train.con[:, 0]
    assert jnp.allclose(
        distribution(raw, criteria_bandwidth).value,
        distribution(criteria_train, criteria_bandwidth).value,
    )


@pytest.mark.parametrize("chunk", [4, 7, (3, 5)])
def test_chunked_matches_unchunked(criteria_train, criteria_bandwidth, chunk):
    ref = distribution(criteria_train, criteria_bandwidth).value
    got = distribution(criteria_train, criteria_bandwidth, chunk=chunk).value
    assert jnp.allclose(got, ref, rtol=1e-6)


def test_jit_grad_and_vmap(criteria_train, criteria_bandwidth):
    fitted = jax.jit(distribution)
    assert jnp.all(jnp.isfinite(fitted(criteria_train, criteria_bandwidth).value))

    grads = jax.grad(lambda b: distribution(criteria_train, b).value.sum())(criteria_bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))

    out = jax.vmap(lambda h: distribution(criteria_train, criteria_bandwidth.replace(h=h)).value)(
        jnp.array([[0.3], [0.5], [0.7]])
    )
    assert out.shape == (3, criteria_train.n)


def test_round_trips_through_jit(criteria_train, criteria_bandwidth):
    fit = distribution(criteria_train, criteria_bandwidth)
    same = jax.jit(lambda f: f)(fit)
    assert isinstance(same, DistributionFit)
    assert same.n_train == fit.n_train
    assert jnp.array_equal(same.value, fit.value)
