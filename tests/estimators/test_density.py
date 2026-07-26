"""Tests for the mixed-type density estimator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import MixedData
from kerneljax.estimators.density import DensityFit, density
from kerneljax.ksum import kweights


def test_returns_a_record_not_a_bare_array(density_data, density_bandwidth):
    fit = density(density_data, density_bandwidth)
    assert isinstance(fit, DensityFit)
    assert fit.value.shape == (density_data.n,)
    assert fit.bandwidth is density_bandwidth


def test_matches_an_independent_numpy_expression(density_data, density_bandwidth):
    got = np.asarray(density(density_data, density_bandwidth).value)

    con = np.asarray(density_data.con)
    uno = np.asarray(density_data.uno)
    orde = np.asarray(density_data.orde)
    h = float(density_bandwidth.h[0])
    lam_uno = float(density_bandwidth.lam_uno[0])
    lam_ord = float(density_bandwidth.lam_ord[0])
    n = con.shape[0]

    diff = (con - con.T) / h
    gaussian = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
    aitchison = np.where(uno == uno.T, 1.0 - lam_uno, lam_uno / 2.0)
    dist = np.abs(orde - orde.T)
    wang = np.where(dist == 0, 1.0 - lam_ord, 0.5 * (1.0 - lam_ord) * lam_ord**dist)

    want = (gaussian * aitchison * wang).sum(axis=1) / (n * h)
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-6


def test_continuous_only_density_integrates_to_approximately_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    data = MixedData.continuous(jnp.asarray(x))
    grid = np.linspace(-6.0, 6.0, 4001).reshape(-1, 1)
    at = MixedData.continuous(jnp.asarray(grid))
    bw = Bandwidth(h=jnp.array([0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

    value = np.asarray(density(data, bw, at=at).value)
    assert np.trapezoid(value, grid[:, 0]) == pytest.approx(1.0, abs=2e-3)


def test_gaussian_estimate_is_strictly_positive(density_data, density_bandwidth):
    value = density(density_data, density_bandwidth).value
    assert jnp.all(value > 0)


def test_doubling_every_bandwidth_changes_the_estimate(density_data, density_bandwidth):
    base = density(density_data, density_bandwidth).value
    doubled = density_bandwidth.replace(
        h=density_bandwidth.h * 2.0,
        lam_uno=density_bandwidth.lam_uno * 2.0,
        lam_ord=density_bandwidth.lam_ord * 2.0,
    )
    changed = density(density_data, doubled).value
    assert not jnp.allclose(base, changed)


def test_large_h_flattens_the_estimate_toward_a_constant():
    rng = np.random.default_rng(4)
    data = MixedData.continuous(jnp.asarray(rng.normal(size=(30, 1))))
    grid = jnp.asarray(np.linspace(-3.0, 3.0, 25)).reshape(-1, 1)
    at = MixedData.continuous(grid)
    bw = Bandwidth(h=jnp.array([50.0]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

    value = density(data, bw, at=at).value
    assert jnp.std(value) < 1e-3 * jnp.mean(value)


def test_leave_one_out_differs_from_full_fit_and_stays_positive(density_data, density_bandwidth):
    fold = jnp.arange(density_data.n)
    loo = density(density_data, density_bandwidth, fold=fold).value
    full = density(density_data, density_bandwidth).value
    assert not jnp.allclose(loo, full)
    assert jnp.all(loo > 0)


def test_leave_one_out_divides_by_the_retained_count(density_data, density_bandwidth):
    fold = jnp.arange(density_data.n)
    got = density(density_data, density_bandwidth, fold=fold).value

    weights = np.asarray(kweights(density_data, density_bandwidth)).copy()
    np.fill_diagonal(weights, 0.0)
    n = density_data.n
    want = weights.sum(axis=1) / ((n - 1) * float(density_bandwidth.h[0]))
    assert np.allclose(np.asarray(got), want, rtol=1e-6)


def test_k_fold_normalizes_by_the_per_row_retained_count(density_data, density_bandwidth):
    fold = jnp.asarray([i % 3 for i in range(density_data.n)])
    got = density(density_data, density_bandwidth, fold=fold).value

    weights = np.asarray(kweights(density_data, density_bandwidth)).copy()
    labels = np.asarray(fold)
    same_fold = labels[:, None] == labels[None, :]
    weights[same_fold] = 0.0
    kept = (~same_fold).sum(axis=1)
    want = weights.sum(axis=1) / (kept * float(density_bandwidth.h[0]))
    assert np.allclose(np.asarray(got), want, rtol=1e-6)


def test_density_fit_is_a_pytree_that_survives_vmap(density_data, density_bandwidth):
    def run(h):
        return density(density_data, density_bandwidth.replace(h=h))

    fits = jax.vmap(run)(jnp.array([[0.4], [0.6], [0.9]]))
    assert isinstance(fits, DensityFit)
    assert fits.value.shape == (3, density_data.n)


@pytest.mark.parametrize("chunk", [4, 6, (5, 4)])
def test_chunked_matches_unchunked(density_data, density_bandwidth, chunk):
    ref = density(density_data, density_bandwidth).value
    got = density(density_data, density_bandwidth, chunk=chunk).value
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_chunked_with_fold_avoids_the_pairwise_fold_memory_blowup():
    n = 4000
    data = MixedData.continuous(jnp.zeros((n, 1)))
    bw = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    fold = jnp.arange(n)

    compiled = jax.jit(lambda d, b, f: density(d, b, fold=f, chunk=64).value).lower(data, bw, fold).compile()
    assert compiled.memory_analysis().temp_size_in_bytes < 5_000_000


def test_jit_grad_vmap(density_data, density_bandwidth):
    total = jax.jit(lambda bw: density(density_data, bw).value.sum())
    assert jnp.isfinite(total(density_bandwidth))
    assert jnp.all(jnp.isfinite(jax.grad(total)(density_bandwidth).h))

    out = jax.vmap(lambda h_values: density(density_data, density_bandwidth.replace(h=h_values)).value.sum())(
        jnp.array([[0.4], [0.6]])
    )
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


def test_density_quartet_eager_jit_grad_vmap(public_api_data, public_api_bandwidth):
    eager = density(public_api_data, public_api_bandwidth)
    assert isinstance(eager, DensityFit)
    assert jnp.all(jnp.isfinite(eager.value))

    jitted = jax.jit(density)(public_api_data, public_api_bandwidth)
    assert jnp.allclose(jitted.value, eager.value, rtol=1e-12)

    def total(bw):
        return density(public_api_data, bw).value.sum()

    grad = jax.grad(total)(public_api_bandwidth)
    assert jnp.all(jnp.isfinite(grad.h))

    out = jax.vmap(lambda h: total(public_api_bandwidth.replace(h=h)))(jnp.array([[0.5], [0.6]]))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))
