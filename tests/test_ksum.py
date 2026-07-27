"""Tests for the kernel sum contraction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.data import MixedData
from kerneljax.ksum import ksum, kweights


def test_default_v_is_ones(ksum_data, ksum_bandwidth):
    got = ksum(ksum_data, ksum_bandwidth)[:, 0]
    want = kweights(ksum_data, ksum_bandwidth).sum(axis=1)
    assert jnp.allclose(got, want, rtol=1e-6)


def test_contracts_against_v(ksum_data, ksum_bandwidth):
    v = jnp.asarray(np.arange(6.0)).reshape(6, 1)
    got = ksum(ksum_data, ksum_bandwidth, v=v)
    want = kweights(ksum_data, ksum_bandwidth) @ v
    assert jnp.allclose(got, want, rtol=1e-6)


def test_sample_weight_scales_columns(ksum_data, ksum_bandwidth):
    sample_weight = jnp.asarray(np.linspace(0.5, 2.0, 6))
    got = ksum(ksum_data, ksum_bandwidth, sample_weight=sample_weight)
    want = (kweights(ksum_data, ksum_bandwidth) * sample_weight[None, :]).sum(axis=1, keepdims=True)
    assert jnp.allclose(got, want, rtol=1e-6)


def test_fold_leave_one_out_zeroes_the_diagonal(ksum_data, ksum_bandwidth):
    fold = jnp.arange(6)
    got = ksum(ksum_data, ksum_bandwidth, fold=fold)
    weights = np.asarray(kweights(ksum_data, ksum_bandwidth)).copy()
    np.fill_diagonal(weights, 0.0)
    assert jnp.allclose(got[:, 0], weights.sum(axis=1), rtol=1e-6)


def test_fold_supports_k_fold_not_just_leave_one_out(ksum_data, ksum_bandwidth):
    fold = jnp.asarray([0, 0, 1, 1, 2, 2])
    got = ksum(ksum_data, ksum_bandwidth, fold=fold)
    weights = np.asarray(kweights(ksum_data, ksum_bandwidth)).copy()
    labels = np.asarray(fold)
    weights[labels[:, None] == labels[None, :]] = 0.0
    assert jnp.allclose(got[:, 0], weights.sum(axis=1), rtol=1e-6)


def test_weight_scale_shared_divides_once(ksum_data, ksum_bandwidth):
    got = ksum(ksum_data, ksum_bandwidth, weight_scale="per_eval")
    assert jnp.allclose(got, ksum(ksum_data, ksum_bandwidth) / 0.7, rtol=1e-6)


def test_weight_scale_per_train_divides_inside_the_sum(ksum_train_indexed_data, ksum_train_indexed_bandwidth):
    got = ksum(ksum_train_indexed_data, ksum_train_indexed_bandwidth, weight_scale="per_train")
    weights = np.asarray(kweights(ksum_train_indexed_data, ksum_train_indexed_bandwidth))
    want = (weights / np.asarray(ksum_train_indexed_bandwidth.h)[None, :, 0]).sum(axis=1)
    assert jnp.allclose(got[:, 0], want, rtol=1e-6)


def test_weight_scale_per_eval_with_eval_indexed_bandwidth(ksum_data, ksum_eval_indexed_bandwidth):
    got = ksum(ksum_data, ksum_eval_indexed_bandwidth, weight_scale="per_eval")
    base = ksum(ksum_data, ksum_eval_indexed_bandwidth)
    assert jnp.allclose(got, base / ksum_eval_indexed_bandwidth.h, rtol=1e-6)


def test_weight_scale_per_train_matches_when_chunked(ksum_train_indexed_data, ksum_train_indexed_bandwidth):
    ref = ksum(ksum_train_indexed_data, ksum_train_indexed_bandwidth, weight_scale="per_train")
    got = ksum(ksum_train_indexed_data, ksum_train_indexed_bandwidth, weight_scale="per_train", chunk=2)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("chunk", [None, 2, 3, 4, (2, 3), (4, 2)])
@pytest.mark.parametrize("use_fold", [False, True])
def test_chunking_is_numerically_identical(ksum_data, ksum_bandwidth, chunk, use_fold):
    fold = jnp.arange(6) if use_fold else None
    ref = ksum(ksum_data, ksum_bandwidth, fold=fold)
    got = ksum(ksum_data, ksum_bandwidth, fold=fold, chunk=chunk)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("chunk", [None, 2, 3, (2, 3)])
def test_chunking_matches_unchunked_with_k_fold(ksum_data, ksum_bandwidth, chunk):
    fold = jnp.asarray([0, 0, 1, 1, 2, 2])
    ref = ksum(ksum_data, ksum_bandwidth, fold=fold)
    got = ksum(ksum_data, ksum_bandwidth, fold=fold, chunk=chunk)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_chunking_handles_a_non_divisible_split(ksum_data, ksum_bandwidth):
    got = ksum(ksum_data, ksum_bandwidth, chunk=4)
    ref = ksum(ksum_data, ksum_bandwidth)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_chunking_matches_unchunked_against_explicit_v(ksum_data, ksum_bandwidth):
    v = jnp.asarray(np.arange(6.0)).reshape(6, 1)
    ref = ksum(ksum_data, ksum_bandwidth, v=v)
    got = ksum(ksum_data, ksum_bandwidth, v=v, chunk=3)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_chunking_handles_a_rectangular_eval_grid(ksum_data, ksum_bandwidth):
    grid = MixedData.from_blocks(
        con=jnp.zeros((5, 1)),
        uno=jnp.zeros((5, 1), jnp.int32),
        uno_levels=(4,),
    )
    ref = ksum(ksum_data, ksum_bandwidth, at=grid)
    got = ksum(ksum_data, ksum_bandwidth, at=grid, chunk=(2, 4))
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_chunking_matches_unchunked_with_eval_indexed_bandwidth(ksum_data, ksum_eval_indexed_bandwidth):
    ref = ksum(ksum_data, ksum_eval_indexed_bandwidth)
    got = ksum(ksum_data, ksum_eval_indexed_bandwidth, chunk=4)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_jit_grad_vmap(ksum_data, ksum_bandwidth):
    total = jax.jit(lambda bw: ksum(ksum_data, bw, fold=jnp.arange(6)).sum())
    assert jnp.isfinite(total(ksum_bandwidth))
    assert jnp.all(jnp.isfinite(jax.grad(total)(ksum_bandwidth).h))

    out = jax.vmap(lambda hh: ksum(ksum_data, ksum_bandwidth.replace(h=hh)).sum())(jnp.array([[0.5], [0.7]]))
    assert out.shape == (2,)


def test_jit_grad_and_vmap_work_when_chunked(ksum_data, ksum_bandwidth):
    def total(bw):
        return ksum(ksum_data, bw, fold=jnp.arange(6), chunk=2).sum()

    jitted = jax.jit(total)
    assert jnp.isfinite(jitted(ksum_bandwidth))

    grads = jax.grad(total)(ksum_bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))

    def per_h(hh):
        return ksum(ksum_data, ksum_bandwidth.replace(h=hh), chunk=2).sum()

    out = jax.vmap(per_h)(jnp.array([[0.5], [0.7]]))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))
