"""Tests for the derivative weight tensor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import MixedData
from kerneljax.kernels import KernelSet
from kerneljax.ksum import kweights, kweights_grad


@pytest.mark.parametrize(
    ("train_fixture", "at_fixture", "bandwidth_fixture", "expected_p_con"),
    [
        ("kweights_grad_mixed_train", "kweights_grad_mixed_at", "kweights_grad_mixed_bandwidth", 1),
        ("kweights_grad_two_con_train", "kweights_grad_two_con_at", "kweights_grad_two_con_bandwidth", 2),
    ],
)
def test_shape_is_p_con_by_n_eval_by_n_train(request, train_fixture, at_fixture, bandwidth_fixture, expected_p_con):
    train = request.getfixturevalue(train_fixture)
    at = request.getfixturevalue(at_fixture)
    bw = request.getfixturevalue(bandwidth_fixture)

    got = kweights_grad(train, bw, at=at)
    assert got.shape == (expected_p_con, at.n, train.n)


@pytest.mark.parametrize(
    ("train_fixture", "at_fixture", "bandwidth_fixture"),
    [
        ("kweights_grad_mixed_train", "kweights_grad_mixed_at", "kweights_grad_mixed_bandwidth"),
        ("kweights_grad_two_con_train", "kweights_grad_two_con_at", "kweights_grad_two_con_bandwidth"),
    ],
)
def test_matches_jacfwd_at_eval_points(request, train_fixture, at_fixture, bandwidth_fixture):
    train = request.getfixturevalue(train_fixture)
    at = request.getfixturevalue(at_fixture)
    bw = request.getfixturevalue(bandwidth_fixture)

    def weights_of_con(con):
        return kweights(train, bw, at=at.replace(con=con))

    jacobian = jax.jacfwd(weights_of_con)(at.con)
    diagonal = jnp.diagonal(jacobian, axis1=0, axis2=2)
    expected = jnp.transpose(diagonal, (1, 2, 0))

    got = kweights_grad(train, bw, at=at)
    assert jnp.allclose(got, expected, rtol=1e-6)


def test_matches_explicit_numpy_two_continuous(
    kweights_grad_two_con_train, kweights_grad_two_con_at, kweights_grad_two_con_bandwidth
):
    train = kweights_grad_two_con_train
    at = kweights_grad_two_con_at
    bw = kweights_grad_two_con_bandwidth

    con_train = np.asarray(train.con)
    con_at = np.asarray(at.con)
    uno_train = np.asarray(train.uno)[:, 0]
    uno_at = np.asarray(at.uno)[:, 0]
    h = np.asarray(bw.h)
    lam_uno = float(bw.lam_uno[0])

    def phi(u):
        return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)

    def dphi(u, h_col):
        return -(u / h_col) * phi(u)

    u_first = (con_at[:, None, 0] - con_train[None, :, 0]) / h[0]
    u_second = (con_at[:, None, 1] - con_train[None, :, 1]) / h[1]
    value_first = phi(u_first)
    value_second = phi(u_second)
    deriv_first = dphi(u_first, h[0])
    deriv_second = dphi(u_second, h[1])

    categorical = np.where(uno_at[:, None] == uno_train[None, :], 1.0 - lam_uno, lam_uno / 2.0)

    expected_first = deriv_first * value_second * categorical
    expected_second = value_first * deriv_second * categorical
    expected = np.stack([expected_first, expected_second], axis=0)

    got = kweights_grad(train, bw, at=at)
    assert np.allclose(np.asarray(got), expected, rtol=1e-6)


def test_purely_continuous_is_finite(kweights_grad_purely_continuous_train, kweights_grad_purely_continuous_bandwidth):
    got = kweights_grad(kweights_grad_purely_continuous_train, kweights_grad_purely_continuous_bandwidth)
    assert got.shape == (2, 6, 6)
    assert jnp.all(jnp.isfinite(got))


def test_mixed_design_gives_finite_results(
    kweights_grad_mixed_train, kweights_grad_mixed_at, kweights_grad_mixed_bandwidth
):
    got = kweights_grad(kweights_grad_mixed_train, kweights_grad_mixed_bandwidth, at=kweights_grad_mixed_at)
    assert got.shape == (1, 4, 5)
    assert jnp.all(jnp.isfinite(got))


def test_single_column_matches_explicit_product(
    kweights_grad_mixed_train, kweights_grad_mixed_at, kweights_grad_mixed_bandwidth
):
    train = kweights_grad_mixed_train
    at = kweights_grad_mixed_at
    bw = kweights_grad_mixed_bandwidth

    got = kweights_grad(train, bw, at=at)
    assert got.shape[0] == 1

    kernels = KernelSet()
    con_train = train.con[:, 0]
    con_at = at.con[:, 0]
    deriv = kernels.continuous.deriv(con_at[:, None], con_train[None, :], bw.h[0])

    uno_train = train.uno[:, 0]
    uno_at = at.uno[:, 0]
    orde_train = train.orde[:, 0]
    orde_at = at.orde[:, 0]

    factor_uno = kernels.unordered.value(uno_at[:, None], uno_train[None, :], bw.lam_uno[0], 3)
    factor_ord = kernels.ordered.value(orde_at[:, None], orde_train[None, :], bw.lam_ord[0], 4)
    categorical = factor_uno * factor_ord

    expected = deriv * categorical
    assert jnp.allclose(got[0], expected, rtol=1e-6)


def test_mask_zeroes_the_same_entries_kweights_would(
    kweights_grad_mixed_train, kweights_grad_mixed_at, kweights_grad_mixed_bandwidth
):
    rng = np.random.default_rng(41)
    mask = jnp.asarray(rng.integers(0, 2, size=(4, 5)).astype(bool))

    unmasked = kweights_grad(kweights_grad_mixed_train, kweights_grad_mixed_bandwidth, at=kweights_grad_mixed_at)
    masked = kweights_grad(
        kweights_grad_mixed_train, kweights_grad_mixed_bandwidth, at=kweights_grad_mixed_at, mask=mask
    )

    broadcast_mask = jnp.broadcast_to(mask[None, :, :], masked.shape)
    assert jnp.allclose(masked[broadcast_mask], unmasked[broadcast_mask])
    assert jnp.all(masked[~broadcast_mask] == 0.0)


@pytest.mark.parametrize("chunk", [None, 1, 2, 3, 4, (2, 3), (3, 2), (4, 4)])
def test_chunking_matches_unchunked(
    kweights_grad_mixed_train, kweights_grad_mixed_at, kweights_grad_mixed_bandwidth, chunk
):
    train = kweights_grad_mixed_train
    at = kweights_grad_mixed_at
    bw = kweights_grad_mixed_bandwidth

    ref = kweights_grad(train, bw, at=at)
    got = kweights_grad(train, bw, at=at, chunk=chunk)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_jit_grad_vmap_are_finite(kweights_grad_mixed_train, kweights_grad_mixed_at, kweights_grad_mixed_bandwidth):
    train = kweights_grad_mixed_train
    at = kweights_grad_mixed_at

    def total(bw):
        return kweights_grad(train, bw, at=at).sum()

    jitted = jax.jit(total)
    assert jnp.isfinite(jitted(kweights_grad_mixed_bandwidth))

    grads = jax.grad(total)(kweights_grad_mixed_bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))
    assert jnp.all(jnp.isfinite(grads.lam_uno))
    assert jnp.all(jnp.isfinite(grads.lam_ord))

    def per_h(h):
        return kweights_grad(train, kweights_grad_mixed_bandwidth.replace(h=h), at=at).sum()

    out = jax.vmap(per_h)(jnp.array([[0.4], [0.6], [0.8]]))
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_finite_when_a_factor_underflows():
    train = MixedData.continuous(jnp.array([[0.0, 0.1], [0.0, -0.2]]))
    at = MixedData.continuous(jnp.array([[500.0, 0.05]]))
    bw = Bandwidth(h=jnp.array([1e-4, 0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

    kernels = KernelSet()
    factor_first = kernels.continuous.value(at.con[:, None, 0], train.con[None, :, 0], bw.h[0])
    assert jnp.all(factor_first == 0.0)

    got = kweights_grad(train, bw, at=at)
    assert jnp.all(jnp.isfinite(got))


@pytest.mark.parametrize("chunk", [None, 1, 2, (2, 2)])
def test_no_continuous_columns_empty_axis(chunk):
    data = MixedData.from_blocks(unordered=jnp.array([[0], [1], [2], [0]]), unordered_levels=(3,))
    bandwidth = Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.3]), lam_ord=jnp.zeros(0))

    result = kweights_grad(data, bandwidth, chunk=chunk)

    assert result.shape == (0, data.n, data.n)
    assert jnp.all(jnp.isfinite(result))
