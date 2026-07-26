"""Tests for the product kernel weight matrix."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.data import Kind, MixedData
from kerneljax.kernels import KernelSet, Op
from kerneljax.ksum import kweights


def test_square_shape_matches_training_size(kweights_train, kweights_bandwidth):
    weights = kweights(kweights_train, kweights_bandwidth)
    assert weights.shape == (6, 6)


def test_square_case_is_symmetric(kweights_train, kweights_bandwidth):
    weights = kweights(kweights_train, kweights_bandwidth)
    assert jnp.allclose(weights, weights.T, rtol=1e-12, atol=1e-12)


def test_matches_an_explicit_three_factor_product(kweights_train, kweights_bandwidth):
    x = np.asarray(kweights_train.con)[:, 0]
    u = np.asarray(kweights_train.uno)[:, 0]
    o = np.asarray(kweights_train.orde)[:, 0]
    h = float(kweights_bandwidth.h[0])
    lam_uno = float(kweights_bandwidth.lam_uno[0])
    lam_ord = float(kweights_bandwidth.lam_ord[0])

    factor_con = np.exp(-0.5 * ((x[:, None] - x[None, :]) / h) ** 2) / np.sqrt(2.0 * np.pi)
    factor_uno = np.where(u[:, None] == u[None, :], 1.0 - lam_uno, lam_uno / 3.0)
    distance = np.abs(o[:, None] - o[None, :])
    factor_ord = np.where(distance == 0, 1.0 - lam_ord, 0.5 * (1.0 - lam_ord) * lam_ord**distance)

    expected = factor_con * factor_uno * factor_ord
    weights = kweights(kweights_train, kweights_bandwidth)
    assert np.allclose(np.asarray(weights), expected, rtol=1e-6)


@pytest.mark.parametrize("n_eval", [1, 3, 5])
def test_rectangular_eval_gives_the_right_shape(kweights_train, kweights_bandwidth, n_eval):
    grid = MixedData.from_blocks(
        con=jnp.zeros((n_eval, 1)),
        uno=jnp.zeros((n_eval, 1), jnp.int32),
        orde=jnp.zeros((n_eval, 1), jnp.int32),
        uno_levels=(4,),
        ord_levels=(3,),
    )
    weights = kweights(kweights_train, kweights_bandwidth, at=grid)
    assert weights.shape == (n_eval, 6)


def test_continuous_only_design(continuous_data, continuous_only_bandwidth):
    weights = kweights(continuous_data, continuous_only_bandwidth)
    assert weights.shape == (5, 5)
    assert jnp.all(jnp.isfinite(weights))


def test_discrete_only_design(discrete_only_data, discrete_only_bandwidth):
    weights = kweights(discrete_only_data, discrete_only_bandwidth)
    assert weights.shape == (3, 3)
    assert jnp.all(jnp.isfinite(weights))


def test_mask_zeroes_exactly_the_masked_entries_and_leaves_the_rest(kweights_train, kweights_bandwidth):
    rng = np.random.default_rng(1)
    mask = jnp.asarray(rng.integers(0, 2, size=(6, 6)).astype(bool))

    unmasked = kweights(kweights_train, kweights_bandwidth)
    masked = kweights(kweights_train, kweights_bandwidth, mask=mask)

    assert jnp.allclose(masked[mask], unmasked[mask])
    assert jnp.all(masked[~mask] == 0.0)


@pytest.mark.parametrize("power", [1, 2, 3])
def test_power_raises_the_weights_elementwise(kweights_train, kweights_bandwidth, power):
    base = kweights(kweights_train, kweights_bandwidth)
    raised = kweights(kweights_train, kweights_bandwidth, power=power)
    assert jnp.allclose(raised, base**power, rtol=1e-12)


def test_op_conv_matches_the_kernels_called_directly(kweights_train, kweights_bandwidth):
    kernels = KernelSet()
    x = kweights_train.con[:, 0]
    u = kweights_train.uno[:, 0]
    o = kweights_train.orde[:, 0]

    factor_con = kernels.continuous.conv(x[:, None], x[None, :], kweights_bandwidth.h[0])
    factor_uno = kernels.unordered.conv(u[:, None], u[None, :], kweights_bandwidth.lam_uno[0], 4)
    factor_ord = kernels.ordered.conv(o[:, None], o[None, :], kweights_bandwidth.lam_ord[0], 3)
    expected = factor_con * factor_uno * factor_ord

    weights = kweights(kweights_train, kweights_bandwidth, op=Op.CONV)
    assert jnp.allclose(weights, expected, rtol=1e-12)


def test_op_per_kind_mapping_differs_from_the_all_value_result(kweights_train, kweights_bandwidth):
    mapping = {Kind.CONTINUOUS: Op.CONV, Kind.UNORDERED: Op.VALUE, Kind.ORDERED: Op.VALUE}
    default = kweights(kweights_train, kweights_bandwidth)
    mapped = kweights(kweights_train, kweights_bandwidth, op=mapping)

    assert mapped.shape == (6, 6)
    assert not jnp.allclose(mapped, default)


def test_op_per_column_tuple_agrees_with_the_equivalent_mapping(kweights_train, kweights_bandwidth):
    mapping = {Kind.CONTINUOUS: Op.CONV, Kind.UNORDERED: Op.VALUE, Kind.ORDERED: Op.VALUE}
    per_column = (Op.CONV, Op.VALUE, Op.VALUE)

    mapped = kweights(kweights_train, kweights_bandwidth, op=mapping)
    tupled = kweights(kweights_train, kweights_bandwidth, op=per_column)

    assert jnp.allclose(tupled, mapped)
    assert not jnp.allclose(tupled, kweights(kweights_train, kweights_bandwidth))


def test_op_tuple_resolves_a_different_operator_per_column_within_one_kind(two_unordered_data, two_unordered_bandwidth):
    kernels = KernelSet()
    first = two_unordered_data.uno[:, 0]
    second = two_unordered_data.uno[:, 1]

    factor_first = kernels.unordered.conv(first[:, None], first[None, :], two_unordered_bandwidth.lam_uno[0], 3)
    factor_second = kernels.unordered.value(second[:, None], second[None, :], two_unordered_bandwidth.lam_uno[1], 2)
    expected = factor_first * factor_second

    weights = kweights(two_unordered_data, two_unordered_bandwidth, op=(Op.CONV, Op.VALUE))
    assert jnp.allclose(weights, expected, rtol=1e-12)


def test_jit_accepts_train_and_kernels_without_static_argnames(kweights_train, kweights_bandwidth):
    def total_weight(data, bw, kernels):
        return kweights(data, bw, kernels=kernels).sum()

    result = jax.jit(total_weight)(kweights_train, kweights_bandwidth, KernelSet())
    assert jnp.isfinite(result)


def test_grad_over_the_bandwidth_is_finite(kweights_train, kweights_bandwidth):
    def total_weight(bw):
        return kweights(kweights_train, bw).sum()

    grads = jax.grad(total_weight)(kweights_bandwidth)
    assert jnp.all(jnp.isfinite(grads.h))
    assert jnp.all(jnp.isfinite(grads.lam_uno))
    assert jnp.all(jnp.isfinite(grads.lam_ord))


def test_vmap_over_the_bandwidth_is_finite(kweights_train, kweights_bandwidth):
    def total_weight(h):
        return kweights(kweights_train, kweights_bandwidth.replace(h=h)).sum()

    out = jax.vmap(total_weight)(jnp.array([[0.4], [0.7], [1.1]]))
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_mismatched_kinds_between_train_and_at_raise_value_error(kweights_train, kweights_bandwidth):
    at = MixedData.continuous(jnp.zeros((4, 1)))
    with pytest.raises(ValueError, match="kinds"):
        kweights(kweights_train, kweights_bandwidth, at=at)


def test_mismatched_levels_between_train_and_at_raise_value_error(kweights_train, kweights_bandwidth):
    at = MixedData.from_blocks(
        con=jnp.zeros((4, 1)),
        uno=jnp.zeros((4, 1), jnp.int32),
        orde=jnp.zeros((4, 1), jnp.int32),
        uno_levels=(5,),
        ord_levels=(3,),
    )
    with pytest.raises(ValueError, match="kinds"):
        kweights(kweights_train, kweights_bandwidth, at=at)
