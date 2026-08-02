"""Invariants across bandwidth representations, code paths, limiting cases, and degenerate inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth, normal_reference
from kerneljax.data import Kind, MixedData
from kerneljax.estimators.density import density
from kerneljax.estimators.regression import local_poly
from kerneljax.kernels import KernelSet, Op
from kerneljax.ksum import ksum, kweights
from kerneljax.tuning.objectives import cv_ls_density, cv_ml_density
from kerneljax.tuning.optimize import select_bandwidth


def _density_value(train, bandwidth):
    return density(train, bandwidth).value


def _train_indexed(bandwidth, sample_size):
    width = bandwidth.h.shape[-1]
    return bandwidth.replace(h=jnp.full((sample_size, width), float(bandwidth.h[0])), h_axis="train")


def _eval_indexed(bandwidth, sample_size):
    width = bandwidth.h.shape[-1]
    return bandwidth.replace(h=jnp.full((sample_size, width), float(bandwidth.h[0])), h_axis="eval")


def _indexed(bandwidth, sample_size, h_axis_label):
    if h_axis_label == "train":
        return _train_indexed(bandwidth, sample_size)
    if h_axis_label == "eval":
        return _eval_indexed(bandwidth, sample_size)
    return bandwidth


def _duplicated(data):
    return data.replace(
        con=jnp.concatenate([data.con, data.con]),
        uno=jnp.concatenate([data.uno, data.uno]),
        orde=jnp.concatenate([data.orde, data.orde]),
    )


def _bootstrap_like_duplicate(data):
    repeated_rows = jnp.concatenate(
        [jnp.arange(data.n), jnp.zeros(4, dtype=jnp.int32), jnp.full((4,), 3, dtype=jnp.int32)]
    )
    return data.replace(
        con=data.con[repeated_rows],
        uno=data.uno[repeated_rows],
        orde=data.orde[repeated_rows],
    )


def _trace_density_at_a_large_sample_size():
    train = MixedData.continuous(jnp.zeros((50000, 1)))
    bandwidth = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    return density(train, bandwidth, fold=jnp.arange(50000), chunk=256).value.sum()


def _trace_cv_ml_density_at_a_large_sample_size():
    train = MixedData.continuous(jnp.zeros((50000, 1)))
    bandwidth = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    return cv_ml_density(train, bandwidth)


def _trace_cv_ls_density_at_a_large_sample_size():
    train = MixedData.continuous(jnp.zeros((50000, 1)))
    bandwidth = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    return cv_ls_density(train, bandwidth)


def _multi_column_ksum(train, bandwidth, *, chunk):
    v = jnp.stack([jnp.ones(train.n, dtype=train.con.dtype), train.con[:, 0]], axis=1)
    return ksum(train, bandwidth, v, fold=jnp.arange(train.n), chunk=chunk)


def _local_poly_mean(train, bandwidth, *, chunk):
    return local_poly(train, train.con[:, 0], bandwidth, degree=1, chunk=chunk).mean


def _density_plain(train, bandwidth, *, chunk):
    return density(train, bandwidth, chunk=chunk).value


def _density_leave_one_out(train, bandwidth, *, chunk):
    return density(train, bandwidth, fold=jnp.arange(train.n), chunk=chunk).value


def _echo_solver(objective, best_candidate, **solver_kwargs):
    del solver_kwargs
    return best_candidate, objective(best_candidate), jnp.asarray(0), jnp.asarray(True)


@pytest.mark.parametrize("estimator", [_density_value, cv_ml_density, cv_ls_density])
def test_uniform_bandwidth_agrees_across_h_axis(cv_mixed_data, cv_mixed_bandwidth, estimator):
    shared_result = estimator(cv_mixed_data, cv_mixed_bandwidth)
    train_result = estimator(cv_mixed_data, _train_indexed(cv_mixed_bandwidth, cv_mixed_data.n))
    eval_result = estimator(cv_mixed_data, _eval_indexed(cv_mixed_bandwidth, cv_mixed_data.n))

    assert jnp.allclose(train_result, shared_result, rtol=1e-12)
    assert jnp.allclose(eval_result, shared_result, rtol=1e-12)


@pytest.mark.parametrize("h_axis_label", ["shared", "train", "eval"])
def test_ksum_default_v_equals_kweights_row_sums(ksum_data, ksum_bandwidth, h_axis_label):
    bandwidth = _indexed(ksum_bandwidth, ksum_data.n, h_axis_label)

    summed = ksum(ksum_data, bandwidth)[:, 0]
    row_sums = kweights(ksum_data, bandwidth).sum(axis=1)
    assert jnp.allclose(summed, row_sums, rtol=1e-6)


@pytest.mark.parametrize("use_fold", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 5, 6, 9, (5, 4)])
def test_ksum_chunking_matches_unchunked(cv_mixed_data, cv_mixed_bandwidth, chunk_size, use_fold):
    fold = jnp.arange(cv_mixed_data.n) if use_fold else None
    reference = ksum(cv_mixed_data, cv_mixed_bandwidth, fold=fold)
    chunked = ksum(cv_mixed_data, cv_mixed_bandwidth, fold=fold, chunk=chunk_size)
    assert jnp.allclose(chunked, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("use_fold", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 5, 6, 9, (5, 4)])
def test_density_chunking_matches_unchunked(cv_mixed_data, cv_mixed_bandwidth, chunk_size, use_fold):
    fold = jnp.arange(cv_mixed_data.n) if use_fold else None
    reference = density(cv_mixed_data, cv_mixed_bandwidth, fold=fold).value
    chunked = density(cv_mixed_data, cv_mixed_bandwidth, fold=fold, chunk=chunk_size).value
    assert jnp.allclose(chunked, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
@pytest.mark.parametrize("chunk_size", [4, 5, 6, 9, (5, 4)])
def test_criterion_chunking_matches_unchunked(cv_mixed_data, cv_mixed_bandwidth, criterion, chunk_size):
    reference = criterion(cv_mixed_data, cv_mixed_bandwidth)
    chunked = criterion(cv_mixed_data, cv_mixed_bandwidth, chunk=chunk_size)
    assert float(chunked) == pytest.approx(float(reference), rel=1e-6)


@pytest.mark.parametrize("operator", [Op.VALUE, Op.CONV])
def test_op_spelling_forms_agree(kweights_train, kweights_bandwidth, operator):
    mapping = {Kind.CONTINUOUS: operator, Kind.UNORDERED: operator, Kind.ORDERED: operator}
    per_column = (operator, operator, operator)

    bare = kweights(kweights_train, kweights_bandwidth, op=operator)
    mapped = kweights(kweights_train, kweights_bandwidth, op=mapping)
    tupled = kweights(kweights_train, kweights_bandwidth, op=per_column)

    assert jnp.allclose(mapped, bare, rtol=1e-12)
    assert jnp.allclose(tupled, bare, rtol=1e-12)


@pytest.mark.parametrize("h_axis_label", ["shared", "train", "eval"])
def test_cv_ml_matches_leave_one_out_density(cv_mixed_data, cv_mixed_bandwidth, h_axis_label):
    bandwidth = _indexed(cv_mixed_bandwidth, cv_mixed_data.n, h_axis_label)
    fold = jnp.arange(cv_mixed_data.n)

    leave_one_out = density(cv_mixed_data, bandwidth, fold=fold).value
    hand_built = -jnp.sum(jnp.log(leave_one_out))
    criterion_value = cv_ml_density(cv_mixed_data, bandwidth)

    assert float(criterion_value) == pytest.approx(float(hand_built), rel=1e-6)


def test_large_h_flattens_mixed_density():
    generator = np.random.default_rng(21)
    train = MixedData.from_blocks(
        con=jnp.asarray(generator.normal(size=(40, 1))),
        uno=jnp.asarray(generator.integers(0, 3, size=(40, 1))),
        orde=jnp.asarray(generator.integers(0, 4, size=(40, 1))),
        uno_levels=(3,),
        ord_levels=(4,),
    )
    grid = jnp.asarray(np.linspace(-3.0, 3.0, 20)).reshape(-1, 1)
    evaluate_at = MixedData.from_blocks(
        con=grid,
        uno=jnp.zeros((20, 1), jnp.int32),
        orde=jnp.zeros((20, 1), jnp.int32),
        uno_levels=(3,),
        ord_levels=(4,),
    )
    bandwidth = Bandwidth(h=jnp.array([80.0]), lam_uno=jnp.array([2.0 / 3.0]), lam_ord=jnp.array([0.999]))

    value = density(train, bandwidth, at=evaluate_at).value
    assert jnp.std(value) < 1e-3 * jnp.mean(value)


@pytest.mark.parametrize("bandwidth_value", [0.2, 0.4, 0.8, 1.5])
def test_density_integrates_to_one(bandwidth_value):
    generator = np.random.default_rng(0)
    sample = generator.normal(size=(200, 1))
    train = MixedData.continuous(jnp.asarray(sample))
    grid = np.linspace(-10.0, 10.0, 8001).reshape(-1, 1)
    evaluate_at = MixedData.continuous(jnp.asarray(grid))
    bandwidth = Bandwidth(h=jnp.array([bandwidth_value]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

    value = np.asarray(density(train, bandwidth, at=evaluate_at).value)
    assert np.trapezoid(value, grid[:, 0]) == pytest.approx(1.0, abs=1e-2)


def test_duplicating_rows_leaves_density_unchanged(cv_mixed_data, cv_mixed_bandwidth):
    duplicated = _duplicated(cv_mixed_data)
    reference = density(cv_mixed_data, cv_mixed_bandwidth, at=cv_mixed_data).value
    doubled = density(duplicated, cv_mixed_bandwidth, at=cv_mixed_data).value
    assert jnp.allclose(doubled, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("estimator", [_density_plain, _density_leave_one_out, _local_poly_mean])
def test_memory_is_not_quadratic(peak_bytes, memory_sample, estimator):
    small, large = 1000, 4000

    def peak_at(sample_size):
        train, bandwidth = memory_sample(sample_size)
        return peak_bytes(lambda t, b: estimator(t, b, chunk=None), train, bandwidth)

    small_bytes = peak_at(small)
    large_bytes = peak_at(large)

    train, _ = memory_sample(large)
    assert large_bytes < 0.01 * large * large * train.con.dtype.itemsize
    assert large_bytes / max(small_bytes, 1) < 0.5 * (large / small) ** 2


@pytest.mark.parametrize("materializing_call", [_multi_column_ksum])
def test_chunked_memory_scales_with_chunk(peak_bytes, memory_sample, materializing_call):
    train, bandwidth = memory_sample(1600)

    chunked_bytes = peak_bytes(lambda t, b: materializing_call(t, b, chunk=64), train, bandwidth)
    unchunked_bytes = peak_bytes(lambda t, b: materializing_call(t, b, chunk=None), train, bandwidth)

    assert chunked_bytes < 0.1 * unchunked_bytes


def test_chunking_bounds_criterion_memory(peak_bytes, memory_sample):
    sample_size = 1600
    train, bandwidth = memory_sample(sample_size)

    chunked_bytes = peak_bytes(lambda t, b: cv_ls_density(t, b, chunk=64), train, bandwidth)

    assert chunked_bytes < 0.05 * sample_size * sample_size * train.con.dtype.itemsize


@pytest.mark.parametrize(
    "trace_builder",
    [
        _trace_density_at_a_large_sample_size,
        _trace_cv_ml_density_at_a_large_sample_size,
        _trace_cv_ls_density_at_a_large_sample_size,
    ],
)
def test_large_n_traces_without_memory_cost(trace_builder):
    traced = jax.eval_shape(trace_builder)
    assert traced.shape == ()


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_exact_duplicate_rows_give_finite_criteria(cv_mixed_data, cv_mixed_bandwidth, criterion):
    duplicated = _bootstrap_like_duplicate(cv_mixed_data)
    assert jnp.isfinite(criterion(duplicated, cv_mixed_bandwidth))


def test_exact_duplicate_rows_finite_bandwidth(cv_mixed_data):
    duplicated = _bootstrap_like_duplicate(cv_mixed_data)
    result = select_bandwidth(duplicated, cv_ml_density, n_starts=2)
    assert jnp.isfinite(result.value)
    assert jnp.all(jnp.isfinite(result.bandwidth.h))


@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_constant_continuous_gives_finite_criteria(criterion):
    train = MixedData.continuous(jnp.ones((12, 1)))
    bandwidth = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    assert jnp.isfinite(criterion(train, bandwidth))


def test_constant_continuous_gives_constant_density():
    train = MixedData.continuous(jnp.ones((12, 1)))
    bandwidth = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    value = density(train, bandwidth).value
    assert jnp.all(jnp.isfinite(value))
    assert jnp.allclose(value, value[0], rtol=1e-6)


@pytest.mark.parametrize("n_starts", [1, 2, 3, 4])
@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_selection_no_worse_than_start(cv_mixed_data, criterion, n_starts):
    start = normal_reference(cv_mixed_data, KernelSet())
    start_value = criterion(cv_mixed_data, start)

    result = select_bandwidth(cv_mixed_data, criterion, n_starts=n_starts)
    assert float(result.value) <= float(start_value) + 1e-6


@pytest.mark.parametrize("n_starts", [1, 2, 3, 4])
@pytest.mark.parametrize("criterion", [cv_ml_density, cv_ls_density])
def test_screening_no_worse_than_start(cv_mixed_data, criterion, n_starts):
    start = normal_reference(cv_mixed_data, KernelSet())
    start_value = criterion(cv_mixed_data, start)

    result = select_bandwidth(cv_mixed_data, criterion, solver=_echo_solver, n_starts=n_starts)
    assert float(result.value) <= float(start_value) + 1e-6
