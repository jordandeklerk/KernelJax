"""Shared pytest fixtures and configuration for the KernelJax test suite."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth, BandwidthTransform, ConditionalBandwidth
from kerneljax.data import ColumnSpec, Kind, MixedData
from kerneljax.kernels import KernelSet
from kerneljax.kernels.base import ContinuousKernel, OrderedKernel, UnorderedKernel
from kerneljax.kernels.continuous import Gaussian
from kerneljax.kernels.discrete import AitchisonAitken, WangVanRyzin


@pytest.fixture
def column_spec():
    return ColumnSpec(
        kinds=(Kind.CONTINUOUS, Kind.UNORDERED, Kind.ORDERED, Kind.CONTINUOUS),
        n_levels=(0, 4, 3, 0),
    )


@pytest.fixture
def continuous_data():
    return MixedData.continuous(jnp.ones((5, 2)))


@dataclasses.dataclass(frozen=True)
class BareContinuousKernel(ContinuousKernel):
    def value(self, x, y, h):
        return x - y + h


@dataclasses.dataclass(frozen=True)
class BareUnorderedKernel(UnorderedKernel):
    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, lam)

    def upper_bound(self, levels):
        return (levels - 1) / levels


@dataclasses.dataclass(frozen=True)
class BareOrderedKernel(OrderedKernel):
    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, lam)

    def upper_bound(self, levels):
        return 1.0


@pytest.fixture
def bare_continuous_kernel_cls():
    return BareContinuousKernel


@pytest.fixture
def bare_unordered_kernel_cls():
    return BareUnorderedKernel


@pytest.fixture
def bare_ordered_kernel_cls():
    return BareOrderedKernel


@dataclasses.dataclass(frozen=True)
class Epanechnikov(ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)


@pytest.fixture
def epanechnikov_kernels():
    return KernelSet(continuous=Epanechnikov())


@dataclasses.dataclass(frozen=True)
class WithExtra(ContinuousKernel):
    def value(self, x, y, h):
        return jnp.ones_like(x * y * h)

    def my_op(self, x, y, h):
        return jnp.full_like(x * y * h, 7.0)


@dataclasses.dataclass(frozen=True)
class ToyOrdered(OrderedKernel):
    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, jnp.power(lam, jnp.abs(x - y)) / levels)

    def upper_bound(self, levels):
        return 1.0


@pytest.fixture
def with_extra_kernel_cls():
    return WithExtra


@pytest.fixture
def toy_ordered_kernel_cls():
    return ToyOrdered


@pytest.fixture
def gaussian():
    return Gaussian()


@pytest.fixture
def aitchison_aitken():
    return AitchisonAitken()


@pytest.fixture
def wang_van_ryzin():
    return WangVanRyzin()


@pytest.fixture
def bandwidth():
    return Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.array([0.2]), lam_ord=jnp.array([0.3]))


@pytest.fixture
def mixed_bandwidth_data():
    return MixedData.from_blocks(
        continuous=jnp.linspace(0.0, 1.0, 12).reshape(12, 1),
        unordered=jnp.array([[i % 4] for i in range(12)]),
        ordered=jnp.array([[i % 3] for i in range(12)]),
        unordered_levels=(4,),
        ordered_levels=(3,),
    )


@pytest.fixture
def mixed_bandwidth_transform(mixed_bandwidth_data):
    return BandwidthTransform(spec=mixed_bandwidth_data.spec, kernels=KernelSet())


@pytest.fixture
def kweights_train():
    rng = np.random.default_rng(0)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(6, 1))),
        unordered=jnp.asarray(rng.integers(0, 4, size=(6, 1))),
        ordered=jnp.asarray(rng.integers(0, 3, size=(6, 1))),
        unordered_levels=(4,),
        ordered_levels=(3,),
    )


@pytest.fixture
def kweights_bandwidth():
    return Bandwidth(h=jnp.array([0.7]), lam_uno=jnp.array([0.15]), lam_ord=jnp.array([0.25]))


@pytest.fixture
def continuous_only_bandwidth():
    return Bandwidth(h=jnp.array([1.0, 1.0]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def discrete_only_data():
    return MixedData.from_blocks(unordered=jnp.array([[0], [1], [2]]), unordered_levels=(3,))


@pytest.fixture
def discrete_only_bandwidth():
    return Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.2]), lam_ord=jnp.zeros(0))


@pytest.fixture
def two_unordered_data():
    return MixedData.from_blocks(
        unordered=jnp.array([[0, 0], [1, 1], [2, 0], [0, 1]]),
        unordered_levels=(3, 2),
    )


@pytest.fixture
def two_unordered_bandwidth():
    return Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.2, 0.3]), lam_ord=jnp.zeros(0))


@pytest.fixture
def ksum_data():
    return MixedData.from_blocks(
        continuous=jnp.asarray(np.linspace(-1.0, 2.0, 6)).reshape(6, 1),
        unordered=jnp.array([[0], [1], [2], [3], [0], [1]]),
        unordered_levels=(4,),
    )


@pytest.fixture
def ksum_bandwidth():
    return Bandwidth(h=jnp.array([0.7]), lam_uno=jnp.array([0.15]), lam_ord=jnp.zeros(0))


@pytest.fixture
def ksum_train_indexed_data():
    return MixedData.continuous(jnp.asarray(np.linspace(0.0, 1.0, 5)).reshape(5, 1))


@pytest.fixture
def ksum_train_indexed_bandwidth():
    h = jnp.asarray(np.linspace(0.4, 0.9, 5)).reshape(5, 1)
    return Bandwidth(h=h, lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0), h_axis="train")


@pytest.fixture
def ksum_eval_indexed_bandwidth():
    h = jnp.asarray(np.linspace(0.5, 1.0, 6)).reshape(6, 1)
    return Bandwidth(h=h, lam_uno=jnp.array([0.15]), lam_ord=jnp.zeros(0), h_axis="eval")


@pytest.fixture
def density_data():
    rng = np.random.default_rng(3)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(15, 1))),
        unordered=jnp.asarray(rng.integers(0, 3, size=(15, 1))),
        ordered=jnp.asarray(rng.integers(0, 4, size=(15, 1))),
        unordered_levels=(3,),
        ordered_levels=(4,),
    )


@pytest.fixture
def density_bandwidth():
    return Bandwidth(h=jnp.array([0.6]), lam_uno=jnp.array([0.25]), lam_ord=jnp.array([0.3]))


@pytest.fixture
def cv_mixed_data():
    rng = np.random.default_rng(11)
    combos = np.array([(u, o) for u in range(3) for o in range(3)] * 2)
    rng.shuffle(combos)
    con = jnp.asarray(rng.normal(size=(combos.shape[0], 1)))
    return MixedData.from_blocks(
        continuous=con,
        unordered=jnp.asarray(combos[:, :1]),
        ordered=jnp.asarray(combos[:, 1:]),
        unordered_levels=(3,),
        ordered_levels=(3,),
    )


@pytest.fixture
def cv_mixed_bandwidth():
    return Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.array([0.3]), lam_ord=jnp.array([0.4]))


@pytest.fixture
def cv_continuous_data():
    rng = np.random.default_rng(12)
    return MixedData.continuous(jnp.asarray(rng.normal(size=(10, 1))))


@pytest.fixture
def cv_continuous_bandwidth():
    return Bandwidth(h=jnp.array([0.6]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def cv_discrete_data():
    rng = np.random.default_rng(13)
    combos = np.array([(u, o) for u in range(3) for o in range(3)] * 2)
    rng.shuffle(combos)
    return MixedData.from_blocks(
        unordered=jnp.asarray(combos[:, :1]),
        ordered=jnp.asarray(combos[:, 1:]),
        unordered_levels=(3,),
        ordered_levels=(3,),
    )


@pytest.fixture
def cv_discrete_bandwidth():
    return Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.3]), lam_ord=jnp.array([0.4]))


@pytest.fixture
def public_api_data():
    return MixedData.from_blocks(
        continuous=jnp.linspace(-1.0, 2.0, 8).reshape(8, 1),
        unordered=jnp.asarray([[i % 3] for i in range(8)]),
        ordered=jnp.asarray([[i % 2] for i in range(8)]),
        unordered_levels=(3,),
        ordered_levels=(2,),
    )


@pytest.fixture
def public_api_bandwidth():
    return Bandwidth(h=jnp.array([0.6]), lam_uno=jnp.array([0.2]), lam_ord=jnp.array([0.3]))


@pytest.fixture
def regression_mixed_response(cv_mixed_data):
    rng = np.random.default_rng(51)
    return jnp.asarray(rng.normal(size=cv_mixed_data.n))


@pytest.fixture
def regression_continuous_response(cv_continuous_data):
    rng = np.random.default_rng(52)
    return jnp.asarray(rng.normal(size=cv_continuous_data.n))


@pytest.fixture
def regression_discrete_response(cv_discrete_data):
    rng = np.random.default_rng(53)
    return jnp.asarray(rng.normal(size=cv_discrete_data.n))


@pytest.fixture
def float64_enabled():
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.fixture
def regression_reference_train(float64_enabled):
    x = jnp.array(
        [
            -0.766796,
            -0.816458,
            -0.141535,
            -0.277605,
            0.436307,
            -1.186873,
            1.191987,
            -0.01819,
            -0.248085,
            -0.362937,
            1.277571,
            -0.468897,
            0.071054,
            -0.266038,
            1.845257,
        ]
    )
    u = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    return MixedData.from_blocks(continuous=x[:, None], unordered=u[:, None], unordered_levels=(3,))


@pytest.fixture
def regression_reference_y(float64_enabled):
    return jnp.array(
        [
            2.084996,
            3.041954,
            4.79571,
            2.614418,
            4.730606,
            3.783158,
            5.356314,
            3.972847,
            4.652491,
            2.508284,
            6.569232,
            4.3846,
            3.1086,
            3.629253,
            9.129875,
        ]
    )


@pytest.fixture
def wls_design():
    rng = np.random.default_rng(7)
    design = jnp.asarray(rng.normal(size=(30, 3)))
    weights = jnp.asarray(rng.uniform(0.5, 2.0, size=30))
    true_coef = jnp.asarray(rng.normal(size=(3, 2)))
    return design, weights, true_coef


@pytest.fixture
def singular_design():
    rng = np.random.default_rng(11)
    base = jnp.asarray(rng.normal(size=(20, 2)))
    design = jnp.concatenate([base, base[:, :1]], axis=1)
    weights = jnp.ones(20)
    return design, weights


@pytest.fixture
def basis_train():
    rng = np.random.default_rng(6)
    return MixedData.continuous(jnp.asarray(rng.normal(size=(6, 2))))


@pytest.fixture
def basis_at():
    return MixedData.continuous(jnp.array([[0.3, -0.2]]))


@pytest.fixture
def basis_bandwidth():
    return Bandwidth(h=jnp.array([0.5, 0.8]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def kweights_grad_mixed_train():
    rng = np.random.default_rng(31)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(5, 1))),
        unordered=jnp.asarray(rng.integers(0, 3, size=(5, 1))),
        ordered=jnp.asarray(rng.integers(0, 4, size=(5, 1))),
        unordered_levels=(3,),
        ordered_levels=(4,),
    )


@pytest.fixture
def kweights_grad_mixed_at():
    rng = np.random.default_rng(32)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(4, 1))),
        unordered=jnp.asarray(rng.integers(0, 3, size=(4, 1))),
        ordered=jnp.asarray(rng.integers(0, 4, size=(4, 1))),
        unordered_levels=(3,),
        ordered_levels=(4,),
    )


@pytest.fixture
def kweights_grad_mixed_bandwidth():
    return Bandwidth(h=jnp.array([0.6]), lam_uno=jnp.array([0.2]), lam_ord=jnp.array([0.25]))


@pytest.fixture
def kweights_grad_two_con_train():
    rng = np.random.default_rng(33)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(5, 2))),
        unordered=jnp.asarray(rng.integers(0, 3, size=(5, 1))),
        unordered_levels=(3,),
    )


@pytest.fixture
def kweights_grad_two_con_at():
    rng = np.random.default_rng(34)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(4, 2))),
        unordered=jnp.asarray(rng.integers(0, 3, size=(4, 1))),
        unordered_levels=(3,),
    )


@pytest.fixture
def kweights_grad_two_con_bandwidth():
    return Bandwidth(h=jnp.array([0.6, 0.4]), lam_uno=jnp.array([0.2]), lam_ord=jnp.zeros(0))


@pytest.fixture
def kweights_grad_purely_continuous_train():
    rng = np.random.default_rng(35)
    return MixedData.continuous(jnp.asarray(rng.normal(size=(6, 2))))


@pytest.fixture
def kweights_grad_purely_continuous_bandwidth():
    return Bandwidth(h=jnp.array([0.5, 0.7]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def poly_train():
    rng = np.random.default_rng(40)
    return MixedData.continuous(jnp.asarray(rng.normal(size=(50, 1))))


@pytest.fixture
def poly_at():
    return MixedData.continuous(jnp.array([[-1.0], [-0.3], [0.4], [1.1]]))


@pytest.fixture
def poly_bandwidth():
    return Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def poly_response(poly_train):
    rng = np.random.default_rng(41)
    x = np.asarray(poly_train.con)[:, 0]
    return jnp.asarray(np.sin(x) + rng.normal(scale=0.2, size=x.shape[0]))


@pytest.fixture
def poly_mixed_train():
    rng = np.random.default_rng(42)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(40, 1))),
        unordered=jnp.asarray(rng.integers(0, 3, size=(40, 1))),
        ordered=jnp.asarray(rng.integers(0, 4, size=(40, 1))),
        unordered_levels=(3,),
        ordered_levels=(4,),
    )


@pytest.fixture
def poly_mixed_bandwidth():
    return Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.array([0.2]), lam_ord=jnp.array([0.3]))


@pytest.fixture
def poly_mixed_response(poly_mixed_train):
    rng = np.random.default_rng(43)
    return jnp.asarray(rng.normal(size=poly_mixed_train.n))


@pytest.fixture
def poly_eval_indexed_bandwidth():
    h = jnp.array([[0.25], [0.4], [0.65], [0.9]])
    return Bandwidth(h=h, lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0), h_axis="eval")


@pytest.fixture
def poly_train_indexed_bandwidth():
    h = jnp.linspace(0.3, 0.9, 50).reshape(-1, 1)
    return Bandwidth(h=h, lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0), h_axis="train")


@pytest.fixture
def poly_flat_train():
    rng = np.random.default_rng(77)
    return MixedData.continuous(jnp.asarray(rng.uniform(-3.0, 3.0, size=3000)).reshape(-1, 1))


@pytest.fixture
def poly_flat_response(poly_flat_train):
    rng = np.random.default_rng(78)
    return jnp.asarray(rng.normal(size=poly_flat_train.n))


@pytest.fixture
def poly_degenerate_train():
    return MixedData.continuous(jnp.zeros((30, 1)))


@pytest.fixture
def poly_degenerate_at():
    return MixedData.continuous(jnp.zeros((1, 1)))


@pytest.fixture
def poly_degenerate_bandwidth():
    return Bandwidth(h=jnp.array([0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def poly_degenerate_response():
    rng = np.random.default_rng(44)
    return jnp.asarray(rng.normal(size=30))


@pytest.fixture
def poly_growing_sample():
    at = MixedData.continuous(jnp.array([[0.0]]))
    bandwidth = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

    trains = []
    responses = []
    for size in (20, 80, 320):
        rng = np.random.default_rng(1000 + size)
        trains.append(MixedData.continuous(jnp.asarray(rng.normal(size=(size, 1)))))
        responses.append(jnp.asarray(rng.normal(size=size)))

    return trains, responses, at, bandwidth


@pytest.fixture
def criteria_train():
    rng = np.random.default_rng(11)
    return MixedData.continuous(jnp.asarray(rng.normal(size=(23, 1))))


@pytest.fixture
def criteria_response(criteria_train):
    return jnp.sin(criteria_train.con[:, 0])


@pytest.fixture
def criteria_bandwidth():
    return Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def grid_sample():
    rng = np.random.default_rng(3)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(40, 2))),
        unordered=jnp.asarray(rng.integers(0, 3, size=(40, 1))),
        ordered=jnp.asarray(rng.integers(0, 4, size=(40, 1))),
        unordered_levels=(3,),
        ordered_levels=(4,),
    )


@pytest.fixture
def noiseless_train():
    return MixedData.continuous(jnp.linspace(-2.0, 2.0, 60).reshape(-1, 1))


@pytest.fixture
def noiseless_response(noiseless_train):
    return jnp.sin(noiseless_train.con[:, 0])


@pytest.fixture
def peak_bytes():
    def measure(call, *args):
        return jax.jit(call).lower(*args).compile().memory_analysis().temp_size_in_bytes

    return measure


@pytest.fixture
def memory_sample():
    def build(sample_size):
        train = MixedData.continuous(jnp.zeros((sample_size, 1)))
        bandwidth = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
        return train, bandwidth

    return build


@pytest.fixture
def cdf_train():
    rng = np.random.default_rng(53)
    return MixedData.from_blocks(
        continuous=jnp.asarray(rng.normal(size=(30, 1))),
        ordered=jnp.asarray(rng.integers(0, 4, size=(30, 1))),
        ordered_levels=(4,),
    )


@pytest.fixture
def cdf_bandwidth():
    return Bandwidth(h=jnp.array([0.35]), lam_uno=jnp.zeros(0), lam_ord=jnp.array([0.3]))


@pytest.fixture
def categorical_relevance():
    def build(effect, sample_size=400, seed=1):
        rng = np.random.default_rng(seed)
        continuous = rng.uniform(0.0, 40.0, size=sample_size)
        category = rng.integers(0, 3, size=sample_size)
        response = (
            0.04 * continuous - 0.0005 * continuous**2 + effect * category + rng.normal(0.0, 0.2, size=sample_size)
        )
        train = MixedData.from_blocks(
            continuous=jnp.asarray(continuous, dtype=jnp.float32)[:, None],
            unordered=jnp.asarray(category)[:, None],
            unordered_levels=(3,),
        )
        return train, jnp.asarray(response, dtype=jnp.float32)

    return build


@pytest.fixture
def selection_reference_train(float64_enabled):
    x = jnp.array(
        [
            9.148060,
            9.370754,
            2.861395,
            8.304476,
            6.417455,
            5.190959,
            7.365883,
            1.346666,
            6.569923,
            7.050648,
            4.577418,
            7.191123,
            9.346722,
            2.554288,
            4.622928,
            9.400145,
            9.782264,
            1.174874,
            4.749971,
            5.603327,
            9.040314,
            1.387102,
            9.888917,
            9.466682,
            0.824376,
            5.142118,
            3.902035,
            9.057381,
            4.469696,
            8.360043,
            7.375956,
            8.110551,
            3.881083,
            6.851697,
            0.039483,
            8.329161,
            0.073341,
            2.076590,
            9.066014,
            6.117786,
            3.795592,
            4.357716,
            0.374310,
            9.735399,
            4.317512,
            9.575766,
            8.877549,
            6.399788,
            9.709666,
            6.188382,
        ]
    )
    g = jnp.array(
        [
            0,
            1,
            1,
            2,
            2,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            2,
            1,
            0,
            1,
            2,
            1,
            1,
            1,
            0,
            0,
            0,
            2,
            0,
            1,
            0,
            0,
            0,
            2,
            2,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            2,
            0,
            0,
            2,
            1,
            1,
            2,
            0,
        ]
    )
    return MixedData.from_blocks(continuous=x[:, None], unordered=g[:, None], unordered_levels=(3,))


@pytest.fixture
def selection_reference_y(float64_enabled):
    return jnp.array(
        [
            0.358665,
            0.343827,
            0.732115,
            1.874779,
            1.353788,
            -1.105842,
            1.273989,
            1.475742,
            0.994376,
            1.370530,
            -0.374642,
            0.875301,
            0.850921,
            1.141175,
            -1.282058,
            0.261782,
            0.624379,
            1.553095,
            -0.460164,
            -0.494416,
            0.045128,
            1.436988,
            -0.370277,
            0.784640,
            0.697855,
            -0.867377,
            -0.505643,
            0.294045,
            -1.025521,
            1.954664,
            1.934437,
            1.384901,
            -0.816764,
            1.133483,
            0.456806,
            0.955974,
            0.215037,
            0.935269,
            -0.086647,
            0.259349,
            -0.412404,
            -0.177470,
            1.479056,
            -0.606613,
            -0.368499,
            0.449553,
            0.951977,
            0.389662,
            0.482245,
            -0.038203,
        ]
    )


@pytest.fixture
def two_continuous():
    rng = np.random.default_rng(0)
    columns = np.column_stack([rng.normal(size=12), rng.uniform(size=12)])
    data = MixedData.continuous(columns)
    bw = Bandwidth(h=jnp.array([0.5, 0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    return data, bw


@pytest.fixture
def divisor_case():
    rng = np.random.default_rng(1)
    columns = np.column_stack([rng.normal(size=10), rng.uniform(size=10)])
    data = MixedData.continuous(columns)
    bw = Bandwidth(h=jnp.array([0.5, 0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    return data, bw


@pytest.fixture
def probe_x():
    return np.linspace(0.05, 0.95, 40)


@pytest.fixture
def probe_y(probe_x):
    return np.sin(2.0 * np.pi * probe_x)


@pytest.fixture
def probe_bw():
    return Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def conditional_sample():
    rng = np.random.default_rng(4242)
    n = 50
    x1 = rng.normal(size=n)
    levels = rng.integers(0, 3, n)
    response = 0.8 * x1 + rng.normal(0, 0.5, n)
    x = MixedData.from_blocks(continuous=x1, unordered=levels, unordered_levels=3)
    return x, MixedData.continuous(response)


@pytest.fixture
def conditional_bandwidth():
    return ConditionalBandwidth(
        x=Bandwidth(h=jnp.array([0.55]), lam_uno=jnp.array([0.25]), lam_ord=jnp.zeros(0)),
        y=Bandwidth(h=jnp.array([0.35]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)),
    )


@pytest.fixture
def categorical_response():
    rng = np.random.default_rng(7)
    n = 60
    score = rng.normal(size=n)
    group = np.clip(np.digitize(score + rng.normal(0, 0.6, n), [-0.7, 0.7]), 0, 2)
    x = MixedData.continuous(score)
    y = MixedData.from_blocks(unordered=group, unordered_levels=3)
    bandwidth = ConditionalBandwidth(
        x=Bandwidth(h=jnp.array([0.4]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)),
        y=Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.25]), lam_ord=jnp.zeros(0)),
    )
    return x, y, bandwidth
