"""Shared pytest fixtures and configuration for the KernelJax test suite."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.bandwidth import Bandwidth, BandwidthTransform
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
        con=jnp.linspace(0.0, 1.0, 12).reshape(12, 1),
        uno=jnp.array([[i % 4] for i in range(12)]),
        orde=jnp.array([[i % 3] for i in range(12)]),
        uno_levels=(4,),
        ord_levels=(3,),
    )


@pytest.fixture
def mixed_bandwidth_transform(mixed_bandwidth_data):
    return BandwidthTransform(spec=mixed_bandwidth_data.spec, kernels=KernelSet())


@pytest.fixture
def kweights_train():
    rng = np.random.default_rng(0)
    return MixedData.from_blocks(
        con=jnp.asarray(rng.normal(size=(6, 1))),
        uno=jnp.asarray(rng.integers(0, 4, size=(6, 1))),
        orde=jnp.asarray(rng.integers(0, 3, size=(6, 1))),
        uno_levels=(4,),
        ord_levels=(3,),
    )


@pytest.fixture
def kweights_bandwidth():
    return Bandwidth(h=jnp.array([0.7]), lam_uno=jnp.array([0.15]), lam_ord=jnp.array([0.25]))


@pytest.fixture
def continuous_only_bandwidth():
    return Bandwidth(h=jnp.array([1.0, 1.0]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))


@pytest.fixture
def discrete_only_data():
    return MixedData.from_blocks(uno=jnp.array([[0], [1], [2]]), uno_levels=(3,))


@pytest.fixture
def discrete_only_bandwidth():
    return Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.2]), lam_ord=jnp.zeros(0))


@pytest.fixture
def two_unordered_data():
    return MixedData.from_blocks(
        uno=jnp.array([[0, 0], [1, 1], [2, 0], [0, 1]]),
        uno_levels=(3, 2),
    )


@pytest.fixture
def two_unordered_bandwidth():
    return Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.2, 0.3]), lam_ord=jnp.zeros(0))


@pytest.fixture
def ksum_data():
    return MixedData.from_blocks(
        con=jnp.asarray(np.linspace(-1.0, 2.0, 6)).reshape(6, 1),
        uno=jnp.array([[0], [1], [2], [3], [0], [1]]),
        uno_levels=(4,),
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
        con=jnp.asarray(rng.normal(size=(15, 1))),
        uno=jnp.asarray(rng.integers(0, 3, size=(15, 1))),
        orde=jnp.asarray(rng.integers(0, 4, size=(15, 1))),
        uno_levels=(3,),
        ord_levels=(4,),
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
        con=con,
        uno=jnp.asarray(combos[:, :1]),
        orde=jnp.asarray(combos[:, 1:]),
        uno_levels=(3,),
        ord_levels=(3,),
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
        uno=jnp.asarray(combos[:, :1]),
        orde=jnp.asarray(combos[:, 1:]),
        uno_levels=(3,),
        ord_levels=(3,),
    )


@pytest.fixture
def cv_discrete_bandwidth():
    return Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.3]), lam_ord=jnp.array([0.4]))


@pytest.fixture
def public_api_data():
    return MixedData.from_blocks(
        con=jnp.linspace(-1.0, 2.0, 8).reshape(8, 1),
        uno=jnp.asarray([[i % 3] for i in range(8)]),
        orde=jnp.asarray([[i % 2] for i in range(8)]),
        uno_levels=(3,),
        ord_levels=(2,),
    )


@pytest.fixture
def public_api_bandwidth():
    return Bandwidth(h=jnp.array([0.6]), lam_uno=jnp.array([0.2]), lam_ord=jnp.array([0.3]))
