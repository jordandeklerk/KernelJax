"""Shared pytest fixtures and configuration for the KernelJax test suite."""

import dataclasses

import jax.numpy as jnp
import pytest

from kerneljax.data import ColumnSpec, Kind, MixedData
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


@pytest.fixture
def gaussian():
    return Gaussian()


@pytest.fixture
def aitchison_aitken():
    return AitchisonAitken()


@pytest.fixture
def wang_van_ryzin():
    return WangVanRyzin()
