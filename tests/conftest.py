"""Shared pytest fixtures and configuration for the KernelJax test suite."""

import jax.numpy as jnp
import pytest

from kerneljax.data import ColumnSpec, Kind, MixedData


@pytest.fixture
def column_spec():
    return ColumnSpec(
        kinds=(Kind.CONTINUOUS, Kind.UNORDERED, Kind.ORDERED, Kind.CONTINUOUS),
        n_levels=(0, 4, 3, 0),
    )


@pytest.fixture
def continuous_data():
    return MixedData.continuous(jnp.ones((5, 2)))
