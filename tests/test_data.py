"""Tests for the mixed-type design matrix."""

import dataclasses
import operator

import jax
import jax.numpy as jnp
import pytest

from kerneljax.data import ColumnSpec, Kind, MixedData


@pytest.mark.parametrize(
    ("attr", "expected"),
    [("p", 4), ("p_con", 2), ("p_uno", 1), ("p_ord", 1)],
)
def test_column_spec_counts(column_spec, attr, expected):
    assert getattr(column_spec, attr) == expected


def test_column_spec_is_hashable():
    spec = ColumnSpec(kinds=(Kind.CONTINUOUS,), n_levels=(0,))
    assert hash(spec) == hash(ColumnSpec(kinds=(Kind.CONTINUOUS,), n_levels=(0,)))


def test_from_blocks_builds_signed_int32_codes():
    data = MixedData.from_blocks(con=jnp.ones((5, 2)), uno=jnp.zeros((5, 1)), uno_levels=(3,))
    assert data.n == 5
    assert data.uno.dtype == jnp.int32
    assert jnp.issubdtype(data.uno.dtype, jnp.signedinteger)


@pytest.mark.parametrize(
    ("attr_path", "expected"),
    [("uno.shape", (5, 0)), ("orde.shape", (5, 0)), ("spec.p_uno", 0)],
)
def test_empty_blocks_have_zero_width(continuous_data, attr_path, expected):
    assert operator.attrgetter(attr_path)(continuous_data) == expected


def test_is_a_pytree_with_static_spec(continuous_data):
    leaves, treedef = jax.tree_util.tree_flatten(continuous_data)
    assert len(leaves) == 3
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.spec == continuous_data.spec


def test_survives_vmap_which_passes_sentinels():
    stacked = MixedData.continuous(jnp.ones((4, 5, 2)))
    out = jax.vmap(lambda d: d.con.sum())(stacked)
    assert out.shape == (4,)


def test_jit_does_not_retrace_on_value_change():
    calls = {"n": 0}

    @jax.jit
    def f(d):
        calls["n"] += 1
        return d.con.sum()

    a = MixedData.continuous(jnp.ones((5, 2)))
    b = MixedData.continuous(jnp.full((5, 2), 3.0))
    f(a)
    f(b)
    assert calls["n"] == 1


def test_from_blocks_rejects_out_of_range_codes():
    with pytest.raises(ValueError, match="outside"):
        MixedData.from_blocks(uno=jnp.array([[0], [9]]), uno_levels=(3,))


def test_frozen(continuous_data):
    with pytest.raises(dataclasses.FrozenInstanceError):
        continuous_data.con = jnp.zeros((5, 2))
