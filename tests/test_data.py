"""Tests for the mixed-type design matrix."""

import dataclasses
import operator

import jax
import jax.numpy as jnp
import pytest

from kerneljax.data import ColumnSpec, Kind, MixedData, grid, quantile_grid


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
    data = MixedData.from_blocks(continuous=jnp.ones((5, 2)), unordered=jnp.zeros((5, 1)), unordered_levels=(3,))
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
        MixedData.from_blocks(unordered=jnp.array([[0], [9]]), unordered_levels=(3,))


def test_frozen(continuous_data):
    with pytest.raises(dataclasses.FrozenInstanceError):
        continuous_data.con = jnp.zeros((5, 2))


@pytest.mark.parametrize("n", [3, 7, 50])
def test_continuous_sweep_has_n_rows(grid_sample, n):
    assert grid(grid_sample, vary=0, n=n).con.shape == (n, 2)


@pytest.mark.parametrize("vary, rows", [(2, 3), (3, 4)])
def test_categorical_sweep_has_one_row_per_level(grid_sample, vary, rows):
    assert grid(grid_sample, vary=vary, n=99).con.shape[0] == rows


def test_swept_column_spans_the_observed_range(grid_sample):
    points = grid(grid_sample, vary=0, n=9)
    assert float(points.con[:, 0].min()) == pytest.approx(float(grid_sample.con[:, 0].min()))
    assert float(points.con[:, 0].max()) == pytest.approx(float(grid_sample.con[:, 0].max()))


def test_pinned_continuous_sits_at_its_quantile(grid_sample):
    points = grid(grid_sample, vary=0, n=5, quantile=0.25)
    want = jnp.quantile(grid_sample.con[:, 1], 0.25)
    assert jnp.allclose(points.con[:, 1], want)


def test_pinned_unordered_sits_at_the_mode(grid_sample):
    points = grid(grid_sample, vary=0, n=5)
    counts = jnp.bincount(grid_sample.uno[:, 0], length=3)
    assert jnp.all(points.uno[:, 0] == jnp.argmax(counts))


def test_swept_categorical_covers_every_level(grid_sample):
    assert jnp.array_equal(grid(grid_sample, vary=2).uno[:, 0], jnp.arange(3))


def test_grid_shares_the_input_spec(grid_sample):
    assert grid(grid_sample).spec == grid_sample.spec


def test_grid_runs_under_jit(grid_sample):
    assert jax.jit(lambda d: grid(d, n=6))(grid_sample).con.shape == (6, 2)


def test_grid_accepts_a_raw_array():
    raw = jnp.linspace(-1.0, 1.0, 10)
    assert grid(raw, n=4).con.shape == (4, 1)


def test_grid_selects_a_column_by_name():
    data = MixedData.continuous(jnp.zeros((5, 2))).replace(
        spec=ColumnSpec(kinds=(Kind.CONTINUOUS, Kind.CONTINUOUS), n_levels=(0, 0), names=("age", "wage"))
    )
    assert grid(data, vary="wage", n=3).con.shape == (3, 2)


@pytest.mark.parametrize("vary", [4, -5, "missing"])
def test_unknown_swept_column_raises(grid_sample, vary):
    with pytest.raises(ValueError):
        grid(grid_sample, vary=vary)


def test_default_sweep_spans_the_full_range(grid_sample):
    points = grid(grid_sample, vary=0, n=9)
    column = grid_sample.con[:, 0]
    assert float(points.con[:, 0].min()) == pytest.approx(float(column.min()), rel=1e-6)
    assert float(points.con[:, 0].max()) == pytest.approx(float(column.max()), rel=1e-6)


@pytest.mark.parametrize("trim", [0.05, 0.1, 0.25])
def test_positive_trim_pulls_the_range_in(grid_sample, trim):
    column = grid_sample.con[:, 0]
    points = grid(grid_sample, vary=0, n=9, trim=trim)
    assert float(points.con[:, 0].min()) > float(column.min())
    assert float(points.con[:, 0].max()) < float(column.max())
    assert float(points.con[:, 0].min()) == pytest.approx(float(jnp.quantile(column, trim)), rel=1e-6)
    assert float(points.con[:, 0].max()) == pytest.approx(float(jnp.quantile(column, 1.0 - trim)), rel=1e-6)


@pytest.mark.parametrize("trim", [-0.05, -0.1, -0.25])
def test_negative_trim_reaches_past_the_data(grid_sample, trim):
    column = grid_sample.con[:, 0]
    points = grid(grid_sample, vary=0, n=9, trim=trim)
    assert float(points.con[:, 0].min()) < float(column.min())
    assert float(points.con[:, 0].max()) > float(column.max())


def test_negative_trim_reflects_the_inner_quantiles(grid_sample):
    column = grid_sample.con[:, 0]
    points = grid(grid_sample, vary=0, n=9, trim=-0.1)
    want_low = 2.0 * column.min() - jnp.quantile(column, 0.1)
    want_high = 2.0 * column.max() - jnp.quantile(column, 0.9)
    assert float(points.con[:, 0].min()) == pytest.approx(float(want_low), rel=1e-6)
    assert float(points.con[:, 0].max()) == pytest.approx(float(want_high), rel=1e-6)


def test_trim_does_not_move_a_categorical_sweep(grid_sample):
    assert jnp.array_equal(grid(grid_sample, vary=2, trim=0.25).uno[:, 0], grid(grid_sample, vary=2).uno[:, 0])


@pytest.mark.parametrize("n", [3, 7, 100])
def test_quantile_grid_has_n_rows(grid_sample, n):
    points = quantile_grid(grid_sample, n=n)
    assert points.con.shape == (n, 2)
    assert points.uno.shape == (n, 1)
    assert points.orde.shape == (n, 1)


def test_quantile_grid_rows_are_aligned_not_crossed():
    data = MixedData.continuous(jnp.stack([jnp.linspace(0.0, 1.0, 20), jnp.linspace(5.0, 6.0, 20)], axis=1))
    points = quantile_grid(data, n=9)
    assert points.con.shape == (9, 2)


def test_quantile_grid_continuous_are_quantiles(grid_sample):
    probs = jnp.linspace(0.0, 1.0, 11)
    points = quantile_grid(grid_sample, n=11)
    for pos in range(2):
        want = jnp.quantile(grid_sample.con[:, pos], probs)
        assert jnp.allclose(points.con[:, pos], want, rtol=1e-6)


def test_quantile_grid_unordered_is_the_mode(grid_sample):
    points = quantile_grid(grid_sample, n=11)
    counts = jnp.bincount(grid_sample.uno[:, 0], length=3)
    assert jnp.all(points.uno[:, 0] == jnp.argmax(counts))


def test_quantile_grid_ordered_is_nondecreasing(grid_sample):
    points = quantile_grid(grid_sample, n=25)
    assert jnp.all(jnp.diff(points.orde[:, 0]) >= 0)


def test_quantile_grid_shares_the_input_spec(grid_sample):
    assert quantile_grid(grid_sample, n=5).spec == grid_sample.spec


def test_quantile_grid_runs_under_jit(grid_sample):
    assert jax.jit(lambda d: quantile_grid(d, n=6))(grid_sample).con.shape == (6, 2)


def test_quantile_grid_accepts_a_raw_array():
    assert quantile_grid(jnp.linspace(-1.0, 1.0, 12), n=4).con.shape == (4, 1)


def test_from_blocks_promotes_one_dimension():
    flat = MixedData.from_blocks(continuous=jnp.arange(4.0), unordered=jnp.array([0, 1, 1, 0]), unordered_levels=(2,))
    shaped = MixedData.from_blocks(
        continuous=jnp.arange(4.0)[:, None], unordered=jnp.array([0, 1, 1, 0])[:, None], unordered_levels=(2,)
    )

    assert flat.con.shape == (4, 1)
    assert flat.uno.shape == (4, 1)
    assert jnp.array_equal(flat.con, shaped.con)
    assert jnp.array_equal(flat.uno, shaped.uno)
    assert flat.spec == shaped.spec


def test_from_blocks_infers_levels():
    codes = jnp.array([0, 1, 2, 1, 0])

    inferred = MixedData.from_blocks(unordered=codes, ordered=codes)
    explicit = MixedData.from_blocks(unordered=codes, ordered=codes, unordered_levels=(3,), ordered_levels=(3,))

    assert inferred.spec.n_levels == (3, 3)
    assert inferred.spec == explicit.spec


def test_explicit_levels_beat_inference():
    codes = jnp.array([0, 1, 1, 0])

    widened = MixedData.from_blocks(unordered=codes, unordered_levels=(4,))

    assert widened.spec.n_levels == (4,)


def test_integer_levels_match_single_entry_tuple():
    codes = jnp.array([0, 1, 1, 0])

    integer = MixedData.from_blocks(unordered=codes, ordered=codes, unordered_levels=4, ordered_levels=4)
    tupled = MixedData.from_blocks(unordered=codes, ordered=codes, unordered_levels=(4,), ordered_levels=(4,))

    assert integer.spec == tupled.spec


def test_integer_levels_broadcast_across_a_block():
    codes = jnp.array([[0, 1], [1, 2], [2, 0]])

    data = MixedData.from_blocks(unordered=codes, unordered_levels=3)

    assert data.spec.n_levels == (3, 3)


def test_integer_levels_over_declaring_a_multi_column_block_are_rejected():
    codes = jnp.array([[0, 0], [1, 1], [2, 2], [0, 5]])

    with pytest.raises(ValueError, match=r"over-declares unordered column 0"):
        MixedData.from_blocks(unordered=codes, unordered_levels=6)


def test_tuple_levels_may_over_declare_a_multi_column_block():
    codes = jnp.array([[0, 0], [1, 1], [2, 2], [0, 5]])

    data = MixedData.from_blocks(unordered=codes, unordered_levels=(6, 6))

    assert data.spec.n_levels == (6, 6)


def test_integer_levels_may_over_declare_a_single_column():
    codes = jnp.array([0, 1, 1, 0])

    data = MixedData.from_blocks(unordered=codes, unordered_levels=6)

    assert data.spec.n_levels == (6,)


def test_integer_levels_too_small_for_a_column_are_rejected():
    codes = jnp.array([[0, 4], [1, 2], [2, 0]])

    with pytest.raises(ValueError, match=r"unordered column 1 has codes outside \[0, 3\)"):
        MixedData.from_blocks(unordered=codes, unordered_levels=3)


@pytest.mark.parametrize("build", [jnp.arange(10), jnp.arange(10).reshape(10, 1)])
def test_integer_continuous_input_is_promoted_to_float(build):
    data = MixedData.from_blocks(continuous=build, names=("x",))
    assert jnp.issubdtype(data.con.dtype, jnp.floating)


def test_integer_continuous_input_flows_through_the_grids():
    data = MixedData.from_blocks(continuous=jnp.arange(10), names=("x",))
    line = grid(data, vary="x", n=5)
    spread = quantile_grid(data, n=5)
    assert float(line.con[-1, 0]) == 9.0
    assert float(spread.con[-1, 0]) == 9.0


def test_spec_names_stay_out_of_equality_and_the_jit_cache():
    with_names = MixedData.from_blocks(continuous=jnp.arange(6.0), names=("x",))
    renamed = MixedData.from_blocks(continuous=jnp.arange(6.0), names=("y",))

    assert with_names.spec == renamed.spec
    assert hash(with_names.spec) == hash(renamed.spec)

    calls = []

    @jax.jit
    def total(data):
        calls.append(1)
        return jnp.sum(data.con)

    total(with_names)
    total(renamed)

    assert len(calls) == 1
