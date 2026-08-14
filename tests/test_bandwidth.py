"""Tests for the bandwidth parameter tree."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.bandwidth import Bandwidth, _search_start, broadcast_h, normal_reference
from kerneljax.data import MixedData
from kerneljax.estimators.density import density
from kerneljax.estimators.distribution import cdf
from kerneljax.estimators.regression import local_poly
from kerneljax.kernels import KernelSet


def test_leaves_are_all_inexact(bandwidth):
    leaves = jax.tree_util.tree_leaves(bandwidth)
    assert len(leaves) == 3
    assert all(jnp.issubdtype(leaf.dtype, jnp.inexact) for leaf in leaves)


def test_h_axis_is_static_metadata(bandwidth):
    _, treedef_a = jax.tree_util.tree_flatten(bandwidth)
    _, treedef_b = jax.tree_util.tree_flatten(bandwidth.replace(h_axis="eval"))
    assert treedef_a != treedef_b


def test_grad_returns_a_bandwidth_of_gradients(bandwidth):
    g = jax.grad(lambda bw: bw.h.sum() * bw.lam_uno.sum())(bandwidth)
    assert isinstance(g, Bandwidth)
    assert g.h.shape == bandwidth.h.shape


def test_replace_swaps_a_field(bandwidth):
    updated = bandwidth.replace(h_axis="train")
    assert updated.h_axis == "train"
    assert bandwidth.h_axis == "shared"
    assert updated.h is bandwidth.h


@pytest.mark.parametrize(
    ("attr", "shape"),
    [("lam_uno", (0,)), ("lam_ord", (0,))],
)
def test_empty_blocks_have_zero_width(attr, shape):
    bw = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    assert getattr(bw, attr).shape == shape


def test_jit_no_retrace_when_leaf_values_change():
    calls = {"n": 0}

    @jax.jit
    def f(bw):
        calls["n"] += 1
        return bw.h.sum()

    a = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    b = Bandwidth(h=jnp.array([1.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    f(a)
    f(b)
    assert calls["n"] == 1


@pytest.mark.parametrize(
    ("axis", "shape", "expected"),
    [
        ("shared", (2,), (1, 1, 2)),
        ("eval", (7, 2), (7, 1, 2)),
        ("train", (5, 2), (1, 5, 2)),
    ],
)
def test_broadcast_h_produces_three_axis_shapes(axis, shape, expected):
    bw = Bandwidth(h=jnp.ones(shape), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0), h_axis=axis)
    assert broadcast_h(bw, p_con=2).shape == expected


@pytest.mark.parametrize(
    ("axis", "shape"),
    [
        ("shared", (3,)),
        ("eval", (7, 3)),
        ("train", (5, 3)),
    ],
)
def test_broadcast_h_rejects_wrong_trailing_width(axis, shape):
    bw = Bandwidth(h=jnp.ones(shape), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0), h_axis=axis)
    with pytest.raises(ValueError, match="p_con"):
        broadcast_h(bw, p_con=2)


@pytest.mark.parametrize(
    ("axis", "shape"),
    [
        ("shared", (2, 2)),
        ("eval", (2,)),
        ("train", (2,)),
    ],
)
def test_broadcast_h_rejects_wrong_rank_for_the_tag(axis, shape):
    bw = Bandwidth(h=jnp.ones(shape), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0), h_axis=axis)
    with pytest.raises(ValueError, match="dimensional"):
        broadcast_h(bw, p_con=2)


def test_ambiguous_per_observation_vector_raises():
    bw = Bandwidth(h=jnp.ones((5,)), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0), h_axis="train")
    with pytest.raises(ValueError, match="two dimensional"):
        broadcast_h(bw, p_con=1)


def test_search_start_lambda_is_interior(mixed_bandwidth_data):
    kernels = KernelSet()
    start = _search_start(mixed_bandwidth_data, kernels)
    spec = mixed_bandwidth_data.spec
    unordered = jnp.asarray([kernels.unordered.upper_bound(levels) for levels in spec.uno_levels])
    ordered = jnp.asarray([kernels.ordered.upper_bound(levels) for levels in spec.ord_levels])

    assert jnp.all(start.lam_uno > 0.0)
    assert jnp.all(start.lam_uno < unordered)
    assert jnp.all(start.lam_ord > 0.0)
    assert jnp.all(start.lam_ord < ordered)


@pytest.mark.parametrize("bad", [jnp.nan, 0.0, -0.5, jnp.inf])
@pytest.mark.parametrize("estimator", ["local_poly", "density", "cdf"])
def test_an_unusable_bandwidth_is_refused_at_the_point_of_use(bad, estimator):
    x = jnp.linspace(0.0, 1.0, 24)
    bw = Bandwidth(h=jnp.array([bad]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

    call = {
        "local_poly": lambda: local_poly(x, jnp.sin(x), bw, degree=1),
        "density": lambda: density(x, bw),
        "cdf": lambda: cdf(x, bw),
    }[estimator]

    with pytest.raises(ValueError, match="finite and positive"):
        call()


def test_a_negative_smoothing_parameter_is_refused():
    data = MixedData.from_blocks(
        continuous=jnp.linspace(0.0, 1.0, 24), unordered=jnp.arange(24) % 3, unordered_levels=3
    )
    bw = Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.array([-0.1]), lam_ord=jnp.zeros(0))

    with pytest.raises(ValueError, match="finite and non-negative"):
        density(data, bw)


def test_a_usable_bandwidth_passes_untouched(bandwidth):
    x = jnp.linspace(0.0, 1.0, 24)
    fit = local_poly(x, jnp.sin(x), Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)))
    assert bool(jnp.all(jnp.isfinite(fit.mean)))


def test_normal_reference_leaves_categorical_smoothing_at_zero(mixed_bandwidth_data):
    bw = normal_reference(mixed_bandwidth_data, KernelSet())

    assert float(bw.h[0]) > 0.0
    assert jnp.all(bw.lam_uno == 0.0)
    assert jnp.all(bw.lam_ord == 0.0)


def test_normal_reference_keeps_float32_data_in_float32_under_x64(float64_enabled):
    import numpy as np

    data = MixedData.continuous(np.random.default_rng(0).normal(size=20).astype(np.float32))
    reference = normal_reference(data, KernelSet())
    assert reference.h.dtype == jnp.float32
