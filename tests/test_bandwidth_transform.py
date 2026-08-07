"""Tests for the constrained/unconstrained bandwidth bijection."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.bandwidth import (
    Bandwidth,
    BandwidthTransform,
    ConditionalBandwidth,
    _search_start,
    normal_reference,
)
from kerneljax.data import MixedData
from kerneljax.kernels import KernelSet


def test_round_trip_is_identity(mixed_bandwidth_transform):
    bw = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.array([0.3]), lam_ord=jnp.array([0.4]))
    back = mixed_bandwidth_transform.from_unconstrained(mixed_bandwidth_transform.to_unconstrained(bw))
    assert jnp.allclose(back.h, bw.h, rtol=1e-6)
    assert jnp.allclose(back.lam_uno, bw.lam_uno, rtol=1e-6)
    assert jnp.allclose(back.lam_ord, bw.lam_ord, rtol=1e-6)


@pytest.mark.parametrize("z", [-40.0, -3.0, 0.0, 3.0, 40.0])
def test_from_unconstrained_respects_the_boxes(mixed_bandwidth_transform, z):
    bw = mixed_bandwidth_transform.from_unconstrained(jnp.full((3,), z))
    assert bw.h[0] > 0.0
    assert 0.0 <= bw.lam_uno[0] <= 0.75
    assert 0.0 <= bw.lam_ord[0] <= 1.0


def test_inverse_is_finite_at_the_boundary(mixed_bandwidth_transform):
    bw = Bandwidth(h=jnp.array([1e-30]), lam_uno=jnp.array([0.75]), lam_ord=jnp.array([1.0]))
    z = mixed_bandwidth_transform.to_unconstrained(bw)
    assert jnp.all(jnp.isfinite(z))


def test_gradient_flows_through_the_transform(mixed_bandwidth_transform):
    g = jax.grad(lambda z: mixed_bandwidth_transform.from_unconstrained(z).h.sum())(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("z", [-40.0, -3.0, 0.0, 3.0, 40.0])
def test_gradient_is_finite_at_extreme_z(mixed_bandwidth_transform, z):
    def loss(z):
        bw = mixed_bandwidth_transform.from_unconstrained(z)
        return bw.h.sum() + bw.lam_uno.sum() + bw.lam_ord.sum()

    g = jax.grad(loss)(jnp.full((3,), z))
    assert jnp.all(jnp.isfinite(g))


def test_bounds_match_the_kernel_upper_bounds(mixed_bandwidth_transform):
    lo, hi = mixed_bandwidth_transform.bounds()
    assert lo.shape == hi.shape == (3,)
    assert jnp.all(lo == 0.0)
    assert hi[0] == jnp.inf
    assert hi[1] == pytest.approx(0.75)
    assert hi[2] == pytest.approx(1.0)


def test_normal_reference_positive_and_shaped(mixed_bandwidth_data):
    bw = normal_reference(mixed_bandwidth_data, KernelSet())
    assert bw.h.shape == (1,)
    assert bw.h[0] > 0.0
    assert bw.lam_uno.shape == (1,)
    assert bw.lam_ord.shape == (1,)
    assert bw.h_axis == "shared"


def test_normal_reference_handles_a_constant_column():
    data = MixedData.continuous(jnp.full((10, 1), 3.0))
    bw = normal_reference(data, KernelSet())
    assert bw.h[0] > 0.0
    assert jnp.all(jnp.isfinite(bw.h))


def test_conditional_bandwidth_is_a_pytree():
    inner = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.array([0.2]), lam_ord=jnp.array([0.3]))
    cond = ConditionalBandwidth(x=inner, y=inner)
    leaves = jax.tree_util.tree_leaves(cond)
    assert len(leaves) == 6
    g = jax.grad(lambda c: c.x.h.sum() + c.y.h.sum())(cond)
    assert isinstance(g, ConditionalBandwidth)


@pytest.mark.parametrize("h_axis", ["eval", "train"])
def test_non_shared_bandwidth_is_rejected(mixed_bandwidth_transform, h_axis):
    bw = Bandwidth(
        h=jnp.full((12, 1), 0.5),
        lam_uno=jnp.array([0.3]),
        lam_ord=jnp.array([0.4]),
        h_axis=h_axis,
    )
    with pytest.raises(ValueError, match="h_axis"):
        mixed_bandwidth_transform.to_unconstrained(bw)


def test_normal_reference_start_is_shared(mixed_bandwidth_data):
    assert normal_reference(mixed_bandwidth_data, KernelSet()).h_axis == "shared"


def test_purely_continuous_spec_round_trips():
    data = MixedData.continuous(jnp.linspace(0.0, 1.0, 10).reshape(10, 1))
    tf = BandwidthTransform(spec=data.spec, kernels=KernelSet())
    bw = Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    back = tf.from_unconstrained(tf.to_unconstrained(bw))
    assert jnp.allclose(back.h, bw.h, rtol=1e-6)
    assert back.lam_uno.shape == (0,)
    assert back.lam_ord.shape == (0,)
    lo, hi = tf.bounds()
    assert lo.shape == hi.shape == (1,)
    assert hi[0] == jnp.inf


def test_purely_categorical_spec_round_trips():
    data = MixedData.from_blocks(
        unordered=jnp.array([[i % 3] for i in range(9)]),
        ordered=jnp.array([[i % 2] for i in range(9)]),
        unordered_levels=(3,),
        ordered_levels=(2,),
    )
    tf = BandwidthTransform(spec=data.spec, kernels=KernelSet())
    bw = Bandwidth(h=jnp.zeros(0), lam_uno=jnp.array([0.4]), lam_ord=jnp.array([0.2]))
    back = tf.from_unconstrained(tf.to_unconstrained(bw))
    assert back.h.shape == (0,)
    assert jnp.allclose(back.lam_uno, bw.lam_uno, rtol=1e-6)
    assert jnp.allclose(back.lam_ord, bw.lam_ord, rtol=1e-6)
    bw_normal_reference = normal_reference(data, KernelSet())
    assert bw_normal_reference.h.shape == (0,)
    assert bw_normal_reference.lam_uno.shape == (1,)
    assert bw_normal_reference.lam_ord.shape == (1,)


@pytest.mark.parametrize("target", ["density", "distribution"])
def test_the_search_start_halves_the_bound(mixed_bandwidth_data, target):
    kernels = KernelSet()
    bw = _search_start(mixed_bandwidth_data, kernels, target=target)
    spec = mixed_bandwidth_data.spec

    assert bw.lam_uno[0] == kernels.unordered.upper_bound(spec.uno_levels[0]) / 2.0
    assert bw.lam_ord[0] == kernels.ordered.upper_bound(spec.ord_levels[0]) / 2.0


def test_distribution_target_uses_its_own_rate(mixed_bandwidth_data):
    data = MixedData.continuous(mixed_bandwidth_data.con)
    n = data.n
    scale = jnp.min(
        jnp.stack(
            [
                jnp.std(data.con, axis=0, ddof=1),
                (jnp.percentile(data.con, 75.0, axis=0) - jnp.percentile(data.con, 25.0, axis=0))
                / (2.0 * jax.scipy.special.ndtri(0.75)),
                jnp.median(jnp.abs(data.con - jnp.median(data.con, axis=0)), axis=0) * 1.4826,
            ]
        ),
        axis=0,
    )

    want = 1.587 * scale * n ** (-1.0 / 3.0)
    got = normal_reference(data, KernelSet(), target="distribution").h
    assert jnp.allclose(got, want, rtol=1e-6)


def test_distribution_rate_ignores_column_count():
    single = MixedData.continuous(jnp.linspace(-2.0, 2.0, 30).reshape(-1, 1))
    double = MixedData.continuous(jnp.tile(jnp.linspace(-2.0, 2.0, 30).reshape(-1, 1), (1, 2)))

    one = normal_reference(single, KernelSet(), target="distribution").h[0]
    two = normal_reference(double, KernelSet(), target="distribution").h[0]
    assert float(one) == pytest.approx(float(two), rel=1e-6)


def test_density_rate_does_depend_on_column_count():
    single = MixedData.continuous(jnp.linspace(-2.0, 2.0, 30).reshape(-1, 1))
    double = MixedData.continuous(jnp.tile(jnp.linspace(-2.0, 2.0, 30).reshape(-1, 1), (1, 2)))

    one = normal_reference(single, KernelSet()).h[0]
    two = normal_reference(double, KernelSet()).h[0]
    assert float(one) != pytest.approx(float(two), rel=1e-3)


@pytest.mark.parametrize(
    "sample",
    ["mixed_bandwidth_data", "discrete_only_data", "two_unordered_data", "cv_continuous_data"],
)
def test_start_is_not_in_a_transform_tail(request, sample):
    data = request.getfixturevalue(sample)
    kernels = KernelSet()
    transform = BandwidthTransform(spec=data.spec, kernels=kernels)

    z = transform.to_unconstrained(_search_start(data, kernels))

    assert jnp.all(jnp.abs(z) < 10.0)
