"""Tests for the constrained/unconstrained bandwidth bijection."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.bandwidth import Bandwidth, BandwidthTransform, ConditionalBandwidth, normal_reference
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
    assert 0.0 < bw.lam_uno[0] < 1.0
    assert 0.0 < bw.lam_ord[0] < 1.0
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
        uno=jnp.array([[i % 3] for i in range(9)]),
        orde=jnp.array([[i % 2] for i in range(9)]),
        uno_levels=(3,),
        ord_levels=(2,),
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
