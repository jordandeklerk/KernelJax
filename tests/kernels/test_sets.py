"""Tests for KernelSet."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.estimators.density import density
from kerneljax.kernels import AitchisonAitken, Gaussian, KernelSet, LiRacine, Op, WangVanRyzin


def test_defaults_are_expected_types():
    ks = KernelSet()
    assert isinstance(ks.continuous, Gaussian)
    assert isinstance(ks.unordered, AitchisonAitken)
    assert isinstance(ks.ordered, LiRacine)


def test_two_instances_compare_equal_and_hash_equal():
    assert KernelSet() == KernelSet()
    assert hash(KernelSet()) == hash(KernelSet())


def test_different_kernel_not_equal_to_default(bare_continuous_kernel_cls):
    assert KernelSet(continuous=bare_continuous_kernel_cls()) != KernelSet()


def test_a_tuple_of_two_kernel_sets_is_hashable(bare_continuous_kernel_cls):
    pair = (KernelSet(), KernelSet(continuous=bare_continuous_kernel_cls()))
    assert isinstance(hash(pair), int)


def test_module_exposes_kernel_set_without_a_cycle():
    import kerneljax.kernels as m

    assert m.KernelSet is KernelSet


def test_static_arg_equal_sets_do_not_recompile():
    calls = []

    def f(x, ks):
        calls.append(1)
        fn = getattr(ks.continuous, Op.VALUE)
        return fn(x, jnp.array(0.0), jnp.array(1.0))

    f_static = jax.jit(f, static_argnames=("ks",))
    first = f_static(jnp.array(0.5), KernelSet())
    second = f_static(jnp.array(0.7), KernelSet())
    assert jnp.isfinite(first)
    assert jnp.isfinite(second)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kernel_obj",
    [Gaussian(), AitchisonAitken(), WangVanRyzin(), LiRacine(), KernelSet()],
)
def test_static_registration_leaves_no_pytree_leaves(kernel_obj):
    assert jax.tree.leaves(kernel_obj) == []


def test_jit_accepts_kernel_set_no_static_argnames():
    def f(x, kernels):
        return kernels.continuous.value(x, jnp.array(0.0), jnp.array(1.0))

    assert jnp.isfinite(jax.jit(f)(jnp.array(0.5), KernelSet()))


def test_equal_sets_do_not_recompile():
    calls = {"n": 0}

    def f(x, kernels):
        calls["n"] += 1
        return kernels.continuous.value(x, jnp.array(0.0), jnp.array(1.0))

    jitted = jax.jit(f)
    jitted(jnp.array(0.5), KernelSet())
    jitted(jnp.array(0.9), KernelSet())
    assert calls["n"] == 1


def test_jit_accepts_user_defined_kernel_in_set(bare_continuous_kernel_cls):
    def f(x, kernels):
        return kernels.continuous.value(x, jnp.array(0.0), jnp.array(1.0))

    out = jax.jit(f)(jnp.array(0.5), KernelSet(continuous=bare_continuous_kernel_cls()))
    assert jnp.isfinite(out)


def test_kernel_set_jit_needs_no_static_argnames(public_api_data, public_api_bandwidth):
    calls = {"n": 0}

    def wrapped(data, bw, kernels):
        calls["n"] += 1
        return density(data, bw, kernels=kernels).value.sum()

    jitted = jax.jit(wrapped)
    jitted(public_api_data, public_api_bandwidth, KernelSet())
    jitted(public_api_data, public_api_bandwidth, KernelSet())

    assert calls["n"] == 1
