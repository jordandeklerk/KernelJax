"""Tests for KernelSet."""

import jax
import jax.numpy as jnp

from kerneljax.kernels import AitchisonAitken, Gaussian, KernelSet, Op, WangVanRyzin


def test_defaults_are_expected_types():
    ks = KernelSet()
    assert isinstance(ks.continuous, Gaussian)
    assert isinstance(ks.unordered, AitchisonAitken)
    assert isinstance(ks.ordered, WangVanRyzin)


def test_two_instances_compare_equal_and_hash_equal():
    assert KernelSet() == KernelSet()
    assert hash(KernelSet()) == hash(KernelSet())


def test_a_different_kernel_is_not_equal_to_the_default(bare_continuous_kernel_cls):
    assert KernelSet(continuous=bare_continuous_kernel_cls()) != KernelSet()


def test_a_tuple_of_two_kernel_sets_is_hashable(bare_continuous_kernel_cls):
    pair = (KernelSet(), KernelSet(continuous=bare_continuous_kernel_cls()))
    assert isinstance(hash(pair), int)


def test_module_exposes_kernel_set_without_a_cycle():
    import kerneljax.kernels as m

    assert m.KernelSet is KernelSet


def test_static_argument_does_not_recompile_for_an_equal_kernel_set():
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
