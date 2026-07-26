"""Tests for the kernel abstract base classes."""

import dataclasses

import jax.numpy as jnp
import pytest

from kerneljax.kernels.base import ContinuousKernel, Op, OrderedKernel


def test_op_constants_are_method_names():
    assert Op.VALUE == "value"
    assert Op.DERIV == "deriv"
    assert Op.CDF == "cdf"
    assert Op.CONV == "conv"


@pytest.mark.parametrize(
    ("kernel_cls_fixture", "op", "call_args"),
    [
        ("bare_continuous_kernel_cls", "deriv", (jnp.array(0.0), jnp.array(0.0), jnp.array(1.0))),
        ("bare_continuous_kernel_cls", "cdf", (jnp.array(0.0), jnp.array(0.0), jnp.array(1.0))),
        ("bare_continuous_kernel_cls", "conv", (jnp.array(0.0), jnp.array(0.0), jnp.array(1.0))),
        ("bare_unordered_kernel_cls", "cdf", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
        ("bare_unordered_kernel_cls", "conv", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
        ("bare_ordered_kernel_cls", "cdf", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
        ("bare_ordered_kernel_cls", "conv", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
    ],
)
def test_unimplemented_operators_raise_not_implemented(request, kernel_cls_fixture, op, call_args):
    kernel = request.getfixturevalue(kernel_cls_fixture)()
    fn = getattr(kernel, op)
    with pytest.raises(NotImplementedError, match=type(kernel).__name__):
        fn(*call_args)


def test_operator_resolves_by_getattr():
    @dataclasses.dataclass(frozen=True)
    class WithExtra(ContinuousKernel):
        def value(self, x, y, h):
            return jnp.ones_like(x * y * h)

        def my_op(self, x, y, h):
            return jnp.full_like(x * y * h, 7.0)

    k = WithExtra()
    op_name = "my_op"
    fn = getattr(k, op_name)
    assert fn(jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)) == 7.0


def test_ordered_kernel_value_uses_position_and_levels():
    @dataclasses.dataclass(frozen=True)
    class ToyOrdered(OrderedKernel):
        def value(self, x, y, lam, levels):
            return jnp.where(x == y, 1.0, jnp.power(lam, jnp.abs(x - y)) / levels)

        def upper_bound(self, levels):
            return 1.0

    k = ToyOrdered()
    same = k.value(jnp.array(2), jnp.array(2), jnp.array(0.5), 4)
    apart = k.value(jnp.array(0), jnp.array(3), jnp.array(0.5), 4)
    assert same == 1.0
    assert jnp.allclose(apart, 0.5**3 / 4)


@pytest.mark.parametrize(
    "kernel_cls_fixture",
    ["bare_continuous_kernel_cls", "bare_unordered_kernel_cls", "bare_ordered_kernel_cls"],
)
def test_kernels_are_hashable_for_the_jit_cache(request, kernel_cls_fixture):
    kernel_cls = request.getfixturevalue(kernel_cls_fixture)
    assert hash(kernel_cls()) == hash(kernel_cls())
