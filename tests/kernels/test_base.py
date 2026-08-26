"""Tests for the kernel abstract base classes."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from kerneljax.kernels import KernelSet
from kerneljax.kernels.base import ContinuousKernel, Op, OrderedKernel, UnorderedKernel


def test_op_constants_are_method_names():
    assert Op.VALUE == "value"
    assert Op.DERIV == "deriv"
    assert Op.CDF == "cdf"
    assert Op.CONV == "conv"


@pytest.mark.parametrize(
    ("kernel_cls_fixture", "op", "call_args"),
    [
        ("bare_unordered_kernel_cls", "cdf", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
        ("bare_unordered_kernel_cls", "conv", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
        ("bare_ordered_kernel_cls", "cdf", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
        ("bare_ordered_kernel_cls", "conv", (jnp.array(0), jnp.array(1), jnp.array(0.5), 3)),
    ],
)
def test_unimplemented_ops_raise_not_implemented(request, kernel_cls_fixture, op, call_args):
    kernel = request.getfixturevalue(kernel_cls_fixture)()
    fn = getattr(kernel, op)
    with pytest.raises(NotImplementedError, match=type(kernel).__name__):
        fn(*call_args)


def test_operator_resolves_by_getattr(with_extra_kernel_cls):
    k = with_extra_kernel_cls()
    op_name = "my_op"
    fn = getattr(k, op_name)
    assert fn(jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)) == 7.0


def test_ordered_value_uses_position_and_levels(toy_ordered_kernel_cls):
    k = toy_ordered_kernel_cls()
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


class Plain(ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)


class WithSetting(ContinuousKernel):
    power: float = 3.0

    def value(self, x, y, h):
        return (1.0 - jnp.abs((x - y) / h)) ** self.power


@dataclasses.dataclass(frozen=True)
class Decorated(ContinuousKernel):
    def value(self, x, y, h):
        return x - y + h


class Derived(WithSetting):
    scale: float = 2.0


class PlainUnordered(UnorderedKernel):
    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, lam)

    def upper_bound(self, levels):
        return 1.0


class PlainOrdered(OrderedKernel):
    def value(self, x, y, lam, levels):
        return lam ** jnp.abs(x - y)

    def upper_bound(self, levels):
        return 1.0


@pytest.mark.parametrize("kernel_cls", [Plain, Decorated, PlainUnordered, PlainOrdered])
def test_a_bare_class_compares_and_hashes_by_value(kernel_cls):
    assert kernel_cls() == kernel_cls()
    assert hash(kernel_cls()) == hash(kernel_cls())


def test_a_setting_is_configuration():
    assert WithSetting(2.0) != WithSetting(3.0)
    assert WithSetting(3.0) == WithSetting()
    assert WithSetting(3.0).power == 3.0
    assert Derived(2.0, 5.0) == Derived(power=2.0, scale=5.0)
    assert Derived() != WithSetting()


@pytest.mark.parametrize("kernel", [Plain(), Decorated(), WithSetting(2.0), Derived(), PlainOrdered()])
def test_a_bare_class_is_frozen(kernel):
    with pytest.raises(dataclasses.FrozenInstanceError):
        kernel.power = 1.0


@pytest.mark.parametrize("kernel", [Plain(), Decorated(), WithSetting(), PlainUnordered(), PlainOrdered()])
def test_a_bare_class_has_no_pytree_leaves(kernel):
    assert jax.tree_util.tree_leaves(kernel) == []


def test_a_bare_class_is_accepted_by_the_kernel_set():
    kernels = KernelSet(continuous=Plain(), unordered=PlainUnordered(), ordered=PlainOrdered())
    assert kernels == KernelSet(continuous=Plain(), unordered=PlainUnordered(), ordered=PlainOrdered())


def test_a_bare_class_passes_through_jit_by_value():
    traces = []

    @jax.jit
    def f(kernel, h):
        traces.append(1)
        return kernel.value(jnp.array(0.5), jnp.array(0.0), h)

    f(WithSetting(2.0), jnp.array(1.0))
    f(WithSetting(2.0), jnp.array(1.0))
    f(WithSetting(3.0), jnp.array(1.0))
    assert len(traces) == 2


class Checked(ContinuousKernel):
    scale: float = 1.0

    def value(self, x, y, h):
        return (x - y) / (h * self.scale)

    def __check_init__(self):
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")


class ForgetfulChild(Checked):
    def __post_init__(self):
        pass


def test_base_invariants_run_without_super():
    with pytest.raises(ValueError, match="scale must be positive"):
        ForgetfulChild(scale=-1.0)
    assert ForgetfulChild(scale=2.0).scale == 2.0
