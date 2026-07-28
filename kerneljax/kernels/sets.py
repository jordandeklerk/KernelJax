"""The kernel set."""

from __future__ import annotations

import dataclasses

import jax

from kerneljax.kernels.base import ContinuousKernel, OrderedKernel, UnorderedKernel
from kerneljax.kernels.continuous import Gaussian
from kerneljax.kernels.discrete import AitchisonAitken, WangVanRyzin

__all__ = ["KernelSet"]


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class KernelSet:
    """One kernel family per column kind.

    Parameters
    ----------
    continuous : ContinuousKernel
        Kernel family applied to every continuous column.
    unordered : UnorderedKernel
        Kernel family applied to every unordered categorical column.
    ordered : OrderedKernel
        Kernel family applied to every ordered categorical column.
    """

    continuous: ContinuousKernel = dataclasses.field(default_factory=Gaussian)
    unordered: UnorderedKernel = dataclasses.field(default_factory=AitchisonAitken)
    ordered: OrderedKernel = dataclasses.field(default_factory=WangVanRyzin)
