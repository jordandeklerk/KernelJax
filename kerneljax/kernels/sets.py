"""The kernel set."""

from __future__ import annotations

import dataclasses

import jax

from kerneljax.kernels.base import ContinuousKernel, OrderedKernel, UnorderedKernel
from kerneljax.kernels.continuous import Gaussian
from kerneljax.kernels.discrete import AitchisonAitken, LiRacine

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

    Examples
    --------
    Defaults cover every column kind, so only the one being changed needs naming.
    Pass the result to any estimator through ``kernels=``.

    .. ipython::
        :okwarning:

        In [1]: import kerneljax as kj
           ...:
           ...: kernels = kj.KernelSet(ordered=kj.WangVanRyzin())
           ...: print(type(kernels.continuous).__name__,
           ...:       type(kernels.unordered).__name__,
           ...:       type(kernels.ordered).__name__)
    """

    continuous: ContinuousKernel = dataclasses.field(default_factory=Gaussian)
    unordered: UnorderedKernel = dataclasses.field(default_factory=AitchisonAitken)
    ordered: OrderedKernel = dataclasses.field(default_factory=LiRacine)
