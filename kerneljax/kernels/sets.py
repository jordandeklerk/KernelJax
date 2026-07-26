"""The kernel set, one kernel family per column kind."""

from __future__ import annotations

import dataclasses

from kerneljax.kernels.base import ContinuousKernel, OrderedKernel, UnorderedKernel
from kerneljax.kernels.continuous import Gaussian
from kerneljax.kernels.discrete import AitchisonAitken, WangVanRyzin

__all__ = ["KernelSet"]


@dataclasses.dataclass(frozen=True)
class KernelSet:
    """One kernel family per column kind.

    A ``KernelSet`` sits in a static position on the jit boundary, so it is a
    plain frozen value rather than a pytree. It can be paired in an ordinary
    tuple, one set for the regressors and one for the response, to seam in
    conditional density estimation later.

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
