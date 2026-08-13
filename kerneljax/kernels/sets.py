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

    continuous: ContinuousKernel = Gaussian()
    unordered: UnorderedKernel = AitchisonAitken()
    ordered: OrderedKernel = LiRacine()

    def __post_init__(self) -> None:
        """Reject unhashable kernels."""
        for kind in ("continuous", "unordered", "ordered"):
            _require_hashable(getattr(self, kind), f"{kind} kernel")


def _require_hashable(obj: object, role: str) -> None:
    """Reject an unhashable kernel."""
    try:
        hash(obj)
    except TypeError as exc:
        raise TypeError(
            f"{role} {type(obj).__name__} is not hashable, so it cannot be a static argument. "
            "Decorate it with @dataclasses.dataclass(frozen=True), and hold any array field as a "
            "tuple of floats rather than as a jax.Array."
        ) from exc


def _resolve_kernels(explicit: KernelSet | None, carried: KernelSet | None) -> KernelSet:
    """Settle on one kernel set."""
    if explicit is None:
        return KernelSet() if carried is None else carried

    if carried is not None and explicit != carried:
        raise ValueError("kernels= contradicts the kernels that bw was selected under")

    return explicit
