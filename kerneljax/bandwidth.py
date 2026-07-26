"""The bandwidth parameter tree and the axis tag that pins its shape."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Literal

import jax
from jaxtyping import Float

from kerneljax.typing import Array, FloatArray

__all__ = ["Bandwidth", "broadcast_h"]

HAxis = Literal["shared", "eval", "train"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["h", "lam_uno", "lam_ord"],
    meta_fields=["h_axis"],
)
@dataclasses.dataclass(frozen=True)
class Bandwidth:
    r"""Bandwidths in natural, constrained scale.

    Values are the bandwidths themselves, so :math:`h > 0` and each
    :math:`\lambda` lies in its own bounded interval. Every leaf is
    floating point, so the whole tree can be differentiated.

    Parameters
    ----------
    h : FloatArray
        Continuous bandwidths. Shape depends on ``h_axis``, either
        ``(p_con,)``, ``(n_eval, p_con)`` or ``(n_train, p_con)``.
    lam_uno : Float[Array, " p_uno"]
        Smoothing parameters for the unordered categorical columns.
    lam_ord : Float[Array, " p_ord"]
        Smoothing parameters for the ordered categorical columns.
    h_axis : {"shared", "eval", "train"}
        Which axis ``h`` is indexed by, static metadata. ``"shared"`` is a
        fixed bandwidth, ``"eval"`` varies with the evaluation point and
        ``"train"`` varies with the training point.
    """

    h: FloatArray
    lam_uno: Float[Array, " p_uno"]
    lam_ord: Float[Array, " p_ord"]
    h_axis: HAxis = "shared"

    def replace(self, **changes: Any) -> Bandwidth:
        """Return a copy with the given fields replaced."""
        return dataclasses.replace(self, **changes)


def broadcast_h(bw: Bandwidth, p_con: int) -> Float[Array, "n_eval n_train p_con"]:
    """Reshape ``bw.h`` to an explicit three axis form for the kernel sum.

    ``bw.h_axis`` selects which of the leading two axes carries the
    bandwidth, and the shape of ``h`` must match that tag.

    Parameters
    ----------
    bw : Bandwidth
        The bandwidth tree to reshape.
    p_con : int
        Number of continuous columns, the expected trailing width of ``h``.

    Returns
    -------
    Float[Array, "n_eval n_train p_con"]
        ``h`` reshaped to ``(1, 1, p_con)`` when shared, ``(n_eval, 1,
        p_con)`` when eval indexed, or ``(1, n_train, p_con)`` when train
        indexed.

    Raises
    ------
    ValueError
        If ``h`` has the wrong rank for ``bw.h_axis`` or a trailing width
        other than ``p_con``.
    """
    h = bw.h
    if bw.h_axis == "shared":
        if h.ndim != 1:
            raise ValueError(f"h_axis='shared' needs a one dimensional h, got shape {h.shape}")
        if h.shape[0] != p_con:
            raise ValueError(f"h has trailing width {h.shape[0]}, expected p_con={p_con}")
        return h.reshape(1, 1, p_con)

    if h.ndim != 2:
        raise ValueError(f"h_axis={bw.h_axis!r} needs a two dimensional h of shape (n, p_con), got shape {h.shape}")
    if h.shape[1] != p_con:
        raise ValueError(f"h has trailing width {h.shape[1]}, expected p_con={p_con}")
    if bw.h_axis == "eval":
        return h[:, None, :]
    return h[None, :, :]
