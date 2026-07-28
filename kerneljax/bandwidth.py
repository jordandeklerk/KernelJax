"""The bandwidth parameter tree, its unconstrained bijection, and starting rules."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.data import ColumnSpec, MixedData
from kerneljax.kernels import KernelSet
from kerneljax.typing import Array, FloatArray

__all__ = ["Bandwidth", "BandwidthTransform", "ConditionalBandwidth", "broadcast_h", "normal_reference"]

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
    :math:`\lambda` lies in its own bounded interval. Every entry is
    floating point, so a bandwidth can be differentiated through.

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


@partial(jax.tree_util.register_dataclass, data_fields=["x", "y"], meta_fields=[])
@dataclasses.dataclass(frozen=True)
class ConditionalBandwidth:
    """A pair of bandwidth trees, one for the regressors and one for the response.

    Parameters
    ----------
    x : Bandwidth
        Bandwidth for the regressor variables.
    y : Bandwidth
        Bandwidth for the response variable.
    """

    x: Bandwidth
    y: Bandwidth


@dataclasses.dataclass(frozen=True)
class BandwidthTransform:
    r"""Bijection between a ``Bandwidth`` and a flat unconstrained vector.

    Continuous entries map through :math:`h = \operatorname{softplus}(z)`.
    Categorical entries map through a scaled logistic
    :math:`\lambda = b \, \sigma(z)`, with :math:`b` the kernel's upper
    bound for that column's level count.

    Parameters
    ----------
    spec : ColumnSpec
        Static column metadata giving the block widths and level counts.
    kernels : KernelSet
        Kernel families, consulted only for their ``upper_bound`` methods.
    """

    spec: ColumnSpec
    kernels: KernelSet

    @property
    def _uno_bounds(self) -> tuple[float, ...]:
        return tuple(self.kernels.unordered.upper_bound(c) for c in self.spec.uno_levels)

    @property
    def _ord_bounds(self) -> tuple[float, ...]:
        return tuple(self.kernels.ordered.upper_bound(c) for c in self.spec.ord_levels)

    def to_unconstrained(self, bw: Bandwidth) -> Float[Array, " k"]:
        """Map a bandwidth to a flat unconstrained vector.

        Entries run continuous first, then unordered, then ordered,
        matching the block order of ``spec``. Values are clamped into the
        open interval before inverting, so the result stays finite at the
        box boundary. Only a bandwidth shared across rows can be mapped,
        since a per row bandwidth holds no single value per column.

        Parameters
        ----------
        bw : Bandwidth
            Bandwidth in natural, constrained scale.

        Returns
        -------
        Float[Array, " k"]
            The unconstrained vector, length ``p_con + p_uno + p_ord``.
        """
        if bw.h_axis != "shared":
            raise ValueError(
                f"to_unconstrained needs h_axis 'shared', got {bw.h_axis!r}. Flattening a per row "
                "bandwidth would feed continuous entries into the categorical blocks."
            )

        parts = [_softplus_inv(jnp.reshape(bw.h, (-1,)))]

        if self._uno_bounds:
            upper = jnp.asarray(self._uno_bounds, dtype=bw.lam_uno.dtype)
            parts.append(_sigmoid_inv(bw.lam_uno / upper))

        if self._ord_bounds:
            upper = jnp.asarray(self._ord_bounds, dtype=bw.lam_ord.dtype)
            parts.append(_sigmoid_inv(bw.lam_ord / upper))

        return jnp.concatenate(parts)

    def from_unconstrained(self, z: Float[Array, " k"]) -> Bandwidth:
        """Map a flat unconstrained vector back to a bandwidth.

        Parameters
        ----------
        z : Float[Array, " k"]
            Unconstrained vector, laid out continuous first, then
            unordered, then ordered, matching the block order of ``spec``.

        Returns
        -------
        Bandwidth
            The bandwidth in natural, constrained scale, with
            ``h_axis="shared"``.
        """
        p_con = self.spec.p_con
        n_uno = len(self._uno_bounds)

        h = _softplus(z[:p_con])

        lam_uno = jnp.zeros(0, dtype=z.dtype)
        if n_uno:
            upper = jnp.asarray(self._uno_bounds, dtype=z.dtype)
            lam_uno = upper * jax.nn.sigmoid(z[p_con : p_con + n_uno])

        lam_ord = jnp.zeros(0, dtype=z.dtype)
        if self._ord_bounds:
            upper = jnp.asarray(self._ord_bounds, dtype=z.dtype)
            lam_ord = upper * jax.nn.sigmoid(z[p_con + n_uno :])

        return Bandwidth(h=h, lam_uno=lam_uno, lam_ord=lam_ord, h_axis="shared")

    def bounds(self) -> tuple[Float[Array, " k"], Float[Array, " k"]]:
        """Return the natural-scale box constraints on the flat vector.

        Returns
        -------
        tuple of Float[Array, " k"]
            Lower and upper bounds. The lower bound is zero everywhere.
            The upper bound is ``inf`` for continuous entries and the
            kernel's ``upper_bound`` for categorical entries.
        """
        lower = jnp.zeros(self.spec.p_con + len(self._uno_bounds) + len(self._ord_bounds))

        upper = jnp.concatenate(
            [
                jnp.full((self.spec.p_con,), jnp.inf),
                jnp.asarray(self._uno_bounds, dtype=lower.dtype) if self._uno_bounds else jnp.zeros(0),
                jnp.asarray(self._ord_bounds, dtype=lower.dtype) if self._ord_bounds else jnp.zeros(0),
            ]
        )

        return lower, upper


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


def normal_reference(data: MixedData, kernels: KernelSet) -> Bandwidth:
    r"""Compute a normal-reference starting bandwidth.

    Continuous entries scale :math:`1.06 \, n^{-1 / (4 + p_{\text{con}})}`
    by :math:`\sigma`, the smallest positive value among the standard
    deviation, the interquartile range divided by
    :math:`2 \, \Phi^{-1}(0.75)`, and the median absolute deviation scaled
    by :math:`1.4826`, for that column. Categorical entries start at
    ``0.1``.

    Parameters
    ----------
    data : MixedData
        Design matrix supplying the sample size and continuous columns.
    kernels : KernelSet
        Unused, accepted for interface uniformity with other starting rules.

    Returns
    -------
    Bandwidth
        A starting bandwidth with ``h_axis="shared"``.
    """
    del kernels
    n = data.n
    spec = data.spec

    if spec.p_con:
        sd = jnp.std(data.con, axis=0, ddof=1)

        q75, q25 = jnp.percentile(data.con, jnp.array([75.0, 25.0]), axis=0)
        iqr = (q75 - q25) / (2.0 * jax.scipy.special.ndtri(0.75))

        mad = jnp.median(jnp.abs(data.con - jnp.median(data.con, axis=0)), axis=0) * 1.4826

        candidates = jnp.stack([sd, iqr, mad])
        scale = jnp.min(jnp.where(candidates > 0, candidates, jnp.inf), axis=0)
        scale = jnp.where(jnp.isfinite(scale), scale, 1.0)

        h = 1.06 * scale * n ** (-1.0 / (4.0 + spec.p_con))
    else:
        h = jnp.zeros(0)

    return Bandwidth(
        h=h,
        lam_uno=jnp.full((spec.p_uno,), 0.1),
        lam_ord=jnp.full((spec.p_ord,), 0.1),
        h_axis="shared",
    )


def _softplus(z: Array) -> Array:
    """Map an unconstrained value to a positive one."""
    return jnp.logaddexp(z, 0.0)


def _softplus_inv(h: Array) -> Array:
    """Invert the softplus map, clamped away from zero."""
    h = jnp.clip(h, 1e-7, None)
    return h + jnp.log(-jnp.expm1(-h))


def _sigmoid_inv(p: Array) -> Array:
    """Invert the logistic map, clamped away from the interval's edges."""
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
    return jnp.log(p) - jnp.log1p(-p)
