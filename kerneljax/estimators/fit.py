"""Result containers for estimators."""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Literal

import jax
from jaxtyping import Float, Int

from kerneljax.bandwidth import Bandwidth, ConditionalBandwidth, SelectionResult
from kerneljax.data import ColumnSpec
from kerneljax.kernels import KernelSet
from kerneljax.typing import Array, ScalarFloat

__all__ = ["ConditionalFit", "DensityFit", "DistributionFit", "LocalPolyFit", "ModeFit", "QuantileFit"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["mean", "grad", "coef", "rcond", "bandwidth", "se", "selection", "r_squared", "residual_se"],
    meta_fields=["degree", "kernels", "spec", "n_train"],
)
@dataclasses.dataclass(frozen=True)
class LocalPolyFit:
    r"""Result of a local polynomial regression fit.

    The polynomial basis is centered and scaled at each evaluation point
    :math:`x`, so the intercept gives the fitted value directly and the
    first-order coefficients recover derivatives after rescaling by the
    bandwidth,

    .. math::

        \hat m(x) = \beta_0, \qquad
        \frac{\partial \hat m}{\partial x_j}(x) = \frac{\beta_j}{h_j}.

    Here :math:`\beta_j` is the coefficient of the first-order term for
    continuous column :math:`j`, and :math:`h_j` is its bandwidth.

    Attributes
    ----------
    mean : Float[Array, " n_eval"]
        Fitted regression value at every evaluation point.
    grad : Float[Array, "n_eval p_con"] or None
        Gradient of the fitted value with respect to every continuous
        column at every evaluation point, or ``None`` when gradients were
        not requested.
    coef : Float[Array, "n_eval k"]
        Full local polynomial coefficient vector at every evaluation point,
        expressed in bandwidth-scaled coordinates.
    rcond : Float[Array, " n_eval"]
        Reciprocal condition number of the weighted moment system at every
        evaluation point, from :func:`~kerneljax.wls`.
    bandwidth : Bandwidth
        Bandwidth used to produce the fit.
    se : Float[Array, " n_eval"] or None
        Standard error of the fitted mean at every evaluation point, or
        ``None`` when standard errors were not requested.
    selection : SelectionResult, optional
        Selection result that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    degree : int
        Total degree of the local polynomial basis. Static.
    kernels : KernelSet
        Kernel families used to produce the fit. Static.
    spec : ColumnSpec, optional
        Column metadata of the training sample. Static.
    n_train : int
        Number of training points. Static.
    r_squared : ScalarFloat, optional
        Squared cosine between the observed and fitted responses after both
        are centered at the sample mean of the observed response. Lies in
        ``[0, 1]`` when defined. This is not generally the squared Pearson
        correlation because the fitted values are not centered at their own
        sample mean. ``None`` when the fit was evaluated away from its
        training points.
    residual_se : ScalarFloat, optional
        Root mean squared residual, ``sqrt(mean((y - mean) ** 2))``. This
        uses ``n`` in the denominator rather than a degrees-of-freedom
        correction. ``None`` when the fit was evaluated away from its
        training points.
    """

    mean: Float[Array, " n_eval"]
    grad: Float[Array, "n_eval p_con"] | None
    coef: Float[Array, "n_eval k"]
    rcond: Float[Array, " n_eval"]
    bandwidth: Bandwidth
    se: Float[Array, " n_eval"] | None
    selection: SelectionResult | None = None
    degree: int = 1
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    spec: ColumnSpec | None = None
    n_train: int = 0
    r_squared: ScalarFloat | None = None
    residual_se: ScalarFloat | None = None


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "bandwidth", "selection"],
    meta_fields=["kernels", "spec", "n_train"],
)
@dataclasses.dataclass(frozen=True)
class DensityFit:
    """Result of a mixed-type density estimate.

    Attributes
    ----------
    value : Float[Array, " n_eval"]
        The density estimate at each evaluation point.
    bandwidth : Bandwidth
        The bandwidth used to produce ``value``.
    selection : SelectionResult, optional
        The selection that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    kernels : KernelSet
        Kernel families the estimate was produced with. Static.
    spec : ColumnSpec, optional
        Column metadata of the training sample. Static.
    n_train : int
        Number of training points. Static.
    """

    value: Float[Array, " n_eval"]
    bandwidth: Bandwidth
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    spec: ColumnSpec | None = None
    n_train: int = 0


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "se", "bandwidth", "selection"],
    meta_fields=["kernels", "spec", "n_train"],
)
@dataclasses.dataclass(frozen=True)
class DistributionFit:
    """Result of a mixed-type cumulative distribution estimate.

    Attributes
    ----------
    value : Float[Array, " n_eval"]
        The distribution estimate at each evaluation point.
    se : Float[Array, " n_eval"]
        The standard error of the estimate at each evaluation point.
    bandwidth : Bandwidth
        The bandwidth used to produce ``value``.
    selection : SelectionResult, optional
        The selection that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    kernels : KernelSet
        Kernel families the estimate was produced with. Static.
    spec : ColumnSpec, optional
        Column metadata of the training sample. Static.
    n_train : int
        Number of training points. Static.
    """

    value: Float[Array, " n_eval"]
    se: Float[Array, " n_eval"]
    bandwidth: Bandwidth
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    spec: ColumnSpec | None = None
    n_train: int = 0


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "bandwidth", "selection"],
    meta_fields=["kernels", "x_spec", "y_spec", "n_train", "target"],
)
@dataclasses.dataclass(frozen=True)
class ConditionalFit:
    """Result of a conditional density or conditional distribution estimate.

    Attributes
    ----------
    value : Float[Array, " n_eval"]
        The estimate at each evaluation pair.
    bandwidth : ConditionalBandwidth
        The bandwidth used to produce ``value``, one block per sample.
    selection : SelectionResult, optional
        The selection that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    kernels : KernelSet
        Kernel families the estimate was produced with. Static.
    x_spec : ColumnSpec, optional
        Column metadata of the conditioning sample. Static.
    y_spec : ColumnSpec, optional
        Column metadata of the response sample. Static.
    n_train : int
        Number of training points. Static.
    target : {"density", "distribution"}
        Which estimate ``value`` holds. Static.
    """

    value: Float[Array, " n_eval"]
    bandwidth: ConditionalBandwidth
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    x_spec: ColumnSpec | None = None
    y_spec: ColumnSpec | None = None
    n_train: int = 0
    target: Literal["density", "distribution"] = "density"


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "bandwidth", "selection"],
    meta_fields=["tau", "kernels", "x_spec", "y_spec", "n_train"],
)
@dataclasses.dataclass(frozen=True)
class QuantileFit:
    """Result of a conditional quantile regression.

    Attributes
    ----------
    value : Float[Array, " n_eval"]
        The conditional quantile at each evaluation point, clamped to the
        observed response range where the distribution never crosses ``tau``.
    bandwidth : ConditionalBandwidth
        The bandwidth used to produce ``value``, one block per sample.
    tau : float
        The quantile level the fit inverts at. Static.
    selection : SelectionResult, optional
        The selection that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    kernels : KernelSet
        Kernel families the estimate was produced with. Static.
    x_spec : ColumnSpec, optional
        Column metadata of the conditioning sample. Static.
    y_spec : ColumnSpec, optional
        Column metadata of the response sample. Static.
    n_train : int
        Number of training points. Static.
    """

    value: Float[Array, " n_eval"]
    bandwidth: ConditionalBandwidth
    tau: float = 0.5
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    x_spec: ColumnSpec | None = None
    y_spec: ColumnSpec | None = None
    n_train: int = 0


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["value", "density", "bandwidth", "accuracy", "selection"],
    meta_fields=["kernels", "x_spec", "y_spec", "n_train"],
)
@dataclasses.dataclass(frozen=True)
class ModeFit:
    """Result of a conditional mode estimate over a categorical response.

    Attributes
    ----------
    value : Int[Array, " n_eval"]
        The modal response level at each evaluation point, as a level code.
    density : Float[Array, " n_eval"]
        The conditional density at the modal level.
    bandwidth : ConditionalBandwidth
        The bandwidth used to produce ``value``, one block per sample.
    accuracy : ScalarFloat, optional
        Share of training observations whose modal level matches the
        observed response, or ``None`` when the fit was evaluated elsewhere.
    selection : SelectionResult, optional
        The selection that produced ``bandwidth``, or ``None`` when the
        bandwidth was supplied directly.
    kernels : KernelSet
        Kernel families the estimate was produced with. Static.
    x_spec : ColumnSpec, optional
        Column metadata of the conditioning sample. Static.
    y_spec : ColumnSpec, optional
        Column metadata of the response sample. Static.
    n_train : int
        Number of training points. Static.
    """

    value: Int[Array, " n_eval"]
    density: Float[Array, " n_eval"]
    bandwidth: ConditionalBandwidth
    accuracy: ScalarFloat | None = None
    selection: SelectionResult | None = None
    kernels: KernelSet = dataclasses.field(default_factory=KernelSet)
    x_spec: ColumnSpec | None = None
    y_spec: ColumnSpec | None = None
    n_train: int = 0
