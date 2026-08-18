"""Goodness of fit reporting for a density estimate or a local polynomial fit."""

from __future__ import annotations

import dataclasses
import inspect

import jax
import jax.numpy as jnp

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import ColumnSpec, Kind
from kerneljax.estimators.fit import ConditionalFit, DensityFit, LocalPolyFit, ModeFit, QuantileFit
from kerneljax.kernels import KernelSet
from kerneljax.typing import Array, ScalarFloat

__all__ = ["Summary", "summary"]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, repr=False)
class Summary:
    """Goodness of fit measures for a fitted estimator.

    Fields a family does not define are left as ``None``, so a density
    summary reports a log likelihood where a regression summary reports an
    R-squared and a residual standard error.

    Parameters
    ----------
    label : str
        Name of the estimator the summary describes. Static.
    method : str, optional
        Selection rule the bandwidth was chosen by, or ``None`` when it was
        supplied directly. Static.
    degree : int, optional
        Total degree of the local polynomial basis, or ``None`` for a
        density. Static.
    n_train : int
        Number of training points. Static.
    spec : ColumnSpec
        Column metadata of the training sample. Static.
    kernels : KernelSet
        Kernel families the fit was produced with. Static.
    bandwidth : Bandwidth
        Bandwidth the fit was produced at.
    criterion_value : ScalarFloat, optional
        Criterion value at the selected bandwidth, or ``None`` when the
        bandwidth was supplied directly.
    converged : Array, optional
        Whether the selection solver stopped because its progress stalled.
    n_iter : Array, optional
        Number of solver iterations used by the selection.
    accuracy : ScalarFloat, optional
        Share of observations whose modal level matches the response, set
        only for a conditional mode estimate.
    r_squared : ScalarFloat, optional
        Squared correlation between fitted and observed responses.
    residual_se : ScalarFloat, optional
        Root mean squared residual of the fit.
    log_likelihood : ScalarFloat, optional
        Summed log density at the training points.
    response_spec : ColumnSpec, optional
        Column metadata of the response sample, set only for a conditional
        estimate. Static.
    response_bandwidth : Bandwidth, optional
        Bandwidth for the response block, set only for a conditional estimate.
    """

    label: str = dataclasses.field(metadata=dict(static=True))
    method: str | None = dataclasses.field(metadata=dict(static=True))
    degree: int | None = dataclasses.field(metadata=dict(static=True))
    n_train: int = dataclasses.field(metadata=dict(static=True))
    spec: ColumnSpec = dataclasses.field(metadata=dict(static=True))
    kernels: KernelSet = dataclasses.field(metadata=dict(static=True))
    bandwidth: Bandwidth
    criterion_value: ScalarFloat | None
    converged: Array | None
    n_iter: Array | None
    r_squared: ScalarFloat | None
    residual_se: ScalarFloat | None
    log_likelihood: ScalarFloat | None
    accuracy: ScalarFloat | None = None
    response_spec: ColumnSpec | None = dataclasses.field(default=None, metadata=dict(static=True))
    response_bandwidth: Bandwidth | None = None

    def __repr__(self) -> str:
        """Render the summary as an aligned report."""
        rows = [
            self.label,
            "",
            _row("Observations", self.n_train),
            _row("Continuous variables", self.spec.p_con),
        ]
        if self.spec.p_uno:
            rows.append(_row("Unordered variables", self.spec.p_uno))
        if self.spec.p_ord:
            rows.append(_row("Ordered variables", self.spec.p_ord))
        if self.degree is not None:
            rows.append(_row("Estimator", _estimator_name(self.degree)))
        rows.append(_row("Bandwidth type", self.bandwidth.h_axis))

        header = f"  {'Variable':<14}{'Kind':<14}{'Bandwidth':>12}"
        if self.response_spec is not None and self.response_bandwidth is not None:
            rows += ["", "  Response", header]
            rows += _column_rows(self.response_spec, self.response_bandwidth)
            rows += ["", "  Conditioning", header]
        else:
            rows += ["", header]
        rows += _column_rows(self.spec, self.bandwidth)

        rows += ["", _row("Continuous kernel", _kernel_name(self.kernels))]

        measures = [
            ("Residual standard error", self.residual_se),
            ("R-squared", self.r_squared),
            ("Correct classification", self.accuracy),
            ("Log likelihood", self.log_likelihood),
        ]
        shown = [_row(name, value) for name, value in measures if value is not None]
        if shown:
            rows += ["", *shown]

        if self.method is not None:
            rows += [
                "",
                _row("Selection", self.method),
                _row("Criterion value", self.criterion_value),
                _row("Solver iterations", self.n_iter),
                _row("Converged", self.converged),
            ]

        return "\n".join(rows)


def summary(fit: ConditionalFit | DensityFit | LocalPolyFit | ModeFit | QuantileFit) -> Summary:
    r"""Measure how well a fitted estimator describes the sample it was fit on.

    For a regression the report carries the squared correlation between fitted
    and observed responses,

    .. math::

        R^2 = \frac{\bigl[\sum_i (y_i - \bar y)(\hat y_i - \bar y)\bigr]^2}
                   {\sum_i (y_i - \bar y)^2 \sum_i (\hat y_i - \bar y)^2}

    together with the root mean squared residual. For a density it carries the
    summed log density at the training points instead.

    The fit must have been evaluated at its own training points, since every
    measure compares fitted values against the sample they came from.

    Parameters
    ----------
    fit : DensityFit or LocalPolyFit
        A fitted estimator, evaluated at its training points.

    Returns
    -------
    Summary
        Object containing the goodness of fit measures:

        - **label**: Name of the estimator the summary describes
        - **method**: Selection rule the bandwidth was chosen by, or None
        - **degree**: Total degree of the local polynomial basis, or None for a density
        - **n_train**: Number of training points
        - **spec**: Column metadata of the training sample
        - **kernels**: Kernel families the fit was produced with
        - **bandwidth**: Bandwidth the fit was produced at
        - **criterion_value**: Criterion value at the selected bandwidth, or None
        - **converged**: Whether the selection solver stalled rather than ran out of budget
        - **n_iter**: Number of solver iterations used by the selection
        - **r_squared**: Squared correlation between fitted and observed responses
        - **residual_se**: Root mean squared residual of the fit
        - **log_likelihood**: Summed log density at the training points

    Examples
    --------
    Summarize a local linear fit at a selected bandwidth.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50)
           ...: y = jnp.sin(x)
           ...: fit = kj.local_poly(x, y, "cv_ls")
           ...: print(kj.summary(fit))

    See Also
    --------
    local_poly : Fit a local polynomial regression.
    density : Estimate a mixed-type probability density.
    """
    if isinstance(fit, (ConditionalFit, ModeFit, QuantileFit)):
        return _conditional_summary(fit)

    fitted = fit.mean if isinstance(fit, LocalPolyFit) else fit.value
    if fit.spec is None:
        raise ValueError(
            "summary needs a fit from local_poly or density, which record the column metadata a report is built from"
        )

    if fitted.shape[0] != fit.n_train:
        raise ValueError(
            f"summary needs a fit evaluated at its own training points, got {fitted.shape[0]} "
            f"evaluation points for {fit.n_train} training points"
        )

    selection = fit.selection
    method = None if selection is None else _criterion_name(selection.criterion)

    if isinstance(fit, DensityFit):
        return Summary(
            label="Mixed-type density estimate",
            method=method,
            degree=None,
            n_train=fit.n_train,
            spec=fit.spec,
            kernels=fit.kernels,
            bandwidth=fit.bandwidth,
            criterion_value=None if selection is None else selection.value,
            converged=None if selection is None else selection.converged,
            n_iter=None if selection is None else selection.n_iter,
            r_squared=None,
            residual_se=None,
            log_likelihood=jnp.sum(jnp.log(fit.value)),
        )

    return Summary(
        label="Local polynomial regression",
        method=method,
        degree=fit.degree,
        n_train=fit.n_train,
        spec=fit.spec,
        kernels=fit.kernels,
        bandwidth=fit.bandwidth,
        criterion_value=None if selection is None else selection.value,
        converged=None if selection is None else selection.converged,
        n_iter=None if selection is None else selection.n_iter,
        r_squared=fit.r_squared,
        residual_se=fit.residual_se,
        log_likelihood=None,
    )


def _row(name: str, value: object) -> str:
    """Lay one labelled measure out on a fixed width line."""
    return f"  {name:<26}{_number(value):>14}"


def _number(value: object) -> str:
    """Render a value for the report and degrade to plain text for a tracer."""
    if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.bool_):
        return str(value)

    if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.integer):
        try:
            return str(int(value))
        except (TypeError, jax.errors.ConcretizationTypeError):
            return str(value)

    try:
        return f"{float(value):.6f}" if isinstance(value, jax.Array | float) else str(value)
    except (TypeError, jax.errors.ConcretizationTypeError):
        return str(value)


def _criterion_name(criterion: object) -> str | None:
    """Name the selection rule and fall back to the class for a criterion the caller wrote."""
    if criterion is None:
        return None
    method = getattr(criterion, "method", None)
    if isinstance(method, str):
        return method
    name = criterion.__name__ if inspect.isfunction(criterion) else type(criterion).__name__
    for suffix in ("_conditional_density", "_conditional_distribution", "_conditional"):
        name = name.removesuffix(suffix)
    return name


def _estimator_name(degree: int) -> str:
    """Name the polynomial order the way a reader thinks of it."""
    return {0: "local constant", 1: "local linear", 2: "local quadratic"}.get(degree, f"degree {degree}")


def _kernel_name(kernels: KernelSet) -> str:
    """Name the continuous kernel family with whatever parameterizes it."""
    kernel = kernels.continuous
    name = type(kernel).__name__

    order = getattr(kernel, "order", None)
    if order is not None:
        return f"{name}, order {order}"

    settings: list[str] = []
    if dataclasses.is_dataclass(kernel):
        settings = [f"{field.name}={getattr(kernel, field.name)}" for field in dataclasses.fields(kernel) if field.repr]

    return f"{name}({', '.join(settings)})" if settings else name


def _column_rows(spec: ColumnSpec, bandwidth: Bandwidth) -> list[str]:
    """Lay one block's columns out against the smoothing parameter each was fit under."""
    names = spec.names or tuple(f"x{i + 1}" for i in range(spec.p))
    widths = jnp.concatenate([jnp.reshape(bandwidth.h, (-1,))[: spec.p_con], bandwidth.lam_uno, bandwidth.lam_ord])

    order = [Kind.CONTINUOUS] * spec.p_con + [Kind.UNORDERED] * spec.p_uno + [Kind.ORDERED] * spec.p_ord
    sorted_names = (
        [n for n, k in zip(names, spec.kinds, strict=True) if k is Kind.CONTINUOUS]
        + [n for n, k in zip(names, spec.kinds, strict=True) if k is Kind.UNORDERED]
        + [n for n, k in zip(names, spec.kinds, strict=True) if k is Kind.ORDERED]
    )
    paired = zip(sorted_names, order, strict=True)
    return [f"  {n:<14}{k.name.lower():<14}{_number(widths[i]):>12}" for i, (n, k) in enumerate(paired)]


def _conditional_summary(fit: ConditionalFit | ModeFit | QuantileFit) -> Summary:
    """Report a conditional estimate."""
    if fit.x_spec is None or fit.y_spec is None:
        raise ValueError("summary needs a conditional fit that recorded the column metadata of both samples")

    if fit.value.shape[0] != fit.n_train:
        raise ValueError(
            f"summary needs a fit evaluated at its own training points, got {fit.value.shape[0]} "
            f"evaluation points for {fit.n_train} training points"
        )

    selection = fit.selection
    if isinstance(fit, QuantileFit):
        label = f"Conditional quantile estimate at tau {fit.tau:g}"
        density = False
    elif isinstance(fit, ModeFit):
        label = "Conditional mode estimate"
        density = False
    else:
        label = f"Conditional {fit.target} estimate"
        density = fit.target == "density"

    return Summary(
        label=label,
        method=None if selection is None else _criterion_name(selection.criterion),
        degree=None,
        n_train=fit.n_train,
        spec=fit.x_spec,
        kernels=fit.kernels,
        bandwidth=fit.bandwidth.x,
        criterion_value=None if selection is None else selection.value,
        converged=None if selection is None else selection.converged,
        n_iter=None if selection is None else selection.n_iter,
        r_squared=None,
        residual_se=None,
        log_likelihood=jnp.sum(jnp.log(fit.value)) if density else None,
        accuracy=fit.accuracy if isinstance(fit, ModeFit) else None,
        response_spec=fit.y_spec,
        response_bandwidth=fit.bandwidth.y,
    )
