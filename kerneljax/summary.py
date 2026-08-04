"""Goodness of fit reporting for a density estimate or a local polynomial fit."""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import ColumnSpec, Kind
from kerneljax.estimators.density import DensityFit
from kerneljax.estimators.regression import LocalPolyFit
from kerneljax.kernels import KernelSet
from kerneljax.typing import Array, ScalarFloat

__all__ = ["Summary", "summary"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "bandwidth",
        "criterion_value",
        "converged",
        "n_iter",
        "r_squared",
        "residual_se",
        "log_likelihood",
    ],
    meta_fields=["label", "method", "degree", "n_train", "spec", "kernels"],
)
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
    r_squared : ScalarFloat, optional
        Squared correlation between fitted and observed responses.
    residual_se : ScalarFloat, optional
        Root mean squared residual of the fit.
    log_likelihood : ScalarFloat, optional
        Summed log density at the training points.
    """

    label: str
    method: str | None
    degree: int | None
    n_train: int
    spec: ColumnSpec
    kernels: KernelSet
    bandwidth: Bandwidth
    criterion_value: ScalarFloat | None
    converged: Array | None
    n_iter: Array | None
    r_squared: ScalarFloat | None
    residual_se: ScalarFloat | None
    log_likelihood: ScalarFloat | None

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

        rows += ["", f"  {'Variable':<14}{'Kind':<14}{'Bandwidth':>12}"]
        rows += [f"  {name:<14}{kind:<14}{_number(width):>12}" for name, kind, width in self._columns()]

        rows += ["", _row("Continuous kernel", _kernel_name(self.kernels))]

        measures = [
            ("Residual standard error", self.residual_se),
            ("R-squared", self.r_squared),
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
                _row("Converged", self.converged),
            ]

        return "\n".join(rows)

    def _columns(self) -> list[tuple[str, str, Array]]:
        """Pair every column with its kind and the smoothing parameter it was fit under."""
        names = self.spec.names or tuple(f"x{i + 1}" for i in range(self.spec.p))
        widths = jnp.concatenate(
            [jnp.reshape(self.bandwidth.h, (-1,))[: self.spec.p_con], self.bandwidth.lam_uno, self.bandwidth.lam_ord]
        )

        order = (
            [Kind.CONTINUOUS] * self.spec.p_con + [Kind.UNORDERED] * self.spec.p_uno + [Kind.ORDERED] * self.spec.p_ord
        )
        sorted_names = (
            [n for n, k in zip(names, self.spec.kinds, strict=True) if k is Kind.CONTINUOUS]
            + [n for n, k in zip(names, self.spec.kinds, strict=True) if k is Kind.UNORDERED]
            + [n for n, k in zip(names, self.spec.kinds, strict=True) if k is Kind.ORDERED]
        )
        return [(n, k.name.lower(), widths[i]) for i, (n, k) in enumerate(zip(sorted_names, order, strict=True))]


def summary(fit: DensityFit | LocalPolyFit) -> Summary:
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
    method = None if selection is None else getattr(selection.criterion, "method", None)

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
    """Render a value for the report, degrading to plain text for a tracer."""
    if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.bool_):
        return str(value)

    try:
        return f"{float(value):.6f}" if isinstance(value, jax.Array | float) else str(value)
    except (TypeError, jax.errors.ConcretizationTypeError):
        return str(value)


def _estimator_name(degree: int) -> str:
    """Name the polynomial order the way a reader thinks of it."""
    return {0: "local constant", 1: "local linear", 2: "local quadratic"}.get(degree, f"degree {degree}")


def _kernel_name(kernels: KernelSet) -> str:
    """Name the continuous kernel family, with its order or whatever else parameterizes it."""
    kernel = kernels.continuous
    name = type(kernel).__name__

    order = getattr(kernel, "order", None)
    if order is not None:
        return f"{name}, order {order}"

    settings: list[str] = []
    if dataclasses.is_dataclass(kernel):
        settings = [f"{field.name}={getattr(kernel, field.name)}" for field in dataclasses.fields(kernel) if field.repr]

    return f"{name}({', '.join(settings)})" if settings else name
