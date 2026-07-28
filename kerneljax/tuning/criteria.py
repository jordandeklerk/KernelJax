"""Criterion objects binding a selection rule to the estimator it selects for."""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol

from jaxtyping import Float

from kerneljax.bandwidth import Bandwidth
from kerneljax.data import MixedData
from kerneljax.kernels import KernelSet
from kerneljax.tuning.objectives import aic_c_regression, cv_ls_density, cv_ls_regression, cv_ml_density
from kerneljax.typing import Array, ScalarFloat

__all__ = ["Criterion", "DensityCriterion", "RegressionCriterion"]


class Criterion(Protocol):
    r"""Interface for a cross-validation criterion minimized over a bandwidth.

    :func:`~kerneljax.select_bandwidth` calls a criterion as
    ``criterion(train, bandwidth, **extra, kernels=kernels, chunk=chunk)`` and
    minimizes what it returns, so a criterion sees one bandwidth at a time and
    reports a single number.

    Anything the criterion needs beyond the training sample and the bandwidth,
    such as a response, arrives through ``extra`` as array data. Settings that
    fix the shape of the estimator instead live on the criterion itself, where
    they stay concrete while the search runs.
    """

    def __call__(
        self,
        train: MixedData,
        bw: Bandwidth,
        *,
        kernels: KernelSet | None = None,
        chunk: int | tuple[int, int] | None = None,
        **extra: Array,
    ) -> ScalarFloat:
        """Evaluate the criterion at a bandwidth and return the value to minimize."""
        ...


@dataclasses.dataclass(frozen=True)
class RegressionCriterion:
    r"""Regression selection rule carrying the local polynomial degree.

    Evaluating an instance runs the chosen rule at a bandwidth, with the degree
    taken from the instance rather than from a default, so one degree governs
    both the search and the fit that follows.

    Parameters
    ----------
    method : {"cv_ls", "aic"}
        Selection rule. ``"cv_ls"`` is the leave-one-out mean squared residual
        of [1]_, ``"aic"`` the corrected Akaike information criterion of [2]_.
    degree : int
        Total degree of the local polynomial basis. Supports 0, 1 and 2, giving
        a local constant, local linear or local quadratic fit. Static.

    Examples
    --------
    Select a bandwidth for a local constant fit.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: y = 3.0 + 2.0 * x[:, 0]
           ...: train = kj.MixedData.continuous(x)
           ...: criterion = kj.RegressionCriterion(method="cv_ls", degree=0)
           ...: result = kj.select_bandwidth(train, criterion, y=y, n_starts=1)
           ...: print(result.bandwidth.h)

    See Also
    --------
    cv_ls_regression : Leave-one-out mean squared residual.
    aic_c_regression : Corrected Akaike information criterion.

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2007). Nonparametric Econometrics: Theory
           and Practice. Princeton University Press.
    .. [2] Hurvich, C. M., Simonoff, J. S., & Tsai, C. L. (1998). "Smoothing
           parameter selection in nonparametric regression using an improved
           Akaike information criterion." Journal of the Royal Statistical
           Society B, 60, 271-293.
    """

    method: Literal["cv_ls", "aic"] = "cv_ls"
    degree: int = 0

    def __post_init__(self) -> None:
        """Reject a method the criterion does not implement."""
        if self.method not in ("cv_ls", "aic"):
            raise ValueError(f"method must be 'cv_ls' or 'aic', got {self.method!r}.")

    def __call__(
        self,
        train: MixedData,
        bw: Bandwidth,
        *,
        y: Float[Array, " n"],
        kernels: KernelSet | None = None,
        chunk: int | tuple[int, int] | None = None,
    ) -> ScalarFloat:
        """Evaluate the selection rule at a bandwidth.

        Parameters
        ----------
        train : MixedData
            Training sample.
        bw : Bandwidth
            Bandwidths for every column.
        y : Float[Array, " n"]
            Response values, one per row of ``train``.
        kernels : KernelSet, optional
            Kernel families, one per column kind. Defaults to ``KernelSet()``.
        chunk : int or tuple of int, optional
            Chunk sizes passed through to the underlying fit.

        Returns
        -------
        ScalarFloat
            The criterion value, minimized over ``bw`` to select a bandwidth.
        """
        rule = cv_ls_regression if self.method == "cv_ls" else aic_c_regression
        return rule(train, bw, y=y, kernels=kernels, degree=self.degree, chunk=chunk)


@dataclasses.dataclass(frozen=True)
class DensityCriterion:
    r"""Density selection rule.

    Evaluating an instance runs the chosen rule at a bandwidth.

    Parameters
    ----------
    method : {"cv_ml", "cv_ls"}
        Selection rule. ``"cv_ml"`` is the leave-one-out log likelihood and
        ``"cv_ls"`` the integrated squared error, both following [1]_.

    Examples
    --------
    Select a density bandwidth by least squares cross validation.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: result = kj.select_bandwidth(train, kj.DensityCriterion(method="cv_ls"), n_starts=1)
           ...: print(result.bandwidth.h)

    See Also
    --------
    cv_ml_density : Leave-one-out log likelihood.
    cv_ls_density : Integrated squared error.

    References
    ----------
    .. [1] Li, Q., & Racine, J. S. (2003). "Nonparametric estimation of
           distributions with categorical and continuous data." Journal of
           Multivariate Analysis, 86, 266-292.
    """

    method: Literal["cv_ml", "cv_ls"] = "cv_ml"

    def __post_init__(self) -> None:
        """Reject a method the criterion does not implement."""
        if self.method not in ("cv_ml", "cv_ls"):
            raise ValueError(f"method must be 'cv_ml' or 'cv_ls', got {self.method!r}.")

    def __call__(
        self,
        train: MixedData,
        bw: Bandwidth,
        *,
        kernels: KernelSet | None = None,
        chunk: int | tuple[int, int] | None = None,
    ) -> ScalarFloat:
        """Evaluate the selection rule at a bandwidth.

        Parameters
        ----------
        train : MixedData
            Training sample.
        bw : Bandwidth
            Bandwidths for every column.
        kernels : KernelSet, optional
            Kernel families, one per column kind. Defaults to ``KernelSet()``.
        chunk : int or tuple of int, optional
            Chunk sizes passed through to the underlying sums.

        Returns
        -------
        ScalarFloat
            The criterion value, minimized over ``bw`` to select a bandwidth.
        """
        rule = cv_ml_density if self.method == "cv_ml" else cv_ls_density
        return rule(train, bw, kernels=kernels, chunk=chunk)
