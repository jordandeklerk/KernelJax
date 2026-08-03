"""Weighted least squares moment system."""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Bool, Float

from kerneljax.typing import Array, FloatArray, ScalarFloat

__all__ = ["WLS", "hat_diagonal", "wls"]


@partial(jax.tree_util.register_dataclass, data_fields=["coef", "cho", "ok", "rcond"], meta_fields=[])
@dataclasses.dataclass(frozen=True)
class WLS:
    """Solution of a weighted least squares moment system.

    Parameters
    ----------
    coef : Float[Array, "k r"]
        Coefficients solving the normal equations, exactly zero when
        ``ok`` is False.
    cho : Float[Array, "k k"]
        Lower Cholesky factor of the regularized Gram matrix, retained
        so a sandwich variance, a hat diagonal or a derivative row can
        reuse it without another factorization.
    ok : Bool[Array, ""]
        Whether the regularized Gram matrix was positive definite.
    rcond : Float[Array, ""]
        A cheap, lower bound estimate of the reciprocal condition
        number of the regularized Gram matrix, built from the diagonal
        of the retained Cholesky factor.

        Near ``1`` for a well conditioned system and small for an ill
        conditioned one. It is ``0.0``, never NaN, when ``ok`` is
        False.
    """

    coef: Float[Array, "k r"]
    cho: Float[Array, "k k"]
    ok: Bool[Array, ""]
    rcond: Float[Array, ""]


def wls(xtwx: FloatArray, xtwy: FloatArray, *, penalty: FloatArray | float = 0.0) -> WLS:
    r"""Solve a weighted least squares moment system by Cholesky factorization.

    Local polynomial regression centers its design at each evaluation point, so this takes
    the normal equation moments already formed, not a design matrix and a response.

    .. math::

        (X^\top W X + P) \beta = X^\top W y

    Here :math:`P` is a ridge penalty, either a scalar added to the diagonal of
    :math:`X^\top W X` or a full matrix added to it directly.

    The reciprocal condition number of the regularized Gram matrix is estimated from the
    diagonal of its Cholesky factor :math:`L`,

    .. math::

        \mathrm{rcond} \approx \left(\frac{\min_i L_{ii}}{\max_i L_{ii}}\right)^{2}

    a lower bound on the true reciprocal condition number, not an exact value.

    Parameters
    ----------
    xtwx : FloatArray
        The Gram matrix :math:`X^\top W X`, shape ``(k, k)``.
    xtwy : FloatArray
        The moment matrix :math:`X^\top W y`, shape ``(k, r)``.
    penalty : FloatArray or float, optional
        The ridge penalty :math:`P`. A scalar is added to the diagonal,
        a ``(k, k)`` matrix is added directly. Defaults to no penalty.

    Returns
    -------
    WLS
        Object containing the weighted least squares solution:

        - **coef**: Fitted coefficients, returned as zero when ``ok`` is False
        - **cho**: Retained Cholesky factor of the regularized Gram matrix
        - **ok**: Whether the regularized Gram matrix was positive definite
        - **rcond**: Estimate of the reciprocal condition number, zero when ``ok`` is False

        A singular system returns zeros rather than whatever a factorization of it
        would otherwise produce, so a failed solve is visible instead of silent.

    Examples
    --------
    Solve a diagonal system by hand to check the coefficients.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: xtwx = jnp.array([[2.0, 0.0], [0.0, 4.0]])
           ...: xtwy = jnp.array([[2.0], [8.0]])
           ...: print(kj.wls(xtwx, xtwy).coef)

    References
    ----------
    .. [1] Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and Its
           Applications. Chapman and Hall.
    """
    dim = xtwx.shape[0]
    diagonal_penalty = jnp.ndim(penalty) == 0

    regularized = xtwx + penalty * jnp.eye(dim, dtype=xtwx.dtype) if diagonal_penalty else xtwx + penalty
    factor = jnp.linalg.cholesky(regularized)
    ok = jnp.all(jnp.isfinite(factor))
    cho = jnp.where(ok, factor, jnp.eye(dim, dtype=regularized.dtype))

    solved = jax.scipy.linalg.cho_solve((cho, True), xtwy)
    coef = jnp.where(ok, solved, jnp.zeros_like(solved))

    factor_diagonal = jnp.diagonal(cho)
    condition_estimate = (jnp.min(factor_diagonal) / jnp.max(factor_diagonal)) ** 2
    rcond = jnp.where(ok, condition_estimate, jnp.zeros_like(condition_estimate))

    return WLS(coef=coef, cho=cho, ok=ok, rcond=rcond)


def hat_diagonal(cho: FloatArray, basis_row: FloatArray, weight_self: ScalarFloat) -> ScalarFloat:
    r"""Return the leverage of an evaluation point from a retained Cholesky factor.

    The hat diagonal, or leverage, of a weighted least squares fit is

    .. math::

        h = w \, b^\top (X^\top W X)^{-1} b

    where :math:`b` is the design row at the evaluation point and :math:`w`
    is its own weight.

    Writing :math:`X^\top W X = L L^\top` for the retained factor :math:`L` and solving
    :math:`L z = b` by forward substitution,

    .. math::

        h = w \, z^\top z

    which avoids ever forming the inverse. Summed across evaluation points, :math:`h` gives
    the effective degrees of freedom that AICc penalizes.

    Parameters
    ----------
    cho : FloatArray
        Lower Cholesky factor of :math:`X^\top W X`, shape ``(k, k)``,
        as returned by :func:`~kerneljax.wls`.
    basis_row : FloatArray
        The design row :math:`b` at the evaluation point, shape ``(k,)``.
    weight_self : ScalarFloat
        The kernel weight :math:`w` of the evaluation point on itself.

    Returns
    -------
    ScalarFloat
        The leverage :math:`h`, lying in the unit interval for a well
        posed weighted least squares problem.

    Examples
    --------
    Compute the leverage implied by a diagonal Gram matrix.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: xtwx = jnp.array([[2.0, 0.0], [0.0, 4.0]])
           ...: xtwy = jnp.array([[2.0], [8.0]])
           ...: fit = kj.wls(xtwx, xtwy)
           ...: basis_row = jnp.array([1.0, 1.0])
           ...: print(kj.hat_diagonal(fit.cho, basis_row, weight_self=1.0))

    References
    ----------
    .. [1] Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and Its
           Applications. Chapman and Hall.
    """
    solved = jax.scipy.linalg.solve_triangular(cho, basis_row, lower=True)
    return weight_self * jnp.sum(solved**2)
