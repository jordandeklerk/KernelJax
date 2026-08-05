"""Bandwidth selection by minimizing a cross-validation criterion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Float

from kerneljax.bandwidth import BandwidthTransform, SelectionResult, normal_reference
from kerneljax.data import MixedData, _as_points
from kerneljax.kernels import KernelSet
from kerneljax.typing import Array, ScalarFloat

__all__ = ["lbfgs", "select_bandwidth"]


@partial(jax.jit, static_argnames=("criterion", "solver", "n_starts", "chunk"))
def select_bandwidth(
    train: MixedData | Array,
    criterion: Callable[..., ScalarFloat],
    *,
    y: Float[Array, " n"] | None = None,
    criterion_kwargs: Mapping[str, Any] | None = None,
    kernels: KernelSet | None = None,
    solver: Callable[..., tuple[Array, ScalarFloat, Array, Array]] | None = None,
    n_starts: int = 3,
    chunk: int | tuple[int, int] | None = None,
) -> SelectionResult:
    r"""Select a bandwidth by minimizing a cross-validation criterion.

    A data-driven bandwidth minimizes a criterion :math:`\mathrm{CV}` such as
    :func:`~kerneljax.cv_ml_density` or
    :func:`~kerneljax.cv_ls_density`, following [1]_ and [2]_.

    .. math::

        (\hat h, \hat\lambda)
        = \arg\min_{h > 0,\ 0 \le \lambda \le \lambda_{\max}} \mathrm{CV}(h, \lambda)

    The box constraints are removed by optimizing over an unconstrained vector :math:`z`.

    .. math::

        h = \operatorname{softplus}(z), \qquad
        \lambda = \lambda_{\max}\, \sigma(z)

    Here :math:`\sigma` is the logistic function and :math:`\lambda_{\max}` is the kernel's
    upper bound for that column, see :class:`~kerneljax.BandwidthTransform`. A
    smoothing parameter returned at its upper bound smooths that column away entirely.

    Parameters
    ----------
    train : MixedData or Array
        Training sample defining the column spec optimized over. A raw array
        is read as continuous columns.
    criterion : callable
        Cross-validation criterion, called as
        ``criterion(train, bandwidth, **criterion_kwargs, kernels=kernels, chunk=chunk)``
        and minimized. Every criterion in :mod:`kerneljax.tuning.criteria`
        and :mod:`kerneljax.tuning.objectives` matches this signature. Static,
        and a :func:`functools.partial` built fresh on each call is never equal
        to an earlier one, so prefer a criterion object or a module level
        function.
    y : Float[Array, " n"], optional
        Response values, one per row of ``train``, forwarded to criteria that
        take one. Traced, so varying it reuses the compiled selector.
    criterion_kwargs : mapping, optional
        Further arguments forwarded to ``criterion``, for criteria that need
        more than a response. Traced on the same terms as ``y``.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    solver : callable, optional
        Optimizer called as ``solver(objective, z0)`` in the unconstrained
        parameterization, returning ``(z, value, n_iter, converged)``.
        Defaults to :func:`~kerneljax.lbfgs`; any callable matching this signature is
        accepted. Static.
    n_starts : int
        Number of starting points the solver runs from, the first being the
        reference rule and the rest perturbations of it. The best solve wins.
        Static.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to ``criterion``. Static.

    Returns
    -------
    SelectionResult
        Object containing the selected bandwidth and the solve that found it:

        - **bandwidth**: Selected bandwidth, in natural constrained scale
        - **value**: Criterion value at the selected bandwidth
        - **n_iter**: Number of solver iterations used by the full solve
        - **converged**: Whether the solver stopped because its progress stalled
        - **criterion**: Criterion that was minimized

    Examples
    --------
    Select a bandwidth by minimizing likelihood cross validation.

    .. ipython::
        :okwarning:

        In [1]: import jax.numpy as jnp
           ...: import kerneljax as kj
           ...:
           ...: x = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
           ...: train = kj.MixedData.continuous(x)
           ...: result = kj.select_bandwidth(train, kj.cv_ml_density, n_starts=1)
           ...: print(result.bandwidth.h)

    References
    ----------
    .. [1] Hall, P., Racine, J., & Li, Q. (2004). "Cross-validation and the
           estimation of conditional probability densities." Journal of the
           American Statistical Association, 99, 1015-1026.
    .. [2] Li, Q., & Racine, J. S. (2007). Nonparametric Econometrics: Theory
           and Practice. Princeton University Press.
    """
    kernels = KernelSet() if kernels is None else kernels
    solver = lbfgs if solver is None else solver
    train = _as_points(train)
    extra = {} if criterion_kwargs is None else dict(criterion_kwargs)

    if y is not None:
        extra["y"] = y

    transform = BandwidthTransform(spec=train.spec, kernels=kernels)
    start = normal_reference(train, kernels)
    z0 = transform.to_unconstrained(start)

    def objective(z: Array) -> ScalarFloat:
        bandwidth = transform.from_unconstrained(z)
        return criterion(train, bandwidth, **extra, kernels=kernels, chunk=chunk)

    perturbations = jnp.linspace(-1.5, 1.5, n_starts - 1) if n_starts > 1 else jnp.zeros(0)
    offsets = jnp.concatenate([jnp.zeros(1), perturbations])[:, None] * jnp.ones_like(z0)[None, :]
    candidates = z0[None, :] + offsets
    solved, values, iterations, flags = jax.lax.map(lambda start: solver(objective, start), candidates)
    usable = jnp.logical_and(jnp.isfinite(values), jnp.all(jnp.isfinite(solved), axis=1))
    ranked = jnp.argmin(jnp.where(usable, values, jnp.inf))

    return SelectionResult(
        bandwidth=transform.from_unconstrained(solved[ranked]),
        value=values[ranked],
        n_iter=iterations[ranked],
        converged=jnp.logical_and(flags[ranked], usable[ranked]),
        criterion=criterion,
    )


def lbfgs(
    fun: Callable[[Array], ScalarFloat],
    z0: Array,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    history: int = 10,
) -> tuple[Array, ScalarFloat, Array, Array]:
    r"""Minimize ``fun`` from ``z0`` with limited-memory BFGS.

    Limited-memory BFGS builds an implicit approximation to the inverse Hessian from the
    ``history`` most recent curvature pairs, following [1]_ and [2]_.

    .. math::

        s_k = z_{k+1} - z_k, \qquad y_k = g_{k+1} - g_k

    Here :math:`s_k` and :math:`y_k` are the step and gradient differences between
    consecutive iterates. The search direction is recovered from these pairs by the two loop
    recursion, without ever forming the Hessian.

    A step length :math:`\alpha` is accepted once backtracking from :math:`\alpha = 1`
    satisfies the Armijo condition

    .. math::

        f(z + \alpha p) \le f(z) + c_1 \alpha\, p^{\top} g

    with :math:`p` the search direction and :math:`g` the gradient at :math:`z`. A step
    whose objective value comes back non-finite is rejected.

    A curvature pair is stored only when :math:`s_k^{\top} y_k` is positive, keeping the
    inverse Hessian approximation well defined.

    Parameters
    ----------
    fun : callable
        Scalar objective, differentiable in its argument.
    z0 : Array
        Starting point, shape ``(k,)``.
    max_iter : int
        Maximum number of iterations. Static.
    tol : float
        Convergence tolerance, satisfied once the gradient's max norm or
        the relative change in the objective value falls below it. Static.
    history : int
        Number of curvature pairs kept for the two loop recursion. Static.

    Returns
    -------
    tuple of Array
        ``(z, value, n_iter, converged)``, the minimizer, the objective
        value there, the number of iterations used and whether ``tol`` was
        reached before ``max_iter``.

    References
    ----------
    .. [1] Nocedal, J. (1980). "Updating quasi-Newton matrices with limited
           storage." Mathematics of Computation, 35, 773-782.
    .. [2] Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS
           method for large scale optimization." Mathematical Programming, 45,
           503-528.
    """
    start_value = fun(z0)
    scale = jnp.where(start_value == 0.0, 1.0, jnp.abs(start_value))

    def scaled(z: Array) -> ScalarFloat:
        return fun(z) / scale

    value_and_grad_fn = jax.value_and_grad(scaled)
    dim = z0.shape[0]

    state_type = tuple[Array, Array, Array, Array, Array, Array, Array]

    def cond_fn(state: state_type) -> Array:
        _, _, _, _, _, iteration, converged = state
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))

    def body_fn(state: state_type) -> state_type:
        z, value, grad, step_diffs, grad_diffs, iteration, _ = state

        direction = _two_loop_direction(grad, step_diffs, grad_diffs)
        search = -direction
        search = jnp.where(search @ grad < 0, search, -grad)

        z_new, _ = _backtracking_step(scaled, z, search, value, grad)

        value_new, grad_new = value_and_grad_fn(z_new)
        step_diff = z_new - z
        grad_diff = grad_new - grad

        curvature = step_diff @ grad_diff
        curvature_is_safe = curvature > 1e-10 * jnp.maximum(step_diff @ step_diff, 1.0)
        step_diff = jnp.where(curvature_is_safe, step_diff, 0.0)
        grad_diff = jnp.where(curvature_is_safe, grad_diff, 0.0)
        step_diffs = jnp.roll(step_diffs, -1, axis=0).at[history - 1].set(step_diff)
        grad_diffs = jnp.roll(grad_diffs, -1, axis=0).at[history - 1].set(grad_diff)

        stalled = jnp.logical_or(
            jnp.max(jnp.abs(grad_new)) < tol, jnp.abs(value_new - value) < tol * (1.0 + jnp.abs(value))
        )
        usable = jnp.logical_and(jnp.isfinite(value_new), jnp.all(jnp.isfinite(grad_new)))
        converged = jnp.logical_and(stalled, usable)
        return z_new, value_new, grad_new, step_diffs, grad_diffs, iteration + 1, converged

    value0, grad0 = value_and_grad_fn(z0)
    init: state_type = (
        z0,
        value0,
        grad0,
        jnp.zeros((history, dim), dtype=z0.dtype),
        jnp.zeros((history, dim), dtype=z0.dtype),
        jnp.asarray(0),
        jnp.asarray(False),
    )
    z, value, _, _, _, n_iter, converged = jax.lax.while_loop(cond_fn, body_fn, init)
    return z, value * scale, n_iter, converged


def _two_loop_direction(gradient: Array, step_history: Array, grad_history: Array) -> Array:
    """Apply the two loop recursion, approximating the inverse Hessian acting on the gradient."""
    history = step_history.shape[0]
    direction = gradient
    alphas = jnp.zeros(history, dtype=gradient.dtype)

    def backward(carry: tuple[Array, Array], index: Array) -> tuple[tuple[Array, Array], None]:
        direction, alphas = carry
        step_diff, grad_diff = step_history[index], grad_history[index]
        rho = jnp.where(jnp.abs(step_diff @ grad_diff) > 1e-12, 1.0 / (step_diff @ grad_diff), 0.0)
        alpha = rho * (step_diff @ direction)
        return (direction - alpha * grad_diff, alphas.at[index].set(alpha)), None

    (direction, alphas), _ = jax.lax.scan(backward, (direction, alphas), jnp.arange(history - 1, -1, -1))

    step_diff_last, grad_diff_last = step_history[history - 1], grad_history[history - 1]
    denom = grad_diff_last @ grad_diff_last
    gamma = jnp.where(denom > 1e-12, (step_diff_last @ grad_diff_last) / denom, 1.0)
    direction = gamma * direction

    def forward(direction: Array, index: Array) -> tuple[Array, None]:
        step_diff, grad_diff = step_history[index], grad_history[index]
        rho = jnp.where(jnp.abs(step_diff @ grad_diff) > 1e-12, 1.0 / (step_diff @ grad_diff), 0.0)
        beta = rho * (grad_diff @ direction)
        return direction + step_diff * (alphas[index] - beta), None

    direction, _ = jax.lax.scan(forward, direction, jnp.arange(history))
    return direction


def _backtracking_step(
    fun: Callable[[Array], ScalarFloat], z: Array, direction: Array, value: ScalarFloat, gradient: Array
) -> tuple[Array, ScalarFloat]:
    """Backtrack along direction from z until the Armijo condition holds, rejecting non-finite steps."""

    def line_search_cond(carry: tuple[Array, Array, Array]) -> Array:
        step, value_new, _ = carry
        insufficient = jnp.logical_or(
            jnp.logical_not(jnp.isfinite(value_new)), value_new > value + 1e-4 * step * (direction @ gradient)
        )
        return jnp.logical_and(step > 1e-14, insufficient)

    def line_search_body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        step, _, _ = carry
        step = step * 0.5
        candidate = z + step * direction
        return step, fun(candidate), candidate

    step0 = jnp.asarray(1.0, dtype=z.dtype)
    candidate0 = z + step0 * direction
    _, value_new, z_new = jax.lax.while_loop(line_search_cond, line_search_body, (step0, fun(candidate0), candidate0))
    z_new = jnp.where(jnp.isfinite(value_new), z_new, z)
    return z_new, value_new
