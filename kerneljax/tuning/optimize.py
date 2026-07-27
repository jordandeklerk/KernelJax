"""Bandwidth selection by minimizing a cross-validation criterion."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from kerneljax.bandwidth import Bandwidth, BandwidthTransform, normal_reference
from kerneljax.data import MixedData
from kerneljax.kernels import KernelSet
from kerneljax.typing import Array, ScalarFloat

__all__ = ["SelectionResult", "lbfgs", "select_bandwidth"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["bandwidth", "value", "n_iter", "converged"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True)
class SelectionResult:
    """Outcome of a bandwidth selection.

    Parameters
    ----------
    bandwidth : Bandwidth
        The selected bandwidth, in natural, constrained scale.
    value : ScalarFloat
        Criterion value at ``bandwidth``.
    n_iter : Array
        Number of solver iterations used by the full solve.
    converged : Array
        Whether the solver stopped because its progress stalled, either
        the gradient or the objective value stopped moving, rather than
        because it ran out of its iteration budget. ``True`` does not by
        itself mean the gradient tolerance was the one that was met.
    """

    bandwidth: Bandwidth
    value: ScalarFloat
    n_iter: Array
    converged: Array


@partial(jax.jit, static_argnames=("criterion", "solver", "n_starts", "chunk"))
def select_bandwidth(
    train: MixedData,
    criterion: Callable[..., ScalarFloat],
    *,
    kernels: KernelSet | None = None,
    solver: Callable[..., tuple[Array, ScalarFloat, Array, Array]] | None = None,
    n_starts: int = 3,
    chunk: int | tuple[int, int] | None = None,
) -> SelectionResult:
    """Select a bandwidth by minimizing ``criterion`` in the unconstrained parameterization.

    A smoothing parameter returned at its upper bound smooths that column
    away entirely.

    Parameters
    ----------
    train : MixedData
        Training sample defining the column spec optimized over.
    criterion : callable
        Cross-validation criterion, called as
        ``criterion(train, bandwidth, kernels=kernels, chunk=chunk)`` and
        minimized. Matches the signature of
        :func:`kerneljax.tuning.objectives.cv_ml_density` and
        :func:`kerneljax.tuning.objectives.cv_ls_density`. Static.
    kernels : KernelSet, optional
        Kernel families, one per column kind. Defaults to ``KernelSet()``.
    solver : callable, optional
        Optimizer called as ``solver(objective, z0)`` in the unconstrained
        parameterization, returning ``(z, value, n_iter, converged)``.
        Defaults to :func:`lbfgs`. Any callable matching this signature is
        accepted, so a custom solver needs no change to this package.
        Static.
    n_starts : int
        Number of perturbed starting points screened before the full
        solve. Static.
    chunk : int or tuple of int, optional
        Chunk sizes passed through to ``criterion``. Static.

    Returns
    -------
    SelectionResult
        The selected bandwidth together with the criterion value and
        solver diagnostics from the full solve.
    """
    kernels = KernelSet() if kernels is None else kernels
    solver = lbfgs if solver is None else solver
    transform = BandwidthTransform(spec=train.spec, kernels=kernels)
    start = normal_reference(train, kernels)
    z0 = transform.to_unconstrained(start)

    def objective(z: Array) -> ScalarFloat:
        return criterion(train, transform.from_unconstrained(z), kernels=kernels, chunk=chunk)

    perturbations = jnp.linspace(-1.5, 1.5, n_starts - 1) if n_starts > 1 else jnp.zeros(0)
    offsets = jnp.concatenate([jnp.zeros(1), perturbations])[:, None] * jnp.ones_like(z0)[None, :]
    candidates = z0[None, :] + offsets
    screened = jax.vmap(objective)(candidates)

    finite = jnp.isfinite(screened)
    ranked = jnp.argmin(jnp.where(finite, screened, jnp.inf))
    best = jnp.where(jnp.any(finite), candidates[ranked], z0)

    z, value, n_iter, converged = solver(objective, best)
    return SelectionResult(
        bandwidth=transform.from_unconstrained(z),
        value=value,
        n_iter=n_iter,
        converged=converged,
    )


def lbfgs(
    fun: Callable[[Array], ScalarFloat],
    z0: Array,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    history: int = 10,
) -> tuple[Array, ScalarFloat, Array, Array]:
    """Minimize ``fun`` from ``z0`` with limited-memory BFGS and a backtracking line search.

    A step whose objective value comes back non-finite is rejected.

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
    """
    value_and_grad_fn = jax.value_and_grad(fun)
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

        z_new, _ = _backtracking_step(fun, z, search, value, grad)

        value_new, grad_new = value_and_grad_fn(z_new)
        step_diff = z_new - z
        grad_diff = grad_new - grad

        curvature = step_diff @ grad_diff
        curvature_is_safe = curvature > 1e-10 * jnp.maximum(step_diff @ step_diff, 1.0)
        step_diff = jnp.where(curvature_is_safe, step_diff, 0.0)
        grad_diff = jnp.where(curvature_is_safe, grad_diff, 0.0)
        step_diffs = jnp.roll(step_diffs, -1, axis=0).at[history - 1].set(step_diff)
        grad_diffs = jnp.roll(grad_diffs, -1, axis=0).at[history - 1].set(grad_diff)

        converged = jnp.logical_or(
            jnp.max(jnp.abs(grad_new)) < tol, jnp.abs(value_new - value) < tol * (1.0 + jnp.abs(value))
        )
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
    return z, value, n_iter, converged


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
