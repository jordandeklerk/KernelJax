"""Tests for the numeric kernel checks."""

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.estimators.density import density
from kerneljax.estimators.distribution import cdf
from kerneljax.estimators.regression import local_poly
from kerneljax.kernels import KernelSet
from kerneljax.kernels._checks import _check_grad_diagonal, _check_value_mass, _passed
from kerneljax.kernels.base import ContinuousKernel


def _epan(x, y, h):
    u = (x - y) / h
    return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)


def _epan_conv(x, y, h):
    u = jnp.abs((x - y) / h)
    return jnp.where(u <= 2.0, (3.0 / 160.0) * (2.0 - u) ** 3 * (u * u + 6.0 * u + 4.0), 0.0)


@dataclasses.dataclass(frozen=True)
class HalfMass(ContinuousKernel):
    def value(self, x, y, h):
        return 0.5 * _epan(x, y, h)


@dataclasses.dataclass(frozen=True)
class CarriesOverH(ContinuousKernel):
    def value(self, x, y, h):
        return _epan(x, y, h) / h


@dataclasses.dataclass(frozen=True)
class UnguardedSinc(ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(u == 0.0, 1.0, jnp.sin(u) / u)


@dataclasses.dataclass(frozen=True)
class ConvOverH(ContinuousKernel):
    def value(self, x, y, h):
        return _epan(x, y, h)

    def conv(self, x, y, h):
        return _epan_conv(x, y, h) / h


@dataclasses.dataclass(frozen=True)
class TruncatedConv(ContinuousKernel):
    def value(self, x, y, h):
        return _epan(x, y, h)

    def conv(self, x, y, h):
        u = jnp.abs((x - y) / h)
        return jnp.where(u <= 1.0, (3.0 / 160.0) * (2.0 - u) ** 3 * (u * u + 6.0 * u + 4.0), 0.0)


@dataclasses.dataclass(frozen=True)
class CdfOverH(ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)

    def cdf(self, x, y, h):
        from jax.scipy.special import ndtr

        return ndtr((x - y) / h) / h


def test_half_mass_is_refused_by_density(probe_x, probe_bw):
    with pytest.raises(ValueError, match=r"integrates to 0\.5"):
        density(probe_x, probe_bw, kernels=KernelSet(continuous=HalfMass()))


def test_half_mass_passes_regression(probe_x, probe_y, probe_bw):
    fit = local_poly(probe_x, probe_y, probe_bw, degree=1, kernels=KernelSet(continuous=HalfMass()))
    assert bool(jnp.all(jnp.isfinite(fit.mean)))


def test_a_one_over_h_factor_is_named(probe_x, probe_bw):
    with pytest.raises(ValueError, match="signature of a kernel carrying its own 1/h factor"):
        density(probe_x, probe_bw, kernels=KernelSet(continuous=CarriesOverH()))


def test_a_nonconforming_kernel_is_refused_inside_jit(probe_x, probe_bw):
    import jax

    with pytest.raises(ValueError, match="signature of a kernel carrying its own 1/h factor"):
        jax.jit(lambda x, bw: density(x, bw, kernels=KernelSet(continuous=CarriesOverH())).value)(probe_x, probe_bw)


def test_half_mass_is_refused_inside_jit(probe_x, probe_bw):
    import jax

    with pytest.raises(ValueError, match=r"integrates to 0\.5"):
        jax.jit(lambda x, bw: density(x, bw, kernels=KernelSet(continuous=HalfMass())).value)(probe_x, probe_bw)


def test_a_conforming_kernel_passes_the_checks_inside_jit(probe_x, probe_bw):
    import jax

    @dataclasses.dataclass(frozen=True)
    class JittedGaussian(ContinuousKernel):
        def value(self, x, y, h):
            u = (x - y) / h
            return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)

    values = jax.jit(lambda x, bw: density(x, bw, kernels=KernelSet(continuous=JittedGaussian())).value)(
        probe_x, probe_bw
    )
    assert bool(jnp.all(jnp.isfinite(values)))


def test_an_unguarded_branch_is_refused_before_selection(probe_x, probe_y):
    with pytest.raises(ValueError, match="differentiates both branches"):
        local_poly(probe_x, probe_y, "cv_ls", degree=1, kernels=KernelSet(continuous=UnguardedSinc()))


def test_an_unguarded_branch_still_fits_at_a_fixed_bandwidth(probe_x, probe_y, probe_bw):
    fit = local_poly(probe_x, probe_y, probe_bw, degree=1, kernels=KernelSet(continuous=UnguardedSinc()))
    assert bool(jnp.all(jnp.isfinite(fit.mean)))


def test_a_conv_carrying_one_over_h_is_refused_by_density_cv_ls(probe_x):
    with pytest.raises(ValueError, match="signature of a 1/h factor"):
        density(probe_x, "cv_ls", n_starts=1, kernels=KernelSet(continuous=ConvOverH()))


def test_a_truncated_conv_is_refused_by_density_cv_ls(probe_x):
    with pytest.raises(ValueError, match="truncating it to the kernel's own support"):
        density(probe_x, "cv_ls", n_starts=1, kernels=KernelSet(continuous=TruncatedConv()))


def test_a_truncated_conv_still_serves_standard_errors(probe_x, probe_y, probe_bw):
    fit = local_poly(probe_x, probe_y, probe_bw, degree=1, se=True, kernels=KernelSet(continuous=TruncatedConv()))
    assert bool(jnp.all(jnp.isfinite(fit.se)))


def test_a_cdf_carrying_one_over_h_is_refused(probe_x, probe_bw):
    with pytest.raises(ValueError, match="cdf runs from"):
        cdf(probe_x, probe_bw, kernels=KernelSet(continuous=CdfOverH()))


def test_a_passing_check_runs_once_per_kernel():
    calls = []

    @dataclasses.dataclass(frozen=True)
    class Counting(ContinuousKernel):
        def value(self, x, y, h):
            calls.append(1)
            u = (x - y) / h
            return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)

    kernel = Counting()
    _check_value_mass(kernel)
    first = len(calls)
    _check_value_mass(kernel)

    assert first > 0
    assert len(calls) == first


def test_a_wrong_conv_at_zero_is_refused_before_standard_errors(probe_x, probe_y, probe_bw):
    @dataclasses.dataclass(frozen=True)
    class WrongRoughness(ContinuousKernel):
        def value(self, x, y, h):
            return _epan(x, y, h)

        def conv(self, x, y, h):
            return 0.5 * _epan_conv(x, y, h)

    with pytest.raises(ValueError, match=r"self-convolution of WrongRoughness\.value at zero"):
        local_poly(probe_x, probe_y, probe_bw, degree=1, se=True, kernels=KernelSet(continuous=WrongRoughness()))


@dataclasses.dataclass(frozen=True)
class Cauchy(ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return 1.0 / (jnp.pi * (1.0 + u * u))

    def conv(self, x, y, h):
        u = (x - y) / h
        return 2.0 / (jnp.pi * (4.0 + u * u))

    def cdf(self, x, y, h):
        return 0.5 + jnp.arctan((x - y) / h) / jnp.pi


def test_the_heaviest_tailed_legitimate_kernel_passes_every_probe(probe_x, probe_bw):
    kernels = KernelSet(continuous=Cauchy())
    density(probe_x, probe_bw, kernels=kernels)
    density(probe_x, "cv_ls", n_starts=1, kernels=kernels)
    cdf(probe_x, probe_bw, kernels=kernels)


def test_a_truncated_conv_is_caught_whatever_the_smoothness(probe_x):
    @dataclasses.dataclass(frozen=True)
    class QuarticTruncatedConv(ContinuousKernel):
        def value(self, x, y, h):
            u = (x - y) / h
            return jnp.where(jnp.abs(u) <= 1.0, (315.0 / 256.0) * (1.0 - u * u) ** 4, 0.0)

        def conv(self, x, y, h):
            u = jnp.abs((x - y) / h)
            return jnp.where(u <= 1.0, 0.7371 * self.value(x, y, h), jnp.zeros_like(u))

    with pytest.raises(ValueError, match="conv is zero at"):
        density(probe_x, "cv_ls", n_starts=1, kernels=KernelSet(continuous=QuarticTruncatedConv()))


def test_a_cdf_that_plateaus_short_of_one_is_refused(probe_x, probe_bw):
    from jax.scipy.special import ndtr

    @dataclasses.dataclass(frozen=True)
    class Plateau(ContinuousKernel):
        def value(self, x, y, h):
            u = (x - y) / h
            return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)

        def cdf(self, x, y, h):
            return 0.98 * ndtr((x - y) / h)

    with pytest.raises(ValueError, match="cdf runs from"):
        cdf(probe_x, probe_bw, kernels=KernelSet(continuous=Plateau()))


def test_a_guard_outside_the_sqrt_is_refused_beyond_the_support(probe_x, probe_y):
    @dataclasses.dataclass(frozen=True)
    class SemicircleClamp(ContinuousKernel):
        def value(self, x, y, h):
            u = (x - y) / h
            return jnp.where(jnp.abs(u) <= 1.0, (2.0 / jnp.pi) * jnp.sqrt(jnp.maximum(1.0 - u * u, 0.0)), 0.0)

    with pytest.raises(ValueError, match="differentiates both branches"):
        local_poly(probe_x, probe_y, "cv_ls", degree=1, kernels=KernelSet(continuous=SemicircleClamp()))


def test_a_guarded_semicircle_selects(probe_x, probe_y):
    @dataclasses.dataclass(frozen=True)
    class Semicircle(ContinuousKernel):
        def value(self, x, y, h):
            u = (x - y) / h
            inside = jnp.abs(u) <= 1.0
            safe = jnp.where(inside, 1.0 - u * u, 1.0)
            return jnp.where(inside, (2.0 / jnp.pi) * jnp.sqrt(safe), 0.0)

    fit = local_poly(probe_x, probe_y, "cv_ls", degree=1, kernels=KernelSet(continuous=Semicircle()))
    assert bool(jnp.all(jnp.isfinite(fit.mean)))


@dataclasses.dataclass(frozen=True)
class Uniform(ContinuousKernel):
    def value(self, x, y, h):
        return jnp.where(jnp.abs((x - y) / h) <= 1.0, 0.5, 0.0)


def _default_tol(check):
    return inspect.signature(check).parameters["tol"].default


def test_mass_tolerance_margins():
    from kerneljax.kernels._checks import _check_value_mass, _mass

    tol = _default_tol(_check_value_mass)
    worst_legitimate = max(
        abs(_mass(Uniform().value, 2.0) - 1.0),
        abs(_mass(Cauchy().value, 2.0) - 1.0),
    )
    smallest_failure = 0.05

    assert worst_legitimate < tol / 5
    assert smallest_failure > 2 * tol


def test_conv_tolerance_margins():
    from kerneljax.kernels._checks import _check_conv_matches, _conv_offsets, _in_u_units, _numeric_conv

    tol = _default_tol(_check_conv_matches)
    offsets = _conv_offsets()
    numeric = _numeric_conv(Cauchy(), offsets)
    stated = _in_u_units(Cauchy().conv, offsets, 1.0)
    worst_legitimate = float(np.max(np.abs(numeric - stated)))

    @dataclasses.dataclass(frozen=True)
    class Quartic(ContinuousKernel):
        def value(self, x, y, h):
            u = (x - y) / h
            return jnp.where(jnp.abs(u) <= 1.0, (315.0 / 256.0) * (1.0 - u * u) ** 4, 0.0)

    beyond_own_support = offsets[np.abs(offsets) >= 1.0]
    truncation_signal = float(np.max(_numeric_conv(Quartic(), beyond_own_support)))

    assert worst_legitimate < tol / 100
    assert truncation_signal > 10 * tol


def test_cdf_tolerance_margins():
    from kerneljax.kernels._checks import _check_cdf_limits, _u_grid

    tol = _default_tol(_check_cdf_limits)
    edge = jnp.asarray(_u_grid()[-1])
    worst_legitimate = float(abs(Cauchy().cdf(edge, jnp.asarray(0.0), jnp.asarray(1.0)) - 1.0))
    smallest_failure = 0.02

    assert worst_legitimate < tol / 3
    assert smallest_failure > 3 * tol


def test_a_criterion_composing_density_survives_the_selection_trace(probe_x):
    @dataclasses.dataclass(frozen=True)
    class FreshGaussian(ContinuousKernel):
        def value(self, x, y, h):
            u = (x - y) / h
            return jnp.exp(-0.5 * u * u) / jnp.sqrt(2.0 * jnp.pi)

    @dataclasses.dataclass(frozen=True)
    class KFoldLikelihood:
        n_folds: int = 5

        def __call__(self, train, bandwidth, *, kernels=None, chunk=None):
            fold = jnp.arange(train.n) % self.n_folds
            fit = density(train, bandwidth, kernels=kernels, chunk=chunk, fold=fold)
            return -jnp.sum(jnp.log(fit.value))

    from kerneljax.selection.optimize import select_bandwidth

    result = select_bandwidth(probe_x, KFoldLikelihood(), kernels=KernelSet(continuous=FreshGaussian()))
    assert bool(jnp.isfinite(result.bandwidth.h[0]))


def test_the_gradient_check_runs_inside_jit():
    @dataclasses.dataclass(frozen=True)
    class Epan(ContinuousKernel):
        def value(self, x, y, h):
            return _epan(x, y, h)

    _passed.clear()

    def probe(h):
        _check_grad_diagonal(Epan(), "value")
        return h

    assert float(jax.jit(probe)(0.2)) == pytest.approx(0.2)
    assert ("grad_value", Epan()) in _passed


def test_an_unguarded_branch_is_still_refused_inside_jit():
    _passed.clear()

    def probe(h):
        _check_grad_diagonal(UnguardedSinc(), "value")
        return h

    with pytest.raises(ValueError, match="non-finite bandwidth gradient"):
        jax.jit(probe)(0.2)
