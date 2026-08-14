"""Tests for discrete kernels."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.kernels.discrete import AitchisonAitken, LiRacine, WangVanRyzin


@pytest.mark.parametrize("levels", [2, 3, 4, 6, 10])
@pytest.mark.parametrize("lam_f", [0.0, 0.1, 0.31, 0.49, 0.6])
def test_aitchison_aitken_value_matches_numpy_formula(aitchison_aitken, lam_f, levels):
    lam = jnp.array(lam_f)
    for x in range(levels):
        for y in range(levels):
            expected = (1.0 - lam_f) if x == y else lam_f / (levels - 1)
            got = float(aitchison_aitken.value(jnp.array(x), jnp.array(y), lam, levels))
            assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("levels", [2, 3, 5, 8])
@pytest.mark.parametrize("lam_f", [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
def test_aitchison_aitken_sums_to_one(aitchison_aitken, lam_f, levels):
    lam = jnp.array(lam_f)
    for x in range(levels):
        total = sum(float(aitchison_aitken.value(jnp.array(x), jnp.array(s), lam, levels)) for s in range(levels))
        assert total == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize(("levels", "expected"), [(2, 0.5), (3, 2.0 / 3.0), (4, 0.75), (5, 0.8), (10, 0.9)])
def test_aitchison_aitken_upper_bound(aitchison_aitken, levels, expected):
    assert aitchison_aitken.upper_bound(levels) == pytest.approx(expected)


@pytest.mark.parametrize("levels", [2, 3, 5, 7])
@pytest.mark.parametrize("lam_f", [0.0, 0.05, 0.31, 0.49, 0.7])
def test_aitchison_aitken_conv_matches_brute_sum(aitchison_aitken, lam_f, levels):
    lam = jnp.array(lam_f)
    for x in range(levels):
        for y in range(levels):
            brute = sum(
                float(aitchison_aitken.value(jnp.array(x), jnp.array(s), lam, levels))
                * float(aitchison_aitken.value(jnp.array(y), jnp.array(s), lam, levels))
                for s in range(levels)
            )
            got = float(aitchison_aitken.conv(jnp.array(x), jnp.array(y), lam, levels))
            assert got == pytest.approx(brute, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("lam_f", [0.0, 0.33, 0.6])
def test_aitchison_aitken_conv_matches_random_pairs(aitchison_aitken, lam_f):
    levels = 12
    lam = jnp.array(lam_f)
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, levels, size=(20, 2))
    for x, y in pairs:
        brute = sum(
            float(aitchison_aitken.value(jnp.array(int(x)), jnp.array(s), lam, levels))
            * float(aitchison_aitken.value(jnp.array(int(y)), jnp.array(s), lam, levels))
            for s in range(levels)
        )
        got = float(aitchison_aitken.conv(jnp.array(int(x)), jnp.array(int(y)), lam, levels))
        assert got == pytest.approx(brute, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize(("x", "y"), [(0, 0), (0, 1)])
@pytest.mark.parametrize("levels", [2, 3, 5, 8])
def test_aitchison_aitken_grads_finite_at_boundaries(aitchison_aitken, levels, x, y):
    for lam_f in (0.0, aitchison_aitken.upper_bound(levels)):
        lam0 = jnp.array(lam_f)
        gv = jax.grad(lambda lam: aitchison_aitken.value(jnp.array(x), jnp.array(y), lam, levels))(lam0)
        gc = jax.grad(lambda lam: aitchison_aitken.conv(jnp.array(x), jnp.array(y), lam, levels))(lam0)
        assert jnp.isfinite(gv)
        assert jnp.isfinite(gc)


@pytest.mark.parametrize(("x", "y"), [(0, 0), (2, 2), (0, 1), (0, 2), (0, 5), (3, 0)])
@pytest.mark.parametrize("lam_f", [0.0, 0.1, 0.4, 0.6, 0.9])
def test_wang_van_ryzin_value_matches_numpy_formula(wang_van_ryzin, lam_f, x, y):
    lam = jnp.array(lam_f)
    d = abs(x - y)
    expected = (1.0 - lam_f) if d == 0 else 0.5 * (1.0 - lam_f) * lam_f**d
    got = float(wang_van_ryzin.value(jnp.array(x), jnp.array(y), lam, 6))
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("levels", [2, 3, 4, 10, 50])
def test_wang_van_ryzin_upper_bound(wang_van_ryzin, levels):
    assert wang_van_ryzin.upper_bound(levels) == pytest.approx(1.0)


@pytest.mark.parametrize("d", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("lam_f", [0.05, 0.2, 0.4, 0.6, 0.9, 0.99])
def test_wang_van_ryzin_conv_matches_lattice_sum(wang_van_ryzin, lam_f, d):
    lattice = np.arange(-4000, 4001)

    def l_np(a, b):
        dd = np.abs(a - b)
        return np.where(dd == 0, 1.0 - lam_f, 0.5 * (1.0 - lam_f) * lam_f**dd)

    brute = float(np.sum(l_np(0, lattice) * l_np(d, lattice)))
    got = float(wang_van_ryzin.conv(jnp.array(0), jnp.array(d), jnp.array(lam_f), 3))
    assert got == pytest.approx(brute, rel=1e-5)


@pytest.mark.parametrize(("x", "y"), [(0, 0), (0, 2), (1, 4)])
@pytest.mark.parametrize("lam_f", [0.0, 0.999])
def test_wang_van_ryzin_grads_finite_near_boundaries(wang_van_ryzin, lam_f, x, y):
    lam0 = jnp.array(lam_f)
    gv = jax.grad(lambda lam: wang_van_ryzin.value(jnp.array(x), jnp.array(y), lam, 6))(lam0)
    gc = jax.grad(lambda lam: wang_van_ryzin.conv(jnp.array(x), jnp.array(y), lam, 6))(lam0)
    assert jnp.isfinite(gv)
    assert jnp.isfinite(gc)


@pytest.mark.parametrize("op", ["value", "conv"])
@pytest.mark.parametrize("kernel_fixture", ["aitchison_aitken", "wang_van_ryzin"])
def test_kernels_are_jit_grad_vmappable(request, kernel_fixture, op):
    kernel = request.getfixturevalue(kernel_fixture)
    fn = getattr(kernel, op)
    f = jax.jit(lambda lam: fn(jnp.array(0), jnp.array(2), lam, 5))
    assert jnp.isfinite(f(jnp.array(0.2)))
    assert jnp.isfinite(jax.grad(f)(jnp.array(0.2)))
    assert jax.vmap(f)(jnp.array([0.1, 0.2, 0.3])).shape == (3,)


@pytest.mark.parametrize("kernel_cls", [AitchisonAitken, WangVanRyzin])
def test_kernel_is_frozen_and_hashable_by_value(kernel_cls):
    a = kernel_cls()
    b = kernel_cls()
    assert a == b
    assert hash(a) == hash(b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.extra = 1


@pytest.mark.parametrize("kernel_cls", [LiRacine, WangVanRyzin])
@pytest.mark.parametrize("lam", [0.1, 0.5, 0.9])
def test_ordered_cdf_is_monotone(kernel_cls, lam):
    kernel = kernel_cls()
    levels = 5
    got = kernel.cdf(jnp.arange(levels), jnp.asarray(2), jnp.asarray(lam), levels)
    assert jnp.all(jnp.diff(got) >= -1e-7)


@pytest.mark.parametrize("kernel_cls", [LiRacine, WangVanRyzin])
@pytest.mark.parametrize("lam", [0.2, 0.6, 0.9])
def test_ordered_cdf_sums_the_full_lattice(kernel_cls, lam):
    kernel = kernel_cls()
    levels, y = 5, 2
    far = jnp.arange(y - 400, y + 1 + 400)

    for x in range(levels):
        weights = kernel.value(far, jnp.asarray(y), jnp.asarray(lam), levels)
        want = float(jnp.sum(jnp.where(far <= x, weights, 0.0)))
        got = float(kernel.cdf(jnp.asarray(x), jnp.asarray(y), jnp.asarray(lam), levels))
        assert got == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("kernel_cls", [LiRacine, WangVanRyzin])
def test_ordered_cdf_spans_the_unit(kernel_cls):
    kernel = kernel_cls()
    lam, y = jnp.asarray(0.5), jnp.asarray(3)
    assert float(kernel.cdf(jnp.asarray(-200), y, lam, 5)) == pytest.approx(0.0, abs=1e-9)
    assert float(kernel.cdf(jnp.asarray(200), y, lam, 5)) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("kernel", [WangVanRyzin(), LiRacine()])
@pytest.mark.parametrize("d", [0, 1, 3])
def test_conv_is_finite_at_the_upper_bound(kernel, d):
    value = float(kernel.conv(jnp.array(0), jnp.array(d), jnp.array(1.0), 4))
    grad = float(jax.grad(lambda lam: jnp.sum(kernel.conv(jnp.array(0), jnp.array(d), lam, 4)))(jnp.array(1.0)))
    assert np.isfinite(value)
    assert np.isfinite(grad)


@pytest.mark.parametrize("kernel", [WangVanRyzin(), LiRacine()])
@pytest.mark.parametrize("d", [0, 1, 3])
def test_conv_is_continuous_at_the_upper_bound(kernel, d):
    at_bound = float(kernel.conv(jnp.array(0), jnp.array(d), jnp.array(1.0), 4))
    just_inside = float(kernel.conv(jnp.array(0), jnp.array(d), jnp.array(0.9999), 4))
    assert at_bound == pytest.approx(just_inside, abs=1e-3)
