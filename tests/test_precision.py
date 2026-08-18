"""Matmul precision tests."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.estimators.conditional import cmode, cv_ls_conditional_density, cv_ls_conditional_distribution
from kerneljax.ksum import ksum


def _subjaxprs(value):
    if hasattr(value, "jaxpr"):
        yield value.jaxpr
    elif hasattr(value, "eqns"):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _subjaxprs(item)


def _dot_precisions(fn, *args):
    collected = []

    def visit(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                collected.append(eqn.params["precision"])
            for value in eqn.params.values():
                for sub in _subjaxprs(value):
                    visit(sub)

    visit(jax.make_jaxpr(fn)(*args).jaxpr)
    return collected


def _assert_all_highest(precisions):
    assert precisions
    for precision in precisions:
        assert precision == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)


@pytest.mark.parametrize("chunk", [None, (2, 2)])
def test_ksum_matmul_precision_is_pinned(kweights_train, kweights_bandwidth, chunk):
    v = jnp.arange(18.0).reshape(6, 3)

    precisions = _dot_precisions(lambda values: ksum(kweights_train, kweights_bandwidth, values, chunk=chunk), v)

    _assert_all_highest(precisions)


@pytest.mark.parametrize("criterion", [cv_ls_conditional_density, cv_ls_conditional_distribution])
def test_conditional_criterion_precision_is_pinned(conditional_sample, conditional_bandwidth, criterion):
    x, y = conditional_sample

    precisions = _dot_precisions(lambda bw: criterion(x, y, bw), conditional_bandwidth)

    _assert_all_highest(precisions)


@pytest.mark.parametrize("criterion", [cv_ls_conditional_density, cv_ls_conditional_distribution])
def test_conditional_criterion_gradient_precision_is_pinned(conditional_sample, conditional_bandwidth, criterion):
    x, y = conditional_sample

    precisions = _dot_precisions(jax.grad(lambda bw: criterion(x, y, bw)), conditional_bandwidth)

    _assert_all_highest(precisions)


def test_cmode_precision_is_pinned(categorical_response):
    x, y, bandwidth = categorical_response

    precisions = _dot_precisions(lambda bw: cmode(x, y, bw).density, bandwidth)

    _assert_all_highest(precisions)
