"""Tests for the criterion objects."""

import jax
import jax.numpy as jnp
import pytest

from kerneljax.tuning.criteria import DensityCriterion, RegressionCriterion
from kerneljax.tuning.objectives import aic_c_regression, cv_ls_density, cv_ls_regression, cv_ml_density
from kerneljax.tuning.optimize import select_bandwidth


@pytest.mark.parametrize(
    "build",
    [
        lambda: RegressionCriterion(method="cv_ls", degree=1),
        lambda: RegressionCriterion(method="aic", degree=0),
        lambda: DensityCriterion(method="cv_ml"),
        lambda: DensityCriterion(method="cv_ls"),
    ],
)
def test_fresh_objects_are_equal(build):
    first, second = build(), build()
    assert first is not second
    assert first == second
    assert hash(first) == hash(second)


@pytest.mark.parametrize("method", ["cvls", "CV_LS", "cv.ls", ""])
def test_regression_rejects_unknown_method(method):
    with pytest.raises(ValueError, match="method must be"):
        RegressionCriterion(method=method)


@pytest.mark.parametrize("method", ["cv_aic", "CV_ML", "cv.ml", ""])
def test_density_rejects_unknown_method(method):
    with pytest.raises(ValueError, match="method must be"):
        DensityCriterion(method=method)


@pytest.mark.parametrize("method, rule", [("cv_ls", cv_ls_regression), ("aic", aic_c_regression)])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_regression_matches_the_rule(
    cv_mixed_data, cv_mixed_bandwidth, regression_mixed_response, method, rule, degree
):
    criterion = RegressionCriterion(method=method, degree=degree)
    got = criterion(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response)
    want = rule(cv_mixed_data, cv_mixed_bandwidth, y=regression_mixed_response, degree=degree)
    assert float(got) == float(want)


@pytest.mark.parametrize("method, rule", [("cv_ml", cv_ml_density), ("cv_ls", cv_ls_density)])
def test_density_matches_the_rule(cv_mixed_data, cv_mixed_bandwidth, method, rule):
    got = DensityCriterion(method=method)(cv_mixed_data, cv_mixed_bandwidth)
    want = rule(cv_mixed_data, cv_mixed_bandwidth)
    assert float(got) == float(want)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_degree_reaches_the_basis(criteria_train, criteria_response, degree):
    criterion = RegressionCriterion(method="cv_ls", degree=degree)
    result = select_bandwidth(criteria_train, criterion, y=criteria_response, n_starts=1)
    assert jnp.all(jnp.isfinite(result.bandwidth.h))


def test_density_criterion_needs_no_response(criteria_train):
    result = select_bandwidth(criteria_train, DensityCriterion(method="cv_ml"), n_starts=1)
    assert jnp.all(jnp.isfinite(result.bandwidth.h))


def test_criterion_is_traceable(criteria_train, criteria_response, criteria_bandwidth):
    criterion = RegressionCriterion(method="cv_ls", degree=1)

    value = jax.jit(criterion)(criteria_train, criteria_bandwidth, y=criteria_response)
    grads = jax.grad(criterion, argnums=1)(criteria_train, criteria_bandwidth, y=criteria_response)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grads.h))


def test_fresh_object_reuses_compilation(criteria_train, criteria_response):
    select_bandwidth(
        criteria_train,
        RegressionCriterion(method="cv_ls", degree=1),
        y=criteria_response,
        n_starts=1,
    )
    compiled = select_bandwidth._cache_size()

    select_bandwidth(
        criteria_train,
        RegressionCriterion(method="cv_ls", degree=1),
        y=criteria_response + 1.0,
        n_starts=1,
    )
    assert select_bandwidth._cache_size() == compiled

    select_bandwidth(
        criteria_train,
        RegressionCriterion(method="aic", degree=1),
        y=criteria_response,
        n_starts=1,
    )
    assert select_bandwidth._cache_size() == compiled + 1
