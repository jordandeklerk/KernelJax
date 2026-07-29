"""Tests for goodness of fit reporting."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.estimators.density import density
from kerneljax.estimators.regression import local_poly
from kerneljax.summary import Summary, summary


def test_r_squared_matches_hand_computed(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth)

    observed = np.asarray(criteria_response)
    fitted = np.asarray(fit.mean)
    centered = observed - observed.mean()
    predicted = fitted - observed.mean()
    want = np.sum(centered * predicted) ** 2 / (np.sum(centered**2) * np.sum(predicted**2))

    assert float(summary(fit, criteria_response).r_squared) == pytest.approx(float(want), rel=1e-6)


def test_residual_se_matches_hand_computed(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth)
    want = np.sqrt(np.mean((np.asarray(criteria_response) - np.asarray(fit.mean)) ** 2))
    assert float(summary(fit, criteria_response).residual_se) == pytest.approx(float(want), rel=1e-6)


def test_density_reports_a_log_likelihood(criteria_train, criteria_bandwidth):
    report = summary(density(criteria_train, criteria_bandwidth))
    assert report.log_likelihood is not None
    assert report.r_squared is None
    assert report.residual_se is None
    assert report.degree is None


def test_regression_leaves_likelihood_unset(criteria_train, criteria_response, criteria_bandwidth):
    report = summary(local_poly(criteria_train, criteria_response, criteria_bandwidth), criteria_response)
    assert report.log_likelihood is None
    assert report.r_squared is not None


def test_selection_fields_travel_onto_the_report(criteria_train, criteria_response):
    report = summary(local_poly(criteria_train, criteria_response, "cv_ls"), criteria_response)
    assert report.method == "cv_ls"
    assert report.criterion_value is not None
    assert report.converged is not None


def test_supplied_bandwidth_leaves_method_unset(criteria_train, criteria_response, criteria_bandwidth):
    report = summary(local_poly(criteria_train, criteria_response, criteria_bandwidth), criteria_response)
    assert report.method is None
    assert report.criterion_value is None


def test_pipeline_runs_inside_one_jit(criteria_train, criteria_response, criteria_bandwidth):
    @jax.jit
    def run(y):
        return summary(local_poly(criteria_train, y, criteria_bandwidth), y)

    report = run(criteria_response)
    assert isinstance(report.r_squared, jax.Array)
    assert jnp.isfinite(report.r_squared)


def test_repr_survives_tracers(criteria_train, criteria_response, criteria_bandwidth):
    rendered = {}

    @jax.jit
    def run(y):
        report = summary(local_poly(criteria_train, y, criteria_bandwidth), y)
        rendered["text"] = repr(report)
        return report.r_squared

    run(criteria_response)
    assert "Local polynomial regression" in rendered["text"]


def test_repr_names_every_column(cv_mixed_data, cv_mixed_bandwidth):
    text = repr(summary(density(cv_mixed_data, cv_mixed_bandwidth)))
    assert "continuous" in text
    assert "unordered" in text
    assert "ordered" in text


def test_fit_away_from_training_points_raises(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth, at=jnp.zeros((3, 1)))
    with pytest.raises(ValueError, match="training points"):
        summary(fit, criteria_response)


def test_regression_without_a_response_raises(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth)
    with pytest.raises(ValueError, match="response"):
        summary(fit)


def test_fit_without_metadata_raises(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth)
    with pytest.raises(ValueError, match="column"):
        summary(dataclasses.replace(fit, spec=None), criteria_response)


def test_report_round_trips_through_jit(criteria_train, criteria_response, criteria_bandwidth):
    report = summary(local_poly(criteria_train, criteria_response, criteria_bandwidth), criteria_response)
    same = jax.jit(lambda r: r)(report)
    assert same.label == report.label
    assert same.n_train == report.n_train
    assert isinstance(same, Summary)
