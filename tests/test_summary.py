"""Tests for goodness of fit reporting."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kerneljax.estimators.conditional import cdensity, cdist, cmode, cquantile
from kerneljax.estimators.density import density
from kerneljax.estimators.regression import local_poly
from kerneljax.kernels import KernelSet
from kerneljax.kernels.base import ContinuousKernel
from kerneljax.selection.optimize import select_bandwidth
from kerneljax.summary import Summary, _kernel_name, summary


def test_r_squared_matches_hand_computed(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth)

    observed = np.asarray(criteria_response)
    fitted = np.asarray(fit.mean)
    centered = observed - observed.mean()
    predicted = fitted - observed.mean()
    want = np.sum(centered * predicted) ** 2 / (np.sum(centered**2) * np.sum(predicted**2))

    assert float(summary(fit).r_squared) == pytest.approx(float(want), rel=1e-6)


def test_residual_se_matches_hand_computed(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth)
    want = np.sqrt(np.mean((np.asarray(criteria_response) - np.asarray(fit.mean)) ** 2))
    assert float(summary(fit).residual_se) == pytest.approx(float(want), rel=1e-6)


def test_density_reports_a_log_likelihood(criteria_train, criteria_bandwidth):
    report = summary(density(criteria_train, criteria_bandwidth))
    assert report.log_likelihood is not None
    assert report.r_squared is None
    assert report.residual_se is None
    assert report.degree is None


def test_regression_leaves_likelihood_unset(criteria_train, criteria_response, criteria_bandwidth):
    report = summary(local_poly(criteria_train, criteria_response, criteria_bandwidth))
    assert report.log_likelihood is None
    assert report.r_squared is not None


def test_selection_fields_travel_onto_the_report(criteria_train, criteria_response):
    report = summary(local_poly(criteria_train, criteria_response, "cv_ls"))
    assert report.method == "cv_ls"
    assert report.criterion_value is not None
    assert report.converged is not None


def test_supplied_bandwidth_leaves_method_unset(criteria_train, criteria_response, criteria_bandwidth):
    report = summary(local_poly(criteria_train, criteria_response, criteria_bandwidth))
    assert report.method is None
    assert report.criterion_value is None


def test_pipeline_runs_inside_one_jit(criteria_train, criteria_response, criteria_bandwidth):
    @jax.jit
    def run(y):
        return summary(local_poly(criteria_train, y, criteria_bandwidth))

    report = run(criteria_response)
    assert isinstance(report.r_squared, jax.Array)
    assert jnp.isfinite(report.r_squared)


def test_repr_survives_tracers(criteria_train, criteria_response, criteria_bandwidth):
    rendered = {}

    @jax.jit
    def run(y):
        report = summary(local_poly(criteria_train, y, criteria_bandwidth))
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
        summary(fit)


def test_fit_without_metadata_raises(criteria_train, criteria_response, criteria_bandwidth):
    fit = local_poly(criteria_train, criteria_response, criteria_bandwidth)
    with pytest.raises(ValueError, match="column"):
        summary(dataclasses.replace(fit, spec=None))


def test_report_round_trips_through_jit(criteria_train, criteria_response, criteria_bandwidth):
    report = summary(local_poly(criteria_train, criteria_response, criteria_bandwidth))
    same = jax.jit(lambda r: r)(report)
    assert same.label == report.label
    assert same.n_train == report.n_train
    assert isinstance(same, Summary)


@dataclasses.dataclass(frozen=True)
class ParameterizedKernel(ContinuousKernel):
    power: int = 3

    def value(self, x, y, h):
        u = jnp.abs((x - y) / h)
        return jnp.where(u < 1.0, (1.0 - u**3) ** self.power, 0.0)


@dataclasses.dataclass(frozen=True)
class BareKernel(ContinuousKernel):
    def value(self, x, y, h):
        return jnp.exp(-0.5 * ((x - y) / h) ** 2)


def test_kernel_name_reports_parameters():
    assert _kernel_name(KernelSet(continuous=ParameterizedKernel(power=8))) == "ParameterizedKernel(power=8)"


def test_kernel_name_omits_empty_parameters():
    assert _kernel_name(KernelSet(continuous=BareKernel())) == "BareKernel"


def test_kernel_name_prefers_the_order():
    assert _kernel_name(KernelSet()) == "Gaussian, order 2"


def test_a_criterion_the_caller_wrote_still_reports_its_selection(criteria_train, criteria_response):
    @dataclasses.dataclass(frozen=True)
    class AbsoluteDeviation:
        degree: int = 1

        def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
            fit = local_poly(train, y, bandwidth, kernels=kernels, chunk=chunk, degree=1, fold=jnp.arange(train.n))
            return jnp.mean(jnp.abs(y - fit.mean))

    result = select_bandwidth(criteria_train, AbsoluteDeviation(), y=criteria_response)
    report = str(summary(local_poly(criteria_train, criteria_response, result)))

    assert "AbsoluteDeviation" in report
    assert "Criterion value" in report
    assert "Converged" in report


def test_the_report_shows_how_many_iterations_selection_took(criteria_train, criteria_response):
    fit = local_poly(criteria_train, criteria_response, "cv_ls", degree=1)
    report = str(summary(fit))

    assert "Solver iterations" in report
    assert f"{int(fit.selection.n_iter)}" in report
    assert "." not in report.split("Solver iterations")[1].splitlines()[0]


def test_a_conditional_summary_renders_both_blocks(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    report = summary(cdensity(x, y, conditional_bandwidth))
    text = repr(report)

    assert report.response_spec is not None
    assert report.response_bandwidth is conditional_bandwidth.y
    assert "Response" in text
    assert "Conditioning" in text
    assert text.index("Response") < text.index("Conditioning")


def test_a_conditional_density_reports_a_log_likelihood(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    fit = cdensity(x, y, conditional_bandwidth)
    report = summary(fit)

    assert float(report.log_likelihood) == pytest.approx(float(jnp.sum(jnp.log(fit.value))))
    assert report.r_squared is None


def test_a_conditional_distribution_leaves_likelihood_unset(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample
    assert summary(cdist(x, y, conditional_bandwidth)).log_likelihood is None


def test_conditional_selection_fields_travel_onto_the_report(conditional_sample):
    x, y = conditional_sample

    fit = cdensity(x, y, "cv_ml", n_starts=1)
    report = summary(fit)

    assert report.method is not None
    assert float(report.criterion_value) == pytest.approx(float(fit.selection.value))


def test_a_conditional_fit_evaluated_elsewhere_is_refused(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    grid = jnp.linspace(-2.0, 2.0, 7)
    head = jax.tree.map(lambda a: a[: grid.size], x)
    fit = cdist(x, y, conditional_bandwidth, at_x=head, at_y=grid[:, None])

    with pytest.raises(ValueError, match="training points"):
        summary(fit)


def test_a_quantile_summary_names_its_level(conditional_sample, conditional_bandwidth):
    x, y = conditional_sample

    report = summary(cquantile(x, y, conditional_bandwidth, tau=0.75))
    text = repr(report)

    assert "tau 0.75" in text
    assert "Response" in text
    assert "Conditioning" in text


def test_a_mode_summary_reports_classification(categorical_response):
    x, y, bandwidth = categorical_response

    report = summary(cmode(x, y, bandwidth))
    text = repr(report)

    assert "Conditional mode estimate" in text
    assert "Correct classification" in text
