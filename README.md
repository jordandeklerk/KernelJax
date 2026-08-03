<div style="text-align: center;" align="center">

<img alt="KernelJax" src="https://raw.githubusercontent.com/jordandeklerk/KernelJax/main/docs/source/_static/kerneljax-logo.png" width="650">

<br>
<br>

<p>
  <em>A low-level JAX interface for nonparametric kernel smoothing of mixed-type data.</em>
</p>

<br>

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jordandeklerk/KernelJax/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/KernelJax/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/KernelJax)
[![Build Status](https://github.com/jordandeklerk/KernelJax/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/KernelJax/actions/workflows/test.yml)
[![Python version](https://img.shields.io/badge/3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/KernelJax)](https://github.com/jordandeklerk/KernelJax/graphs/commit-activity)

</div>

__KernelJax__ is a low-level JAX library for nonparametric kernel smoothing of
mixed-type data, built for researchers who develop new methodology from
composable, differentiable building blocks. Its kernels, bandwidth selectors, and
smoothers run natively on CPUs, GPUs, and TPUs through JAX and XLA, and fit
naturally into the wider JAX ecosystem.

> [!WARNING]
> KernelJax is in early development and has not been released to PyPI. The API is
> unstable and the documentation site is still being built out.

## Installation

```bash
uv pip install git+https://github.com/jordandeklerk/KernelJax.git
```

JAX itself is not pinned, so install the build matching your hardware first. The
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) covers
the CPU, CUDA, and TPU wheels.

## Quickstart

A continuous fit needs nothing beyond the data. `"cv_ls"` selects the bandwidth by
least squares cross-validation, and `degree=1` fits a local line.

```python
import numpy as np
import kerneljax as kj

rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)

fit = kj.local_poly(x, y, "cv_ls", degree=1)
print(kj.summary(fit, y=y))
```

```
Local polynomial regression

  Observations                         200
  Continuous variables                   1
  Estimator                   local linear
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  x1            continuous        0.036019

  Continuous kernel         Gaussian, order 2

  Residual standard error         0.174543
  R-squared                       0.947953

  Selection                          cv_ls
  Criterion value                 0.034301
  Converged                           True
```

Selection is its own function when you want the bandwidth itself. The result goes to
any estimator over the same columns.

```python
bw = kj.select_bandwidth(x, kj.RegressionCriterion(method="cv_ls", degree=1), y=y)
refit = kj.local_poly(x, y, bw)
print(float(refit.bandwidth.h[0]))
```

```
0.03601887822151184
```

### Mixed types

Categorical columns sit alongside continuous ones, each with a smoothing parameter
chosen jointly with the bandwidths instead of the sample being split per category.
Level counts are read from the data unless you give them.

```python
experience = rng.uniform(0, 40, size=200)
region = rng.integers(0, 3, size=200)
log_wage = 2.0 + 0.04 * experience - 0.0005 * experience**2 + 0.1 * region + rng.normal(0, 0.2, 200)

covariates = kj.MixedData.from_blocks(continuous=experience, unordered=region)
mixed = kj.local_poly(covariates, log_wage, "cv_ls", degree=1, se=True)
print(kj.summary(mixed, y=log_wage))
```

```
Local polynomial regression

  Observations                         200
  Continuous variables                   1
  Unordered variables                    1
  Estimator                   local linear
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  x1            continuous        6.623884
  x2            unordered         0.124881

  Continuous kernel         Gaussian, order 2

  Residual standard error         0.204661
  R-squared                       0.586877

  Selection                          cv_ls
  Criterion value                 0.046175
  Converged                           True
```

That parameter is a dial. Near zero keeps the categories apart, at its upper bound it
pools them and drops the column. Region matters for wages, so it stays low. It says
nothing about how experience is distributed, so the density below pools it away.

```python
density = kj.density(covariates, "cv_ml")
print(kj.summary(density))
```

```
Mixed-type density estimate

  Observations                         200
  Continuous variables                   1
  Unordered variables                    1
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  x1            continuous        1.394879
  x2            unordered         0.666663

  Continuous kernel         Gaussian, order 2

  Log likelihood               -955.369202

  Selection                          cv_ml
  Criterion value               966.392151
  Converged                           True
```

### Composing with JAX

Criteria are ordinary JAX functions. Differentiating one with respect to its bandwidth
returns a `Bandwidth` of gradients, one per parameter.

```python
import jax

grads = jax.grad(kj.cv_ml_density, argnums=1)(covariates, density.bandwidth)
print("d/dh     ", grads.h)
print("d/dlambda", grads.lam_uno)
```

```
d/dh      [0.00796509]
d/dlambda [-10.769775]
```

`jit` and `vmap` compose the same way. Selection uses these gradients internally,
running L-BFGS where these criteria are usually minimized by a derivative-free search.

## What is implemented

| | |
| --- | --- |
| Densities | `density`, with likelihood and least squares cross-validation |
| Distributions | `cdf`, with its own cross-validation criterion |
| Regression | `local_poly` at any degree, with standard errors and derivatives |
| Bandwidths | `select_bandwidth`, `normal_reference`, and a jittable `lbfgs` |
| Primitives | `ksum` and `kweights`, the generalized product kernel |
| Kernels | Gaussian, Aitchison and Aitken, Wang and van Ryzin, Li and Racine |
| Reporting | `summary`, for densities and regressions |

Data may mix continuous, unordered categorical, and ordered categorical columns
in any combination.

## Design notes

Estimators contract the kernel weights as they go rather than holding the pairwise
weight matrix. On an A100 at n=16000 a density fit needs a few bytes where that matrix
would take 0.954 GiB. Results are checked against established implementations to
machine precision in double precision, and the test suite covers that agreement.
