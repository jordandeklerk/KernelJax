<div style="text-align: center;" align="center">

<img alt="KernelJax" src="https://raw.githubusercontent.com/jordandeklerk/KernelJax/main/docs/source/_static/kerneljax-logo.png" width="650">

<br>
<br>

<p>
  <em>A low-level JAX interface for nonparametric kernel smoothing of mixed-type data.</em>
</p>

<p>
  <a href="https://kerneljax.readthedocs.io/en/latest/" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://kerneljax.readthedocs.io/en/latest/quickstart.html" target="_blank"><strong>Quickstart</strong></a> ·
  <a href="https://kerneljax.readthedocs.io/en/latest/background/smoothing.html" target="_blank"><strong>Background</strong></a> ·
  <a href="https://kerneljax.readthedocs.io/en/latest/api.html" target="_blank"><strong>API Reference</strong></a>
</p>

<br>

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jordandeklerk/KernelJax/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/KernelJax/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/KernelJax)
[![Build Status](https://github.com/jordandeklerk/KernelJax/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/KernelJax/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/kerneljax/badge/?version=latest)](https://kerneljax.readthedocs.io/en/latest/)
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
> unstable and may change without notice.

## Features

- Estimators for samples mixing continuous, unordered categorical, and ordered
  categorical columns.
  - [Density](kerneljax/estimators/density.py) and
    [cumulative distribution](kerneljax/estimators/distribution.py) estimation.
  - [Conditional density, distribution, and quantile
    regression](kerneljax/estimators/conditional.py) given mixed covariates.
  - [Local polynomial regression](kerneljax/estimators/regression.py) with
    derivatives and pointwise standard errors.
- Bandwidths and categorical smoothing parameters selected jointly from the data by
  least squares or likelihood cross validation, a corrected AIC, or a closed-form
  plug-in rule.
- Cross-validation criteria are ordinary JAX functions that compose with
  `jax.grad`, `jax.jit`, and `jax.vmap`, so a bandwidth can be learned inside a larger
  model rather than fixed by a derivative-free search.
- Composable primitives (`kweights`, `ksum`, the kernel base classes) are public,
  so estimators the high-level interface does not ship can be built directly.

## Installation

```bash
uv pip install git+https://github.com/jordandeklerk/KernelJax.git
```

JAX itself is not pinned, so install the build matching your hardware first. The
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) covers
the CPU, CUDA, and TPU wheels.

## Quickstart

```python
import numpy as np
import kerneljax as kj

rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)
fit = kj.local_poly(x, y, "cv_ls", degree=1)
print(kj.summary(fit))
```

```text
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
  Solver iterations                      8
  Converged                           True
```

### Custom kernels

A kernel sets how observations are weighted. Subclass `ContinuousKernel` or the
categorical base for that column kind, implement `value`, and pass the result through
`KernelSet`. Bandwidth selection then runs through the kernel you wrote.

```python
import dataclasses
import jax.numpy as jnp

@dataclasses.dataclass(frozen=True)
class Epanechnikov(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)

epan = kj.local_poly(x, y, "cv_ls", degree=1, kernels=kj.KernelSet(continuous=Epanechnikov()))
print(f"Epanechnikov  h={epan.bandwidth.h[0]:.6f}  r2={epan.r_squared:.6f}")
print(f"Gaussian      h={fit.bandwidth.h[0]:.6f}  r2={fit.r_squared:.6f}")
```

```text
Epanechnikov  h=0.084645  r2=0.947231
Gaussian      h=0.036019  r2=0.947953
```

### Custom bandwidth selection

Bandwidth selection minimizes a scalar criterion. The built-in rules
(`cv_ls`, corrected AIC) are ordinary JAX callables, and so is anything you
write in their place. Implement `__call__`, pass it to
`select_bandwidth`, and the optimizer finds `h` by differentiating the
loss you wrote.

```python
@dataclasses.dataclass(frozen=True)
class AbsoluteDeviation:
    degree: int = 1

    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        fit = kj.local_poly(train, y, bandwidth, kernels=kernels, chunk=chunk,
                            degree=self.degree, fold=jnp.arange(train.n))
        return jnp.mean(jnp.abs(y - fit.mean))

squared = kj.select_bandwidth(x, kj.RegressionCriterion(method="cv_ls", degree=1), y=y)
absolute = kj.select_bandwidth(x, AbsoluteDeviation(degree=1), y=y)
print(f"squared error       h = {squared.bandwidth.h[0]:.4f}")
print(f"absolute deviation  h = {absolute.bandwidth.h[0]:.4f}")
```

```text
squared error       h = 0.0360
absolute deviation  h = 0.0348
```

[density]: kerneljax/estimators/density.py#L58
[distribution]: kerneljax/estimators/distribution.py#L60
[regression]: kerneljax/estimators/regression.py#L99
[bandwidth]: kerneljax/bandwidth.py#L69
[normal_reference]: kerneljax/bandwidth.py#L296
[objectives]: kerneljax/selection/objectives.py
[cv_ls]: kerneljax/selection/objectives.py#L173
[aic]: kerneljax/selection/objectives.py#L385
[select_bandwidth]: kerneljax/selection/optimize.py#L22
[selection]: kerneljax/selection
[kweights]: kerneljax/ksum.py#L23
[ksum]: kerneljax/ksum.py#L232
[kernels]: kerneljax/kernels/base.py#L21
[kernelset]: kerneljax/kernels/sets.py#L18
