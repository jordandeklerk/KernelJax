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

```python
import numpy as np
import kerneljax as kj

rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)

fit = kj.local_poly(x, y, "cv_ls", degree=1)
print(kj.summary(fit))
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

Or select the bandwidth on its own and reuse it.

```python
bw = kj.select_bandwidth(x, kj.RegressionCriterion(method="cv_ls", degree=1), y=y)
refit = kj.local_poly(x, y, bw)
print(float(refit.bandwidth.h[0]))
```

```
0.03601887822151184
```

### Mixed Types

Categorical columns sit alongside continuous ones, with their smoothing parameters
chosen jointly.

```python
experience = rng.uniform(0, 40, size=200)
region = rng.integers(0, 3, size=200)
log_wage = 2.0 + 0.04 * experience - 0.0005 * experience**2 + 0.1 * region + rng.normal(0, 0.2, 200)

covariates = kj.MixedData.from_blocks(continuous=experience, unordered=region)

mixed = kj.local_poly(covariates, log_wage, "cv_ls", degree=1, se=True)
density = kj.density(covariates, "cv_ml")

print("regression  h", mixed.bandwidth.h, " lambda", mixed.bandwidth.lam_uno)
print("density     h", density.bandwidth.h, " lambda", density.bandwidth.lam_uno)
print("standard errors", mixed.se[:3])
```

```
regression  h [6.623884]  lambda [0.12488104]
density     h [1.3948787]  lambda [0.6666628]
standard errors [0.03195919 0.03875961 0.03553929]
```

### Custom Kernels

Subclass the base for the column kind and implement the `value` function.

> [!IMPORTANT]
> `value` is elementwise and must never reduce. It receives `(x, y)` already broadcast
> and scaled by `h`, and returns that same shape.
>
> Scale to unit variance, as the built-in kernels are, or bandwidths will not be
> comparable across kernels.

`deriv`, `cdf` and `conv` are optional.

```python
import dataclasses
import jax.numpy as jnp

@dataclasses.dataclass(frozen=True)
class Tricube(kj.ContinuousKernel):
    power: int = 3

    def value(self, x, y, h):
        u = jnp.abs((x - y) / h)
        return jnp.where(u < 1.0, (1.0 - u**3) ** self.power, 0.0)

own = kj.local_poly(x, y, "cv_ls", degree=1, kernels=kj.KernelSet(continuous=Tricube(power=8)))
print(kj.summary(own))
```

```
Local polynomial regression

  Observations                         200
  Continuous variables                   1
  Estimator                   local linear
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  x1            continuous        0.123780

  Continuous kernel         Tricube(power=8)

  Residual standard error         0.175727
  R-squared                       0.947239

  Selection                          cv_ls
  Criterion value                 0.034177
  Converged                           True
```

### Composing with JAX

Every criterion is differentiable, and the gradient comes back shaped like the bandwidth.

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

`jit` and `vmap` compose too.
