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
  <a href="https://kerneljax.readthedocs.io/en/latest/user-guide/custom-kernels.html" target="_blank"><strong>Custom Kernels</strong></a> ·
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

- Density, cumulative distribution, and local polynomial regression over samples mixing
  continuous, unordered categorical, and ordered categorical columns.
- Bandwidths and categorical smoothing parameters selected jointly from the data by
  least squares or likelihood cross validation, a corrected AIC, or a closed-form
  plug-in rule.
- Cross-validation criteria are ordinary JAX functions that compose with `jax.grad`,
  `jax.jit`, and `jax.vmap`, so a bandwidth can be learned inside a larger model rather
  than fixed by a derivative-free search.
- Composable primitives (`kweights`, `ksum`, the kernel base classes) are public, so
  estimators the high-level interface does not ship can be built directly.

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
