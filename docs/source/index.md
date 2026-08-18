---
hide-navigation: true
---

# Welcome to <span class="kj-wordmark">KernelJax</span>

**Nonparametric kernel smoothing, differentiable end to end.**

KernelJax is a low-level JAX library for nonparametric kernel smoothing with continuous and categorical data. It provides estimators and bandwidth-selection criteria for researchers developing and extending statistical methodology in JAX.

For example, the following fits a local linear regression with its bandwidth chosen by
least-squares cross-validation, then renders the fit as a summary report carrying the
selected bandwidth, the goodness of fit, and the selection diagnostics in one table.

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

That criterion is minimized with gradient-based optimization, which is why the report can count solver iterations and say whether the search converged.

In addition to local polynomial regression, KernelJax supports kernel density and distribution estimation, conditional density and distribution estimation, mixed continuous and categorical kernels, and automatic bandwidth selection.

## Why KernelJax exists

The core idea of KernelJax is that an estimator should be an ordinary JAX program rather
than a procedure you call. Fits, bandwidths, and design matrices are pytrees, so
{func}`jax.jit`, {func}`jax.grad`, and {func}`jax.vmap` work without special handling.
Cross-validation criteria are ordinary differentiable functions, allowing smoothing
parameters to be optimized within larger models rather than selected in a separate step.

The same principle applies to the parts. Kernels, criteria, solvers, and low-level
primitives such as {func}`~kerneljax.kweights`, {func}`~kerneljax.ksum`, and
{func}`~kerneljax.wls` remain accessible through public interfaces. Everything lowers
through XLA, runs on hardware supported by JAX, and is derived from first principles in the
[Background](background/smoothing.md) documentation using the same concepts exposed by the
API.

## Next steps

- [Install](install.md) covers installation, GPU and TPU support, and double precision.
- The [user guide](user-guide/index.md) develops every estimator one topic per page on a
  shared example, starting from [what KernelJax is](user-guide/intro.md) and ending with the
  extension points.
- [Custom kernels](user-guide/custom-kernels.md) covers writing your own kernel and the
  requirements it has to meet, and
  [custom bandwidth selection](user-guide/custom-criteria.md) does the same for the rule
  that picks the bandwidth.
- [Background](background/smoothing.md) is a four-part introduction to kernel smoothing,
  running from densities through regression, mixed-type data, and bandwidth selection.
- The [API reference](api.md) documents every exported object, grouped by topic.
- The [GitHub repository](https://github.com/jordandeklerk/KernelJax) has the source code,
  development history, and issue tracker.

```{toctree}
:hidden:

install
user-guide/index
performance
background/index
api
development/index
release-notes
development/acknowledgments
```
