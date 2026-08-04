# KernelJax

KernelJax is a low-level JAX library for nonparametric kernel smoothing with mixed-type
data, built for researchers developing new statistical methodology from composable,
differentiable primitives.

It provides density, cumulative distribution, and regression estimators over continuous,
unordered categorical, and ordered categorical variables, choosing bandwidths and
categorical smoothing parameters from the data rather than by hand. Everything runs
natively on CPUs, GPUs, and TPUs through [JAX](https://docs.jax.dev/en/latest/) and
[XLA](https://openxla.org/xla).

```{warning}
KernelJax is under active development and has not yet been released on PyPI. The API may change without notice.
```

## Why KernelJax exists

The R package [np](https://cran.r-project.org/package=np) is the reference implementation
for this class of estimators, and KernelJax targets computational parity with it. Four
things here are different:

- **Differentiable criteria.** Cross-validation objectives are ordinary JAX functions that
  compose with {func}`jax.grad`, {func}`jax.jit`, and {func}`jax.vmap`, so a bandwidth can
  be learned inside a larger model instead of being fixed by a derivative-free search.
- **Accelerated execution.** Estimators lower through XLA and run unchanged on CPUs, GPUs,
  and TPUs.
- **Composable primitives.** Every estimator is one contraction of kernel weights against a
  vector, and that primitive is public. `kweights`, `ksum`, and the kernels themselves are
  exported, so methodology that no shipped estimator covers can be built directly.
- **Matrix-free contraction.** Weights are reduced as they are computed rather than stored,
  so peak memory grows with the sample rather than with its square. A dense
  $16{,}000 \times 16{,}000$ matrix of 32-bit weights alone would need 0.954 GiB.

## Where to go next

- [Installation](installation.md) covers installing from the repository and choosing the
  [JAX build](https://docs.jax.dev/en/latest/installation.html) for your hardware.
- [Quickstart](quickstart.md) walks through the main API, from a first fit to building
  estimators out of the exported primitives.
- [Background](background/smoothing.md) is a four-part introduction to kernel smoothing,
  running from densities through regression, mixed-type data, and bandwidth selection.
- The [API reference](api.md) documents every exported object, grouped by topic.
- The [GitHub repository](https://github.com/jordandeklerk/KernelJax) has the source code,
  development history, and issue tracker.

```{toctree}
:hidden:

Home <self>
```

```{toctree}
:caption: Getting Started
:hidden:

installation
quickstart
custom-kernels
```

```{toctree}
:caption: Background
:hidden:

background/smoothing
background/regression
background/mixed-data
background/selection
```

```{toctree}
:caption: Reference
:hidden:

api
```
