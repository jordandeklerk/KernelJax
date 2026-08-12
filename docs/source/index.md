# Welcome to KernelJax

KernelJax is a low-level JAX library for nonparametric kernel smoothing with mixed-type
data, built for researchers developing new statistical methodology from composable,
differentiable primitives.

```{warning}
KernelJax is under active development and has not yet been released on PyPI. The API may change without notice.
```

## Why KernelJax exists

The core idea of KernelJax is that an estimator should be an ordinary JAX program rather than a procedure you call. Fits, bandwidths, and design matrices are pytrees, so {func}`jax.jit`,
{func}`jax.grad`, and {func}`jax.vmap` work without special handling. Cross-validation criteria
are ordinary differentiable functions, allowing smoothing parameters to be optimized within
larger models rather than selected in a separate step.

The same principle applies to the parts. Kernels, criteria, solvers, and low-level primitives
such as {func}`~kerneljax.kweights`, {func}`~kerneljax.ksum`, and {func}`~kerneljax.wls` remain
accessible through public interfaces. Everything lowers through XLA, runs on hardware supported
by JAX, and is derived from first principles in the [Background](background/smoothing.md)
documentation using the same concepts exposed by the API.

## Installation

```bash
uv pip install kerneljax
```

The latest development version installs straight from the repository.

```bash
uv pip install git+https://github.com/jordandeklerk/KernelJax.git
```

```{tip}
Check the install with `python -c 'import kerneljax; print(kerneljax.__version__)'`.
```

### GPU and TPU support

JAX is not pinned, so install the build that matches your hardware before KernelJax. The
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) covers the CPU,
CUDA and TPU wheels.

```bash
uv pip install --upgrade "jax[cuda12]"
```

Everything in KernelJax runs on whichever device JAX is configured for, with no change to
the calling code.

### Double precision

JAX defaults to 32 bit floats. Bandwidth selection and the cross-validation criteria are
sensitive to that near an optimum, and agreement with established implementations is
pinned in double precision, so enable 64 bit before importing if you are comparing
numbers rather than exploring.

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## Where to go next

- [Quickstart](quickstart.md) walks through the main API, from a first fit to building
  estimators out of the exported primitives.
- [Custom kernels](user-guide/custom-kernels.md) covers writing your own kernel and the four
  requirements it has to meet, and [Custom bandwidth selection](user-guide/custom-criteria.md) does the
  same for the rule that picks the bandwidth.
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
:hidden:

quickstart
```

```{toctree}
:caption: User Guide
:hidden:

user-guide/custom-kernels
user-guide/custom-criteria
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
:caption: Development
:hidden:

development/contributing
development/design
```

```{toctree}
:hidden:

api
```

```{toctree}
:hidden:

release-notes
```

```{toctree}
:hidden:

acknowledgments
```
