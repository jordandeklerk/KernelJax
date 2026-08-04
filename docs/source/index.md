# Welcome to KernelJax

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

Nonparametric kernel smoothing of mixed continuous and categorical data is a staple of applied
econometrics and statistics, and the software that made it practical deserves much of the
credit. The R package [np](https://cran.r-project.org/package=np) is the reference
implementation for the whole class, and
[statsmodels](https://www.statsmodels.org/stable/nonparametric.html) brings a good part of
that family to Python. Both are careful and well tested, and computational parity with np is
the standard KernelJax holds its own numbers to.

Both were also designed around a model of computation that has since moved. An estimator is a
procedure you call, its bandwidth is settled by a derivative-free search sealed inside that
call, and the pieces it is assembled from stay internal. No implementation of this class of
estimators currently exists in the JAX ecosystem. KernelJax is written to close that gap.

KernelJax is built to compose rather than to be called. The mixed-type design matrix, the
bandwidth and every fit object are registered pytrees, so {func}`jax.jit`, {func}`jax.grad`
and {func}`jax.vmap` apply to any of them without special handling. The cross-validation
criteria are ordinary functions of the data and a bandwidth, which means they can be minimized
by the built-in L-BFGS, handed to any optimizer in the JAX ecosystem, or added as one term to
a larger loss so that a smoothing parameter is trained alongside a neural network rather than
fixed before it starts. Everything lowers through XLA and runs on whichever hardware JAX
targets, from a laptop core to an accelerator, with no change to the calling code.

The parts are open on purpose. A kernel, a selection criterion and the optimizer that
minimizes it are each supplied as an argument against a published contract, covered in
[Custom kernels](user-guide/custom-kernels.md) and
[custom criteria](user-guide/custom-criteria.md), and the primitives the shipped estimators
are built from, {func}`~kerneljax.kweights`, {func}`~kerneljax.ksum` and
{func}`~kerneljax.wls`, are exported rather than hidden. A method nobody has written yet can
be assembled from the same pieces the shipped ones use.

The closeness between the code and the mathematics is meant to be useful in its own right.
[Background](background/smoothing.md) derives these estimators from first principles, and the
primitives carry the names of the objects in those derivations, so a reader can hold a
textbook open beside the code they are writing.

## Installation

KernelJax has not been released to PyPI yet, so install it from the repository.

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
  requirements it has to meet, and [Custom criteria](user-guide/custom-criteria.md) does the
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
