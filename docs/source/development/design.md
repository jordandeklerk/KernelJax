# Design principles

KernelJax has a definite shape, and this page explains where that shape comes from. None of
what follows is required reading for using the library. It is meant for the moment you want
to change something and find yourself wondering why the code resists being changed in one
direction and welcomes it in another. Each section states a principle, then shows it in
running code.

The examples share one small mixed sample.

```python
import jax
import jax.numpy as jnp
import numpy as np

import kerneljax as kj

rng = np.random.default_rng(3)
n = 150
x = rng.uniform(0.0, 1.0, n)
group = rng.integers(0, 3, n)
y = np.sin(2 * np.pi * x) + 0.5 * group + rng.normal(0.0, 0.2, n)
data = kj.MixedData.from_blocks(continuous=x, unordered=group, unordered_levels=3)
```

## One contraction underneath everything

Every estimator is a thin layer over the same two operations. {func}`~kerneljax.kweights`
builds the matrix of kernel weights between evaluation and training points, and
{func}`~kerneljax.ksum` contracts those weights against whatever the estimator cares about.
A density contracts against nothing but the weights themselves.

```python
bw = kj.select_bandwidth(data, kj.DensityCriterion(method="cv_ml"))
weights = kj.kweights(data, bw.bandwidth)

by_hand = jnp.mean(weights, axis=1) / jnp.prod(bw.bandwidth.h)
shipped = kj.density(data, bw).value

print(f"largest gap to density={float(jnp.max(jnp.abs(by_hand - shipped))):.2e}")
```

```text
largest gap to density=2.98e-08
```

A regression contracts the same weights against the responses and divides.

```python
numerator = kj.ksum(data, bw.bandwidth, y[:, None])
denominator = kj.ksum(data, bw.bandwidth)

nadaraya_watson = (numerator / denominator).ravel()
local_constant = kj.local_poly(data, y, bw.bandwidth).mean

print(f"largest gap to local_poly={float(jnp.max(jnp.abs(nadaraya_watson - local_constant))):.2e}")
```

```text
largest gap to local_poly=7.15e-07
```

Even the conditional family reduces to the same parts. A conditional density is a ratio of
two weight contractions, one over the conditioning columns and one over the response.

```python
response = kj.MixedData.continuous(y)
cbw = kj.ConditionalBandwidth(
    x=bw.bandwidth,
    y=kj.Bandwidth(h=jnp.array([0.3]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)),
)

weights_x = kj.kweights(data, cbw.x)
weights_y = kj.kweights(response, cbw.y)
ratio = jnp.sum(weights_x * weights_y, axis=1) / (0.3 * jnp.sum(weights_x, axis=1))

print(f"largest gap to cdensity={float(jnp.max(jnp.abs(ratio - kj.cdensity(data, response, cbw).value))):.2e}")
```

```text
largest gap to cdensity=5.96e-08
```

The consequence we care about is that correctness concentrates. The weight matrix handles
mixed column kinds, per column operators, and held out folds in one place, so an estimator
built on top inherits all of that without restating any of it. The primitives are public for
the same reason. An estimator we never wrote can still be assembled from the same verified
parts, and the [custom guides](../user-guide/custom-kernels.md) lean on exactly this.

## Differentiable end to end

The decision that most shapes the library is that every bandwidth selection criterion is an
ordinary JAX function of the bandwidth. The reference implementations in the literature
treat the criterion as a black box and search with derivative free methods. We instead
require every criterion, every kernel, and every estimator in the selection path to be
differentiable, so the criterion has a gradient anywhere we care to ask for one.

```python
def criterion(bandwidth):
    return kj.cv_ls_regression(data, bandwidth, y=y, degree=1)


gradient = jax.grad(criterion)(bw.bandwidth)
print(f"d/dh={float(gradient.h[0]):+.4f}  d/dlam={float(gradient.lam_uno[0]):+.4f}")
```

```text
d/dh=+0.1507  d/dlam=+0.4590
```

In exchange, selection is a gradient based search that converges in a handful of iterations
rather than hundreds of criterion evaluations, and a bandwidth can sit inside a larger model
and be learned along with it.

```python
selected = kj.select_bandwidth(data, kj.RegressionCriterion(method="cv_ls", degree=1), y=y)
print(f"converged={bool(selected.converged)} in n_iter={int(selected.n_iter)} iterations")
```

```text
converged=True in n_iter=15 iterations
```

Differentiability has demands of its own, and two of them explain corners of the code that
otherwise look arbitrary. The search runs in unconstrained coordinates, with a softplus for
continuous bandwidths and a scaled logistic for categorical smoothing parameters, because a
gradient method needs the whole real line to move in. And the search never starts a
categorical parameter at zero, even though zero is the natural reference value, because zero
sits in the flat tail of the logistic where the gradient cannot lift it out. Where a
derivative free search can start anywhere, a gradient based one must start where the
geometry lets it move.

## Static structure, traced values

JAX compiles a program once per distinct structure and reuses the compilation for new
values. KernelJax leans into that split. Anything that fixes the shape of the computation,
the kernel families, the polynomial degree, the criterion configuration, travels as a
hashable frozen dataclass and is declared static. Anything that varies per call, the data,
the responses, the candidate bandwidth, flows through as an array and is traced.

The split is why criterion settings live on the criterion object rather than in a keyword
bag. Data routed through `criterion_kwargs` is traced, so a setting routed the same way
arrives at the estimator as a tracer and is refused the moment it reaches a static slot.

```python
def with_degree_argument(train, bandwidth, *, y, degree, kernels=None, chunk=None):
    fit = kj.local_poly(train, y, bandwidth, degree=degree, fold=jnp.arange(train.n))
    return jnp.mean(jnp.abs(y - fit.mean))


kj.select_bandwidth(data, with_degree_argument, y=y, criterion_kwargs={"degree": 1})
```

```text
ValueError: Non-hashable static arguments are not supported. An error occurred while trying to
hash an object of type <class 'kerneljax.basis.LocalPolyBasis'>,
LocalPolyBasis(degree=JitTracer(~int32[])). The error was:
TypeError: unhashable type: 'DynamicJaxprTracer'
```

The alternative to raising is silent recompilation on every candidate bandwidth, which turns
a fast search into a slow one without telling anyone. The
[custom bandwidth selection](../user-guide/custom-criteria.md) guide walks the failure and
the pattern that avoids it.

## Results carry their provenance

Selection is the expensive step, so its result is built to travel. A
{class}`~kerneljax.SelectionResult` remembers the criterion and the kernels it was selected
under, every fit remembers its bandwidth and its selection, and every estimator accepts any
of them wherever it accepts a bandwidth. Settings recover automatically from whatever is
handed in, so a bandwidth selected once flows through a whole analysis without any setting
being restated.

```python
sel = kj.select_bandwidth(data, kj.RegressionCriterion(method="cv_ls", degree=2), y=y)
fit = kj.local_poly(data, y, sel)

print(f"degree={fit.degree} recovered from the criterion the bandwidth was selected under")
```

```text
degree=2 recovered from the criterion the bandwidth was selected under
```

Recovery has a sharp edge, and the library keeps it honest. A setting restated explicitly
must agree with the one the result carries, because silently preferring either would let an
estimator report numbers produced under settings the caller never chose.

```python
kj.local_poly(data, y, sel, degree=1)
```

```text
ValueError: degree=1 contradicts the degree 2 that bw was selected under
```

The same applies to kernels, which travel inside the selection and are refused if restated
differently, and to the conditional family, where a fit selected under
{func}`~kerneljax.cdensity` hands its bandwidth blocks straight to {func}`~kerneljax.cdist`
and {func}`~kerneljax.cquantile`.

## Refuse rather than guess

Where KernelJax cannot do the right thing, it declines loudly and explains itself. A kernel
that loses half its mass can still produce a regression, because a regression is a ratio and
the constant cancels. Taken to a density, where the constant no longer cancels, the same
kernel is rejected with the measured value in the message.

```python
import dataclasses


@dataclasses.dataclass(frozen=True)
class HalfMass(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return 0.5 * jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)


kj.density(x, "cv_ml", kernels=kj.KernelSet(continuous=HalfMass()))
```

```text
ValueError: HalfMass.value integrates to 0.5000 in u units rather than one, so
every density it produces is scaled by that factor. A regression fit is a ratio
and cancels the constant, which is why this only fires from a density.
```

The same posture covers statistical traps. Likelihood selection for a conditional
distribution is refused outright, because the likelihood of a distribution value rewards
oversmoothing without bound, and the error points to the criterion that works.

```python
kj.cdist(data, response, "cv_ml")
```

```text
ValueError: cv_ml cannot select a bandwidth for a conditional distribution, since the
likelihood of a CDF value rewards oversmoothing without bound. Select with
cv_ls, reuse a cdensity fit, or supply a ConditionalBandwidth
```

There are no flags to bypass these checks. A refusal a user can switch off is a warning, and
warnings train people to ignore them. The checks follow one placement rule. They run when a
feature first depends on the property they verify, not before, so everything that was
already correct stays out of their way.

## Held to the literature

Every estimator and criterion in KernelJax is checked numerically against the R package
[np](https://cran.r-project.org/package=np), which has been the reference implementation of
mixed-type kernel smoothing for two decades. The standard is machine precision at matched
inputs, not statistical closeness, and the difference matters. A criterion that agrees with
the reference to within half a percent is easily close enough to pass a statistical eye, yet
a gap that size can hide a structural difference in which observations enter which sum.
Agreement to ten digits is not pedantry. It is how a formula transcribed from a paper is
distinguished from the formula the reference actually computes. The
[contributing](contributing.md) page describes how the comparison is run.

The same standard extends to the documentation. Every number printed on these pages is
produced by executing the page, and every error message shown is the message the library
actually raises, verified on every documentation build. The blocks above are no exception.
A claim that cannot survive being executed does not ship.
