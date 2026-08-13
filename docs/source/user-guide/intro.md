# What is KernelJax?

KernelJax estimates smooth relationships, densities, distributions, and their conditional versions from mixed continuous and categorical data. You do not choose a parametric functional form in advance, and the library can select its smoothing parameters directly from the data.

The example below fits a local linear regression with one continuous covariate and one categorical covariate.

```python
import numpy as np
import kerneljax as kj

rng = np.random.default_rng(7)
n = 200
x = rng.uniform(size=n)
group = rng.integers(0, 3, n)
y = np.sin(2 * np.pi * x) + 0.6 * group + rng.normal(0, 0.2, n)

data = kj.MixedData.from_blocks(
    continuous=x,
    unordered=group,
    unordered_levels=3,
    names=("x", "group"),
)

fit = kj.local_poly(data, y, "cv_ls", degree=1)
print(kj.summary(fit))
```

```text
Local polynomial regression

  Observations                         200
  Continuous variables                   1
  Unordered variables                    1
  Estimator                   local linear
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  x             continuous        0.054234
  group         unordered         0.014085

  Continuous kernel         Gaussian, order 2

  Residual standard error         0.182911
  R-squared                       0.957603

  Selection                          cv_ls
  Criterion value                 0.043569
  Solver iterations                     18
  Converged                           True
```

Nothing in that call was tuned by hand. `"cv_ls"` selects both smoothing parameters jointly by least-squares cross-validation. KernelJax minimizes that differentiable criterion with gradient-based optimization, and the report keeps the selection diagnostics alongside the goodness-of-fit statistics.

This is the central idea behind the library. Criteria, kernels, and estimators remain ordinary JAX computations, so selection can use gradients, fits can run under `jit`, and bandwidths can participate in larger differentiable models.

The categorical column also required no dummy encoding or separate per-group fits. {class}`~kerneljax.MixedData` records which variables are continuous, unordered, or ordered, and each kind is smoothed with an appropriate kernel. Its smoothing parameter is selected jointly with the continuous bandwidths, so a categorical variable carrying no information about the response can be smoothed away automatically.

## Reading a fit

The printed report is a view of the fit object underneath, and the numerical results remain available directly as arrays.

```python
residuals = y - fit.mean

print(f"fit.mean carries {fit.mean.shape[0]} fitted values, one per observation")
print(f"largest absolute residual {float(np.max(np.abs(residuals))):.3f}")
```

```text
fit.mean carries 200 fitted values, one per observation
largest absolute residual 0.543
```

Those arrays can feed directly into whatever comes next, whether that is a plot, another calculation, or another JAX transformation. The [regression guide](regression.md) works with the same fit object to obtain bandwidths, standard errors, predictions, and derivatives.

## Evaluating somewhere new

The opening fit was evaluated at its own training points. To predict somewhere else, construct evaluation data and pass it through `at=`.

Here we vary `x` across its observed range while holding the categorical variable fixed.

```python
points = kj.grid(data, vary="x", n=5)
pred = kj.local_poly(data, y, fit, at=points)

print(pred.mean)
```

```text
[ 0.09101082  0.99185854 -0.03735442 -0.8732855  -0.11484411]
```

The existing fit is passed back as the bandwidth rule, so selection is not run a second time. {func}`~kerneljax.grid` sweeps `x` across its observed range and holds `group` at its most common level.

The [regression guide](regression.md#predicting-on-a-grid) develops predictions like these further, including [standard errors](regression.md#predicting-on-a-grid) and [derivatives](regression.md#derivatives).

## Selecting and reusing bandwidths

Every estimator accepts its bandwidth rule in the same argument. A string such as `"cv_ls"` is the shortest form, but the same argument can also take a previous selection result, a fitted estimator, or an explicit {class}`~kerneljax.Bandwidth`.

That makes bandwidth selection reusable rather than something each estimator has to repeat.

```python
sel = kj.select_bandwidth(
    data,
    kj.RegressionCriterion(method="cv_ls", degree=1),
    y=y,
)

chosen = kj.Bandwidth(
    h=np.array([0.1]),
    lam_uno=np.array([0.05]),
    lam_ord=np.zeros(0),
)

again = kj.local_poly(data, y, sel)
fixed = kj.local_poly(data, y, chosen)

print(f"selected r_squared={float(again.r_squared):.3f}")
print(f"held fixed r_squared={float(fixed.r_squared):.3f}")
```

```text
selected r_squared=0.958
held fixed r_squared=0.928
```

The first fit reuses the result returned by {func}`~kerneljax.select_bandwidth`. The second skips selection entirely and uses the supplied smoothing parameters as fixed values.

The string form is shorthand for a configured criterion. Settings already named by the estimator call, such as `degree`, travel with the selection result so they do not need to be restated later. This matters because bandwidth selection is usually the expensive part of the computation.

The [selection guide](selection.md) covers the available criteria, the optimizer, explicit selection and reuse, and [what to check](selection.md#reading-a-selection) before relying on a selected bandwidth.

So far, we have used KernelJax from the outside. The rest of this page looks underneath the estimator API at the choices that make those pieces compose. These details are not required to fit a model, but they matter if you want to understand the library deeply or build on top of it.

## Build on shared primitives

Every estimator is a thin layer over the same small set of operations. {func}`~kerneljax.kweights` builds the matrix of kernel weights between evaluation and training points, and {func}`~kerneljax.ksum` contracts those weights against whatever the estimator needs.

A density uses only the weights themselves.

```python
import jax
import jax.numpy as jnp

bw = kj.select_bandwidth(data, kj.DensityCriterion(method="cv_ml"))
weights = kj.kweights(data, bw.bandwidth)

by_hand = jnp.mean(weights, axis=1) / jnp.prod(bw.bandwidth.h)
shipped = kj.density(data, bw).value

print(f"largest gap to density={float(jnp.max(jnp.abs(by_hand - shipped))):.2e}")
```

```text
largest gap to density=2.98e-08
```

Regression uses the same weights but contracts them against the response. The [primitives guide](primitives.md) rebuilds the shipped local constant estimator directly from {func}`~kerneljax.ksum`. Conditional estimators use the same decomposition again, with a conditional density formed as a ratio of contractions over the conditioning and response kernels rather than through a separate computational path.

This concentrates correctness in a small set of primitives. Mixed column types, per-column operators, and held-out folds are handled once in `kweights` and `ksum`, and estimators built on top inherit that behavior automatically.

The primitives are public for the same reason. An estimator that KernelJax does not provide can still be assembled from the same pieces, which is the pattern used throughout the [primitives](primitives.md) and [custom kernel](custom-kernels.md) guides.

## Keep the full path differentiable

Bandwidth selection is where KernelJax's use of JAX matters most. Traditional implementations generally treat the selection criterion as a black box and optimize it with derivative-free methods. KernelJax keeps the criterion itself differentiable with respect to the bandwidth.

The same least-squares criterion used in the opening example can be differentiated directly.

```python
def criterion(bandwidth):
    return kj.cv_ls_regression(data, bandwidth, y=y, degree=1)


gradient = jax.grad(criterion)(chosen)
print(f"d/dh={float(gradient.h[0]):+.4f}  d/dlam={float(gradient.lam_uno[0]):+.4f}")
```

```text
d/dh=+0.3965  d/dlam=+0.0542
```

The solver iterations reported by the opening fit are optimization steps driven by this gradient. In practice, that lets bandwidth selection reach a solution in far fewer criterion evaluations than a derivative-free search. The [primitives guide](primitives.md) evaluates the same gradient at a plug-in starting point and at the selected solution.

Differentiability also constrains how the optimization is parameterized. Continuous bandwidths are represented through a softplus transform, while categorical smoothing parameters use a scaled logistic transform. Optimization therefore happens over unconstrained coordinates while the transformed values remain inside their valid domains.

The same geometry explains why categorical smoothing parameters do not initialize exactly at zero. Under the logistic transform, zero lies in a flat tail where the gradient has little leverage, so optimization starts slightly inside the admissible region instead.

## Separate structure from values

JAX needs to know which parts of a computation define its structure and which parts are values that can change between calls. KernelJax keeps that boundary explicit.

Kernel families, polynomial degree, and criterion configuration are static. Data, responses, and candidate bandwidths are traced values. JAX can then compile a computation for a particular structure and reuse it as those values change.

This is why criterion settings live on the criterion object rather than in `criterion_kwargs`. The keyword arguments are intended for traced values such as responses. Routing a structural setting such as `degree` through them turns that setting into a tracer, which cannot later occupy a static JAX argument.

Keeping the distinction explicit prevents failures from appearing far from the configuration that caused them and makes compilation behavior predictable. The [custom bandwidth selection](custom-criteria.md) guide walks through the failure case and the pattern that avoids it.

## Preserve selection context

Bandwidth selection is usually the expensive part of an analysis, so its result is designed to travel.

A {class}`~kerneljax.SelectionResult` remembers the criterion and kernels it was selected under, and fitted estimators retain both their bandwidth and the selection that produced it. Passing either one back into another estimator reuses that information rather than reconstructing it from separate arguments.

The earlier prediction example relied on exactly this behavior. Passing `fit` as the bandwidth rule reused the selected bandwidth and its settings without running selection again.

Provenance is only useful if it cannot be silently overridden. An explicitly supplied setting therefore has to agree with the one carried by the result.

```python
kj.local_poly(data, y, fit, degree=2)
```

```text
ValueError: degree=2 contradicts the degree 1 that bw was selected under
```

The same rule applies to kernels. If a selection result carries one kernel configuration and the caller supplies another, KernelJax refuses the mismatch rather than silently choosing one.

The conditional family follows the same pattern. A fit selected under {func}`~kerneljax.cdensity` can hand its bandwidth blocks directly to {func}`~kerneljax.cdist` and {func}`~kerneljax.cquantile` without restating the settings that produced them.

## Refuse invalid computations

KernelJax also refuses requests when the requested computation is not mathematically meaningful. Those checks are applied only when an operation first depends on the property being verified.

Kernel normalization is one example. Regression is a ratio of weighted sums, so a common multiplicative constant in the kernel cancels. Density estimation has no such cancellation. A kernel with the wrong total mass can therefore still be valid for regression but is rejected when a density first requires proper normalization.

The [custom kernels guide](custom-kernels.md#normalization-is-checked-when-it-matters) shows that check directly.

The same principle applies to statistical objectives. Likelihood selection for a conditional distribution is rejected because the likelihood of a CDF value rewards oversmoothing without bound. Rather than return a bandwidth from an ill-posed objective, KernelJax points to the supported alternatives.

```python
kj.cdist(data, y, "cv_ml")
```

```text
ValueError: cv_ml cannot select a bandwidth for a conditional distribution, since the likelihood
of a CDF value rewards oversmoothing without bound. Select with cv_ls, reuse a cdensity fit, or
supply a ConditionalBandwidth
```

These checks are not optional because bypassing them would allow the library to return results it knows are invalid. They are also placed as narrowly as possible, so computations that are already mathematically valid are not restricted unnecessarily.

The rest of the guide develops these ideas one topic at a time. The estimator pages build on the shared example above, while the selection, primitives, and custom guides go deeper into bandwidth optimization and the pieces used to construct and extend those estimators.

If you are comparing KernelJax numerically with another implementation, enable [double precision](../install.md#double-precision) first. JAX uses 32-bit floating-point arithmetic by default.
