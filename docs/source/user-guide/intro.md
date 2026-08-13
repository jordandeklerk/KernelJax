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

The categorical column also required no dummy encoding or separate per-group fits. {class}`~kerneljax.MixedData` records which variables are continuous, unordered, or ordered, and each kind is smoothed with an appropriate kernel. Its smoothing parameter is selected jointly with the continuous bandwidths, so a categorical variable carrying little useful information can be smoothed toward its maximum-smoothing limit, where distinctions between its levels contribute progressively less to the fit.

## Reading a fit

The printed report is a view of the fit object underneath. Two of the reported statistics describe the fit itself.

The reported residual standard error is the root mean squared residual

$$
\operatorname{RSE} = \left[
\frac{1}{n}
\sum_{i=1}^{n}
\bigl(Y_i - \hat m(X_i)\bigr)^2
\right]^{1/2},
$$

where $Y_i$ is the response at observation $i$, $\hat m(X_i)$ is its fitted value, and $n$ is the sample size. Despite the name in the report, this is the root mean squared residual rather than a degrees-of-freedom adjusted estimate of the error standard deviation. The [background page on regression](../background/regression.md#the-smoother-matrix) explains why KernelJax uses $n$ in the denominator.

The reported $R^2$ is also slightly different from the familiar linear-model definition. KernelJax measures both the response and the fitted values relative to the response mean $\bar Y$ and takes their squared cosine,

$$
R^2 = \frac{
\left[
\sum_{i=1}^{n}
(Y_i-\bar Y)
\bigl(\hat m(X_i)-\bar Y\bigr)
\right]^2
}{
\left[
\sum_{i=1}^{n}(Y_i-\bar Y)^2
\right]
\left[
\sum_{i=1}^{n}\bigl(\hat m(X_i)-\bar Y\bigr)^2
\right]
},
\qquad
\bar Y = \frac{1}{n}\sum_{i=1}^{n}Y_i.
$$

This is not, in general, the squared Pearson correlation because the fitted values are centered at $\bar Y$ rather than at their own sample mean. Whenever the denominator is nonzero, the Cauchy-Schwarz inequality keeps the statistic in $[0, 1]$.

The more familiar definition

$$
1 -
\frac{
\sum_i \bigl(Y_i-\hat m(X_i)\bigr)^2
}{
\sum_i (Y_i-\bar Y)^2
}
$$

agrees with the statistic above when the residuals are orthogonal to the fitted deviations $\hat m(X_i)-\bar Y$. An ordinary least-squares projection with an intercept has that property. A local polynomial smoother does not generally have it, so the two definitions need not coincide.

The numerical results behind the report remain available directly as arrays.

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

The string form is shorthand for a configured criterion. Settings already named by the estimator call, such as `degree`, travel with the selection result so they do not need to be reconstructed later. Reusing that result also avoids running bandwidth selection again, which is usually the expensive part of the computation.

The [selection guide](selection.md) covers the available criteria, the optimizer, explicit selection and reuse, and [what to check](selection.md#reading-a-selection) before relying on a selected bandwidth.

So far, we have used KernelJax from the outside. Underneath that API, criteria, kernels, and estimators remain ordinary JAX computations rather than opaque estimator internals. That design lets bandwidth selection use automatic differentiation, fits run under `jit`, and bandwidths participate directly in larger JAX programs. The rest of this page looks at the choices that make those pieces compose. These details are not required to fit a model, but they matter if you want to understand the library deeply or build on top of it.

## Build on shared primitives

Every estimator is a thin layer over the same small set of operations. {func}`~kerneljax.kweights` builds the matrix of kernel weights between evaluation and training points, and {func}`~kerneljax.ksum` contracts those weights against whatever the estimator needs.

For a mixed sample, an entry of the weight matrix can be written as

$$
W(x,X_i) = \prod_{d\in\mathcal C}
k\!\left(
\frac{x_d-X_{id}}{h_d}
\right)
\prod_{d\in\mathcal U}
L_d^{\mathrm{uno}}(x_d,X_{id};\lambda_d)
\prod_{d\in\mathcal O}
L_d^{\mathrm{ord}}(x_d,X_{id};\lambda_d),
$$

where $\mathcal C$, $\mathcal U$, and $\mathcal O$ index the continuous, unordered, and ordered columns. The continuous kernel is $k$, $h_d$ is its bandwidth, and the categorical kernels use smoothing parameters $\lambda_d$.

The important detail is that {func}`~kerneljax.kweights` leaves out the continuous $1/h_d$ scale factors. Categorical kernels have no corresponding bandwidth divisor, and leaving the continuous factors unscaled lets each estimator apply the normalization it needs exactly once.

To see where that normalization enters, density estimation gives the simplest example. For a density estimate,

$$
\hat f(x) = \frac{1}{
n\prod_{d\in\mathcal C}h_d
}
\sum_{i=1}^{n} W(x,X_i).
$$

The reconstruction below is exactly that calculation.

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

Reusing the same weighting and contraction path concentrates correctness in a small set of primitives. Mixed column types, per-column operators, and held-out folds are handled once in `kweights` and `ksum`, and estimators built on top inherit that behavior automatically.

Making those primitives public extends the same benefit to code outside the shipped estimators. An estimator that KernelJax does not provide can still be assembled from the same pieces, which is the pattern used throughout the [primitives](primitives.md) and [custom kernel](custom-kernels.md) guides.

## Keep the full path differentiable

Bandwidth selection is where KernelJax's use of JAX matters most. Many traditional implementations treat the selection criterion as a black box and optimize it with derivative-free methods. KernelJax keeps the criterion itself differentiable with respect to the bandwidth.

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

These are derivatives with respect to the bandwidths in their natural constrained scale. The optimizer itself works in unconstrained coordinates, so it sees these derivatives through the transformation that maps an unconstrained vector back to a valid bandwidth.

Write $z_d$ for the unconstrained coordinate associated with column $d$. Continuous bandwidths use the softplus map

$$
h_d = \operatorname{softplus}(z_d) = \log\bigl(1+e^{z_d}\bigr),
$$

while categorical smoothing parameters use a scaled logistic map

$$
\lambda_d = \bar\lambda_d \sigma(z_d),
\qquad
\sigma(z) = \frac{1}{1+e^{-z}},
$$

where $\bar\lambda_d$ is the upper bound imposed by the categorical kernel. For the default Aitchison-Aitken kernel on an unordered variable with $c$ levels, that bound is $\bar\lambda_d = (c-1)/c$.

The softplus maps the real line onto $(0,\infty)$, while the scaled logistic maps it onto $(0,\bar\lambda_d)$. The optimizer can therefore move freely in $z$ without stepping outside the admissible bandwidth region.

The chain rule gives the gradient the optimizer actually follows. Differentiating the two maps gives $h_d'(z_d) = \sigma(z_d)$ for the softplus, since the derivative of the softplus is the logistic function, and $\lambda_d'(z_d) = \bar\lambda_d \sigma(z_d)\bigl(1-\sigma(z_d)\bigr)$ for the scaled logistic. Composing each with the criterion gives

$$
\begin{aligned}
\frac{\partial \mathrm{CV}}{\partial z_d}
&= \frac{\partial \mathrm{CV}}{\partial h_d} \cdot \sigma(z_d)
&& \text{continuous},
\\
\frac{\partial \mathrm{CV}}{\partial z_d}
&= \frac{\partial \mathrm{CV}}{\partial \lambda_d} \cdot \bar\lambda_d \sigma(z_d)\bigl(1-\sigma(z_d)\bigr)
&& \text{categorical}.
\end{aligned}
$$

The solver iterations reported by the opening fit are driven by these transformed gradients rather than by the natural-scale gradient printed above. This lets the optimizer use automatic derivatives of the criterion while keeping every candidate bandwidth valid.

The same geometry explains the categorical starting rule. The boundary value $\lambda_d=0$ corresponds to $z_d\to-\infty$, where

$$
\sigma(z_d)\bigl(1-\sigma(z_d)\bigr)\to 0.
$$

Starting at that boundary would therefore leave almost no gradient in the unconstrained coordinate. KernelJax instead starts categorical parameters at $\lambda_d = \bar\lambda_d / 2$, which corresponds to $z_d=0$, where the logistic derivative is largest and the search has room to move in either direction.

## Separate structure from values

Automatic differentiation is only one part of keeping the estimator stack compatible with JAX. Compilation introduces another distinction between the structure of a computation and the values that can change between calls.

KernelJax keeps that boundary explicit. Kernel families, polynomial degree, and criterion configuration are static. Data, responses, and candidate bandwidths are traced values. JAX can then compile a computation for a particular structure and reuse it as those values change.

That distinction also shapes the criterion API. Structural settings such as `degree` live on the criterion object, while `criterion_kwargs` is intended for traced values such as responses. Routing a structural setting such as `degree` through those keyword arguments turns that setting into a tracer, which cannot later occupy a static JAX argument.

Keeping the distinction explicit prevents failures from appearing far from the configuration that caused them and makes compilation behavior predictable. The [custom bandwidth selection](custom-criteria.md) guide walks through the failure case and the pattern that avoids it.

## Preserve selection context

The reuse shown earlier depends on preserving more than the numerical bandwidth. A {class}`~kerneljax.SelectionResult` remembers the criterion and kernels it was selected under, and fitted estimators retain both their bandwidth and the selection that produced it. Passing either one back into another estimator reuses that information rather than reconstructing it from separate arguments.

The earlier prediction example relied on exactly this behavior. Passing `fit` as the bandwidth rule reused the selected bandwidth and its settings without running selection again.

That context is only useful if it cannot be silently overridden. An explicitly supplied setting therefore has to agree with the one carried by the result.

```python
kj.local_poly(data, y, fit, degree=2)
```

```text
ValueError: degree=2 contradicts the degree 1 that bw was selected under
```

The same rule applies to kernels. If a selection result carries one kernel configuration and the caller supplies another, KernelJax refuses the mismatch rather than silently choosing one.

The same context-preserving rule extends to the conditional estimators. A fit selected under {func}`~kerneljax.cdensity` can hand its bandwidth blocks directly to {func}`~kerneljax.cdist` and {func}`~kerneljax.cquantile` without restating the settings that produced them.

## Refuse invalid computations

Configuration mismatches are one kind of invalid request. KernelJax applies the same principle when the requested computation itself is not mathematically meaningful. Those checks are applied only when an operation first depends on the property being verified.

Kernel normalization is one example. Regression is a ratio of weighted sums, so a common multiplicative constant in the kernel cancels. Density estimation has no such cancellation. A kernel with the wrong total mass can therefore still be valid for regression but is rejected when a density first requires proper normalization.

The [custom kernels guide](custom-kernels.md#normalization-is-checked-when-it-matters) shows that check directly.

A second example comes from bandwidth selection for conditional distributions. Likelihood selection for a conditional distribution is rejected because the likelihood of a CDF value rewards oversmoothing without bound. Rather than return a bandwidth from an ill-posed objective, KernelJax points to the supported alternatives.

```python
kj.cdist(data, y, "cv_ml")
```

```text
ValueError: cv_ml cannot select a bandwidth for a conditional distribution, since the likelihood
of a CDF value rewards oversmoothing without bound. Select with cv_ls, reuse a cdensity fit, or
supply a ConditionalBandwidth
```

These checks are not optional because bypassing them would allow the library to return results it knows are invalid. They are also placed as narrowly as possible, so computations that are already mathematically valid are not restricted unnecessarily.

If you are comparing KernelJax numerically with another implementation, enable [double precision](../install.md#double-precision) first. JAX uses 32-bit floating-point arithmetic by default.

The rest of the guide develops these ideas one topic at a time. The estimator pages build on the shared example above, while the selection, primitives, and custom guides go deeper into bandwidth optimization and the pieces used to construct and extend those estimators.
