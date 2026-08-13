# Custom bandwidth selection

Kernels determine how observations are weighted. Bandwidth criteria determine which smoothing parameters KernelJax chooses. Those are separate extension points. A custom criterion can use the built-in kernels, custom kernels can use the built-in criteria, and you can replace both when you need to.

KernelJax does not require custom criteria to inherit from a base class. A criterion is simply a callable that receives the training data and a candidate bandwidth and returns a scalar objective for {func}`~kerneljax.select_bandwidth` to minimize.

This page builds a regression criterion first, uses it in bandwidth selection, and then works through the small interface needed to create more specialized criteria. For custom weighting schemes, see [Custom kernels](custom-kernels.md). For the statistical ideas behind the built-in selectors, see [Bandwidth selection](../background/selection.md).

## Your first custom criterion

The built-in least-squares cross-validation criterion chooses the bandwidth that minimizes held-out squared prediction error. We can replace squared error with absolute deviation in a few lines.

Written out, the built-in criterion averages the squared leave-one-out residuals,

$$
\mathrm{CV}_{ls}(h, \lambda) = \frac{1}{n} \sum_{i=1}^{n} \bigl(Y_i - \hat m_{-i}(X_i)\bigr)^2,
$$

over the $n$ observed pairs $(X_i, Y_i)$, where $\hat m_{-i}$ is the local polynomial fit evaluated at $X_i$ and formed without the observation sitting there, $h$ collects the continuous bandwidths, and $\lambda$ collects the categorical smoothing parameters. Replacing the square by an absolute value leaves every other part of that expression alone,

$$
\mathrm{CV}_{\mathrm{lad}}(h, \lambda) = \frac{1}{n} \sum_{i=1}^{n} \bigl| Y_i - \hat m_{-i}(X_i) \bigr|,
$$

and that one change is the whole of the criterion below. [Bandwidth selection](../background/selection.md#cross-validation-for-regression) derives the squared version and says what it targets.

First, create a small regression problem with a handful of large outliers.

```python
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import kerneljax as kj

n = 150
rng = np.random.default_rng(1)
x = rng.uniform(size=n)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
outliers = rng.choice(n, 8, replace=False)
y[outliers] += 6.0
train = kj.MixedData.continuous(x)
```

Now define the criterion.

```python
@dataclasses.dataclass(frozen=True)
class AbsoluteDeviation:
    degree: int = 1

    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        fold = jnp.arange(train.n)

        fit = kj.local_poly(
            train, y, bandwidth, degree=self.degree, kernels=kernels, chunk=chunk, fold=fold
        )

        return jnp.mean(jnp.abs(y - fit.mean))
```

There are only three ideas in that class.

`fold = jnp.arange(train.n)` gives every observation its own fold, producing leave-one-out predictions.

`local_poly(...)` evaluates those predictions at the candidate bandwidth supplied by the optimizer.

The final line reduces the held-out errors to one scalar objective.

We can hand the criterion directly to {func}`~kerneljax.select_bandwidth`.

```python
squared = kj.select_bandwidth(train, kj.RegressionCriterion(method="cv_ls", degree=1), y=y)
absolute = kj.select_bandwidth(train, AbsoluteDeviation(degree=1), y=y)
truth = np.sin(2 * np.pi * x)

for name, result in [("squared error", squared), ("absolute deviation", absolute)]:
    fit = kj.local_poly(train, y, result)

    error = jnp.median(jnp.abs(fit.mean - truth))

    print(f"{name:19s} h = {result.bandwidth.h[0]:.4f}   median |error| = {error:.4f}")
```

```text
squared error       h = 0.0323   median |error| = 0.1223
absolute deviation  h = 0.0628   median |error| = 0.2141
```

The two criteria choose very different bandwidths. The absolute-deviation loss is less sensitive to individual large residuals, but that does not automatically make the resulting estimator more robust. The regression underneath is still fitted by weighted least squares, and here the wider selected bandwidth spreads the effect of the contaminated observations over a larger neighborhood.

```{important}
KernelJax minimizes the objective you give it. Deciding whether that objective targets the behavior you actually want remains part of the statistical problem.
```

## What a criterion must satisfy

A custom bandwidth criterion has a deliberately small interface. At a minimum, it should do the following.

1. Accept `train` and `bandwidth` as its first two arguments.
2. Accept the keyword arguments `kernels` and `chunk`.
3. Accept any additional data it needs through keyword arguments such as `y`.
4. Return a single scalar.
5. Be compatible with JAX differentiation with respect to the bandwidth.
6. Be hashable when represented as a stateful callable object.

Conceptually, {func}`~kerneljax.select_bandwidth` does this.

```python
def objective(z):
    bandwidth = transform.from_unconstrained(z)

    return criterion(train, bandwidth, **extra, kernels=kernels, chunk=chunk)
```

The optimizer works in unconstrained coordinates, transforms each candidate back into a valid {class}`~kerneljax.Bandwidth`, and differentiates the scalar your criterion returns. Everything inside the criterion should therefore be ordinary JAX-compatible computation.

## Functions work too

A dataclass is convenient when the criterion has settings such as `degree`, but it is not required. A plain function works just as well.

```python
def absolute_deviation(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1, kernels=kernels, chunk=chunk, fold=jnp.arange(train.n)
    )

    return jnp.mean(jnp.abs(y - fit.mean))
```

```python
result = kj.select_bandwidth(train, absolute_deviation, y=y)
```

Use a callable object when the criterion has configuration that should travel with it. Use a function when it does not.

## Why callable objects are frozen

When a criterion does carry configuration, a frozen dataclass like `AbsoluteDeviation` above is usually the easiest representation. The criterion is treated as a static argument by JAX, so it must be hashable. A mutable dataclass is not.

```python
@dataclasses.dataclass
class MutableCriterion:
    degree: int = 1

    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        fit = kj.local_poly(
            train,
            y,
            bandwidth,
            degree=self.degree,
            kernels=kernels,
            chunk=chunk,
            fold=jnp.arange(train.n),
        )

        return jnp.mean(jnp.abs(y - fit.mean))
```

```python
kj.select_bandwidth(train, MutableCriterion(), y=y)
```

```text
ValueError: Non-hashable static arguments are not supported. An error occurred while trying to
hash an object of type <class '__main__.MutableCriterion'>, MutableCriterion(degree=1). The
error was:
TypeError: unhashable type: 'MutableCriterion'
```

A frozen dataclass also gives configuration values predictable equality and hashing behavior, which helps JAX reuse compiled functions.

## Return one scalar

The optimizer needs one number to minimize. This criterion returns one loss per observation.

```python
def returns_a_vector(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1, kernels=kernels, chunk=chunk, fold=jnp.arange(train.n)
    )

    return jnp.abs(y - fit.mean)
```

Passing it to the selector fails when JAX first tries to differentiate it.

```python
kj.select_bandwidth(train, returns_a_vector, y=y)
```

```text
TypeError: Gradient only defined for scalar-output functions.
Output had shape: (150,).
```

Reduce the observation-level contributions inside the criterion using whatever loss definition is appropriate for your problem.

## Accept `kernels` and `chunk`

KernelJax passes both keywords to every criterion call, whether or not your criterion uses them. Here is a criterion that omits them.

```python
def forgets_the_keywords(train, bandwidth, *, y):
    fit = kj.local_poly(train, y, bandwidth, degree=1, fold=jnp.arange(train.n))

    return jnp.mean(jnp.abs(y - fit.mean))
```

It fails immediately.

```python
kj.select_bandwidth(train, forgets_the_keywords, y=y)
```

```text
TypeError: forgets_the_keywords() got an unexpected keyword argument 'kernels'
```

Even when your criterion does not use them directly, include `kernels=None` and `chunk=None` in its signature. Passing them onward to the estimator underneath also ensures that selection is evaluating the same kernel configuration and chunking behavior that the eventual fit will use.

## A custom density criterion

The same interface works for density estimation. For example, the built-in likelihood criterion uses leave-one-out density estimates. We can instead write a five-fold version.

The objective the class below returns is the leave-one-out log likelihood with a fold in place of the single point. Writing $F_i$ for the fold label of observation $i$,

$$
\begin{aligned}
\mathrm{CV}^{(F)}_{ml}(h, \lambda) &= -\sum_{i=1}^{n} \log \hat f_{-F_i}(X_i), \\
\hat f_{-F_i}(X_i) &= \frac{1}{n_i} \sum_{j \,:\, F_j \neq F_i} K_{h,\lambda}(X_i, X_j), \\
n_i &= \#\{\, j \,:\, F_j \neq F_i \,\},
\end{aligned}
$$

where $K_{h,\lambda}$ is the [product kernel](../background/mixed-data.md#the-product-kernel) across the columns, already carrying a factor $1/h_d$ for each continuous column, and $n_i$ counts the observations that survive the exclusion at row $i$. The built-in `cv_ml` criterion is the same expression with every label distinct, so its divisor is $n - 1$ at every row while the five-fold version divides by roughly $4n/5$. KernelJax forms $n_i$ itself, which is why the class never touches the denominator, and [Bandwidth selection](../background/selection.md#likelihood-cross-validation) explains what the leave-one-out form is estimating.

```python
@dataclasses.dataclass(frozen=True)
class KFoldLikelihood:
    n_folds: int = 5

    def __call__(self, train, bandwidth, *, kernels=None, chunk=None):
        fold = jnp.arange(train.n) % self.n_folds
        fit = kj.density(train, bandwidth, kernels=kernels, chunk=chunk, fold=fold)
        return -jnp.sum(jnp.log(fit.value))
```

Select the bandwidth exactly as before.

```python
five_fold = kj.select_bandwidth(train, KFoldLikelihood(n_folds=5))
cv_ml = kj.select_bandwidth(train, kj.DensityCriterion(method="cv_ml"))

for name, result in [("five-fold likelihood", five_fold), ("leave-one-out cv_ml", cv_ml)]:
    print(f"{name:20s}  h = {result.bandwidth.h[0]:.4f}")
```

```text
five-fold likelihood  h = 0.0655
leave-one-out cv_ml   h = 0.0533
```

Five-fold cross validation removes more observations from each training fit than leave-one-out cross validation, so each held-out density is estimated from a smaller effective sample here. The selected bandwidth is correspondingly wider.

The important API point is that density criteria are not a separate extension mechanism. They are ordinary callables returning scalar functions of a bandwidth, just like regression criteria.

## Holding observations out

KernelJax estimators expose fold assignment directly rather than treating leave-one-out as a special mode. {func}`~kerneljax.ksum`, {func}`~kerneljax.local_poly`, and {func}`~kerneljax.density` accept a `fold` array containing one label per observation. Pairs with matching evaluation and training labels are excluded.

Every estimator that accepts folds restricts its sum to the retained pairs, so a quantity evaluated at $X_i$ is built from

$$
\sum_{j \,:\, F_j \neq F_i} W(X_i, X_j)\, v_j ,
$$

where $F_i$ is again the label the `fold` array gives training point $i$, and $v_j$ is whatever the estimator contracts those weights against, a column of ones for a density and the response for a regression. A local polynomial applies the same restriction one level up, to the weighted least squares problem itself,

$$
\begin{aligned}
\hat\beta_{-F_i}(X_i) &= \arg\min_{\beta} \sum_{j \,:\, F_j \neq F_i}
K_{h,\lambda}(X_i, X_j)\bigl(Y_j - \beta^\top b\bigl((X_j - X_i) / h\bigr)\bigr)^2, \\
\hat m_{-F_i}(X_i) &= e_1^\top \hat\beta_{-F_i}(X_i),
\end{aligned}
$$

with $b$ the local polynomial basis centered at the evaluation point and $e_1$ the first standard basis vector, so that the intercept is again the fitted mean.

For leave-one-out, every observation receives its own label.

```python
fold = jnp.arange(train.n)
```

For five-fold cross validation, observations sharing a label are held out together.

```python
fold = jnp.arange(train.n) % 5
```

This means a custom criterion does not need a Python loop over folds or observations. The estimator handles all exclusions inside the JAX computation. Density estimation also adjusts its normalization for the omitted observations, so the criterion does not need to correct the denominator itself.

Two built-in criteria work differently. {func}`~kerneljax.cdf` does not expose `fold`. Its leave-one-out distribution criterion removes the observation's own contribution directly from the cumulative estimate. Likewise, `aic_c_regression` uses the full-sample regression fit and penalizes model complexity through the hat-matrix trace rather than holding observations out.

## Keep static settings on the criterion

There are two kinds of information a criterion may need.

* data that vary with the problem
* settings that determine the structure of the computation

Those should travel differently. Array-like data belong in arguments such as `y` or `criterion_kwargs`. Settings that change the static structure of the estimator should generally live on the criterion object itself. The local polynomial degree is the clearest example.

The `AbsoluteDeviation` class at the top of this page already follows the recommended pattern, holding `degree` as a frozen field and reading it inside `__call__`.

Routing the degree dynamically through `criterion_kwargs` instead can turn it into a traced JAX value.

```python
def with_degree_argument(train, bandwidth, *, y, degree, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=degree, kernels=kernels, chunk=chunk, fold=jnp.arange(train.n)
    )

    return jnp.mean(jnp.abs(y - fit.mean))
```

This fails because `degree` reaches an estimator that expects it to remain static.

```python
kj.select_bandwidth(train, with_degree_argument, y=y, criterion_kwargs={"degree": 1})
```

```text
ValueError: Non-hashable static arguments are not supported. An error occurred while trying to
hash an object of type <class 'kerneljax.basis.LocalPolyBasis'>,
LocalPolyBasis(degree=JitTracer(~int32[])). The error was:
TypeError: unhashable type: 'DynamicJaxprTracer'
```

If a setting controls the shape or structure of the JAX computation, keep it as immutable configuration on the criterion object.

## Settings can travel with the selection result

Keeping configuration on the criterion has another benefit. A {class}`~kerneljax.SelectionResult` retains the criterion used to select its bandwidth. When that result is passed back to an estimator, KernelJax can recover compatible settings automatically.

For example, pass the earlier selection back to the estimator.

```python
fit = kj.local_poly(train, y, absolute)
print(f"degree read off the criterion: {fit.degree}")
```

```text
degree read off the criterion: 1
```

The same applies to kernels. If the bandwidth was selected using a custom kernel set, reusing the selection result preserves that kernel configuration.

For local polynomial regression, KernelJax specifically looks for a criterion attribute named `degree`. That means this works.

```python
@dataclasses.dataclass(frozen=True)
class MyCriterion:
    degree: int = 1
```

while a semantically equivalent field named `poly_degree` will not be discovered automatically.

If no `degree` attribute is available, `local_poly` falls back to its default degree unless you pass one explicitly. KernelJax also protects against contradictory settings.

```python
kj.local_poly(train, y, absolute, degree=2)
```

```text
ValueError: degree=2 contradicts the degree 1 that bw was selected under
```

This keeps the estimator used after selection aligned with the estimator the criterion actually optimized.

## Inspect a criterion directly

A criterion is an ordinary callable. You do not need to run the optimizer to evaluate it. That is useful both for understanding a new criterion and for debugging one.

```python
criterion = AbsoluteDeviation(degree=1)
bandwidth = kj.Bandwidth(h=jnp.array([0.02]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

for h in [0.02, 0.06, 0.15, 0.40]:
    candidate = bandwidth.replace(h=jnp.array([h]))
    value = criterion(train, candidate, y=y)
    print(f"h = {h:.2f}   criterion = {value:.4f}")
```

```text
h = 0.02   criterion = 0.6511
h = 0.06   criterion = 0.6414
h = 0.15   criterion = 0.6799
h = 0.40   criterion = 0.7403
```

The lowest value in this small sweep is near $h = 0.06$, close to the 0.0628 selected earlier. For a one-dimensional bandwidth, evaluating a few candidate values is often the fastest way to check whether the objective behaves the way you expect.

## Bandwidth selection and gradients

Optimization-based selection differentiates your criterion with respect to the bandwidth. A criterion can therefore have a perfectly reasonable forward value while still producing an unusable gradient. The following criterion has exactly that flaw.

```python
@dataclasses.dataclass(frozen=True)
class NanGradient(AbsoluteDeviation):
    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        loss = super().__call__(train, bandwidth, y=y, kernels=kernels, chunk=chunk)
        zero = loss - loss
        return loss + jnp.where(zero == 0.0, 0.0, jnp.sqrt(zero))
```

The selected branch contributes zero to the forward value, but the other branch contains a problematic derivative. The same principle appears in custom kernels. Guard unsafe expressions themselves rather than assuming an untaken `jnp.where` branch cannot affect differentiation.

You can inspect the value and gradient directly.

```python
criterion = NanGradient()

value, grad = jax.value_and_grad(
    lambda h: criterion(
        train, kj.Bandwidth(h=jnp.array([h]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)), y=y
    )
)(0.1)

print(f"value={value:.4f}  gradient={grad}")
```

```text
value=0.6563  gradient=nan
```

For a custom criterion, checking `jax.value_and_grad` at a reasonable bandwidth is often worth doing before launching a full multi-start search.

## Reading a selection result

A bandwidth search can fail without raising an exception, so do not judge the result from the bandwidth alone. A {class}`~kerneljax.SelectionResult` gives you four pieces of information that matter together.

```text
bandwidth
value
n_iter
converged
```

Consider two intentionally broken criteria, the `NanGradient` class from above and a companion whose forward value is never finite.

```python
@dataclasses.dataclass(frozen=True)
class NanValue(AbsoluteDeviation):
    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        return super().__call__(train, bandwidth, y=y, kernels=kernels, chunk=chunk) * jnp.nan
```

```python
start = kj.normal_reference(train, kj.KernelSet())
print(f"the search starts at h = {start.h[0]:.4f}")

for name, criterion in [("nan gradient", NanGradient()), ("nan value", NanValue())]:
    result = kj.select_bandwidth(train, criterion, y=y)

    print(
        f"{name:12s}  "
        f"h = {result.bandwidth.h[0]:<7.4f} "
        f"value = {result.value:<7.4f} "
        f"n_iter = {result.n_iter}  "
        f"converged = {result.converged}"
    )
```

```text
the search starts at h = 0.1081
nan gradient  h = nan     value = 0.9693  n_iter = 200  converged = False
nan value     h = 0.1081  value = nan     n_iter = 200  converged = False
```

The two failures look very different. With a bad gradient, the objective value can remain finite even while the optimizer moves to an invalid bandwidth. With a bad objective value, the bandwidth can look completely ordinary because the solver never found a usable step and returned the starting point.

```{important}
Read `bandwidth`, `value`, `n_iter`, and `converged` together. A plausible bandwidth by itself is not evidence of a successful solve.
```

## Where the search starts

Bandwidth-selection objectives are generally nonconvex, so KernelJax does not rely on one optimization trajectory. {func}`~kerneljax.select_bandwidth` uses `n_starts=3` by default.

The first start uses {func}`~kerneljax.normal_reference` for continuous bandwidths. Categorical parameters begin halfway between zero and their kernel-specific upper bounds rather than at zero, since zero can sit in a flat region of the optimization transform.

Additional starts perturb that initial point in unconstrained coordinates. Each start gets a full solve, and KernelJax keeps the best finite result.

That means the fields in the returned `SelectionResult` describe the **winning start**. They are not aggregates over all optimization runs.

If a custom criterion appears to converge to a poor local solution, increasing `n_starts` is usually more useful than changing the solver immediately. For debugging a single trajectory, use one start.

```python
result = kj.select_bandwidth(train, criterion, y=y, n_starts=1)
```

## Swapping the solver

The optimizer itself is also replaceable. By default, {func}`~kerneljax.select_bandwidth` uses {func}`~kerneljax.lbfgs`, but any callable with the expected solver interface can be supplied. A minimal gradient-descent solver looks like this.

```python
def gradient_descent(objective, start, *, steps=300, rate=0.05):
    def step(z, _):
        grad = jax.grad(objective)(z)
        return (z - rate * grad, None)

    z, _ = jax.lax.scan(step, start, length=steps)
    converged = jnp.all(jnp.isfinite(z))
    return (z, objective(z), jnp.asarray(steps), converged)
```

The solver receives `objective` and `start` positionally, and returns

```text
coordinates
objective value
iteration count
convergence flag
```

We can compare it with the default.

```python
criterion = kj.RegressionCriterion(method="cv_ls", degree=1)
descent = kj.select_bandwidth(train, criterion, y=y, solver=gradient_descent, n_starts=1)
lbfgs = kj.select_bandwidth(train, criterion, y=y, n_starts=1)

for name, result in [("gradient descent", descent), ("L-BFGS", lbfgs)]:
    print(
        f"{name:16s} "
        f"value = {result.value:.4f}  "
        f"h = {result.bandwidth.h[0]:.4f}  "
        f"steps = {result.n_iter:d}"
    )
```

```text
gradient descent value = 1.9194  h = 0.1336  steps = 300
L-BFGS           value = 1.9185  h = 0.1606  steps = 5
```

The custom solver works, but L-BFGS reaches a slightly lower objective in far fewer iterations. More importantly, both one-start solutions are worse than what the default multi-start search finds on this example. That is a useful reminder that, for a nonconvex bandwidth criterion, where the optimization begins can matter as much as the optimizer itself. A different solver is therefore usually the second thing to try. More starts are the first.

## Common mistakes

Most custom-criterion problems come from a small number of interface mismatches, and each row of this table is demonstrated earlier on the page with the error it produces.

| Mistake                                           | Symptom                                             | Fix                                                        |
| ------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| Returning a vector of losses                      | `Gradient only defined for scalar-output functions` | Reduce to one scalar inside the criterion                  |
| Omitting `kernels` or `chunk`                     | `unexpected keyword argument 'kernels'`             | Include both keywords in the signature, even if unused     |
| Routing static settings through `criterion_kwargs`| `Non-hashable static arguments are not supported`   | Keep structural settings as frozen fields on the criterion |
| Returning `nan`                                   | Solver stays at its start with `converged = False`  | Inspect `value` and `converged` together                   |
| A non-finite gradient                             | Finite `value` alongside `h = nan`                  | Check `jax.value_and_grad` before blaming the solver       |
| Too few starts                                    | A plausible result at a poor local solution         | Compare criterion values across more starts                |

## Interface at a glance

A custom criterion is any JAX-compatible callable with this general shape.

```text
criterion(train, bandwidth, *, kernels=None, chunk=None, **data) -> scalar
```

| Requirement                          | Why it matters                                                     |
| ------------------------------------ | ------------------------------------------------------------------ |
| Accept `train` and `bandwidth`       | They define the data and candidate smoothing parameters            |
| Accept `kernels` and `chunk`         | KernelJax passes both on every call                                |
| Return one scalar                    | The optimizer needs a scalar objective                             |
| Remain differentiable in `bandwidth` | Optimization follows the bandwidth gradient                        |
| Return finite values                 | Non-finite objectives cannot be minimized                          |
| Be hashable when stateful            | Criteria are static JAX arguments                                  |
| Keep structural settings static      | Estimator configuration such as degree cannot become traced values |

The shortest useful custom criterion can therefore be just a function.

```python
def my_criterion(train, bandwidth, *, kernels=None, chunk=None):
    ...
    return loss
```

Use a frozen callable object when the criterion needs configuration of its own, then pass it directly to {func}`~kerneljax.select_bandwidth`. From there, the selector treats your criterion exactly like one of the built-in rules.
