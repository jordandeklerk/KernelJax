# Custom bandwidth selection

Kernels determine how observations are weighted. Bandwidth criteria determine which smoothing parameters KernelJax chooses. Those are separate extension points. A custom criterion can use the built-in kernels, a custom kernel can use the built-in criteria, and both can be replaced when a method requires it.

KernelJax does not require custom criteria to inherit from a base class. A criterion is any callable that receives the training data and a candidate bandwidth and returns one scalar objective for {func}`~kerneljax.select_bandwidth` to minimize.

That small interface leaves two separate questions in your control. The criterion determines what statistical objective a good bandwidth should minimize, while the selector determines how KernelJax searches that objective. This page starts by changing the loss used to select a regression bandwidth, then develops the callable contract, held-out computations, static configuration, diagnostics, and optimization machinery needed for more specialized selectors.

For custom weighting schemes, see [Custom kernels](custom-kernels.md). For the statistical ideas behind the built-in criteria, see [Bandwidth selection](../background/selection.md).

## Your first custom criterion

The built-in least-squares regression criterion chooses a bandwidth by minimizing the mean squared leave-one-out residual,

$$
\mathrm{CV}_{\mathrm{LS}}(h,\lambda)
= \frac{1}{n}
\sum_{i=1}^{n}
\left[
Y_i-\hat m_{-i}(X_i)
\right]^2,
$$

where $\hat m_{-i}(X_i)$ is the local polynomial prediction at $X_i$ obtained without allowing observation $i$ to contribute to its own fit.

We can change the **selection loss** without changing the estimator underneath. Replacing the square with an absolute value gives

$$
\mathrm{CV}_{\mathrm{abs}}(h,\lambda)
= \frac{1}{n}
\sum_{i=1}^{n}
\left|
Y_i-\hat m_{-i}(X_i)
\right|.
$$

The regression being evaluated is still a local polynomial fit obtained by weighted least squares. Only the criterion used to choose its bandwidth has changed.

First, create a small regression problem containing several large response outliers.

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
            train, y, bandwidth, degree=self.degree,
            kernels=kernels, chunk=chunk, fold=fold,
        )

        return jnp.mean(jnp.abs(y - fit.mean))
```

There are three pieces to it. `fold = jnp.arange(train.n)` assigns every observation its own fold, so each prediction excludes the observation being predicted. `local_poly(...)` evaluates those leave-one-out predictions at the candidate bandwidth supplied by the optimizer. The final line reduces the held-out residuals to the single scalar that bandwidth selection minimizes.

The criterion can be passed directly to {func}`~kerneljax.select_bandwidth`.

```python
squared = kj.select_bandwidth(
    train, kj.RegressionCriterion(method="cv_ls", degree=1), y=y
)

absolute = kj.select_bandwidth(train, AbsoluteDeviation(degree=1), y=y)

truth = np.sin(2 * np.pi * x)

for name, result in [
    ("squared error", squared),
    ("absolute deviation", absolute),
]:
    fitted = kj.local_poly(train, y, result)
    error = jnp.median(jnp.abs(fitted.mean - truth))

    print(
        f"{name:19s} "
        f"h = {result.bandwidth.h[0]:.4f}   "
        f"median |error| = {error:.4f}"
    )
```

```text
squared error       h = 0.0323   median |error| = 0.1223
absolute deviation  h = 0.0628   median |error| = 0.2141
```

The criteria choose quite different bandwidths. Absolute deviation makes the **selection objective** less sensitive to individual large residuals, but that does not make the resulting regression estimator robust to outliers. The local polynomial itself is still fitted by weighted least squares.

Here the absolute-deviation criterion chooses a wider bandwidth, which spreads the influence of the contaminated observations across a larger neighborhood and produces a larger median error against the known regression function. Changing the criterion therefore changes what the selector rewards, not the estimator being evaluated.

```{important}
KernelJax minimizes the objective you give it. Whether that objective targets the statistical behavior you want remains part of the method you are designing.
```

There is a second distinction worth making early. A criterion can be computationally valid while silently measuring the wrong quantity. For example, removing `fold=fold` from the call above would still produce a scalar loss, but each observation would then contribute to its own fitted value.

```python
def in_sample_absolute_deviation(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1, kernels=kernels, chunk=chunk
    )

    return jnp.mean(jnp.abs(y - fit.mean))
```

This function runs, but it is an in-sample fitting criterion rather than the leave-one-out criterion defined above. KernelJax can enforce the callable interface, but it cannot infer whether the scalar you return represents the statistical objective you intended.

## What a criterion must satisfy

A custom criterion has a deliberately small interface. It should

1. accept `train` and `bandwidth` as its first two arguments,
2. accept the keyword arguments `kernels` and `chunk`,
3. accept any additional data it needs through keyword arguments such as `y`,
4. return one scalar,
5. be compatible with JAX automatic differentiation through the candidate bandwidth, and
6. be hashable when represented by a callable object with configuration.

Conceptually, {func}`~kerneljax.select_bandwidth` constructs an objective like the one below.

```python
def objective(z):
    bandwidth = transform.from_unconstrained(z)

    return criterion(train, bandwidth, **extra, kernels=kernels, chunk=chunk)
```

The optimizer does not work directly on constrained values such as $h > 0$ and bounded categorical $\lambda$. It works on an unconstrained vector `z`, transforms that vector into a valid {class}`~kerneljax.Bandwidth`, and differentiates the resulting scalar objective.

A criterion therefore does not need to be differentiable in the classical sense at every possible point, but it does need to produce usable JAX derivatives along the optimization path. Nonsmooth losses such as absolute deviation can work, while non-finite values or gradients cannot.

The next few sections make the pieces of that contract concrete.

## Functions work too

A dataclass is useful when a criterion carries configuration such as polynomial degree, but it is not required. A plain function works just as well.

```python
def absolute_deviation(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1,
        kernels=kernels, chunk=chunk, fold=jnp.arange(train.n),
    )

    return jnp.mean(jnp.abs(y - fit.mean))
```

```python
result = kj.select_bandwidth(train, absolute_deviation, y=y)
```

Use a callable object when configuration should travel with the criterion. A module-level function is usually the simplest choice when there is no configuration to retain.

That distinction becomes important because the criterion itself is part of JAX's static configuration.

## Why callable objects are frozen

A criterion passed to {func}`~kerneljax.select_bandwidth` is static JAX configuration. A configurable criterion object must therefore be hashable.

A mutable dataclass is not.

```python
@dataclasses.dataclass
class MutableCriterion:
    degree: int = 1

    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        fit = kj.local_poly(
            train, y, bandwidth, degree=self.degree,
            kernels=kernels, chunk=chunk, fold=jnp.arange(train.n),
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

A frozen dataclass provides stable equality and hashing for immutable configuration. The same compilation consideration applies to functions and other callables. A callable recreated as a new object on every invocation may appear to JAX as new static configuration and trigger another compilation. Reusing criterion objects or module-level functions avoids that unnecessary churn.

## Return one scalar

The optimizer needs one objective value for each candidate bandwidth. Observation-level losses can be part of that calculation, but the criterion itself must reduce them to a scalar.

This criterion instead returns one loss per observation.

```python
def returns_a_vector(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1,
        kernels=kernels, chunk=chunk, fold=jnp.arange(train.n),
    )

    return jnp.abs(y - fit.mean)
```

Passing it to the selector fails when JAX differentiates the objective.

```python
kj.select_bandwidth(train, returns_a_vector, y=y)
```

```text
TypeError: Gradient only defined for scalar-output functions.
Output had shape: (150,).
```

Reduce observation-level contributions inside the criterion according to the loss you intend to optimize. The reduction is itself part of the statistical definition of the criterion, not merely an API requirement.

## Accept `kernels` and `chunk`

KernelJax passes `kernels` and `chunk` on every criterion evaluation. A custom criterion therefore needs to accept both keywords even when it does not use them.

```python
def forgets_the_keywords(train, bandwidth, *, y):
    fit = kj.local_poly(train, y, bandwidth, degree=1, fold=jnp.arange(train.n))

    return jnp.mean(jnp.abs(y - fit.mean))
```

```python
kj.select_bandwidth(train, forgets_the_keywords, y=y)
```

```text
TypeError: forgets_the_keywords() got an unexpected keyword argument 'kernels'
```

Even if the criterion does not use those values directly, include `kernels=None` and `chunk=None` in its signature.

When the criterion calls another KernelJax operation internally, passing them onward is important. `kernels` ensures that selection evaluates the same weighting scheme the eventual estimator will use. `chunk` preserves the requested memory and computation strategy without changing the mathematical result.

## Keep static settings on the criterion

The criterion may also need information of two fundamentally different kinds.

* data that vary from one problem to another
* configuration that changes the structure of the computation

They should not travel through the selector in the same way. Array-like data such as a response belong in `y` or `criterion_kwargs`. They are traced values, so changing them can reuse the compiled structure.

Settings such as polynomial degree should normally be immutable fields on the criterion object. The `AbsoluteDeviation` class at the top of this page already follows that pattern, holding `degree` as a frozen field and reading it inside `__call__`.

Routing `degree` through `criterion_kwargs` instead causes it to become a traced value inside the compiled selector.

```python
def with_degree_argument(train, bandwidth, *, y, degree, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=degree,
        kernels=kernels, chunk=chunk, fold=jnp.arange(train.n),
    )

    return jnp.mean(jnp.abs(y - fit.mean))
```

```python
kj.select_bandwidth(
    train, with_degree_argument, y=y, criterion_kwargs={"degree": 1}
)
```

```text
ValueError: Non-hashable static arguments are not supported. An error occurred while trying to
hash an object of type <class 'kerneljax.basis.LocalPolyBasis'>,
LocalPolyBasis(degree=JitTracer(~int32[])). The error was:
TypeError: unhashable type: 'DynamicJaxprTracer'
```

The problem is not that `criterion_kwargs` is generally unsafe. It is specifically the wrong route for information that a downstream JAX operation needs as static structure. If a setting affects shapes, dispatch, polynomial degree, or other compile-time behavior, put it on the static criterion object instead.

Keeping that configuration on the criterion also determines what information can travel with the selected bandwidth later.

## Settings travel with the selection result

A {class}`~kerneljax.SelectionResult` retains the criterion that produced its bandwidth. When that result is reused by an estimator, compatible settings can therefore be recovered automatically.

```python
fit = kj.local_poly(train, y, absolute)

print(f"degree read off the criterion: {fit.degree}")
```

```text
degree read off the criterion: 1
```

For local polynomial regression, KernelJax looks specifically for an attribute named `degree` on the carried criterion.

A custom object such as

```python
@dataclasses.dataclass(frozen=True)
class MyCriterion:
    degree: int = 1
```

therefore supplies enough metadata for `local_poly` to recover that setting. An otherwise equivalent field named `poly_degree` would not be discovered automatically. If no `degree` is carried, `local_poly` uses its usual default unless a degree is supplied explicitly.

An explicit setting must agree with carried configuration.

```python
kj.local_poly(train, y, absolute, degree=2)
```

```text
ValueError: degree=2 contradicts the degree 1 that bw was selected under
```

The same agreement rule applies to kernels. A `SelectionResult` retains the {class}`~kerneljax.KernelSet` used during selection, so reusing the result does not silently switch back to another kernel configuration.

The criterion object therefore serves two purposes. During selection it defines the static structure of the objective. After selection it provides context for interpreting and safely reusing the bandwidth that objective produced.

## Holding observations out

The first criterion used `fold` to obtain leave-one-out predictions. KernelJax exposes that exclusion mechanism directly in several lower-level estimators so custom criteria can define other held-out schemes without writing Python loops over observations or folds.

{func}`~kerneljax.ksum`, {func}`~kerneljax.local_poly`, and {func}`~kerneljax.density` accept a `fold` array carrying one label per training observation. The array is positional, so the label in row $i$ belongs to training observation $i$. A pair is dropped whenever an evaluation row and a training row carry the same label, which requires the evaluation points to line up with the training sample, so `at` must be left out or match `train` in length.

For a generic weighted contraction, the retained part of row $i$ is

$$
\sum_{j : F_j \ne F_i} W(X_i, X_j) v_j,
$$

where $F_i$ is the fold label of row $i$ and $v_j$ is the value attached to training observation $j$. For a density, $v_j = 1$. For a kernel regression numerator, $v_j = Y_j$.

A local polynomial applies the same exclusion to the entire weighted least-squares problem.

$$
\hat\beta_{-F_i}(X_i)
= \arg\min_\beta
\sum_{j : F_j \ne F_i}
K_{h, \lambda}(X_i, X_j)
\left[
Y_j - \beta^\top b_h(X_j, X_i)
\right]^2.
$$

The fitted conditional mean is the intercept of that local polynomial.

The labels are ordinary integers. Leave-one-out gives every observation a distinct label, while a five-fold assignment repeats labels across rows.

```python
leave_one_out = jnp.arange(train.n)
five_fold = jnp.arange(train.n) % 5

print(f"leave one out  {np.asarray(leave_one_out)[:10]}")
print(f"five fold      {np.asarray(five_fold)[:10]}")
```

```text
leave one out  [0 1 2 3 4 5 6 7 8 9]
five fold      [0 1 2 3 4 0 1 2 3 4]
```

The choice changes the fitted values, because it changes how much of the sample is available at each row.

```python
bandwidth = kj.Bandwidth(
    h=jnp.array([0.1]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)
)

folds = [
    ("in sample", None),
    ("leave one out", leave_one_out),
    ("five fold", five_fold),
]

for name, fold in folds:
    fit = kj.local_poly(train, y, bandwidth, degree=1, fold=fold)
    residual = float(jnp.mean(jnp.abs(y - fit.mean)))

    print(f"{name:15s} mean |residual| = {residual:.4f}")
```

```text
in sample       mean |residual| = 0.6263
leave one out   mean |residual| = 0.6563
five fold       mean |residual| = 0.6622
```

The in-sample residuals are smallest because every observation contributes to its own fitted value. Leave-one-out removes that single contribution. Five-fold removes every observation sharing a label, about a fifth of the sample, so each fitted value is formed from noticeably less local information.

No Python loop over observations or folds is required. The exclusion is handled inside the JAX computation. Density estimation also changes the divisor to the number of retained observations on each row, so a custom density criterion does not need to make that correction itself.

Not every estimator implements held-out observations through the public `fold` argument. The unconditional CDF criterion, for example, constructs its leave-one-out values directly from cumulative kernel weights. Corrected AIC for regression uses the full-sample fit instead and penalizes complexity through the smoother matrix.

With that mechanism in place, extending the same criterion interface beyond regression is straightforward.

## A custom density criterion

The same extension mechanism works for density estimation. The built-in likelihood criterion uses leave-one-out densities. As an example, we can instead evaluate a five-fold log-likelihood criterion.

Let $F_i$ be the fold containing observation $i$, and let $W_{ij}(h,\lambda)$ denote the **unscaled** product-kernel weight between $X_i$ and $X_j$. If $\mathcal{C}$ indexes the continuous columns, the held-out density is

$$
\hat f_{-F_i}(X_i)
= \frac{
\sum_{j : F_j \ne F_i} W_{ij}(h, \lambda)
}{
n_i \prod_{d \in \mathcal{C}} h_d
},
\qquad
n_i = \#\{j : F_j \ne F_i\}
$$

The corresponding criterion is

$$
\mathrm{CV}^{(F)}_{\mathrm{ML}}(h,\lambda)
= -\sum_{i=1}^{n}
\log \hat f_{-F_i}(X_i).
$$

This normalization matters. KernelJax kernel values do not carry their own continuous $1/h$ factors. The density estimator inserts $1/\prod h_d$ itself and divides each row by the number of observations left after its fold has been removed. The built-in likelihood criterion uses the same construction with one observation per fold, so $n_i = n-1$ for every row.

The criterion can therefore delegate the held-out density calculation to {func}`~kerneljax.density` rather than reconstructing those normalizations itself.

```python
@dataclasses.dataclass(frozen=True)
class KFoldLikelihood:
    n_folds: int = 5

    def __call__(self, train, bandwidth, *, kernels=None, chunk=None):
        fold = jnp.arange(train.n) % self.n_folds

        fit = kj.density(
            train, bandwidth, kernels=kernels, chunk=chunk, fold=fold
        )

        return -jnp.sum(jnp.log(fit.value))
```

The bandwidth is selected through the same callable interface used for the regression criterion.

```python
five_fold = kj.select_bandwidth(train, KFoldLikelihood(n_folds=5))

cv_ml = kj.select_bandwidth(train, kj.DensityCriterion(method="cv_ml"))

for name, result in [
    ("five-fold likelihood", five_fold),
    ("leave-one-out cv_ml", cv_ml),
]:
    print(
        f"{name:20s}  "
        f"h = {result.bandwidth.h[0]:.4f}"
    )
```

```text
five-fold likelihood  h = 0.0655
leave-one-out cv_ml   h = 0.0533
```

Five-fold cross-validation estimates each held-out density from fewer observations than leave-one-out cross-validation. In this sample, the five-fold criterion chooses the wider bandwidth. That direction is plausible because the smaller fitting sample can benefit from more smoothing, but it is not a general rule that k-fold selection must always produce a larger bandwidth.

The fold assignment above is intentionally simple for demonstration. `jnp.arange(train.n) % 5` deterministically assigns rows by their position. In a real analysis, construct folds in a way that makes sense for the sampling design rather than relying on row order.

The important API point is that a density criterion is not a separate kind of extension. It is still an ordinary scalar function of a bandwidth. What changes is the statistical calculation performed inside that function.

## Inspect a criterion directly

Once a criterion is written, the optimizer should not be the first place you inspect its behavior. A criterion is an ordinary callable, so you can evaluate it directly at candidate bandwidths.

```python
criterion = AbsoluteDeviation(degree=1)

bandwidth = kj.Bandwidth(
    h=jnp.array([0.02]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)
)

for h in [0.02, 0.06, 0.15, 0.40]:
    candidate = bandwidth.replace(
        h=jnp.array([h])
    )

    value = criterion(train, candidate, y=y)

    print(
        f"h = {h:.2f}   "
        f"criterion = {value:.4f}"
    )
```

```text
h = 0.02   criterion = 0.6511
h = 0.06   criterion = 0.6414
h = 0.15   criterion = 0.6799
h = 0.40   criterion = 0.7403
```

The lowest value in this small sweep occurs near $h=0.06$, close to the `0.0628` found by the optimizer. For a one-dimensional bandwidth, a small grid like this is often more informative than looking only at an optimizer exit flag. For several bandwidth dimensions, evaluating selected slices can serve the same purpose.

Forward values are only half of the check when the criterion will be optimized with gradients.

## Check the gradient too

A criterion can have a reasonable forward value while still producing a gradient that cannot be optimized.

The following example deliberately introduces that problem.

```python
@dataclasses.dataclass(frozen=True)
class NanGradient(AbsoluteDeviation):
    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        loss = super().__call__(train, bandwidth, y=y, kernels=kernels, chunk=chunk)

        zero = loss - loss

        return loss + jnp.where(zero == 0.0, 0.0, jnp.sqrt(zero))
```

The selected branch adds zero to the forward value, but the unused square-root branch has a problematic derivative at zero.

Inspect both directly.

```python
criterion = NanGradient()

value, grad = jax.value_and_grad(
    lambda h: criterion(
        train,
        kj.Bandwidth(h=jnp.array([h]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0)),
        y=y,
    )
)(0.1)

print(
    f"value={value:.4f}  "
    f"gradient={grad}"
)
```

```text
value=0.6563  gradient=nan
```

As with custom kernels, guard the unsafe expression itself rather than assuming that an untaken `jnp.where` branch cannot affect differentiation.

For a custom criterion, evaluating `jax.value_and_grad` at several plausible bandwidths is often worth doing before launching a full multi-start optimization. Remember that this direct check differentiates in the natural bandwidth scale. {func}`~kerneljax.select_bandwidth` ultimately differentiates the criterion through its unconstrained bandwidth transform, so the numerical gradient seen by the solver also contains the derivative of that transformation.

Direct evaluation and direct differentiation tell us whether the objective behaves sensibly at chosen points. The selection result then tells us what happened when KernelJax actually searched it.

## Reading a selection result

A search can return a result without producing a useful optimum, so the bandwidth should not be read in isolation.

A {class}`~kerneljax.SelectionResult` exposes four numerical pieces that belong together.

```text
bandwidth
value
n_iter
converged
```

Consider the `NanGradient` criterion above and a second criterion whose objective is never finite.

```python
@dataclasses.dataclass(frozen=True)
class NanValue(AbsoluteDeviation):
    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        return (
            super().__call__(train, bandwidth, y=y, kernels=kernels, chunk=chunk)
            * jnp.nan
        )
```

```python
start = kj.normal_reference(train, kj.KernelSet())

print(
    f"the continuous reference starts at "
    f"h = {start.h[0]:.4f}"
)

for name, criterion in [
    ("nan gradient", NanGradient()),
    ("nan value", NanValue()),
]:
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
the continuous reference starts at h = 0.1081
nan gradient  h = nan     value = 0.9693  n_iter = 200  converged = False
nan value     h = 0.1081  value = nan     n_iter = 200  converged = False
```

The failures look different. With an unusable gradient, an objective value may remain finite while the coordinates themselves become invalid. With an unusable objective, the coordinates can look perfectly ordinary even though no meaningful optimization took place.

```{important}
Read `bandwidth`, `value`, `n_iter`, and `converged` together. A plausible bandwidth by itself is not evidence that selection succeeded.
```

For the default L-BFGS solver, `converged=True` means its stopping tolerance was reached with a finite final objective and gradient before the iteration limit. It does not mean that a global minimum was found.

That last distinction matters because a bandwidth search is generally not one optimization trajectory.

## Where the search starts

Bandwidth-selection objectives are generally nonconvex, so KernelJax runs multiple optimization trajectories by default. {func}`~kerneljax.select_bandwidth` uses `n_starts=3`.

The first continuous bandwidth comes from the normal-reference scale. Categorical parameters start halfway between zero and their kernel-specific upper bounds so that they begin in a region where the logistic transform still has useful slope.

Additional starts shift that initial point in the **unconstrained** coordinate system. Each start receives a complete solver run. KernelJax then ranks runs whose objective values and coordinates are finite and keeps the one with the smallest objective.

An important consequence is that the winner is not chosen according to its `converged` flag. A finite non-converged run with the lowest objective can be selected, in which case the returned {class}`~kerneljax.SelectionResult` correctly carries `converged=False`.

The result fields describe that one winning trajectory, not averages or summaries across all starts. If a criterion appears to land at a poor local solution, comparing more starting points is often the first useful diagnostic.

For inspecting one trajectory in isolation,

```python
result = kj.select_bandwidth(train, criterion, y=y, n_starts=1)
```

removes the multi-start comparison.

So far, the criterion has been replaceable while the optimizer itself has remained fixed. The final extension point separates those two pieces as well.

## Swapping the solver

The optimizer is replaceable too. By default, {func}`~kerneljax.select_bandwidth` uses {func}`~kerneljax.lbfgs`, but any static callable matching the solver interface can be supplied.

A minimal gradient-descent example is

```python
def gradient_descent(objective, start, *, steps=300, rate=0.05):
    def step(z, _):
        grad = jax.grad(objective)(z)
        return z - rate * grad, None

    z, _ = jax.lax.scan(step, start, length=steps)

    converged = jnp.all(jnp.isfinite(z))

    return z, objective(z), jnp.asarray(steps), converged
```

A solver receives `objective` and the starting unconstrained coordinates positionally and must return

```text
coordinates
objective value
iteration count
convergence flag
```

The interpretation of the last two fields belongs to the solver. In this deliberately simple example, `converged` only means that the final coordinates are finite, so it is not a numerical convergence test comparable to L-BFGS's stopping tolerance.

For a custom solver, the meaning of `converged` is whatever boolean that solver returns, so document its stopping rule accordingly.

We can compare the two solvers from the same starting point.

```python
criterion = kj.RegressionCriterion(method="cv_ls", degree=1)

descent = kj.select_bandwidth(
    train, criterion, y=y, solver=gradient_descent, n_starts=1
)

lbfgs = kj.select_bandwidth(train, criterion, y=y, n_starts=1)

for name, result in [
    ("gradient descent", descent),
    ("L-BFGS", lbfgs),
]:
    print(
        f"{name:16s} "
        f"value = {result.value:.4f}  "
        f"h = {result.bandwidth.h[0]:.4f}  "
        f"steps = {int(result.n_iter)}"
    )
```

```text
gradient descent value = 1.9194  h = 0.1336  steps = 300
L-BFGS           value = 1.9185  h = 0.1606  steps = 5
```

The custom solver works, but L-BFGS reaches a slightly smaller objective in far fewer iterations here. More importantly, both are one-start solutions. On a nonconvex bandwidth objective, changing the initialization can matter as much as changing the local optimizer.

It is therefore useful to keep those questions separate when diagnosing a poor result. First ask whether the criterion itself has sensible values and gradients. Then ask whether different starts reach different parts of that objective. Only then does changing the local solver address a distinct part of the search.

## Common mistakes

| Mistake                                                | Symptom                                              | Fix                                                    |
| ------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------ |
| Returning a vector of losses                           | `Gradient only defined for scalar-output functions`  | Reduce to one scalar inside the criterion              |
| Omitting `kernels` or `chunk`                          | Unexpected keyword argument                          | Accept both keywords                                   |
| Routing structural settings through `criterion_kwargs` | Static argument becomes a tracer                     | Store structural settings on a frozen criterion object |
| Returning a non-finite objective                       | `value` is `nan` or `inf`                            | Inspect the criterion directly before optimization     |
| Producing a non-finite gradient                        | Finite forward value but unusable search coordinates | Check `jax.value_and_grad` at plausible bandwidths     |
| Treating `converged=True` as a global guarantee        | Plausible but inferior local solution                | Compare objective values across starts                 |
| Assuming the winning start converged                   | `converged=False` on a finite selected result        | Read the winner's status together with its objective   |
| Using arbitrary row-order folds                        | Cross-validation does not match the sampling design  | Construct folds appropriate to the data                |

## Interface at a glance

A custom criterion has the general shape

```text
criterion(train, bandwidth, *, kernels=None, chunk=None, **data) -> scalar
```

| Requirement                           | Why it matters                                                 |
| ------------------------------------- | -------------------------------------------------------------- |
| Accept `train` and `bandwidth`        | They define the sample and candidate smoothing parameters      |
| Accept `kernels` and `chunk`          | KernelJax supplies them on every criterion call                |
| Return one scalar                     | The optimizer minimizes a scalar objective                     |
| Be JAX-autodiff compatible            | Selection differentiates through the bandwidth transform       |
| Return finite values along the search | Non-finite objectives cannot be meaningfully ranked            |
| Produce usable gradients              | Gradient-based solvers depend on them                          |
| Be hashable when stateful             | The criterion is static JAX configuration                      |
| Keep structural settings static       | Configuration such as polynomial degree cannot become a tracer |

The shortest useful custom criterion can therefore be just a function.

```python
def my_criterion(train, bandwidth, *, kernels=None, chunk=None):
    ...
    return loss
```

Use a frozen callable object when the criterion needs static configuration of its own, then pass it directly to {func}`~kerneljax.select_bandwidth`. From that point onward, the selector treats it through the same callable interface as the built-in criteria.
