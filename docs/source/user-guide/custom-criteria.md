# Custom bandwidth selection

Kernels determine how observations are weighted. Bandwidth criteria determine which smoothing parameters KernelJax chooses. Those are separate extension points. A custom criterion can use the built-in kernels, a custom kernel can use the built-in criteria, and both can be replaced when a method requires it.

KernelJax does not require custom criteria to inherit from a base class. A criterion is any callable that receives the training data and a candidate bandwidth and returns one scalar objective for {func}`~kerneljax.select_bandwidth` to minimize.

That small interface leaves two separate questions in your control. The criterion determines what statistical objective a good bandwidth should minimize, while the selector determines how KernelJax searches that objective. After mapping the interface and its scope, this page changes the loss used to select a regression bandwidth, then develops held-out computations, static configuration, diagnostics, and optimization machinery for more specialized selectors.

For custom weighting schemes, see [Custom kernels](custom-kernels.md). For the statistical ideas behind the built-in criteria, see [Bandwidth selection](../background/selection.md).

## Interface at a glance

A custom criterion has the general shape

```text
criterion(train, bandwidth, *, kernels=None, chunk=None, **data) -> scalar
```

{func}`~kerneljax.select_bandwidth` converts a raw training array to {class}`~kerneljax.MixedData` before calling the criterion. The candidate {class}`~kerneljax.Bandwidth` contains the continuous, unordered, and ordered smoothing parameters for the complete training specification, and the selector optimizes them jointly.

| Requirement                           | Why it matters                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------- |
| Accept `train` and `bandwidth`        | They define the sample and candidate smoothing parameters                       |
| Accept `kernels` and `chunk`          | KernelJax supplies them on every criterion call                                 |
| Return a floating JAX scalar          | The optimizer differentiates and minimizes one value                            |
| Return finite values and gradients    | Non-finite runs cannot be searched or compared meaningfully                     |
| Be hashable when stateful             | The criterion is static JAX configuration                                       |
| Keep structural settings static       | Configuration such as polynomial degree cannot become a tracer                  |
| Pass array-like extras as data        | `y` and `criterion_kwargs` remain traced rather than becoming static constants  |

Follow this safe workflow.

1. define the criterion,
2. evaluate it directly at several plausible bandwidths,
3. check its scalar value and full-bandwidth gradient,
4. pass it to `select_bandwidth`, and
5. pass the resulting {class}`~kerneljax.SelectionResult` to the intended estimator.

The criterion itself is never passed as an estimator's `bw` argument.

```{important}
This callable extension point covers the public `select_bandwidth` workflow for regression, density, and unconditional distribution objectives. It selects one shared smoothing parameter per column. Conditional estimators currently expose their built-in selection rules or accept a fixed or previously selected conditional bandwidth. Their public interface does not accept an arbitrary custom criterion.
```

The selector works in unconstrained coordinates internally. Softplus and logistic transforms convert those coordinates to natural-scale continuous and categorical bandwidths before calling the criterion, so a criterion should not transform them again. At extreme coordinates, finite-precision arithmetic can still saturate at zero or a categorical boundary. Criteria and result checks should handle those values safely.

## Your first custom criterion

### Define the criterion

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

A plain function is sufficient when no configuration needs to travel with the result. This first example uses a frozen callable object because its polynomial `degree` must remain static during selection and should be recovered when the selected bandwidth is reused.

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

### Inspect the value and gradient

Do not make the optimizer the first place that exercises a new criterion. Evaluate a small range of plausible bandwidths first.

```python
criterion = AbsoluteDeviation(degree=1)

probe = kj.Bandwidth(h=jnp.array([0.02]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

value_and_gradient = jax.value_and_grad(criterion, argnums=1)

for h in [0.02, 0.06, 0.15, 0.40]:
    candidate = probe.replace(h=jnp.array([h]))
    value, gradient = value_and_gradient(train, candidate, y=y)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradient)
    )

    print(f"h = {h:.2f}   criterion = {value:.4f}")
```

```text
h = 0.02   criterion = 0.6511
h = 0.06   criterion = 0.6414
h = 0.15   criterion = 0.6799
h = 0.40   criterion = 0.7403
```

The lowest value in this small sweep occurs near $h=0.06$. Each pass also checks the complete `Bandwidth` gradient. For a one-dimensional bandwidth, this view is often more informative than an optimizer flag alone. For several dimensions, inspect representative slices.

For mixed data, the same gradient object also contains `lam_uno` and `lam_ord`. Probe several plausible bandwidths, including categorical values near their bounds. This direct calculation uses the natural bandwidth scale. The selector's gradient also includes its internal softplus and logistic transformations.

### Select and reuse

The criterion can be passed directly to {func}`~kerneljax.select_bandwidth`.

This comparison is deliberately about whether the criterion changes the selected bandwidth, not whether it creates a robust estimator. That distinction is interpreted immediately after the output.

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

Recompute the winning custom objective before interpreting it.

```python
recomputed = criterion(train, absolute.bandwidth, y=y)

np.testing.assert_allclose(recomputed, absolute.value, rtol=1e-5, atol=1e-6)
assert bool(jnp.isfinite(absolute.value))
assert all(
    bool(jnp.all(jnp.isfinite(leaf)))
    for leaf in jax.tree_util.tree_leaves(absolute.bandwidth)
)
assert bool(jnp.all(absolute.bandwidth.h > 0.0))
assert bool(jnp.all(absolute.bandwidth.lam_uno >= 0.0))
assert bool(jnp.all(absolute.bandwidth.lam_ord >= 0.0))

print(
    f"value = {absolute.value:.4f}  "
    f"n_iter = {int(absolute.n_iter)}  "
    f"converged = {bool(absolute.converged)}"
)
```

```text
value = 0.6413  n_iter = 21  converged = True
```

Finiteness and valid bounds are necessary, but the optimizer diagnostics still matter. The [diagnostics section](#diagnose-the-search) explains how to interpret all four result fields and clarifies what `converged` establishes and what it does not.

The `absolute` result carries the `degree=1` field from `AbsoluteDeviation`. That is why `local_poly(train, y, absolute)` can safely recover the degree without another argument.

### Criterion versus estimator

The criteria choose quite different bandwidths. Absolute deviation makes the **selection objective** less sensitive to individual large residuals, but that does not make the resulting regression estimator robust to outliers. The local polynomial itself is still fitted by weighted least squares.

Here the absolute-deviation criterion chooses a wider bandwidth, which spreads the influence of the contaminated observations across a larger neighborhood and produces a larger median error against the known regression function. Changing the criterion therefore changes what the selector rewards, not the estimator being evaluated.

```{important}
KernelJax minimizes the objective you give it. Whether that objective targets the statistical behavior you want remains part of the method you are designing.
```

There is a second distinction worth making early. A criterion can be computationally valid while silently measuring the wrong quantity. For example, removing `fold=fold` from the call above would still produce a scalar loss, but each observation would then contribute to its own fitted value.

```python
# Wrong: this measures in-sample error rather than held-out error.
def in_sample_absolute_deviation(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1, kernels=kernels, chunk=chunk
    )

    return jnp.mean(jnp.abs(y - fit.mean))
```

This function runs, but it is an in-sample fitting criterion rather than the leave-one-out criterion defined above. KernelJax can enforce the callable interface, but it cannot infer whether the scalar you return represents the statistical objective you intended.

## Follow the callable contract

Each requirement in the interface summary has a characteristic failure mode. A criterion does not need to be differentiable in the classical sense at every possible point, but it must produce usable JAX derivatives along the selector's path. Nonsmooth losses such as absolute deviation can work. Non-finite values or gradients cannot.

### Functions work too

A dataclass is useful when a criterion carries configuration such as polynomial degree, but it is not required. A plain function works just as well.

```python
def absolute_deviation(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1,
        kernels=kernels, chunk=chunk, fold=jnp.arange(train.n),
    )

    return jnp.mean(jnp.abs(y - fit.mean))
```

The function hardcodes `degree=1`, but KernelJax cannot infer that setting from its body. Select and reuse it with the degree stated explicitly.

```python
plain_result = kj.select_bandwidth(train, absolute_deviation, y=y)
plain_fit = kj.local_poly(train, y, plain_result, degree=1)
```

Use a callable object with a field literally named `degree` when that setting should travel automatically. A module-level function is usually the simplest choice when there is no configuration to retain.

That distinction matters for the next requirement.

### Why callable objects are frozen

A criterion passed to {func}`~kerneljax.select_bandwidth` is static JAX configuration. A configurable criterion object must therefore be hashable.

A mutable dataclass is not.

```python
# Wrong: a mutable dataclass is not hashable.
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
ValueError: Non-hashable static arguments are not supported ...
TypeError: unhashable type: 'MutableCriterion'
```

A frozen dataclass provides stable equality and hashing only when all its fields are themselves hashable. Keep arrays such as responses, observation weights, and fold labels in `y` or `criterion_kwargs`. Truly static sequences can be stored as tuples.

The same compilation consideration applies to functions and other callables. A callable recreated as a new object on every invocation may appear to JAX as new static configuration and trigger another compilation. Reusing criterion objects or module-level functions avoids that unnecessary churn.

### Return one scalar

The optimizer needs one objective value for each candidate bandwidth. Observation-level losses can be part of that calculation, but the criterion itself must reduce them to a scalar.

This criterion instead returns one loss per observation.

```python
# Wrong: the observation-level losses are not reduced.
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

Reduce observation-level contributions inside the criterion according to the loss you intend to optimize. The reduction is itself part of the statistical definition of the criterion, not merely an API requirement. The result should be floating point and vary usefully with the bandwidth. Autodiff rejects an integer-valued objective, while a bandwidth-independent floating constant has a zero gradient and gives the solver no direction.

### Keep the returned scalar inside JAX

A common mistake is to read “return one scalar” as “return a Python `float`.” The following function can appear to work eagerly, but selection traces it before differentiating it.

```python
# Wrong: converting a traced value to a Python scalar.
def python_scalar(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(
        train, y, bandwidth, degree=1,
        kernels=kernels, chunk=chunk, fold=jnp.arange(train.n),
    )

    return float(jnp.mean(jnp.abs(y - fit.mean)))
```

```python
kj.select_bandwidth(train, python_scalar, y=y, n_starts=1)
```

```text
ConcretizationTypeError: Abstract tracer value encountered where concrete
value is expected ...
The problem arose with the `float` function ...
```

Return the zero-dimensional JAX array directly. The same rule excludes `np.asarray(loss)`, NumPy reductions over traced values, and Python branching on a candidate bandwidth. Use `jax.numpy` operations and JAX control flow inside the criterion.

### Accept `kernels` and `chunk`

KernelJax passes `kernels` and `chunk` on every criterion evaluation. A custom criterion therefore needs to accept both keywords, with `kernels=None` and `chunk=None` defaults, even when it does not use them.

```python
# Wrong: KernelJax supplies keywords this function does not accept.
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

When the criterion calls another KernelJax operation internally, passing them onward is important. Merely accepting the keywords is not enough. Failing to forward `kernels` can silently optimize the default weights instead of the requested custom kernels. Forwarding `chunk` preserves the requested memory strategy without changing the mathematical result.

Chunked operations remain chunk-bounded under differentiation by recomputing checkpointed blocks during the backward pass. See [Performance and compilation](../performance.md#bounding-memory-with-chunk) for the memory tradeoff.

### Keep static settings on the criterion

The criterion may also need information of two fundamentally different kinds.

* data that vary from one problem to another
* configuration that changes the structure of the computation

They should not travel through the selector in the same way. Array-like data such as a response belong in `y` or `criterion_kwargs`. They are traced values, so changing them can reuse the compiled structure when their pytree structure, shapes, and dtypes remain compatible.

Settings such as polynomial degree should normally be immutable fields on the criterion object. The `AbsoluteDeviation` class at the top of this page already follows that pattern, holding `degree` as a frozen field and reading it inside `__call__`.

Routing `degree` through `criterion_kwargs` instead causes it to become a traced value inside the compiled selector.

```python
# Wrong: degree becomes traced data even though local_poly needs it statically.
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
ValueError: Non-hashable static arguments are not supported ...
LocalPolyBasis(degree=JitTracer(~int32[])) ...
TypeError: unhashable type: 'DynamicJaxprTracer'
```

The problem is not that `criterion_kwargs` is generally unsafe. It is specifically the wrong route for information that a downstream JAX operation needs as static structure. If a setting affects shapes, dispatch, polynomial degree, or other compile-time behavior, put it on the static criterion object instead.

Keeping that configuration on the criterion also determines what information can travel with the selected bandwidth later.

### Settings travel with the selection result

A {class}`~kerneljax.SelectionResult` retains the criterion that produced its bandwidth. The first example used that fact when `local_poly(train, y, absolute)` recovered degree one.

KernelJax looks specifically for an attribute named `degree` on the carried criterion. An otherwise equivalent field named `poly_degree` would not be discovered automatically. If no `degree` is carried, `local_poly` uses its usual default unless a degree is supplied explicitly.

An explicit setting must agree with carried configuration.

```python
kj.local_poly(train, y, absolute, degree=2)
```

```text
ValueError: degree=2 contradicts the degree 1 that bw was selected under
```

The same agreement rule applies to kernels. A `SelectionResult` retains the {class}`~kerneljax.KernelSet` used during selection, so reusing the result does not silently switch back to another kernel configuration.

Arbitrary custom fields are retained for inspection, but estimators do not automatically interpret them. Today, `local_poly` recognizes `degree`, while kernels travel separately on the result. A custom criterion also carries no automatic target tag. KernelJax cannot tell whether its bandwidth was designed for regression, density, or a distribution. Reuse the result only with the estimator whose objective the criterion actually evaluates.

The criterion object therefore serves two purposes. During selection, it defines the static structure of the objective. Afterward, it records the objective and any specifically recognized metadata.

## Holding observations out

The first criterion used `fold` to obtain leave-one-out predictions. KernelJax exposes that exclusion mechanism directly in several lower-level estimators so custom criteria can define other held-out schemes without writing Python loops over observations or folds.

{func}`~kerneljax.ksum`, {func}`~kerneljax.local_poly`, and {func}`~kerneljax.density` accept a `fold` array. The following four rules matter.

* supply a one-dimensional integer array with shape `(train.n,)`,
* any evaluation-training pair with equal labels is excluded,
* omit `at` unless its row $i$ corresponds to the same observation as `fold[i]`, and
* for density, KernelJax divides each row by the number of retained observations.

The API checks the length of `at` when folds are present, but it cannot detect a permutation or other row misalignment with the fold labels.

For a generic weighted contraction, the retained part of row $i$ is

$$
\sum_{j \mid F_j \ne F_i} W(X_i, X_j) v_j,
$$

where $F_i$ is the fold label of row $i$ and $v_j$ is the value attached to training observation $j$. For a density, $v_j = 1$. For a kernel regression numerator, $v_j = Y_j$.

A local polynomial applies the same exclusion to the entire weighted least-squares problem.

$$
\hat\beta_{-F_i}(X_i)
= \arg\min_\beta
\sum_{j \mid F_j \ne F_i}
K_{h, \lambda}(X_i, X_j)
\left[
Y_j - \beta^\top b_h(X_j, X_i)
\right]^2.
$$

The fitted conditional mean is the intercept of that local polynomial.

The labels are ordinary integers. Leave-one-out gives every observation a distinct label, while a five-fold assignment repeats labels across rows.

```python
leave_one_out = jnp.arange(train.n)
five_fold_labels = jnp.arange(train.n) % 5

print(f"leave one out  {np.asarray(leave_one_out)[:10]}")
print(f"five fold      {np.asarray(five_fold_labels)[:10]}")
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
    ("five fold", five_fold_labels),
]

for name, fold in folds:
    fit = kj.local_poly(train, y, bandwidth, degree=1, fold=fold)
    residual = jnp.mean(jnp.abs(y - fit.mean))

    print(f"{name:15s} mean |residual| = {residual:.4f}")
```

```text
in sample       mean |residual| = 0.6263
leave one out   mean |residual| = 0.6563
five fold       mean |residual| = 0.6622
```

The in-sample residuals are smallest because every observation contributes to its own fitted value. Leave-one-out removes that single contribution. Five-fold removes every observation sharing a label, about a fifth of the sample, so each fitted value is formed from noticeably less local information.

No Python loop over observations or folds is required. The exclusion is handled inside the JAX computation. Density estimation also changes the divisor to the number of retained observations on each row, so a custom density criterion does not need to make that correction itself.

Construct fold labels outside the criterion according to the sampling design. Randomized, grouped, spatial, and blocked folds answer different questions. Equality-based masking is not forward-chaining validation. For time-series data, ordinary fold labels can still let future observations predict earlier ones.

Not every estimator implements held-out observations through the public `fold` argument. The unconditional CDF criterion, for example, constructs its leave-one-out values directly from cumulative kernel weights. Corrected AIC for regression uses the full-sample fit instead and penalizes complexity through the smoother matrix.

With that mechanism in place, extending the same criterion interface beyond regression is straightforward.

### A custom density criterion

The same extension mechanism works for density estimation. The built-in likelihood criterion uses leave-one-out densities. As an example, we can instead evaluate a five-fold log-likelihood criterion.

Let $F_i$ be the fold containing observation $i$, and let $W_{ij}(h,\lambda)$ denote the **unscaled** product-kernel weight between $X_i$ and $X_j$. If $\mathcal{C}$ indexes the continuous columns, the held-out density is

$$
\hat f_{-F_i}(X_i)
= \frac{
\sum_{j \mid F_j \ne F_i} W_{ij}(h, \lambda)
}{
n_i \prod_{d \in \mathcal{C}} h_d
},
\qquad
n_i = \#\{j \mid F_j \ne F_i\}
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
def held_out_likelihood(train, bandwidth, *, fold, kernels=None, chunk=None):
    fit = kj.density(train, bandwidth, kernels=kernels, chunk=chunk, fold=fold)

    safe = jnp.maximum(fit.value, jnp.finfo(fit.value.dtype).tiny)

    return -jnp.sum(jnp.log(safe))
```

The positive floor matches the built-in unconditional likelihood criterion. A zero or negative held-out density receives a saturated large penalty instead of producing an infinite objective with a `nan` gradient. Strict positivity is still preferable because a saturated penalty contains little useful optimization information. The floor does not guarantee a useful derivative at every extreme bandwidth, so retain the direct gradient checks from the first example.

Fold assignments are problem data, so pass them through `criterion_kwargs` rather than storing an array on a static criterion object. Check before selection that the labels define at least two nonempty folds. A single fold excludes every observation and leaves the density denominator at zero.

```python
five_fold = kj.select_bandwidth(
    train,
    held_out_likelihood,
    criterion_kwargs={"fold": five_fold_labels},
)

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

The fold assignment above is intentionally simple for demonstration. `jnp.arange(train.n) % 5` deterministically assigns rows by their position. In a real analysis, replace `five_fold_labels` with randomized, grouped, or otherwise design-appropriate labels.

The important API point is that a density criterion is not a separate kind of extension. It is still an ordinary scalar function of a bandwidth. With mixed data, the candidate `Bandwidth` carries all continuous and categorical parameters and `density` uses them together. What changes is the statistical calculation performed inside the criterion.

## Stress-test a criterion

### Catch a poisoned gradient

A criterion can have a reasonable forward value while still producing a gradient that cannot be optimized.

The following example deliberately introduces that problem.

```python
# Wrong: the forward value is finite but the unused branch poisons its gradient.
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
bad_criterion = NanGradient()

value, grad = jax.value_and_grad(
    lambda h: bad_criterion(
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

As with custom kernels, guard the unsafe expression itself rather than assuming that an untaken `jnp.where` branch cannot affect differentiation. The corrected criterion makes the square root safe before selecting between branches.

```python
@dataclasses.dataclass(frozen=True)
class GuardedGradient(AbsoluteDeviation):
    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        loss = super().__call__(train, bandwidth, y=y, kernels=kernels, chunk=chunk)
        zero = loss - loss
        safe_root = jnp.sqrt(jnp.maximum(zero, jnp.finfo(loss.dtype).tiny))

        return loss + jnp.where(zero == 0.0, 0.0, safe_root)
```

As before, probe `jax.value_and_grad` at several plausible bandwidths before launching a full multi-start optimization, remembering that the solver additionally differentiates through the selector's bandwidth transform.

Direct evaluation and direct differentiation tell us whether the objective behaves sensibly at chosen points. The selection result then tells us what happened when KernelJax actually searched it.

For mixed data, repeat the direct probes with categorical parameters near zero, at representative interior values, and near their kernel-specific upper bounds. Validation should use the same response, extra arrays, kernels, and chunk strategy intended for the real selection.

### Check compilation and chunking

Finally, compare eager, compiled, and chunked evaluations of the working criterion. These paths should agree within a tolerance appropriate for floating-point accumulation order.

```python
candidate = probe.replace(h=jnp.array([0.1]))


def objective(bandwidth):
    return criterion(train, bandwidth, y=y)


eager_value = objective(candidate)
compiled_value = jax.jit(objective)(candidate)
chunked_value = criterion(train, candidate, y=y, chunk=32)

np.testing.assert_allclose(compiled_value, eager_value, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(chunked_value, eager_value, rtol=1e-5, atol=1e-6)
```

## Diagnose the search

### Reading a selection result

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
# Wrong: every candidate objective is non-finite.
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

for name, bad_criterion in [
    ("nan gradient", NanGradient()),
    ("nan value", NanValue()),
]:
    result = kj.select_bandwidth(train, bad_criterion, y=y)

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

For the default L-BFGS solver, `converged=True` means the finite final state satisfied either the gradient-max-norm tolerance or the relative-objective-change tolerance. It does not mean that a global minimum was found.

That last distinction matters because a bandwidth search is generally not one optimization trajectory.

### Where the search starts

Bandwidth-selection objectives are generally nonconvex, so KernelJax runs multiple optimization trajectories by default. {func}`~kerneljax.select_bandwidth` uses `n_starts=3`.

The first continuous start uses the density-target normal-reference formula. Categorical parameters start halfway between zero and their kernel-specific upper bounds so that they begin in a region where the logistic transform still has useful slope.

Additional starts shift that initial point in the **unconstrained** coordinate system. For each extra start, the same scalar offset is added to every coordinate. These are diagonal perturbations rather than independent random samples of a multidimensional search space. Each start receives a complete solver run. KernelJax then ranks runs whose objective values and coordinates are finite and keeps the one with the smallest objective.

An important consequence is that the winner is not chosen according to its `converged` flag. A finite non-converged run with the lowest objective can be selected, in which case the returned {class}`~kerneljax.SelectionResult` correctly carries `converged=False`.

The result fields describe that one winning trajectory, not averages or summaries across all starts. If a criterion appears to land at a poor local solution, comparing more starting points is often the first useful diagnostic.

For inspecting one trajectory in isolation,

```python
result = kj.select_bandwidth(
    train,
    AbsoluteDeviation(degree=1),
    y=y,
    n_starts=1,
)
```

removes the multi-start comparison.

So far, the criterion has been replaceable while the optimizer itself has remained fixed. The final extension point separates those two pieces as well.

## Swapping the solver

The optimizer is an advanced extension point. By default, {func}`~kerneljax.select_bandwidth` uses {func}`~kerneljax.lbfgs`. A replacement runs inside `jit` and `jax.lax.map`, so it must use JAX-traceable, fixed-shape computation. An ordinary SciPy optimizer or a Python data-dependent loop does not satisfy this contract.

A frozen callable binds solver settings while keeping them hashable and static.

This is a mechanics example, not a recommended general-purpose optimizer. Its fixed learning rate is sensitive to the objective's scale, and it has no line search.

```python
@dataclasses.dataclass(frozen=True)
class GradientDescent:
    steps: int = 300
    rate: float = 0.05
    tol: float = 1e-5

    def __call__(self, objective, start):
        def step(z, _):
            gradient = jax.grad(objective)(z)

            return z - self.rate * gradient, None

        z, _ = jax.lax.scan(step, start, length=self.steps)

        value, gradient = jax.value_and_grad(objective)(z)
        usable = (
            jnp.all(jnp.isfinite(z))
            & jnp.isfinite(value)
            & jnp.all(jnp.isfinite(gradient))
        )
        converged = usable & (jnp.max(jnp.abs(gradient)) < self.tol)

        return z, value, jnp.asarray(self.steps), converged
```

A solver receives only `objective` and the one-dimensional starting coordinate vector positionally. It must return

* coordinates with the same shape as `start`,
* the scalar value `objective(coordinates)` used to rank starts,
* a scalar integer iteration count, and
* a scalar boolean convergence flag.

The interpretation of the last two fields belongs to the solver and should be documented. Here, `converged` means the fixed iteration budget ended with finite state and a final gradient max norm below `tol`. The loop does not stop early. KernelJax additionally forces the selected result to `converged=False` when the solver's reported coordinates or objective are non-finite.

We can configure and compare the two solvers from the same starting point.

```python
criterion = kj.RegressionCriterion(method="cv_ls", degree=1)
solver = GradientDescent(steps=300, rate=0.05, tol=1e-5)

descent = kj.select_bandwidth(
    train, criterion, y=y, solver=solver, n_starts=1
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
        f"steps = {int(result.n_iter)}  "
        f"converged = {bool(result.converged)}"
    )
```

```text
gradient descent value = 1.9194  h = 0.1336  steps = 300  converged = False
L-BFGS           value = 1.9185  h = 0.1606  steps = 5  converged = True
```

The custom solver executes correctly, but it does not reach its requested gradient tolerance within 300 steps. L-BFGS reaches a slightly smaller objective in far fewer iterations here. More importantly, both are one-start solutions. On a nonconvex bandwidth objective, changing the initialization can matter as much as changing the local optimizer.

It is therefore useful to keep those questions separate when diagnosing a poor result. First ask whether the criterion itself has sensible values and gradients. Then ask whether different starts reach different parts of that objective. Only then does changing the local solver address a distinct part of the search.

## Common mistakes

Use this checklist before relying on a custom criterion.

* confirm that its holdout labels exclude exactly the intended observations,
* probe several plausible continuous and categorical bandwidths,
* verify a floating scalar, finite full-tree gradients, JIT compatibility, and chunk equivalence,
* pass the same response, extra data, kernels, and chunk setting used by the intended selection,
* recompute the criterion at the winning bandwidth and compare it with `result.value`, and
* run with multiple starts, inspect all four fields on the winning trajectory, and compare winners from different `n_starts` settings when diagnosing sensitivity.

| Mistake                                                | Symptom                                              | Fix                                                       |
| ------------------------------------------------------ | ---------------------------------------------------- | --------------------------------------------------------- |
| Returning a vector of losses                           | `Gradient only defined for scalar-output functions`  | Reduce to one floating JAX scalar                         |
| Converting the result with `float` or NumPy            | Tracer conversion error                              | Keep traced calculations in `jax.numpy`                   |
| Omitting `kernels` or `chunk`                          | Unexpected keyword argument                          | Accept both keywords                                      |
| Accepting but not forwarding `kernels`                 | Selection silently evaluates the default kernels     | Forward the supplied kernel set                           |
| Storing arrays on a criterion object                   | Frozen object is still unhashable                     | Pass arrays through `y` or `criterion_kwargs`             |
| Routing structural settings through `criterion_kwargs` | Static argument becomes a tracer                     | Store structural settings on a frozen criterion object    |
| Logging a density that can be zero                     | Infinite objective and `nan` gradient                | Define a finite penalty or floor before `log`              |
| Producing a non-finite gradient                        | Finite forward value but unusable search coordinates | Check the full `Bandwidth` gradient at plausible points    |
| Treating `converged=True` as a global guarantee        | Plausible but inferior local solution                | Compare objective values across starts                    |
| Using arbitrary row-order folds                        | Cross-validation does not match the sampling design  | Construct folds appropriate to the data                   |

A callable interface can only enforce computational structure. KernelJax cannot determine whether the scalar represents the statistical target you intended, whether the holdout design is valid for the sample, or whether the selected local optimum is scientifically useful.
