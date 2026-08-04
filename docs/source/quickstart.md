# Quickstart

This page walks through the main API. It assumes KernelJax is
[installed](index.md#installation) and working properly. Every estimator follows the same shape. You hand it training data, a rule for choosing the smoothing parameters, and optionally the points to evaluate at, and you get back a fit object carrying the estimate, the selected bandwidth, and the selection diagnostics.

## A first fit

An estimator is a kernel and a bandwidth rule, and the call says exactly that. The two tabs
below are the same fit written in code and in mathematics.

::::{tab-set}
:class: fit-tabs

:::{tab-item} Python

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
  Converged                           True
```

:::

:::{tab-item} Math

Weight each observation by its distance from $x$, one factor per column.

$$
K_{h,\lambda}(x, X_i) = \prod_{d} K_d(x_d, X_{id})
$$

Fit a polynomial centered at $x$ under those weights, rather than a constant.

$$
\hat\beta(x) = \arg\min_{\beta} \sum_{i=1}^{n}
  K_{h,\lambda}(x, X_i)\, \bigl(Y_i - \beta^\top b(X_i - x)\bigr)^2
$$

Read the estimate off its intercept, and the derivatives off the rest.

$$
\hat m(x) = e_1^\top \hat\beta(x)
$$

Set the widths by how well they predict each observation from the others. This last
line is what `"cv_ls"` names, and it is the part the data answers rather than you.

$$
(\hat h, \hat\lambda) = \arg\min_{h, \lambda} \frac{1}{n} \sum_{i=1}^{n}
  \bigl(Y_i - \hat m_{-i}(X_i)\bigr)^2
$$

:::

::::

The third argument is the bandwidth, and it accepts either a rule or a value. A string names a
rule to run now, while a `Bandwidth`, a `SelectionResult` or an earlier fit reuses a value
already chosen, which is how you evaluate a fitted model somewhere new without paying for
selection twice.

`degree` sets the order of the local polynomial. Degree 0 is a local constant fit, the
Nadaraya-Watson estimator, and is what you get if you say nothing. Degree 1 is local linear,
and it is worth passing explicitly on most problems, since it is the lowest degree that keeps
the boundary bias at the same order as the interior. The
[regression page](background/regression.md) works through why.

## Reading the fit

The fit is a dataclass, and a JAX pytree, so its fields are plain arrays.

```python
fit.bandwidth.h          # continuous bandwidths      -> [0.03601888]
fit.bandwidth.lam_uno    # unordered smoothing parameters
fit.bandwidth.lam_ord    # ordered smoothing parameters
fit.mean                 # fitted values, shape (200,)
fit.r_squared            # 0.947953
fit.residual_se          # 0.174543
fit.selection.converged  # True
fit.selection.value      # the criterion at the optimum
fit.selection.n_iter     # L-BFGS iterations
```

`converged` deserves a habit rather than a glance. It is `True` when the solver stopped
because progress stalled at a finite iterate, and `False` when it exhausted its iteration
budget. It does not claim the minimum is global, and a `False` here is the signal that a
reported bandwidth was never really selected.

`fit.bandwidth.h_axis` is what the summary prints as the bandwidth type. It is `"shared"`
when one bandwidth per column serves every point, which is the default and the only thing the
selectors produce, and `"eval"` or `"train"` when the bandwidth varies across evaluation or
training points instead. The latter two exist for bandwidths you build yourself, and a
`"train"` bandwidth is what an adaptive density estimate needs.

## Choosing the selection rule

Each estimator accepts a different set of criteria.

| Estimator | Accepted strings |
| --- | --- |
| {func}`~kerneljax.local_poly` | `"cv_ls"`, `"aic"`, `"normal_reference"` |
| {func}`~kerneljax.density` | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |
| {func}`~kerneljax.cdf` | `"cv_cdf"`, `"normal_reference"` |

`"normal_reference"` is the odd one out. It is a closed-form plug-in rule that runs no
optimization at all, which makes it a fast starting point and the thing to reach for when you
want a bandwidth without waiting. The rest minimize a cross-validation criterion with L-BFGS
from `n_starts` restarts, defaulting to three. Those restarts are not decoration. The criteria
are not convex, and on a covariate weakly related to the response a single start can settle
into a spurious interior minimum that three starts escape.

For finer control, build the criterion yourself and call
{func}`~kerneljax.select_bandwidth`.

```python
criterion = kj.RegressionCriterion(method="cv_ls", degree=1)
result = kj.select_bandwidth(x, criterion, y=y, n_starts=5)
fit = kj.local_poly(x, y, result)
```

The `SelectionResult` carries the criterion it was built from, which is why the `local_poly`
call on the last line does not need `degree` repeated. Reusing the result reuses the whole
specification, not just the number.

## Evaluating at new points

Pass `at` to evaluate somewhere other than the training sample. Passing the previous fit as
the bandwidth reuses what it selected, so no second selection runs.

```python
xs = np.linspace(0, 1, 5)
pred = kj.local_poly(x, y, fit, at=xs)
print(np.asarray(pred.mean))
```

```text
[-0.00260663  0.92215514 -0.01005817 -1.0386817  -0.00386816]
```

A fit produced with `at` describes points that have no observed response, so it carries no
`r_squared` and no `residual_se`, and {func}`~kerneljax.summary` will decline it. Summarize
the training fit and predict from it separately.

For a dense evaluation grid, {func}`~kerneljax.grid` varies one column and holds the others
fixed, and {func}`~kerneljax.quantile_grid` places points at sample quantiles. What "held
fixed" means depends on the kind of column. A continuous column is pinned at a quantile, an
unordered one at its most common level, and an ordered one at the first level reaching that
quantile. Varying a categorical column ignores the requested row count and emits exactly one
row per level.

## Standard errors and derivatives

Both are opt-in, since each costs extra work.

```python
pred = kj.local_poly(x, y, fit, at=xs, se=True, gradient=True)
print(np.asarray(pred.se))
print(np.asarray(pred.grad).ravel())
```

```text
[0.04331535 0.0346276  0.05581196 0.03439153 0.05658244]
[ 6.3323874  0.8423864 -7.0216484  1.0322222  7.8586035]
```

Two shapes to note. `se` is one number per evaluation point, while `grad` carries one column
per continuous covariate and so comes back as `(5, 1)` here rather than flat, which is why the
example ravels it. And `se` describes the fitted mean only, not the gradient beside it. It
follows the convention [np](https://cran.r-project.org/package=np) uses, a one-pass variance
taken about the constant basis row whatever the degree, so it is not the residual variance of
the polynomial actually fitted.

The derivative comes from the fitted polynomial coefficients rather than by differencing, so
it requires `degree >= 1` and costs nothing beyond reading off a coefficient the fit already
computed.

## Mixed-type data

{class}`~kerneljax.MixedData` is the design matrix, and it is the one object worth
understanding properly, because everything downstream reads its metadata rather than
inspecting the arrays. A bare array handed to an estimator is promoted to a `MixedData` of
continuous columns, which is why the examples above worked without mentioning it. Anything
with a categorical column has to be built explicitly through
{meth}`~kerneljax.MixedData.from_blocks`, the only constructor that validates.

```python
rng = np.random.default_rng(2)
region = rng.integers(0, 3, 300)
exper = rng.uniform(0, 30, 300)
wage = 1.5 + 0.05 * exper - 0.0008 * exper**2 + 0.3 * region + rng.normal(0, 0.2, 300)

data = kj.MixedData.from_blocks(continuous=exper, unordered=region,
                                names=("exper", "region"))
print(kj.summary(kj.local_poly(data, wage, "cv_ls", degree=1)))
```

```text
Local polynomial regression

  Observations                         300
  Continuous variables                   1
  Unordered variables                    1
  Estimator                   local linear
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  exper         continuous        7.722590
  region        unordered         0.012039

  Continuous kernel         Gaussian, order 2

  Residual standard error         0.192066
  R-squared                       0.757928

  Selection                          cv_ls
  Criterion value                 0.039218
  Converged                           True
```

The unordered smoothing parameter came back at 0.012, near zero, because region genuinely
matters here and the criterion declines to pool across regions. Had it come back near its
upper bound of $(c-1)/c = 0.667$ instead, that would be the criterion reporting the variable
as irrelevant, which [Bandwidth selection](background/selection.md#what-cross-validation-buys)
explains. Read "near the bound" rather than "at the bound", since the optimizer approaches it
without arriving.

### Blocks, order, and names

`from_blocks` always assembles columns in block order, continuous first, then unordered, then
ordered, regardless of the order you typed the keywords. Everything downstream inherits that
order. `bandwidth.h` holds one entry per continuous column, `lam_uno` per unordered column,
and the summary table concatenates them the same way, so `h[0]` is the first *continuous*
column and not the first column of whatever you started with.

`names` must be given in that same block order. Nothing checks it against the blocks, so
mislabeled names are accepted in silence and then corrupt every readout that uses them, the
summary table and the `vary=` argument of {func}`~kerneljax.grid` included. Passing names at
all is worth the habit, since the alternative is reading `x1` and `x2` off a summary and
mapping them back by hand.

### Level counts

When you omit `unordered_levels` or `ordered_levels`, the level count is inferred as the
largest code present plus one. That inference is right more often than it looks, and wrong in
one specific way.

```python
print(kj.MixedData.from_blocks(unordered=region).spec.n_levels)        # (3,)
print(kj.MixedData.from_blocks(unordered=region + 1).spec.n_levels)    # (4,)
```

Codes must run from zero. A gap in the middle costs nothing, because the count is still
correct and the missing level simply goes unpopulated. A level absent from the *top* of the
range undercounts, and one-based codes overcount by inventing an empty level zero. Both are
accepted without complaint.

That matters more than a bookkeeping error usually would, because for an unordered column the
level count $c$ enters two things at once. It sets the upper end of the search, at $(c-1)/c$
for the Aitchison-Aitken kernel, and it sets the kernel itself, since a non-matching level
carries weight $\lambda/(c-1)$. So the same $\lambda$ means different smoothing under a
different $c$. Declare the count whenever the sample might not contain every level, which is
the common case for a held-out split or a small subgroup.

Ordered columns are different. Neither shipped ordered kernel consults the level count at all,
so `ordered_levels` affects validation and the rows {func}`~kerneljax.grid` emits, and nothing
else.

### Carrying the spec forward

Training data and evaluation points must agree on column kinds and level counts, and the check
is exact.

```python
held_out = slice(0, 50)

at = kj.MixedData.from_blocks(
    continuous=exper[held_out],
    unordered=region[held_out],
    unordered_levels=data.spec.uno_levels,
    names=data.spec.names,
)
print(at.spec == data.spec)   # True
```

Inferring the levels of the evaluation block independently is the usual way to trip this,
since a subset that happens to omit the top level infers a smaller count and no longer matches.
Passing the training spec's counts through explicitly reproduces it exactly. Grids built by
{func}`~kerneljax.grid` and {func}`~kerneljax.quantile_grid` reuse the training spec already,
so they drop into `at` with nothing extra.

## Densities and distributions

```python
rng = np.random.default_rng(0)
z = np.concatenate([rng.normal(-2, 0.7, 150), rng.normal(2, 1.0, 150)])

at = np.array([-2.0, 0.0, 2.0])
dens = kj.density(z, "cv_ml", at=at)
dist = kj.cdf(z, "cv_cdf", at=at)

print(np.asarray(dens.value))
print(np.asarray(dist.value))
```

```text
[0.25493127 0.02736566 0.17541169]
[0.23390172 0.51609    0.7928456 ]
```

Evaluated at the two modes and the trough between them, the density is high at each mode
and low in the middle, while the distribution rises monotonically through the same points.

`density` returns the estimate in `.value`, `cdf` returns it in `.value` with standard
errors in `.se`, and both accept `at` and carry the same `.bandwidth` and `.selection`
fields as a regression fit. {func}`~kerneljax.summary` handles densities and regressions but
not distribution fits, so read a `cdf` result off its fields directly.

## Custom kernels

Every estimator takes `kernels=`, so any kernel can be swapped for another or for one you
write yourself. [Custom kernels](user-guide/custom-kernels.md) covers how, and what to be
careful of.

## Building from primitives

The two primitives underneath every estimator are exported.
{func}`~kerneljax.kweights` returns the full `(n_eval, n_train)` kernel weight matrix, and
{func}`~kerneljax.ksum` contracts those weights against a vector. Reach for `kweights` when
the matrix itself is the object you want, and for `ksum` when it is an intermediate, since
`ksum` against a single column reduces as it goes and never materializes the matrix. That is
where the memory claim in the library's pitch comes from, and it is worth knowing that it
applies to the single-column case, a many-column `v` falls back to forming the matrix and
multiplying.

Nadaraya-Watson is one ratio of two such contractions.

```python
import jax.numpy as jnp

train = kj.MixedData.from_blocks(continuous=x)
at = kj.MixedData.from_blocks(continuous=xs)

num = kj.ksum(train, fit.bandwidth, jnp.asarray(y)[:, None], at=at)
den = kj.ksum(train, fit.bandwidth, at=at)
print(np.asarray((num / den).ravel()))
```

```text
[ 0.15152554  0.9244799   0.00252233 -1.0318325  -0.20996527]
```

With `v` omitted, `ksum` defaults it to a column of ones, so `den` is the row sums of the
weight matrix as an `(n_eval, 1)` column rather than a flat vector, which is why the ratio is
raveled.

The endpoints are the interesting part. The data were generated from
$y = \sin(2\pi x)$, so the truth at both $x = 0$ and $x = 1$ is exactly zero. Local linear
returned $-0.0026$ and $-0.0039$ there, while this local constant fit returns $0.1515$ and
$-0.2100$. That gap is the boundary bias of the Nadaraya-Watson estimator, which is $O(h)$ at
an edge against $O(h^2)$ in the interior, and it is exactly the defect local linear was
introduced to remove. Local linear is not unbiased at the boundary either, it simply keeps the
same order there as inside, as [Kernel regression](background/regression.md) sets out.

When memory is the binding constraint rather than time, `chunk` bounds the working set by
processing the weight matrix in blocks. It takes an `(eval, train)` pair, or a bare integer to
chunk the evaluation axis alone, and it changes nothing about the answer.

## Composing with JAX

The criteria are not sealed inside the selection routine. Each is an ordinary JAX function
of the data and a bandwidth, so it differentiates, compiles and vectorizes like any other
array code. This is what lets a bandwidth be learned as part of something larger rather than
fixed beforehand.

```python
import jax
import jax.numpy as jnp

train = kj.MixedData.from_blocks(continuous=x)
bw = kj.normal_reference(train, kj.KernelSet())

print(f"{float(kj.cv_ls_regression(train, bw, y=jnp.asarray(y), degree=1)):.6f}")
print(np.asarray(jax.grad(
    lambda b: kj.cv_ls_regression(train, b, y=jnp.asarray(y), degree=1))(bw).h))
```

```text
0.047626
[0.4289686]
```

Note the `degree=1`, which the criterion needs stated explicitly and defaults to 0 without.
The gradient comes back shaped like the bandwidth itself, one entry per continuous column,
because `Bandwidth` is a registered pytree and differentiating a scalar function of it returns
the same structure. `select_bandwidth` does not hand this gradient to L-BFGS directly. It
reparameterizes first, mapping each bandwidth to an unconstrained coordinate through a
softplus and each categorical parameter through a scaled logistic, and the solver
differentiates through that map instead, which is how the box constraints are enforced without
a constrained solver.

Evaluating the criterion across many bandwidths at once is a {func}`jax.vmap` away.

```python
def criterion_at(h):
    b = kj.Bandwidth(h=jnp.array([h]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
    return kj.cv_ls_regression(train, b, y=jnp.asarray(y), degree=1)

grid = jnp.array([0.02, 0.036, 0.05, 0.10, 0.20])
print(np.asarray(jax.jit(jax.vmap(criterion_at))(grid)))
```

```text
[0.03553576 0.03430145 0.03484762 0.04650113 0.10823295]
```

The criterion is smallest at 0.036, which is the bandwidth selection found, and the surface is
visibly flat on one side of it and steep on the other. That asymmetry is the usual shape of a
cross-validation criterion in the bandwidth and the reason a plot of it is worth more than a
single number.

## Where to go next

- [Background](background/smoothing.md) develops the statistics from first principles.
- [Custom kernels](user-guide/custom-kernels.md) covers writing your own.
- Enable [double precision](index.md#double-precision) before comparing numbers
  against an established implementation, since JAX defaults to 32-bit floats.
- The [GitHub repository](https://github.com/jordandeklerk/KernelJax) has the source code
  and issue tracker.
