# Quickstart

This page walks through the main API. It assumes KernelJax is
[installed](installation.md) and takes no position on the underlying statistics, which
[Background](background/smoothing.md) covers from first principles.

Every estimator follows the same shape. You hand it training data, a rule for choosing the
smoothing parameters, and optionally the points to evaluate at, and you get back a fit
object carrying the estimate, the selected bandwidth, and the selection diagnostics.

## A first fit

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

The third argument is the bandwidth. Passing a string selects one by cross validation;
passing a `Bandwidth`, a `SelectionResult` or an earlier fit reuses it instead.

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

For a dense evaluation grid, {func}`~kerneljax.grid` builds one by varying a single column
and holding the others at a quantile, and {func}`~kerneljax.quantile_grid` places points at
sample quantiles.

## Standard errors and derivatives

Both are opt-in, since each costs extra work.

```python
kj.local_poly(x, y, fit, at=xs, se=True).se
kj.local_poly(x, y, fit, at=xs, gradient=True).grad
```

```text
[0.04331535 0.0346276  0.05581196 0.03439153 0.05658244]
[ 6.3323874  0.8423864 -7.0216484  1.0322222  7.8586035]
```

The derivative comes from the fitted polynomial coefficients rather than by differencing, so
it requires `degree >= 1`.

## Mixed-type data

Columns of different kinds go in through {meth}`~kerneljax.MixedData.from_blocks`, which
infers the level counts of categorical columns from the data.

```python
rng = np.random.default_rng(2)
region = rng.integers(0, 3, 300)
exper = rng.uniform(0, 30, 300)
wage = 1.5 + 0.05 * exper - 0.0008 * exper**2 + 0.3 * region + rng.normal(0, 0.2, 300)

data = kj.MixedData.from_blocks(continuous=exper, unordered=region)
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
  x1            continuous        7.722590
  x2            unordered         0.012039

  Continuous kernel         Gaussian, order 2

  Residual standard error         0.192066
  R-squared                       0.757928

  Selection                          cv_ls
  Criterion value                 0.039218
  Converged                           True
```

The unordered smoothing parameter came back at 0.012, near zero, because region genuinely
matters here and the criterion declines to pool across regions. Had it run to its upper
bound of $(c-1)/c = 0.667$ instead, that would be the criterion reporting the variable as
irrelevant, which [Bandwidth selection](background/selection.md) explains.

Use `ordered=` for categorical columns whose levels have a natural order, and pass
`unordered_levels` or `ordered_levels` when a level is absent from the sample.

## Densities and distributions

```python
rng = np.random.default_rng(0)
z = np.concatenate([rng.normal(-2, 0.7, 150), rng.normal(2, 1.0, 150)])

dens = kj.density(z, "cv_ml")
dist = kj.cdf(z, "cv_cdf", at=np.array([-2.0, 0.0, 2.0]))
print(np.asarray(dist.value))
```

```text
[0.23390172 0.51609    0.7928456 ]
```

`density` returns the estimate in `.value`, `cdf` returns it in `.value` with standard
errors in `.se`, and both accept `at` and carry the same `.bandwidth` and `.selection`
fields as a regression fit.

## Choosing the selection rule

Each estimator accepts a different set of criteria.

| Estimator | Accepted strings |
| --- | --- |
| {func}`~kerneljax.local_poly` | `"cv_ls"`, `"aic"`, `"normal_reference"` |
| {func}`~kerneljax.density` | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |
| {func}`~kerneljax.cdf` | `"cv_cdf"`, `"normal_reference"` |

`"normal_reference"` is a closed-form plug-in rule and needs no optimization, which makes it
useful as a fast starting point. The rest minimize a cross-validation criterion with L-BFGS
from `n_starts` restarts, defaulting to three.

For finer control, build the criterion yourself and call
{func}`~kerneljax.select_bandwidth`:

```python
criterion = kj.RegressionCriterion(method="cv_ls", degree=1)
result = kj.select_bandwidth(x, criterion, y=y, n_starts=5)
fit = kj.local_poly(x, y, result)
```

## Custom kernels

Every estimator takes `kernels=`, so any kernel can be swapped for another or for one you
write yourself. [Custom kernels](custom-kernels.md) covers how, and what to be careful of.

## Building from primitives

The two primitives underneath every estimator are exported.
{func}`~kerneljax.kweights` returns the kernel weight matrix, and
{func}`~kerneljax.ksum` contracts those weights against a vector without ever forming the
matrix. Nadaraya-Watson is one ratio of two such contractions:

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

Compare the two endpoints against the local linear fit above, which gave `-0.0026` and
`-0.0039` where this gives `0.1515` and `-0.2100`. Both estimate the same function, but the
local constant fit is biased at the boundary while local linear is not, which is exactly the
$O(h)$ against $O(h^2)$ distinction derived in
[Kernel regression](background/regression.md).

Because these are ordinary JAX functions, they compose with {func}`jax.jit`,
{func}`jax.grad` and {func}`jax.vmap` like any other array code.

## Where to go next

- [Background](background/smoothing.md) develops the statistics from first principles.
- Enable [double precision](installation.md#double-precision) before comparing numbers
  against an established implementation, since JAX defaults to 32-bit floats.
- The [GitHub repository](https://github.com/jordandeklerk/KernelJax) has the source code
  and issue tracker.
