# Quickstart

KernelJax estimates relationships without committing to a functional form. This page works one
problem end to end on a wage dataset with a continuous covariate, an ordered one and an unordered
one.

## The data

How does pay vary with experience, and how much of what we see is really about education or region
instead? A parametric answer commits to a shape up front, a quadratic in experience and dummies
for the rest, and then argues about that shape for the whole analysis. The nonparametric answer
commits to nothing and lets the data say how sharply the surface bends.

We simulate it so we know the truth. Wage rises with experience and flattens off, education shifts
the whole curve, and region is drawn independently of everything. The estimator is not told any of
this.

```python
import jax
import matplotlib.pyplot as plt
import numpy as np

import kerneljax as kj

rng = np.random.default_rng(0)
n = 300
exper = rng.uniform(0, 30, n)
educ = rng.integers(0, 4, n)
region = rng.integers(0, 4, n)
wage = 8 + 0.35 * exper - 0.008 * exper**2 + 1.2 * educ + rng.normal(0, 0.6, n)

data = kj.MixedData.from_blocks(
    continuous=exper,
    unordered=region,
    ordered=educ,
    unordered_levels=4,
    ordered_levels=4,
    names=("exper", "region", "educ"),
)
for name, kind in zip(data.spec.names, data.spec.kinds):
    print(f"{name:8s} {kind.value}")
```

```text
exper    continuous
region   unordered
educ     ordered
```

Declaring the kinds is the one piece of structure the estimator needs, since it decides which
kernel each column gets. A continuous column is smoothed by distance, an ordered one by how many
levels apart two values are, and an unordered one only by whether the levels match. Columns come
back in block order, continuous then unordered then ordered, whatever order you passed them, and
every bandwidth vector follows that layout.

```python
colors = ["#482475", "#2d708e", "#2ab07f", "#bddf26"]

fig, ax = plt.subplots()
for level, color in enumerate(colors):
    mask = educ == level
    ax.scatter(exper[mask], wage[mask], color=color, s=18, alpha=0.85,
               label=f"educ {level}")
ax.set_xlabel("experience (years)")
ax.set_ylabel("wage")
ax.set_title("Synthetic wage data")
ax.legend(title="education", loc="lower right", ncol=2)
plt.show()
```

![Synthetic wage data](_static/figures/quickstart-0.svg)

The bend and the education bands are obvious, and neither is something we want to have to assume.

## A first fit

One call does the whole thing. Every observation gets a weight built from one factor per column,
close observations counting for more than distant ones,

$$
K_{h,\lambda}(x, X_i) = \prod_{d} K_d(x_d, X_{id}),
$$

and at each point those weights define a small least squares problem, fitting a polynomial
centered there rather than a constant,

$$
\hat\beta(x) = \arg\min_{\beta} \sum_{i=1}^{n}
  K_{h,\lambda}(x, X_i)\, \bigl(Y_i - \beta^\top b(X_i - x)\bigr)^2 .
$$

The estimate is that polynomial's intercept, $\hat m(x) = e_1^\top \hat\beta(x)$, and its
remaining coefficients are the derivatives up to a factorial, which is why they come free.

None of that says how wide the weights should be. They are chosen by holding each observation out
and asking which values predict it best,

$$
(\hat h, \hat\lambda) = \arg\min_{h, \lambda} \frac{1}{n} \sum_{i=1}^{n}
  \bigl(Y_i - \hat m_{-i}(X_i)\bigr)^2 ,
$$

which is the line `"cv_ls"` names, and the third argument is where that choice goes. A string
selects now, while a `Bandwidth`, a `SelectionResult` or an earlier fit reuses a value already
chosen. `degree` defaults to 0, the Nadaraya-Watson estimator, and we ask for 1, local linear,
which keeps the boundary bias at the same order as the interior.

```python
fit = kj.local_poly(data, wage, "cv_ls", degree=1)
print(kj.summary(fit))
```

```text
Local polynomial regression

  Observations                         300
  Continuous variables                   1
  Unordered variables                    1
  Ordered variables                      1
  Estimator                   local linear
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  exper         continuous        2.792196
  region        unordered         0.749997
  educ          ordered           0.036101

  Continuous kernel         Gaussian, order 2

  Residual standard error         0.537311
  R-squared                       0.908360

  Selection                          cv_ls
  Criterion value                 0.334907
  Converged                           True
```

Each column has its own smoothing parameter, chosen jointly rather than one at a time, and those
three numbers are the entire fitted model. Everything below them is diagnostics.

## What the bandwidths say

For a categorical column the smoothing parameter runs between two extremes. At zero, only
observations sharing the level count, which is the same as splitting the sample into cells and
throwing the rest away. At its upper bound every level is weighted identically, a constant factor
that cancels from the ratio and leaves the column out of the model.

So the selected value is a verdict on the variable.

```python
bound = kj.AitchisonAitken().upper_bound(4)
print(f"region  lam={fit.bandwidth.lam_uno[0]:.6f}  bound={bound:.6f}")
print(f"educ    lam={fit.bandwidth.lam_ord[0]:.6f}")
```

```text
region  lam=0.749997  bound=0.750000
educ    lam=0.036101
```

Region came back sitting on its bound, so cross validation removed it. Education went the other
way, close enough to zero that its levels are kept almost separate. No test was run to arrive at
that distinction. It falls out of asking which bandwidths predict held-out observations best,
because an irrelevant column can only add variance, so the criterion pools it away.

Including a covariate you are unsure about is ordinarily expensive, since every extra dimension
slows the rate at which a nonparametric estimator learns. Here a column that carries nothing costs
nothing, which [Bandwidth selection](background/selection.md#what-cross-validation-buys) sets out
properly.

## Reading the fit

Before interpreting a fit, check that there is one. Bandwidth selection is nonconvex, so it can
fail, and it fails quietly rather than by raising.

```python
print(jax.tree.map(lambda a: a.shape, fit.bandwidth))
print(f"converged={fit.selection.converged}  n_iter={fit.selection.n_iter}  "
      f"criterion={fit.selection.value:.6f}")
print(f"degree={fit.degree}  r_squared={fit.r_squared:.4f}  "
      f"residual_se={fit.residual_se:.4f}")
```

```text
Bandwidth(h=(1,), lam_uno=(1,), lam_ord=(1,), h_axis='shared')
converged=True  n_iter=38  criterion=0.334907
degree=1  r_squared=0.9084  residual_se=0.5373
```

`converged` is `True` when the solver stopped because progress stalled at a finite iterate and
`False` when it ran out of iterations, the signal that the reported bandwidth was never really
selected. It does not promise the minimum is global. The fit is also a pytree, so mapping over it
shows how it is laid out, and that is what lets {func}`jax.grad` differentiate through the
criteria later.

## Predicting on a grid

A fitted surface over three covariates cannot be drawn, so we vary one and hold the others
somewhere representative. The curve you get is conditional on where the others were pinned.
{func}`~kerneljax.grid` pins a continuous column at a quantile, an unordered one at its most
common level, and an ordered one at the level reaching that quantile, and it carries the training
spec forward so the result drops straight into `at`.

```python
points = kj.grid(data, vary="exper", n=200)
pred = kj.local_poly(data, wage, fit, at=points, se=True)
exper_grid = points.con[:, 0]

print(f"exper varies over {exper_grid.size} points "
      f"from {exper_grid[0]:.2f} to {exper_grid[-1]:.2f}")
print(f"region pinned at {points.uno[0, 0]}, educ pinned at {points.orde[0, 0]}")

fig, ax = plt.subplots()
ax.scatter(exper, wage, s=14, alpha=0.30, color="#8a8f98", label="observations")
ax.fill_between(exper_grid, pred.mean - 2 * pred.se, pred.mean + 2 * pred.se,
                alpha=0.25, color="#4c78a8", linewidth=0, label="±2 se")
ax.plot(exper_grid, pred.mean, color="#4c78a8", lw=2.2, label="local linear fit")
ax.set_xlabel("experience (years)")
ax.set_ylabel("wage")
ax.set_title("Fit at the modal region and median education")
ax.legend()
plt.show()
```

```text
exper varies over 200 points from 0.01 to 29.92
region pinned at 2, educ pinned at 2
```

![Fit at the modal region and median education](_static/figures/quickstart-1.svg)

The curve recovers the concave shape without ever having been told to look for one, and the band
widens where the data thin out.

## Derivatives

In a parametric wage model the return to experience is a coefficient. Here it is a function, and
reading it off is where local linear earns its extra degree over local constant. Fitting a line
rather than a level at each point makes the slope a derivative estimate the fit has already
computed.

```python
slope = kj.local_poly(data, wage, fit, at=points, gradient=True)
estimated = slope.grad[:, 0]
truth = 0.35 - 2 * 0.008 * exper_grid

fig, ax = plt.subplots()
ax.plot(exper_grid, estimated, color="#e45756", lw=2.2, label="estimated")
ax.plot(exper_grid, truth, ls="--", lw=1.6, color="#8a8f98", label="truth")
ax.axhline(0, lw=0.8, color="#8a8f98", alpha=0.5)
ax.set_xlabel("experience (years)")
ax.set_ylabel("d wage / d experience")
ax.set_title("Estimated marginal effect against the truth")
ax.legend()
plt.show()
```

![Estimated marginal effect against the truth](_static/figures/quickstart-2.svg)

The estimate tracks the truth through the interior and pulls away at both ends. That is not a bug
to tune out. Near an edge there are neighbors on one side only, so the slope comes from a lopsided
window, and the derivative feels it before the level does.

## Densities and distributions

Regression is one question about a sample. The distribution of pay is another, and it takes the
same machinery, since a density is the same kernel weights contracted against a column of ones
instead of a response.

```python
wage_data = kj.MixedData.continuous(wage)
wage_grid = np.linspace(wage.min(), wage.max(), 200)
wage_points = kj.MixedData.continuous(wage_grid)

dens = kj.density(wage_data, "cv_ml", at=wage_points)
dist = kj.cdf(wage_data, "cv_cdf", at=wage_points)
print(f"density       h={dens.bandwidth.h[0]:.6f}")
print(f"distribution  h={dist.bandwidth.h[0]:.6f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6))
left.hist(wage, bins=30, density=True, color="#8a8f98", alpha=0.35)
left.plot(wage_grid, dens.value, color="#4c78a8", lw=2.2)
left.set_title("density")
left.set_xlabel("wage")
right.plot(wage_grid, dist.value, color="#54a24b", lw=2.2)
right.set_title("distribution")
right.set_xlabel("wage")
fig.tight_layout()
plt.show()
```

```text
density       h=0.454527
distribution  h=0.443713
```

![Density and distribution of wage](_static/figures/quickstart-3.svg)

## Choosing the selection rule

Those two calls named `"cv_ml"` and `"cv_cdf"` where the regression named `"cv_ls"`. Which
criterion you minimize is a modelling decision rather than a setting.

| Estimator | Accepted strings |
| --- | --- |
| {func}`~kerneljax.local_poly` | `"cv_ls"`, `"aic"`, `"normal_reference"` |
| {func}`~kerneljax.density` | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |
| {func}`~kerneljax.cdf` | `"cv_cdf"`, `"normal_reference"` |

`"normal_reference"` is the cheap one, a closed-form rule that assumes roughly normal data and
runs no optimization. It is the quickest way to get a fit in front of you, and here it lands near
the cross validated answer.

```python
quick = kj.local_poly(data, wage, "normal_reference", degree=1)
print(f"normal reference  h={quick.bandwidth.h[0]:.6f}")
print(f"cross validated   h={fit.bandwidth.h[0]:.6f}")
```

```text
normal reference  h=3.026807
cross validated   h=2.792196
```

The rest minimize a criterion with L-BFGS from `n_starts` restarts, defaulting to three. These
criteria are not convex, and while the restarts usually agree, when they disagree they disagree
completely.

```python
noise_rng = np.random.default_rng(9)
x_noise = noise_rng.uniform(size=150)
y_noise = noise_rng.normal(0, 1, 150)

criterion = kj.RegressionCriterion(method="cv_ls", degree=1)
one_start = kj.select_bandwidth(x_noise, criterion, y=y_noise, n_starts=1)
three_starts = kj.select_bandwidth(x_noise, criterion, y=y_noise, n_starts=3)
print(f"1 start   h={one_start.bandwidth.h[0]:8.4f}  criterion={one_start.value:.6f}")
print(f"3 starts  h={three_starts.bandwidth.h[0]:8.4f}  criterion={three_starts.value:.6f}")
```

```text
1 start   h=  0.1392  criterion=0.989901
3 starts  h= 41.8218  criterion=0.984698
```

On that draw `y_noise` is unrelated to `x_noise`, so the honest answer is a very large bandwidth,
the continuous analogue of what happened to region. A single start settles two orders of magnitude
short of it, at a worse criterion value, and reports success. This is why `converged` alone is not
enough, and why one start is rarely worth the time it saves.

## Under the hood

Underneath every estimator is a single operation, contracting kernel weights against a vector, and
it is exported. The Nadaraya-Watson estimator is a ratio of two of them, the response in the
numerator and a column of ones in the denominator, which is how you would build a method the
library does not ship.

```python
numerator = kj.ksum(data, fit.bandwidth, wage[:, None], at=points)
denominator = kj.ksum(data, fit.bandwidth, at=points)
nadaraya_watson = (numerator / denominator).ravel()

local_constant = kj.local_poly(data, wage, fit.bandwidth, at=points)
gap = abs(nadaraya_watson - local_constant.mean).max()
print(f"largest gap to local_poly={gap:.3e}")
```

```text
largest gap to local_poly=1.526e-05
```

The two are the same estimator, so the gap is float32 arithmetic.

The criteria are open in the same way, and they are differentiable. A bandwidth chosen by a
derivative-free search has to be settled before anything else happens, whereas a criterion with a
gradient can sit inside a larger objective and be learned along with everything else in it.

```python
def cv_ls(bandwidth):
    return kj.cv_ls_regression(data, bandwidth, y=wage, degree=1)


at_plug_in = jax.grad(cv_ls)(quick.bandwidth)
at_selected = jax.grad(cv_ls)(fit.bandwidth)

for label, grad in [("plug-in", at_plug_in), ("selected", at_selected)]:
    print(f"{label:9s} d/dh={grad.h[0]:+.3e}  d/dlam_uno={grad.lam_uno[0]:+.3e}  "
          f"d/dlam_ord={grad.lam_ord[0]:+.3e}")
```

```text
plug-in   d/dh=-9.776e-03  d/dlam_uno=-1.576e-01  d/dlam_ord=+1.888e+00
selected  d/dh=+1.799e-07  d/dlam_uno=-6.443e-05  d/dlam_ord=+1.384e-05
```

The gradient comes back shaped like the bandwidth, one entry per column with the categorical
parameters included. At the plug-in bandwidth it points somewhere; at the one cross validation
chose it has all but vanished.

## Where to go next

- [Custom kernels](user-guide/custom-kernels.md) and
  [Custom criteria](user-guide/custom-criteria.md) cover replacing the weighting scheme and
  the rule that picks the bandwidth.
- [Background](background/smoothing.md) derives all of it from first principles.
- The [API reference](api.md) documents every exported object.
- Enable [double precision](index.md#double-precision) before comparing numbers against an
  established implementation, since JAX defaults to 32-bit floats.
