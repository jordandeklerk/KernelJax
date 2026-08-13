# Local polynomial regression

{func}`~kerneljax.local_poly` estimates how the mean of a response moves with mixed
covariates, without committing to a functional form. This page fits one, reads the
bandwidths it learns, predicts along a grid with standard errors, and estimates derivatives.

The setup is the shared wage example from [Working with data](data.md).

```python
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
```

## A first fit

A local linear regression takes one line.

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
  exper         continuous        2.792168
  region        unordered         0.750000
  educ          ordered           0.036096

  Continuous kernel         Gaussian, order 2

  Residual standard error         0.537311
  R-squared                       0.908361

  Selection                          cv_ls
  Criterion value                 0.334907
  Solver iterations                     27
  Converged                           True
```

There are two important choices in that call.

`degree=1` asks for a local linear fit. Instead of fitting one global line to the entire sample, KernelJax fits a small weighted line around each point where the regression function is evaluated.

`"cv_ls"` tells KernelJax to choose the bandwidths by least-squares cross validation. Each observation is left out in turn, and the selected bandwidths are the ones that predict those held-out observations best.

Formally, observations receive weights

$$
K_{h,\lambda}(x, X_i) = \prod_d K_d(x_d, X_{id}),
$$

where each covariate contributes a kernel appropriate to its data type. The local polynomial is then obtained from

$$
\hat\beta(x) = \arg\min_{\beta} \sum_{i=1}^{n} K_{h,\lambda}(x, X_i) \bigl(Y_i - \beta^\top b(X_i-x)\bigr)^2.
$$

The intercept gives the estimated conditional mean,

$$
\hat m(x) = e_1^\top\hat\beta(x),
$$

while the remaining coefficients provide derivatives with respect to the continuous covariates.

The bandwidths are selected by minimizing

$$
(\hat h,\hat\lambda) = \arg\min_{h,\lambda} \frac{1}{n} \sum_{i=1}^{n} \bigl(Y_i-\hat m_{-i}(X_i)\bigr)^2.
$$

You do not need these equations to use the API, but they provide the useful mental model. KernelJax decides how local each fit should be, then estimates the relationship using nearby observations.

## Reading the bandwidths

The fitted bandwidths are worth looking at closely.

```text
Variable      Kind             Bandwidth
exper         continuous        2.792168
region        unordered         0.750000
educ          ordered           0.036096
```

For a continuous variable such as experience, the bandwidth controls the width of the local neighborhood. Larger values pool observations over a wider range, and smaller values make the fit more local.

Categorical bandwidths have a slightly different interpretation. At one extreme, observations from different categories receive little or no weight. At the other, category membership stops affecting the weights at all.

For the default Aitchison-Aitken kernel used with unordered variables, that upper bound is $(c-1)/c$ when there are $c$ levels, so four regions put it at $3/4$.

```python
bound = kj.AitchisonAitken().upper_bound(4)
print(f"region  lam={fit.bandwidth.lam_uno[0]:.6f}  bound={bound:.6f}")
print(f"educ    lam={fit.bandwidth.lam_ord[0]:.6f}")
```

```text
region  lam=0.750000  bound=0.750000
educ    lam=0.036096
```

Region lands exactly on its upper bound. At that value, every region receives the same weight, so region has effectively been smoothed out of the regression.

Education behaves very differently. Its bandwidth is close to zero, so observations at different education levels are kept relatively distinct.

That is exactly what we hoped to recover from the simulated data. Region was generated independently of wage, while education has a real effect. We did not need a separate significance test or variable-selection step to produce this behavior. It emerged from choosing the bandwidths that best predicted held-out observations.

This ability to smooth across uninformative variables is especially useful in nonparametric models, where unnecessary dimensions can otherwise make estimation much harder. [Bandwidth selection](../background/selection.md#what-cross-validation-buys) develops that idea in more detail.

## Predicting on a grid

Next, let us look at the fitted relationship with experience.

Because the model contains three covariates, we cannot draw the entire fitted surface directly. Instead, we vary experience while holding region and education at representative values.

`grid` does this for us.

```python
points = kj.grid(data, vary="exper", n=200)
pred = kj.local_poly(data, wage, fit, at=points, se=True)
exper_grid = points.con[:, 0]

print(
    f"exper varies over {exper_grid.size} points from {exper_grid[0]:.2f} to {exper_grid[-1]:.2f}"
)
print(f"region pinned at {points.uno[0, 0]}, educ pinned at {points.orde[0, 0]}")
```

```text
exper varies over 200 points from 0.01 to 29.92
region pinned at 2, educ pinned at 2
```

By default, `grid` holds continuous variables at their median, unordered variables at their most common level, and ordered variables at the level corresponding to the requested quantile.

Passing the fitted object back to `local_poly` reuses its selected bandwidths and polynomial degree. Setting `se=True` also returns pointwise standard errors for the fitted mean.

```python
fig, ax = plt.subplots()
ax.scatter(exper, wage, s=14, alpha=0.30, color="#8a8f98", label="observations")

ax.fill_between(
    exper_grid,
    pred.mean - 2 * pred.se,
    pred.mean + 2 * pred.se,
    alpha=0.25,
    color="#4c78a8",
    linewidth=0,
    label="±2 se",
)

ax.plot(exper_grid, pred.mean, color="#4c78a8", lw=2.2, label="local linear fit")
ax.set_xlabel("experience (years)")
ax.set_ylabel("wage")
ax.set_title("Fit at the modal region and median education")
ax.legend()
plt.show()
```

![Fit at the modal region and median education](../_static/figures/quickstart-1.svg)

The fitted curve recovers the nonlinear relationship even though we never specified a quadratic, spline basis, or other global functional form.

Uncertainty increases near the edge of the sample, where fewer nearby observations are available to support the local fit.

## Derivatives

A local polynomial gives us more than the fitted level. Its slope also provides an estimate of the derivative with respect to each continuous covariate.

For this example, that means we can estimate how the expected wage changes with another year of experience at each point along the curve.

```python
slope = kj.local_poly(data, wage, fit, at=points, gradient=True)
estimated = slope.grad[:, 0]
truth = 0.35 - 2 * 0.008 * exper_grid
```

Because the data are simulated, we can compare the estimated derivative with the true one.

```python
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

![Estimated marginal effect against the truth](../_static/figures/quickstart-2.svg)

The derivative is recovered well through most of the sample. The largest differences appear near the boundaries, where local derivative estimation is more difficult because observations are available primarily on one side of the evaluation point.
