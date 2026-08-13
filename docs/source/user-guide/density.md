# Density and distribution

{func}`~kerneljax.density` and {func}`~kerneljax.cdf` describe a sample on its own
terms, where its mass concentrates and what fraction lies below any cut. Both are built
from the same smoothing machinery as the regression, and both select their bandwidths from
the data.

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

Regression asks how the conditional mean of a response changes with its covariates. KernelJax also exposes density and distribution estimators built from the same smoothing machinery.

First, put wage into a `MixedData` object and create a grid over its range.

```python
wage_data = kj.MixedData.from_blocks(continuous=wage, names=("wage",))
wage_grid = np.linspace(wage.min(), wage.max(), 200)
wage_points = kj.MixedData.continuous(wage_grid)
```

A density and CDF differ mainly in the criterion used to select their bandwidths.

```python
dens = kj.density(wage_data, "cv_ml", at=wage_points)
dist = kj.cdf(wage_data, "cv_cdf", at=wage_points)
print(f"density       h={dens.bandwidth.h[0]:.6f}")
print(f"distribution  h={dist.bandwidth.h[0]:.6f}")
```

```text
density       h=0.454527
distribution  h=0.443713
```

```python
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

![Density and distribution of wage](../_static/figures/quickstart-3.svg)

The density tells us where wages are concentrated. The distribution function tells us what fraction lies below any given wage.

## Mixed-type density

Nothing restricts a density to continuous columns. The joint density of the full sample
smooths experience by distance, region by level match, and education by level distance,
and every smoothing parameter is selected jointly.

```python
joint = kj.density(data, "cv_ml")
print(kj.summary(joint))
```

```text
Mixed-type density estimate

  Observations                         300
  Continuous variables                   1
  Unordered variables                    1
  Ordered variables                      1
  Bandwidth type                    shared

  Variable      Kind             Bandwidth
  exper         continuous        1.493567
  region        unordered         0.749997
  educ          ordered           0.000000

  Continuous kernel         Gaussian, order 2

  Log likelihood              -1844.200806

  Selection                          cv_ml
  Criterion value              1878.115845
  Solver iterations                     21
  Converged                           True
```

The categorical smoothing parameters read the same way as in the regression. A value near
zero keeps a variable's levels distinct, and a value at the kernel's upper bound pools them
completely.

## Distribution standard errors

Every distribution fit carries pointwise standard errors, no extra argument needed, and
they are largest near the median where the binomial variance of an empirical proportion
peaks.

```python
dist_se = kj.cdf(wage_data, "cv_cdf", at=wage_points)
middle = wage_grid.size // 2
print(f"se at the low end={float(dist_se.se[0]):.4f}")
print(f"se near the median={float(dist_se.se[middle]):.4f}")
```

```text
se at the low end=0.0025
se near the median=0.0273
```
