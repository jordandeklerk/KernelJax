# Working with data

Every estimator in KernelJax starts from the same object. A {class}`~kerneljax.MixedData`
sample declares which columns are continuous, which are unordered categories, and which are
ordered ones, and that declaration decides which kernel smooths each column.

The pages in this guide share one simulated wage example. Wage rises with experience before
flattening and eventually declining slightly, education shifts wages upward, and region is
generated independently so it carries no information about wage. Simulating the data means
every page can compare what the estimators recover against what is actually there.

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

```python
for name, kind in zip(data.spec.names, data.spec.kinds):
    print(f"{name:8s} {kind.value}")
```

```text
exper    continuous
region   unordered
educ     ordered
```

`MixedData` tells KernelJax how each covariate should be treated. Continuous variables are smoothed according to distance, ordered variables according to how far apart their levels are, and unordered variables according to whether their levels match.

Internally, columns are stored in block order. Continuous columns come first, then unordered, then ordered, and bandwidths follow the same layout.

```python
colors = ["#482475", "#2d708e", "#2ab07f", "#bddf26"]
fig, ax = plt.subplots()
for level, color in enumerate(colors):
    mask = educ == level
    ax.scatter(exper[mask], wage[mask], color=color, s=18, alpha=0.85, label=f"educ {level}")

ax.set_xlabel("experience (years)")
ax.set_ylabel("wage")
ax.set_title("Synthetic wage data")
ax.legend(title="education", loc="lower right", ncol=2)
plt.show()
```

![Synthetic wage data](../_static/figures/quickstart-0.svg)

The nonlinear relationship with experience is already visible, as are the shifts between education levels. We would like the estimator to recover both without specifying either pattern directly.

## Evaluation grids

Estimators evaluate at their training points unless told otherwise, and the `at` argument
takes any sample of the same column layout. Two constructors build the common cases.
{func}`~kerneljax.grid` varies one continuous column over its observed range while pinning
every other column at a representative value.

```python
line = kj.grid(data, vary="exper", n=5)
print(np.asarray(line.con[:, 0]).round(2))
print(f"region pinned at {line.uno[0, 0]}, educ pinned at {line.orde[0, 0]}")
```

```text
[1.000e-02 7.490e+00 1.496e+01 2.244e+01 2.992e+01]
region pinned at 2, educ pinned at 2
```

{func}`~kerneljax.quantile_grid` instead walks every column through its own quantiles at
once, which spreads evaluation points where the data actually sit.

```python
spread = kj.quantile_grid(data, n=5)
print(np.asarray(spread.con[:, 0]).round(2))
```

```text
[1.000e-02 8.980e+00 1.683e+01 2.431e+01 2.992e+01]
```

A bare array also works anywhere a sample does. A one dimensional array is promoted to a
single continuous column, which is why the small examples on other pages can pass `wage`
around without building a `MixedData` first.
