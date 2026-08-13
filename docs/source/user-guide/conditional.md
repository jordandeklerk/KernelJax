# Conditional estimation

The regression describes how the **mean** of wage moves with the covariates, and the
unconditional density describes wage across the whole sample. The conditional family
combines the two views. {func}`~kerneljax.cdensity` estimates how the full distribution
of a response changes with the covariates, {func}`~kerneljax.cdist` its distribution
function, {func}`~kerneljax.cquantile` its quantiles, and {func}`~kerneljax.cmode` the
most likely level of a categorical response.

The setup is the shared wage example from [Working with data](data.md), with wage as the
response.

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
wage_data = kj.MixedData.from_blocks(continuous=wage, names=("wage",))
wage_grid = np.linspace(wage.min(), wage.max(), 200)
wage_points = kj.MixedData.continuous(wage_grid)
```

## Conditional density

```python
cfit = kj.cdensity(data, wage_data, "cv_ml")
print(kj.summary(cfit))
```

```text
Conditional density estimate

  Observations                         300
  Continuous variables                   1
  Unordered variables                    1
  Ordered variables                      1
  Bandwidth type                    shared

  Response
  Variable      Kind             Bandwidth
  wage          continuous        0.403852

  Conditioning
  Variable      Kind             Bandwidth
  exper         continuous        1.508078
  region        unordered         0.675291
  educ          ordered           0.000000

  Continuous kernel         Gaussian, order 2

  Log likelihood               -217.261520

  Selection                          cv_ml
  Criterion value                 0.955425
  Solver iterations                     30
  Converged                           True
```

A conditional density has two sets of bandwidths. One controls smoothing over the response and the other controls smoothing over the conditioning variables.

As before, passing the fitted object back to the estimator reuses the selected bandwidths. We can therefore fix education and region, choose an experience level, and evaluate the conditional density across the wage grid.

```python
def pay_distribution(years):
    pinned = kj.MixedData.from_blocks(
        continuous=np.full(wage_grid.size, years),
        unordered=np.full(wage_grid.size, 0),
        ordered=np.full(wage_grid.size, 2),
        unordered_levels=4,
        ordered_levels=4,
    )

    return kj.cdensity(data, wage_data, cfit, at_x=pinned, at_y=wage_points).value
```

```python
early = pay_distribution(5.0)
late = pay_distribution(25.0)
```

```python
fig, ax = plt.subplots()
ax.plot(wage_grid, early, color="#4c78a8", lw=2.2, label="5 years")
ax.plot(wage_grid, late, color="#e45756", lw=2.2, label="25 years")
ax.set_xlabel("wage")
ax.set_ylabel("conditional density")
ax.set_title("Distribution of pay, given experience")
ax.legend(title="experience")
plt.show()
```

![Conditional density of wage given experience](../_static/figures/quickstart-4.svg)

The distribution at 25 years of experience is shifted toward higher wages relative to the distribution at 5 years. Unlike the regression, which summarizes this change through the conditional mean, the density lets us see how the entire distribution moves.

## Conditional distribution

{func}`~kerneljax.cdist` estimates the corresponding conditional distribution function with
the same general interface. It refuses likelihood selection, since the likelihood of a CDF
value rewards oversmoothing without bound, and selects with `"cv_ls"` instead, a least
squares criterion scored against the response indicator. A fit selected under `cdensity`
can also be handed to `cdist` directly.

Pinning education and region at the same representative levels as the plot above, the
fitted distribution answers probability questions directly.

```python
pinned = kj.MixedData.from_blocks(
    continuous=np.array([5.0, 25.0]),
    unordered=np.zeros(2, dtype=int),
    ordered=np.full(2, 2),
    unordered_levels=4,
    ordered_levels=4,
)

below_12 = kj.cdist(data, wage_data, cfit, at_x=pinned, at_y=np.full((2, 1), 12.0))
print(f"P(wage <= 12 | 5 years)  = {float(below_12.value[0]):.3f}")
print(f"P(wage <= 12 | 25 years) = {float(below_12.value[1]):.3f}")
```

```text
P(wage <= 12 | 5 years)  = 0.626
P(wage <= 12 | 25 years) = 0.000
```

At five years of experience just under two thirds of the distribution sits below a wage of
twelve, while at twenty five years essentially none of it does, the same shift as the
density plot expressed as a probability.

## Conditional quantiles

A conditional density shows the whole distribution moving. {func}`~kerneljax.cquantile`
reads specific points off that movement by inverting the conditional distribution, so the
fitted value at level `tau` is the wage below which that share of the distribution sits.
The selection carried by the conditional density fit hands straight in.

```python
for tau in (0.25, 0.5, 0.75):
    q = kj.cquantile(data, wage_data, cfit, tau=tau, at_x=pinned)
    print(f"tau={tau:.2f}  5 years={float(q.value[0]):.2f}  25 years={float(q.value[1]):.2f}")
```

```text
tau=0.25  5 years=11.35  25 years=13.82
tau=0.50  5 years=11.79  25 years=14.27
tau=0.75  5 years=12.23  25 years=14.73
```

Every quartile sits higher at twenty five years of experience than at five, which is the
distribution shift from the plot above read off as numbers. Quantile regression through a
conditional distribution needs no loss function choice and no separate model per level.

## Conditional modes

When the response is categorical, the natural summary is the most likely level.
{func}`~kerneljax.cmode` evaluates the conditional density at every response level and
takes the largest, so it acts as a nonparametric classifier. Here it recovers education
from experience and wage, the two things education influences in the simulation.

```python
covariates = kj.MixedData.from_blocks(
    continuous=np.column_stack([exper, wage]), names=("exper", "wage")
)
labels = kj.MixedData.from_blocks(ordered=educ, ordered_levels=4, names=("educ",))

mode = kj.cmode(covariates, labels, "cv_ml")
print(f"correct classification={float(mode.accuracy):.3f}")
```

```text
correct classification=0.790
```

When the fit is evaluated at its own training points, the fit reports the share of
observations whose modal level matches the observed one, the same measure a confusion
matrix would summarize.
