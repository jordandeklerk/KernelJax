# Density and distribution

{func}`~kerneljax.density` and {func}`~kerneljax.cdf` describe the distribution of a sample directly. A density shows where probability mass is concentrated, while a cumulative distribution function gives the probability of falling at or below an evaluation point. Both use the same mixed-type kernel machinery as regression, but they combine those kernels differently and use criteria suited to the quantity being estimated.

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

Regression asks how the conditional mean of one variable changes with others. Here we instead treat wage itself as the sample whose distribution we want to estimate.

```python
wage_data = kj.MixedData.from_blocks(
    continuous=wage,
    names=("wage",),
)

wage_grid = np.linspace(wage.min(), wage.max(), 200)
wage_points = kj.MixedData.continuous(wage_grid)
```

## Density and distribution estimates

For a mixed sample, KernelJax forms the unscaled product-kernel weight $W_i(x)$, taking one kernel factor per column and leaving the continuous factors without their $1/h_d$ divisors. The [introduction](intro.md#build-on-shared-primitives) writes that product out one column kind at a time. Writing $\mathcal{C}$ for the continuous columns, a density applies those divisors once after forming the product,

$$
\hat f(x)
=
\frac{1}{n \prod_{d\in\mathcal{C}} h_d}
\sum_{i=1}^{n}
W_i(x).
$$

The [Kernel smoothing](../background/smoothing.md#building-a-density-estimator) page develops this estimator from the one-dimensional case, while [Mixed-type data](../background/mixed-data.md#the-product-kernel) introduces the product kernel used across column types.

A cumulative distribution replaces each kernel value with its cumulative counterpart. For supported column types,

$$
\hat F(x)
=
\frac{1}{n}
\sum_{i=1}^{n}
G_{h, \lambda}(x, X_i),
$$

with

$$
G_{h, \lambda}(x, X_i)
=
\prod_d G_d(x_d, X_{id}).
$$

For a continuous Gaussian kernel,

$$
G_d(x_d, X_{id})
=
\Phi\left(
\frac{x_d - X_{id}}{h_d}
\right),
$$

where $\Phi$ is the standard normal distribution function. Integrating the scaled density kernel absorbs the $1/h_d$ factor, so no bandwidth divisor remains in the CDF.

For an ordered categorical column, $G_d$ instead sums the ordered kernel over levels at or below the requested value. Each observation therefore contributes a smooth analogue of the step that it would contribute to the empirical distribution function. An unordered category has no notion of “at or below.” For that reason, {func}`~kerneljax.cdf` supports continuous and ordered columns but not unordered ones.

For the one-dimensional wage sample, both estimators are straightforward.

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

The density shows where wages are concentrated. The distribution gives the estimated fraction of the population lying at or below any particular wage.

The bandwidths happen to be similar here, but they solve different optimization problems. There is no reason for density and distribution bandwidths to agree in general.

## Selecting a density bandwidth

The density above uses likelihood cross-validation through `"cv_ml"`.

At a candidate bandwidth, observation $i$ is removed from its own density estimate,

$$
\hat f_{-i}(X_i)
=
\frac{1}
{(n-1)\prod_{d\in\mathcal{C}} h_d}
\sum_{k\ne i}
W_k(X_i),
$$

and the criterion is the negative leave-one-out log likelihood

$$
\operatorname{CV}_{\mathrm{ML}}(h, \lambda)
=
- \sum_{i=1}^{n}
\log \hat f_{-i}(X_i).
$$

Minimizing this criterion asks the rest of the sample to place high density at each held-out observation. The [Bandwidth selection](../background/selection.md#likelihood-cross-validation) page relates the population version of this objective to Kullback-Leibler divergence.

KernelJax also supports least-squares cross-validation for densities through `"cv_ls"`. The two criteria estimate density fit in different ways, so they need not choose the same bandwidth.

## Selecting a distribution bandwidth

A CDF requires a different objective. Using the CDF value itself as though it were a likelihood does not produce a useful smoothing criterion. Instead, KernelJax compares the smoothed distribution estimate with the indicator that an empirical CDF is trying to estimate.

For evaluation points $x_1, \ldots, x_N$,

$$
\operatorname{CV}_{\mathrm{CDF}}(h, \lambda)
=
\frac{1}{nN}
\sum_{j=1}^{N}
\sum_{i=1}^{n}
\left[
\mathbf{1}\{ X_i \le x_j \}
-
\hat F_{-i}(x_j)
\right]^2,
$$

where

$$
\hat F_{-i}(x_j)
=
\frac{1}{n-1}
\sum_{k \ne i}
G_{h, \lambda}(x_j, X_k).
$$

For multiple continuous or ordered columns, $\mathbf{1}\{ X_i \le x_j \}$ is one only when observation $i$ lies at or below evaluation point $x_j$ in every column. By default, the criterion evaluates this loss at $N=100$ points from {func}`~kerneljax.quantile_grid`. In the one-dimensional wage example, those are 100 sample quantiles spanning the observed distribution.

Each squared term therefore compares a leave-one-out smooth CDF with the step indicator it is meant to reproduce. This plays the same role for distribution estimation that held-out prediction error plays in regression.

## Mixed-type density

Density estimation does not require every column to be continuous. The joint density of the full sample combines a continuous kernel for experience, an unordered kernel for region, and an ordered kernel for education, with all three smoothing parameters selected jointly. More precisely, the result is a density with respect to the product of Lebesgue measure on the continuous coordinates and counting measure on the categorical coordinates.

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

The categorical smoothing parameters have the same interpretation they did in regression. Values near zero preserve distinctions between levels, while values approaching the kernel's upper bound increasingly pool across them. Region is again essentially at the Aitchison–Aitken upper bound of $0.75$, so its levels receive nearly equal weight. Education is at the opposite extreme, with a smoothing parameter effectively equal to zero, so its ordered levels remain distinct in the estimated joint density.

These bandwidths describe the **joint distribution of the covariates**, not their relationship with wage. Their interpretation should therefore remain tied to the density criterion being optimized rather than read as a general statement of variable importance.

## Distribution standard errors

Every {func}`~kerneljax.cdf` fit also carries a pointwise standard error.

KernelJax treats the estimated CDF value as a sample proportion and uses the plug-in expression

$$
\widehat{\operatorname{se}}\left(\hat F(x)\right)
=
\sqrt{\frac{
\hat F(x) \bigl(1 - \hat F(x)\bigr)
}{n}},
$$

where $n$ is the number of training observations. Under this formula, the variance term $\hat F(x)\bigl(1 - \hat F(x)\bigr)$ is largest when $\hat F(x) = 1/2$. The standard error is therefore bounded above by $\frac{1}{2\sqrt{n}}$.

With $n = 300$,

$$
\frac{1}{2\sqrt{300}}
\approx 0.0289.
$$

The standard error is largest near the middle of the estimated distribution and shrinks toward the tails as $\hat F(x)$ approaches zero or one. It is important to read this as the plug-in approximation KernelJax reports. It treats $\hat F(x)$ like an ordinary empirical proportion and does not separately account for smoothing bias, bandwidth-selection uncertainty, or other effects introduced by estimating the smoothed CDF.

We can compare a point near the lower tail with the sample median.

```python
median_idx = np.argmin(np.abs(wage_grid - np.median(wage)))

print(f"se at the low end={float(dist.se[0]):.4f}")
print(f"se near the median={float(dist.se[median_idx]):.4f}")
```

```text
se at the low end=0.0025
se near the median=0.0289
```

Near the sample median, $\hat F(x)$ is close to one half and the standard error approaches its maximum. Near the lower end of the observed range, the estimated CDF is much closer to zero, so the plug-in standard error is correspondingly smaller.
