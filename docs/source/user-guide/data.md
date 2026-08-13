# Working with data

KernelJax needs to know how each covariate should be treated before it can choose the corresponding kernel. For mixed samples, that information lives in a {class}`~kerneljax.MixedData` object, which records which columns are continuous, which are unordered categories, and which are ordered categories.

The pages in this guide share one simulated wage example. Wage rises with experience before flattening and declining slightly, education shifts wages upward, and region is generated independently so it carries no information about wage. Because the data-generating process is known, later pages can compare what the estimators recover with what is actually there.

Write $Y_i$ for wage, $e_i$ for years of experience, $s_i \in \{0,1,2,3\}$ for education level, and $r_i \in \{0,1,2,3\}$ for region. The sample is generated from

$$
\begin{aligned}
Y_i &= m(e_i, s_i, r_i) + \varepsilon_i \\
m(e, s, r) &= 8 + 0.35e - 0.008e^2 + 1.2s \\
\varepsilon_i \mid e_i, s_i, r_i &\sim \mathcal{N}(0, 0.6^2).
\end{aligned}
$$

The function $m$ is the conditional mean

$$
m(e, s, r) = \mathbb{E}[Y \mid E = e, S = s, R = r]
$$

from [Kernel smoothing](../background/smoothing.md#what-do-we-mean-by-nonparametric), written out for this particular example.

Two features of this conditional mean will matter throughout the guide. First, its slope with respect to experience is

$$
\frac{\partial m}{\partial e} = 0.35 - 0.016e,
$$

which is zero at $e = \frac{0.35}{0.016} = 21.875$. The wage curve therefore rises with experience, flattens near twenty-two years, and declines gently afterward.

Second, $r$ does not appear on the right-hand side of $m(e,s,r)$. Holding experience and education fixed, $m(e, s, r_1) = m(e, s, r_2)$ for any two regions $r_1$ and $r_2$. Region is therefore irrelevant to the conditional mean even though it remains part of the sample. No estimator in the guide is told either of these facts.

The code below generates the sample and declares the role of each covariate.

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

The resulting `MixedData` object keeps that declaration with the sample.

```python
for name, kind in zip(data.spec.names, data.spec.kinds):
    print(f"{name:8s} {kind.value}")
```

```text
exper    continuous
region   unordered
educ     ordered
```

Those distinctions determine how KernelJax compares observations along each column. Continuous variables are smoothed according to distance, ordered variables according to distance between their levels, and unordered variables according to whether their levels match.

Categorical columns use zero-based integer codes. A column with $c$ levels therefore uses codes in $\{0,1,\ldots,c-1\}$. The `unordered_levels` and `ordered_levels` arguments record the corresponding values of $c$. They can be omitted when the level counts are recoverable from the observed codes, but passing them explicitly is useful when a valid level may be absent from the sample. For example, a four-level variable should still be declared with `ordered_levels=4` if level `3` happens not to appear in a particular dataset.

The declaration also determines the layout used when the underlying arrays are inspected directly. Internally, columns are stored in block order, continuous first, then unordered, then ordered. Names, level counts, and bandwidths follow the same layout.

The statistical structure built into the example is already visible in the data.

```python
colors = ["#482475", "#2d708e", "#2ab07f", "#bddf26"]
fig, ax = plt.subplots()

for level, color in enumerate(colors):
    mask = educ == level
    ax.scatter(
        exper[mask],
        wage[mask],
        color=color,
        s=18,
        alpha=0.85,
        label=f"educ {level}",
    )

ax.set_xlabel("experience (years)")
ax.set_ylabel("wage")
ax.set_title("Synthetic wage data")
ax.legend(title="education", loc="lower right", ncol=2)
plt.show()
```

![Synthetic wage data](../_static/figures/quickstart-0.svg)

The nonlinear relationship with experience is visible, as are the shifts between education levels. Later estimators will be asked to recover both without being given either pattern directly. Region provides a different test because it is present as an unordered covariate even though the true conditional mean does not depend on it.

## Purely continuous data

`MixedData` is necessary when the sample contains categorical columns because KernelJax needs their kinds and level counts to interpret them correctly. Purely continuous samples have no such ambiguity, so they can use a shorter form.

You do not need to construct a `MixedData` object when every column is continuous. A raw array is interpreted as a purely continuous sample, and a one-dimensional array is promoted to a single continuous column. That is why small examples elsewhere in the guide can pass arrays such as `wage` directly.

Raw arrays are not interchangeable with `MixedData` when categorical columns are part of the sample. In that case, the column kinds and level counts are needed to interpret the data, so evaluation points must preserve the corresponding `MixedData` structure.

## Evaluation grids

Constructing the training sample is only one part of working with data. Estimators evaluate at their training points unless told otherwise, but predictions, derivatives, and other summaries are often easier to interpret along a deliberately chosen set of evaluation points.

The `at` argument accepts another sample with the same column layout. KernelJax provides two helpers for constructing the most common kinds of evaluation paths.

{func}`~kerneljax.grid` is useful when one column should vary while the others remain fixed. It sweeps the chosen column and holds every other column at a representative value.

For a continuous column, the default `trim=0` produces $n$ equally spaced points between its smallest and largest observed values,

$$
x_j = x_{\min} + \frac{j-1}{n-1} \left(x_{\max} - x_{\min}\right),
\qquad
j=1,\ldots,n.
$$

A positive `trim` replaces those endpoints with inner sample quantiles. For example, `trim=0.1` sweeps from the tenth to the ninetieth percentile. A negative value extends the sweep beyond the observed range. If the swept column is categorical, `grid` instead visits each of its levels once and ignores `n`.

The remaining columns need values of their own. Write $q$ for the `quantile` argument, which defaults to $1/2$, $\hat F_d^{-1}$ for the sample quantile function of continuous column $d$, and $\hat p_{ds}$ for the observed share at level $s$ of categorical column $d$. The pinned value is

$$
\begin{aligned}
&\hat F_d^{-1}(q) && \text{continuous}, \\
&\arg\max_s \hat p_{ds} && \text{unordered}, \\
&\min\left\{ s : \sum_{\ell \le s}\hat p_{d\ell} \ge q \right\} && \text{ordered}.
\end{aligned}
$$

For continuous columns, KernelJax uses the sample quantile with linear interpolation between order statistics. An unordered column is held at its most common level, while an ordered column uses the first level whose cumulative sample share reaches $q$.

For the wage sample, sweeping experience therefore gives a one-dimensional path through the mixed covariate space.

```python
line = kj.grid(data, vary="exper", n=5)

print(np.asarray(line.con[:, 0]).round(2))
print(f"region pinned at {line.uno[0, 0]}, educ pinned at {line.orde[0, 0]}")
```

```text
[1.000e-02 7.490e+00 1.496e+01 2.244e+01 2.992e+01]
region pinned at 2, educ pinned at 2
```

The two categorical columns happen to be pinned at the same level, but for different reasons. Education has sample shares $0.233$, $0.233$, $0.283$, and $0.250$ across its four levels, so its cumulative shares are $0.233$, $0.467$, $0.750$, and $1$, and level $2$ is the first to reach the default quantile $q=0.5$. Region is unordered, so `grid` uses the mode, and level $2$ happens to be the most common region with 84 of the 300 observations.

### Following the sample quantiles

A one-column sweep is not always the path of interest. When several continuous or ordered columns should move together through their marginal distributions, {func}`~kerneljax.quantile_grid` constructs a different sequence of evaluation points.

Rather than spacing one continuous variable evenly over its range, it evaluates the continuous and ordered columns along a shared ladder of probabilities while holding unordered columns at their modes.

The probabilities are

$$
p_j = \frac{j-1}{n-1},
\qquad
j=1,\ldots,n.
$$

For each $p_j$, the $d$th component of evaluation point $\mathbf{x}^{(j)}$ is

$$
x_d^{(j)} =
\begin{cases}
\hat F_d^{-1}(p_j), & d\text{ continuous}, \\[6pt]
\arg\max_s \hat p_{ds}, & d\text{ unordered}, \\[6pt]
\displaystyle
\min\left\{ s : \sum_{\ell \le s}\hat p_{d\ell} \ge p_j \right\}, & d\text{ ordered}.
\end{cases}
$$

Each row therefore pairs the marginal quantiles of the columns at one common probability. The result always has $n$ rows regardless of the number of columns.

This is not a Cartesian product grid. It traces one path through the marginal distributions, so it also should not be interpreted as reproducing their empirical joint distribution. A Cartesian grid grows multiplicatively with the number of dimensions, while `quantile_grid` remains a one-dimensional sequence of $n$ evaluation points.

For the current sample, the distinction from `grid` is already visible in the experience values.

```python
spread = kj.quantile_grid(data, n=5)
print(np.asarray(spread.con[:, 0]).round(2))
```

```text
[1.000e-02 8.980e+00 1.683e+01 2.431e+01 2.992e+01]
```

The two vectors differ because {func}`~kerneljax.grid` and {func}`~kerneljax.quantile_grid` space the continuous column differently. `grid` places experience evenly over its observed range, so its middle point is the range midpoint at $14.96$. `quantile_grid` places it evenly in probability, so its middle point is the sample median at $16.83$. They agree at the endpoints because the sample quantiles at probabilities zero and one are the smallest and largest observed values.
