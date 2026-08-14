# Conditional estimation

Regression describes how the **conditional mean** of a response changes with its covariates. An unconditional density describes the response across the sample as a whole. Conditional estimation combines those two views by allowing the entire response distribution to change with the covariates.

{func}`~kerneljax.cdensity` estimates a conditional density, {func}`~kerneljax.cdist` its cumulative distribution, and {func}`~kerneljax.cquantile` selected quantiles of that distribution. When the response is categorical, {func}`~kerneljax.cmode` instead returns its most likely level.

The setup is the shared wage example from [Working with data](data.md), with wage now treated as the response.

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

wage_data = kj.MixedData.from_blocks(
    continuous=wage,
    names=("wage",),
)

wage_grid = np.linspace(wage.min(), wage.max(), 200)
wage_points = kj.MixedData.continuous(wage_grid)
```

Here `data` contains the conditioning variables and `wage_data` contains the response. Keeping those two samples separate matters throughout this page because conditional estimators smooth in both spaces. The conditioning variables determine which observations are locally relevant, while the response sample determines the distribution being estimated. `wage_points` provides the response values where we will evaluate that distribution.

## Conditional density

A conditional density is the most direct way to describe how the full response distribution changes with the conditioning variables. We start by fitting one to wage given experience, region, and education.

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
  wage          continuous        0.403857

  Conditioning
  Variable      Kind             Bandwidth
  exper         continuous        1.507978
  region        unordered         0.675305
  educ          ordered           0.000000

  Continuous kernel         Gaussian, order 2

  Log likelihood               -217.261276

  Selection                          cv_ml
  Criterion value                 0.955425
  Solver iterations                     32
  Converged                           True
```

Unlike the unconditional estimators from the previous page, a conditional fit has two bandwidth blocks. One controls smoothing across the conditioning variables $x$, and the other controls smoothing across the response $y$.

Write $W_i^x(x) = K_{h_x, \lambda_x}(x, X_i)$ for the unscaled product-kernel weight contributed by the conditioning variables and $W_i^y(y) = K_{h_y, \lambda_y}(y, Y_i)$ for the corresponding response-kernel value.

If $\mathcal{C}_y$ indexes the continuous response columns, KernelJax estimates

$$
\hat{f}(y\mid x) =
\frac{
\displaystyle\sum_{i=1}^n
W_i^x(x) W_i^y(y)
}{
\displaystyle
\left(\prod_{d\in\mathcal{C}_y} h_{y,d}\right)
\sum_{i=1}^n W_i^x(x)
}
$$

This is the usual joint-over-marginal construction,

$$
\hat{f}(y\mid x) = \frac{\hat{f}(x, y)}{\hat{f}(x)}.
$$

The continuous bandwidth scale associated with the conditioning variables appears in both the joint and marginal estimates and cancels. The response scale does **not** cancel, which is why only

$$
\frac{1}{\prod_{d\in\mathcal{C}_y} h_{y,d}}
$$

remains in the conditional density.

Another way to read the estimator is as a weighted average of response kernels. Every observed response places a small kernel bump around itself, while $W_i^x(x)$ determines how much observation $i$ contributes at the conditioning point $x$. The response bandwidth controls the width of those bumps, while the conditioning bandwidths determine which observations are treated as locally relevant. [Mixed-type data](../background/mixed-data.md#the-product-kernel) develops the product kernel column by column.

The `"cv_ml"` rule used for `cfit` selects both bandwidth blocks jointly by leave-one-out likelihood. At each observed pair $(X_i, Y_i)$, KernelJax evaluates the response under a conditional-density estimate that excludes observation $i$ and minimizes the negative mean log density,

$$
\operatorname{CV}_{\mathrm{ML}}
= -\frac{1}{n} \sum_{i=1}^{n} \log \hat{f}_{-i}(Y_i \mid X_i).
$$

The conditioning and response bandwidths are therefore selected together rather than in two independent optimization problems.

With the fit and its bandwidths established, we can reuse them at new evaluation points without running selection again. Here we fix education and region and compare the wage distribution at five and twenty-five years of experience.

```python
def pay_distribution(years):
    pinned = kj.MixedData.from_blocks(
        continuous=np.full(wage_grid.size, years),
        unordered=np.full(wage_grid.size, 0),
        ordered=np.full(wage_grid.size, 2),
        unordered_levels=4,
        ordered_levels=4,
    )

    return kj.cdensity(
        data,
        wage_data,
        cfit,
        at_x=pinned,
        at_y=wage_points,
    ).value
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

The distribution at twenty-five years of experience is shifted toward higher wages relative to the distribution at five years. Regression summarized this relationship through a conditional mean. The conditional density shows how the entire fitted response distribution changes instead.

## Conditional distribution

A conditional density describes where the response distribution is concentrated. {func}`~kerneljax.cdist` turns that same local weighting idea into cumulative probabilities by replacing the response density kernel with its cumulative form.

For the continuous wage response,

$$
\hat{F}(y\mid x) =
\frac{
\displaystyle\sum_{i=1}^{n}
W_i^x(x) G_{h_y}(y, Y_i)
}{
\displaystyle\sum_{i=1}^n W_i^x(x)
}
$$

where

$$
G_{h_y}(y, Y_i) =
\int_{-\infty}^y
\frac{1}{h_y}
k\left(
\frac{t - Y_i}{h_y}
\right)
\, dt.
$$

With the Gaussian kernel,

$$
G_{h_y}(y, Y_i) = \Phi\left( \frac{y - Y_i}{h_y} \right).
$$

Integrating the response density kernel absorbs its $1/h_y$ scale factor. The result is therefore a weighted average of cumulative kernel values rather than a density. Each observation contributes a smooth version of the step it would contribute to an empirical conditional distribution.

The conditioning weights remain unchanged, so the same conditional bandwidth object can be reused between density and distribution estimators. Bandwidth **selection**, however, uses a different objective when the target is a distribution.

KernelJax does not allow `"cv_ml"` to select a conditional CDF bandwidth because treating CDF values as likelihood contributions rewards excessive smoothing. Instead, `"cv_ls"` compares the leave-one-out conditional distribution with the indicator it is estimating.

For a response grid $y_1, \ldots, y_G$,

$$
\operatorname{CV}_{\mathrm{LS}} =
\frac{1}{nG}
\sum_{i=1}^n
\sum_{g=1}^G
\left[
\mathbf{1}\{ Y_i \le y_g \}
- \hat{F}_{-i}(y_g \mid X_i)
\right]^2.
$$

A bandwidth selected under {func}`~kerneljax.cdensity` can still be reused by {func}`~kerneljax.cdist`. Doing so evaluates the CDF with those existing bandwidths rather than claiming that they are the bandwidths a CDF-specific criterion would necessarily have selected.

That distinction matters in the example below. We deliberately pass `cfit`, so the conditional distribution reuses the bandwidths selected for the density. Pinning education and region at the same values as above then lets us ask cumulative probability questions at the same two experience levels.

```python
pinned = kj.MixedData.from_blocks(
    continuous=np.array([5.0, 25.0]),
    unordered=np.zeros(2, dtype=int),
    ordered=np.full(2, 2),
    unordered_levels=4,
    ordered_levels=4,
)

below_12 = kj.cdist(
    data,
    wage_data,
    cfit,
    at_x=pinned,
    at_y=np.full((2, 1), 12.0),
)

print(f"P(wage <= 12 | 5 years)  = {float(below_12.value[0]):.3f}")
print(f"P(wage <= 12 | 25 years) = {float(below_12.value[1]):.3f}")
```

```text
P(wage <= 12 | 5 years)  = 0.626
P(wage <= 12 | 25 years) = 0.000
```

At five years of experience, about 63% of the fitted conditional distribution lies below a wage of twelve. At twenty-five years, the fitted probability is effectively zero. This is the same shift visible in the density plot, now expressed as a cumulative probability.

## Conditional quantiles

Once a conditional distribution is available, we can reverse the previous question. Instead of choosing a wage and asking for the probability below it, {func}`~kerneljax.cquantile` chooses a probability and finds the corresponding location in the fitted distribution.

For a quantile level $\tau\in(0,1)$,

$$
\hat{q}_\tau(x) =
\inf \left\{
y : \hat{F}(y \mid x) \ge \tau
\right\}
$$

KernelJax currently performs this inversion for a **single continuous response column**. It searches by bisection over the observed response range. If the fitted CDF already exceeds $\tau$ at the smallest observed response, the estimate is clamped to that lower endpoint, and if it never reaches $\tau$ by the largest observed response, it is clamped to the upper endpoint.

The bandwidths can again come from an existing conditional fit. Here we reuse `cfit` and the same conditioning points to read several quartiles from the fitted wage distribution.

```python
for tau in (0.25, 0.5, 0.75):
    q = kj.cquantile(
        data,
        wage_data,
        cfit,
        tau=tau,
        at_x=pinned,
    )

    print(
        f"tau={tau:.2f}  "
        f"5 years={float(q.value[0]):.2f}  "
        f"25 years={float(q.value[1]):.2f}"
    )
```

```text
tau=0.25  5 years=11.35  25 years=13.82
tau=0.50  5 years=11.79  25 years=14.27
tau=0.75  5 years=12.23  25 years=14.73
```

Every quartile is higher at twenty-five years of experience than at five. The density showed the overall shift, the CDF translated it into probabilities below chosen wage values, and the quantiles now show where chosen probabilities fall within each conditional distribution.

Once a conditional distribution and its bandwidths are available, many quantile levels can be obtained from the same fitted distribution. There is no need to fit a separate check-loss model for each value of $\tau$.

## Conditional modes

Everything above used wage as a continuous response. When the response is categorical, there is no continuous distribution to invert for a quantile. A natural summary is instead the response level with the largest fitted conditional density.

{func}`~kerneljax.cmode` evaluates that conditional density at every declared response level and returns the maximizer,

$$
\hat{c}(x) = \arg\max_{\ell\in\{0, \ldots, c-1\}} \hat{f}(\ell\mid x),
$$

with ties resolved toward the lowest level.

For a categorical response, there is no continuous response-bandwidth divisor. Writing $L_{\lambda_y}$ for its categorical response kernel gives

$$
\hat{f}(\ell\mid x) =
\frac{
\displaystyle\sum_{i=1}^n
W_i^x(x) L_{\lambda_y}(\ell, Y_i)
}{
\displaystyle\sum_{i=1}^n W_i^x(x)
}
$$

These values are the fitted conditional-density scores that {func}`~kerneljax.cmode` compares across the declared levels. For an unordered finite-support kernel such as Aitchison-Aitken, they can be interpreted directly as probabilities across those levels. For ordered kernels whose normalization is defined on the wider integer lattice, the values over only the declared finite levels need not sum exactly to one, so it is safer to interpret them as conditional density values used to rank the candidate levels.

The response smoothing parameter $\lambda_y$ determines how strongly information is shared across response levels. Near zero, the fit remains concentrated around matching levels. As $\lambda_y$ increases, the response kernel spreads weight farther across categories according to the kernel's notion of similarity.

To make the response categorical, we now change the roles of the variables from the wage example. Education becomes an ordered response, while experience and wage become the conditioning variables.

```python
covariates = kj.MixedData.from_blocks(
    continuous=np.column_stack([exper, wage]),
    names=("exper", "wage"),
)

labels = kj.MixedData.from_blocks(
    ordered=educ,
    ordered_levels=4,
    names=("educ",),
)

mode = kj.cmode(covariates, labels, "cv_ml")
print(f"correct classification={float(mode.accuracy):.3f}")
```

```text
correct classification=0.790
```

Because the fit is evaluated at its training points, `accuracy` compares the modal level with the observed response for each row,

$$
\operatorname{accuracy} =
\frac{1}{n}
\sum_{i=1}^n
\mathbf{1}
\left\{
\hat{c}(X_i) = Y_i
\right\}
$$

Here the fitted mode agrees with the observed education level for 79% of the sample. This is an in-sample classification rate, not a held-out measure of predictive performance. A confusion matrix would break the same predictions down by observed and predicted level, while `accuracy` reports only their overall agreement.
