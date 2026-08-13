# Bandwidth selection

How much to smooth is usually the most consequential choice in kernel estimation. KernelJax can select that smoothing automatically, and every estimator exposes the selection rule through the same bandwidth argument.

The choice is statistical before it is computational. Different rules optimize different criteria, so choosing a selector means deciding what kind of error the bandwidth should trade off. When that selector requires numerical optimization, there is then a second question of how KernelJax searches for a good solution.

The setup is the shared wage example from [Working with data](data.md), with the local linear fit from [Local polynomial regression](regression.md).

```python
import jax
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

fit = kj.local_poly(data, wage, "cv_ls", degree=1)
```

## Choosing a selector

The selection criterion is part of the statistical method, not merely an optimizer setting. The accepted shorthand depends on the estimator being fitted.

| Estimator                     | Accepted strings                           |
| ----------------------------- | ------------------------------------------ |
| {func}`~kerneljax.local_poly` | `"cv_ls"`, `"aic"`, `"normal_reference"`   |
| {func}`~kerneljax.density`    | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |
| {func}`~kerneljax.cdf`        | `"cv_cdf"`, `"normal_reference"`           |
| {func}`~kerneljax.cdensity`   | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |
| {func}`~kerneljax.cdist`      | `"cv_ls"`, `"normal_reference"`            |
| {func}`~kerneljax.cquantile`  | `"cv_ls"`, `"normal_reference"`            |
| {func}`~kerneljax.cmode`      | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |

These names determine what KernelJax asks a candidate bandwidth to do well.

`"cv_ml"` uses leave-one-out likelihood and is natural for density estimation. `"cv_ls"` denotes a least-squares criterion suited to the estimator being selected. For regression, that is the leave-one-out mean squared residual. For density and conditional density, it is an integrated-squared-error criterion. For conditional distributions, it compares the estimated CDF with the indicator the CDF is trying to reproduce.

`"aic"` is available for local polynomial regression. It does not hold observations out. Instead, it balances in-sample fit against the effective degrees of freedom spent by the smoother.

The [Bandwidth selection](../background/selection.md) background page derives these criteria in detail, while [Density and distribution](density.md) develops the CDF criterion directly.

### The normal-reference rule

`"normal_reference"` is different from the criteria above because it does not optimize a cross-validation or information criterion at all. It constructs a bandwidth directly from the scale and size of the sample, making it much cheaper than running a numerical search.

For the density-style reference rule used by the local polynomial fit in this example, each continuous bandwidth is

$$
h_j
=
1.059224\, s_j\,
n^{-1/(2P+p_{\mathrm{con}})},
$$

where $P$ is the order of the continuous kernel, $p_{\mathrm{con}}$ is the number of continuous columns, and $s_j$ is a robust scale estimate for column $j$.

KernelJax takes

$$
s_j
=
\min_{+}
\left\{
\hat\sigma_j,\;
\frac{\operatorname{IQR}_j}{2\Phi^{-1}(0.75)},\;
1.4826\,\operatorname{MAD}_j
\right\},
$$

where $\min_{+}$ means the smallest strictly positive candidate. Here

$$
\operatorname{MAD}_j =
\operatorname{median}_i \left|
X_{ij}-\operatorname{median}_k X_{kj}
\right|.
$$

Using several scale estimates makes the reference width less sensitive to an unusually large sample standard deviation.

For this example there is one continuous column and the default Gaussian kernel has order $P=2$, so

$$
-\frac{1}{2P+p_{\mathrm{con}}}
=
-\frac{1}{5}.
$$

"Experience" has the three candidate scale estimates $8.94$, $11.36$, and $11.22$, so KernelJax uses $s=8.94$. The resulting bandwidth is

$$
h
=
1.059224
\times 8.94
\times 300^{-1/5}
\approx 3.03.
$$

The factor $1.059224$ is the familiar one-dimensional Gaussian normal-reference constant. KernelJax combines it with the dimension-dependent rate above and its robust scale rule when constructing this reference bandwidth.

This should be read as a **rule of thumb**, not as the bandwidth that optimizes the regression criterion. In particular, the local polynomial estimator uses it as a cheap reference-scale bandwidth even though it was not derived by minimizing the regression's prediction error.

Categorical variables require a different treatment. A normal reference model supplies no analogous scale for a category label, so `"normal_reference"` returns zero categorical smoothing parameters. Region and education therefore remain completely separated under this rule.

Cross-validation is not restricted that way. It can move the categorical smoothing parameters anywhere within their admissible ranges, including all the way to the maximum-smoothing limit. [Local polynomial regression](regression.md#reading-the-bandwidths) shows that happening for region.

```python
quick = kj.local_poly(data, wage, "normal_reference", degree=1)

print(f"normal reference  h={quick.bandwidth.h[0]:.6f}")
print(f"cross validated   h={fit.bandwidth.h[0]:.6f}")
```

```text
normal reference  h=3.026807
cross validated   h=2.792168
```

For this example, the reference and cross-validated continuous bandwidths happen to be fairly close. Nothing requires that agreement in general.

The reference rule also has a second role. When a selector such as `"cv_ls"` or `"cv_ml"` does require numerical optimization, its bandwidth provides KernelJax with a sensible scale from which to begin the search.

## Why KernelJax uses multiple starts

For optimization-based selectors, the reference bandwidth is a starting point rather than the final answer. KernelJax runs L-BFGS from several starting points and keeps the resulting solution with the best criterion value.

KernelJax uses three starts by default, and the estimators expose `n_starts` when you want to change that. The search takes place in the unconstrained coordinates described in the [introduction](intro.md#keep-the-full-path-differentiable). The first start comes from the reference rule, with categorical coordinates moved to the middle of their admissible ranges. Additional starts perturb that unconstrained vector before running the solver again.

Multiple starts matter because bandwidth-selection criteria are not generally convex. Two runs can both satisfy the optimizer's stopping rule while ending at different local solutions.

A deliberately difficult example makes that distinction visible. Here the response is generated independently of the covariate, so there is no systematic relationship for a local regression to recover.

```python
noise_rng = np.random.default_rng(9)
x_noise = noise_rng.uniform(size=150)
y_noise = noise_rng.normal(0, 1, 150)
```

```python
one_start = kj.local_poly(
    x_noise,
    y_noise,
    "cv_ls",
    degree=1,
    n_starts=1,
)

three_starts = kj.local_poly(
    x_noise,
    y_noise,
    "cv_ls",
    degree=1,
)

for label, run in [("1 start", one_start), ("3 starts", three_starts)]:
    s = run.selection

    print(
        f"{label:8s}  "
        f"h={s.bandwidth.h[0]:8.4f}  "
        f"criterion={s.value:.6f}  "
        f"converged={s.converged}"
    )
```

```text
1 start   h=  0.1392  criterion=0.989901  converged=True
3 starts  h= 33.5028  criterion=0.984698  converged=True
```

Both searches report convergence, but they arrive at very different bandwidths and criterion values. Because `y_noise` contains no systematic relationship with `x_noise`, there is little reason to fit a strongly local curve. The much larger bandwidth found by the three-start search is therefore plausible.

The large-bandwidth limit makes that interpretation precise.

For a one-dimensional local linear regression, the fit at $x$ solves a weighted regression on

$$
1
\qquad\text{and}\qquad
\frac{X_i-x}{h}.
$$

As $h\to\infty$, the Gaussian kernel weights become equal across observations. For any finite $h$, the columns

$$
1
\qquad\text{and}\qquad
\frac{X_i-x}{h}
$$

span the same linear space as $1$ and $X_i$. Once the weights become constant, the local weighted regression therefore approaches the global ordinary least-squares line $\hat m_h(x) \longrightarrow \hat\alpha + \hat\beta x$, where

$$
(\hat\alpha,\hat\beta)
=
\arg\min_{a,b}
\sum_{i=1}^{n}
\left(
Y_i-a-bX_i
\right)^2.
$$

The cross-validation criterion approaches the leave-one-out mean squared error of that line.

Let $\mathbf Z$ be the $n\times2$ design matrix with rows $(1,X_i)$ and let $H = \mathbf Z(\mathbf Z^\top\mathbf Z)^{-1}\mathbf Z^\top$ be its hat matrix. The ordinary least-squares leave-one-out identity gives

$$
Y_i-\hat m_{-i}(X_i)
=
\frac{
Y_i-\hat\alpha-\hat\beta X_i
}{
1-H_{ii}
}.
$$

Consequently,

$$
\operatorname{CV}_{\mathrm{LS}}(h)
\longrightarrow
\frac{1}{n}
\sum_{i=1}^{n}
\left(
\frac{
Y_i-\hat\alpha-\hat\beta X_i
}{
1-H_{ii}
}
\right)^2
=
0.984697.
$$

The three-start search reports `0.984698` at $h=33.5$, essentially the same value at the displayed precision. The optimizer has not simply wandered toward a meaningless large number. The criterion has moved close to its global-linear limit.

The one-start fit settles at a much smaller bandwidth and a worse criterion value even though it also reports `converged=True`. Convergence therefore tells us something about how a particular solver run stopped, not whether that run found the best solution available. That distinction is important when reading the selection diagnostics.

## Reading a selection

For an optimization-based bandwidth, it is worth inspecting the selection result before relying on the fitted estimator. The wage fit from the beginning of the page exposes both the selected bandwidth structure and the diagnostics for the run that produced it.

```python
print(jax.tree.map(lambda a: a.shape, fit.bandwidth))

print(
    f"converged={fit.selection.converged}  "
    f"n_iter={fit.selection.n_iter}  "
    f"criterion={fit.selection.value:.6f}"
)

print(
    f"degree={fit.degree}  "
    f"r_squared={fit.r_squared:.4f}  "
    f"residual_se={fit.residual_se:.4f}"
)
```

```text
Bandwidth(h=(1,), lam_uno=(1,), lam_ord=(1,), h_axis='shared')
converged=True  n_iter=27  criterion=0.334907
degree=1  r_squared=0.9084  residual_se=0.5373
```

For KernelJax's default L-BFGS solver, `converged=True` means the selected run stopped before exhausting its iteration budget because either $\max_j |g_j| < \mathrm{tol}$ or the change in the scaled objective satisfied

$$
|f_{k+1}-f_k|
<
\mathrm{tol}\,
(1+|f_k|),
$$

with a finite final objective and gradient.

It does **not** mean that the gradient criterion specifically was met, and it does not imply that the returned point is the global minimum. The noise example above shows why those distinctions matter. `n_iter` reports the number of iterations taken by the selected run, while `criterion` is the original, unscaled objective evaluated at its chosen bandwidth.

The result also carries the structure needed to use that bandwidth later. The bandwidth itself is a JAX pytree. Here, `h_axis='shared'` means one continuous bandwidth is shared across all evaluation and training points rather than varying by row. The arrays inside the bandwidth are traced JAX values, so bandwidths and selection results can participate naturally in JAX computations. Structural information such as the criterion and kernels remains attached as static metadata.

## Selecting once and reusing

Because a selection result carries both the chosen bandwidth and the context under which it was selected, it can be reused without reconstructing that information from separate arguments. This is especially useful because selection is usually the expensive step.

{func}`~kerneljax.select_bandwidth` exposes the same selection machinery that estimator shortcuts use internally. Criterion objects configure what is minimized, and the returned {class}`~kerneljax.SelectionResult` retains the criterion and kernels under which the bandwidth was selected.

```python
sel = kj.select_bandwidth(
    data,
    kj.RegressionCriterion(method="cv_ls", degree=1),
    y=wage,
)

reused = kj.local_poly(data, wage, sel)

print(
    f"degree={reused.degree} recovered, "
    f"r_squared={float(reused.r_squared):.4f}"
)
```

```text
degree=1 recovered, r_squared=0.9084
```

The degree is recovered from the criterion carried by the selection, so it does not have to be stated again. The same principle applies to kernels. A selection is meaningful only together with the configuration under which it was obtained, so KernelJax carries that context forward rather than silently falling back to defaults.

If a setting is restated explicitly, it must agree with the one already attached to the result. A contradiction raises rather than allowing one configuration to silently override the other. The [introduction](intro.md#preserve-selection-context) shows that guard directly.
