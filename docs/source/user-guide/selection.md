# Bandwidth selection

How much to smooth matters more than any other choice in kernel estimation, and this page
covers how KernelJax makes it. Every estimator accepts a selection rule by name, and the
alternatives differ statistically rather than mechanically.

The setup is the shared wage example from [Working with data](data.md), with the local
linear fit from [Local polynomial regression](regression.md).

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
```

```python
fit = kj.local_poly(data, wage, "cv_ls", degree=1)
```

## Choosing a selector

The selection criterion is part of the statistical model rather than just an optimization setting.

| Estimator                     | Accepted strings                           |
|-------------------------------|--------------------------------------------|
| {func}`~kerneljax.local_poly` | `"cv_ls"`, `"aic"`, `"normal_reference"`   |
| {func}`~kerneljax.density`    | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |
| {func}`~kerneljax.cdf`        | `"cv_cdf"`, `"normal_reference"`           |
| {func}`~kerneljax.cdensity`   | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |
| {func}`~kerneljax.cdist`      | `"cv_ls"`, `"normal_reference"`            |
| {func}`~kerneljax.cquantile`  | `"cv_ls"`, `"normal_reference"`            |
| {func}`~kerneljax.cmode`      | `"cv_ml"`, `"cv_ls"`, `"normal_reference"` |

`"normal_reference"` is the inexpensive option. It uses a closed-form plug-in rule based on a normal reference model and does not run numerical optimization.

That makes it useful when you want a quick first fit.

```python
quick = kj.local_poly(data, wage, "normal_reference", degree=1)
print(f"normal reference  h={quick.bandwidth.h[0]:.6f}")
print(f"cross validated   h={fit.bandwidth.h[0]:.6f}")
```

```text
normal reference  h=3.026807
cross validated   h=2.792168
```

For this example, the plug-in and cross-validated bandwidths are fairly close.

The optimization-based selectors use L-BFGS from multiple starting points. By default, KernelJax uses three starts, and every estimator exposes `n_starts` if you want to change that.

Multiple starts matter because bandwidth-selection objectives are generally nonconvex. Different initial values can lead the optimizer to different local solutions.

Here is a deliberately difficult example in which the response is unrelated to the covariate.

```python
noise_rng = np.random.default_rng(9)
x_noise = noise_rng.uniform(size=150)
y_noise = noise_rng.normal(0, 1, 150)
```

```python
one_start = kj.local_poly(x_noise, y_noise, "cv_ls", degree=1, n_starts=1)
three_starts = kj.local_poly(x_noise, y_noise, "cv_ls", degree=1)

for label, run in [("1 start", one_start), ("3 starts", three_starts)]:
    s = run.selection

    print(
        f"{label:8s}  h={s.bandwidth.h[0]:8.4f}  criterion={s.value:.6f}  converged={s.converged}"
    )
```

```text
1 start   h=  0.1392  criterion=0.989901  converged=True
3 starts  h= 33.5028  criterion=0.984698  converged=True
```

Since `y_noise` contains no relationship with `x_noise`, a very large bandwidth is sensible. There is little reason to estimate a local relationship.

The one-start optimization settles on a much smaller bandwidth and a worse criterion value even though the optimizer itself reports convergence. This is why a successful optimizer exit does not guarantee that the best available solution has been found.

## Reading a selection

For optimization-based bandwidth selection, it is worth checking the selection result before relying on the fitted model.

```python
print(jax.tree.map(lambda a: a.shape, fit.bandwidth))

print(
    f"converged={fit.selection.converged}  "
    f"n_iter={fit.selection.n_iter}  "
    f"criterion={fit.selection.value:.6f}"
)

print(f"degree={fit.degree}  r_squared={fit.r_squared:.4f}  residual_se={fit.residual_se:.4f}")
```

```text
Bandwidth(h=(1,), lam_uno=(1,), lam_ord=(1,), h_axis='shared')
converged=True  n_iter=27  criterion=0.334907
degree=1  r_squared=0.9084  residual_se=0.5373
```

`converged=True` means the optimizer terminated successfully at a finite solution. It does not imply that the solution is the global minimum, which is one reason KernelJax uses multiple starts by default.

The bandwidth itself is a JAX pytree. Here, `h_axis='shared'` means the same bandwidth is used at every evaluation point rather than assigning a separate bandwidth to each row.

Because these objects are registered pytrees, they also compose naturally with JAX transformations such as {func}`jax.grad`.

## Selecting once and reusing

Selection is the expensive step, so its result is built to travel.
{func}`~kerneljax.select_bandwidth` runs the same search the estimators run internally, and
the criterion classes configure what it minimizes. The returned
{class}`~kerneljax.SelectionResult` remembers the criterion and kernels it was selected
under, so handing it to an estimator recovers compatible settings without restating them.

```python
sel = kj.select_bandwidth(data, kj.RegressionCriterion(method="cv_ls", degree=1), y=wage)
reused = kj.local_poly(data, wage, sel)
print(f"degree={reused.degree} recovered, r_squared={float(reused.r_squared):.4f}")
```

```text
degree=1 recovered, r_squared=0.9084
```

A setting restated explicitly must agree with the one the result carries, and a
contradiction raises rather than silently preferring either. The
[introduction](intro.md#preserve-selection-context) shows that guard firing.
