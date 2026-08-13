# Building from primitives

The estimators above are built from lower-level primitives that KernelJax exposes directly.

The setup is the shared wage example from [Working with data](data.md), together with the
fits and evaluation grid the earlier pages produced.

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
quick = kj.local_poly(data, wage, "normal_reference", degree=1)
points = kj.grid(data, vary="exper", n=200)
```

The central operation is a kernel-weighted contraction. A density contracts the kernel weights against a column of ones, while a Nadaraya-Watson regression contracts them against the response and divides by the corresponding weight sum.

We can reproduce KernelJax's local-constant regression directly.

```python
numerator = kj.ksum(data, fit.bandwidth, wage[:, None], at=points)
denominator = kj.ksum(data, fit.bandwidth, at=points)
nadaraya_watson = (numerator / denominator).ravel()
local_constant = kj.local_poly(data, wage, fit.bandwidth, at=points)
gap = abs(nadaraya_watson - local_constant.mean).max()
print(f"largest gap to local_poly={gap:.3e}")
```

```text
largest gap to local_poly=1.335e-05
```

These are the same estimator. The small numerical difference comes from float32 arithmetic.

Notice that we passed `fit.bandwidth` rather than `fit`. Passing the full fit would also carry its `degree=1`, while passing only the bandwidth causes `local_poly` to use its default degree of zero, giving the local-constant estimator we want to reproduce.

The selection criteria themselves are exposed in the same way and remain differentiable.

```python
def cv_ls(bandwidth):
    return kj.cv_ls_regression(data, bandwidth, y=wage, degree=1)

at_plug_in = jax.grad(cv_ls)(quick.bandwidth)
at_selected = jax.grad(cv_ls)(fit.bandwidth)

for label, grad in [("plug-in", at_plug_in), ("selected", at_selected)]:
    print(
        f"{label:9s} "
        f"d/dh={grad.h[0]:+.3e}  "
        f"d/dlam_uno={grad.lam_uno[0]:+.3e}  "
        f"d/dlam_ord={grad.lam_ord[0]:+.3e}"
    )
```

```text
plug-in   d/dh=-3.008e-02  d/dlam_uno=-3.304e+00  d/dlam_ord=+3.895e-07
selected  d/dh=+2.045e-07  d/dlam_uno=-6.394e-05  d/dlam_ord=+1.146e-06
```

The gradient has the same structure as the bandwidth itself, including one component for each categorical smoothing parameter.

At the cross-validated solution, the gradients for the interior parameters are close to zero. The unordered bandwidth is sitting on its upper bound, so its gradient need not vanish. That is the optimization view of the same result we saw earlier, that region has been smoothed out.

This lower-level interface is useful when the estimator you need is not one that KernelJax ships directly. The package provides the kernels, bandwidth objects, contractions, and differentiable criteria so they can be composed into new estimators rather than treated as a closed procedure.

## The weight matrix and per column operators

{func}`~kerneljax.ksum` contracts, and {func}`~kerneljax.kweights` exposes the matrix it
contracts with, one row per evaluation point and one column per training point.

```python
weights = kj.kweights(data, fit.bandwidth, at=points)
print(weights.shape)
```

```text
(200, 300)
```

The `op` argument takes each continuous column through a different reading of its kernel,
the value itself, its integral for distribution estimators, its derivative for gradients,
or its self convolution for least squares criteria. One operator per column means a single
weight matrix can integrate over one variable while differentiating another, which is what
the conditional estimators are built from.

```python
accumulated = kj.kweights(data, fit.bandwidth, at=points, op=("cdf", "value", "value"))
print(f"value weights sum on the first row  {float(weights[0].sum()):8.3f}")
print(f"cdf weights sum on the first row    {float(accumulated[0].sum()):8.3f}")
```

```text
value weights sum on the first row     1.268
cdf weights sum on the first row       1.052
```
