# Custom kernels

Every estimator takes `kernels=`, a {class}`~kerneljax.KernelSet` holding one kernel per
column kind. The defaults are a second-order Gaussian for continuous columns,
Aitchison-Aitken for unordered ones and Li-Racine for ordered ones, and swapping any of them
changes the smoothing without touching the estimator.

This page covers writing one of your own. The [Quickstart](quickstart.md) covers the rest of
the API, and [Kernel smoothing](background/smoothing.md) covers what a kernel is doing.

## Writing one

Subclass the base class for the column kind you are targeting and implement `value`.
Here is the Epanechnikov kernel, which is optimal in the sense described in
[Kernel smoothing](background/smoothing.md#why-the-kernel-matters-far-less-than-the-bandwidth).

```python
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import kerneljax as kj

rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)

@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class Epanechnikov(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)

kernels = kj.KernelSet(continuous=Epanechnikov())
fit = kj.local_poly(x, y, "cv_ls", degree=1, kernels=kernels)

print(f"{float(fit.bandwidth.h[0]):.6f}")   # 0.084645
print(f"{float(fit.r_squared):.6f}")        # 0.947231
```

The selected bandwidth is larger than the Gaussian default gives on the same data because the two kernels
carry different scales, not because either is smoothing more.

## What a kernel must satisfy

```{warning}
Four requirements are easy to miss and only one of them fails loudly.

1. **Return no** $1/h$ **factor.** The estimator divides by $\prod_d h_d$ exactly once, so a
   kernel that normalizes itself is applied twice and every density comes out wrong by a
   constant.
2. **Be a frozen dataclass registered as static.** Kernels are static arguments under
   `jit`, so they must be hashable. Omitting either decorator raises
   `Non-hashable static arguments are not supported`.
3. **Be elementwise.** `value` receives arrays and must broadcast. Use `jnp.where` rather
   than a Python `if`, which cannot see traced values.
4. **Be differentiable in the smoothing parameter.** Selection differentiates the criterion
   through your kernel, so a NaN gradient stops the optimizer without any error being
   raised.
```

That last one deserves a demonstration, because `jnp.where` evaluates *both* branches and
differentiates both. An unsafe expression in the branch that is not taken poisons the
gradient while leaving the value correct:

```python
levels = jnp.float32(1.0)

def unsafe(lam):
    return jnp.where(lam < 2.0, 1.0 - lam, lam / (levels - 1.0))

def safe(lam):
    guarded = jnp.where(levels - 1.0 == 0.0, 1.0, levels - 1.0)
    return jnp.where(lam < 2.0, 1.0 - lam, lam / guarded)
```

```text
unsafe  value=0.7000  d/dlam=nan
safe    value=0.7000  d/dlam=-1.0
```

The two agree on the value and disagree on the gradient, so a forward-only check will not
catch it. Guard the denominator, not just the branch.

## Optional methods

`value` alone is enough for local polynomial regression and for likelihood cross validation.
The rest of the interface is optional, and each method unlocks one thing.

| Method | Needed for | Without it |
| --- | --- | --- |
| `conv` | `density(..., "cv_ls")` | `NotImplementedError` on the $\int \hat f^2$ term |
| `cdf` | {func}`~kerneljax.cdf` | `NotImplementedError` when integrating the kernel |
| `deriv` | derivative weight tensors through {func}`~kerneljax.ksum` | `NotImplementedError` |

Note that `gradient=True` on a regression does *not* need `deriv`, since the derivative
comes from the fitted polynomial coefficients rather than from the kernel.

A categorical kernel additionally must implement `upper_bound(levels)`, which sets the range
selection searches over. It is abstract on both categorical base classes, so omitting it
fails at instantiation rather than silently.
