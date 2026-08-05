# Custom kernels

Every estimator takes `kernels=`, a {class}`~kerneljax.KernelSet` holding one kernel per
column kind. The defaults are a second-order Gaussian for continuous columns,
Aitchison-Aitken for unordered ones and Li-Racine for ordered ones, and swapping any of them
changes the smoothing without touching the estimator.

That substitution is the point of the design. An estimator in KernelJax is one contraction of
kernel weights against a vector, and it never asks what produced the weights, so a kernel you
write reaches every estimator, every criterion and the bandwidth selector at once. This page
covers what that costs you, one method and a short list of conventions that nothing enforces,
so most of what goes wrong here goes wrong quietly. The [Quickstart](../quickstart.md) covers
the rest of the API, and [Kernel smoothing](../background/smoothing.md) covers what a kernel
is doing.

## Writing one

Subclass the base class for the column kind you are targeting and implement `value`.
Here is the Epanechnikov kernel, which is optimal in the sense described in
[Kernel smoothing](../background/smoothing.md#why-the-kernel-hardly-matters).

```python
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import kerneljax as kj

rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)

@dataclasses.dataclass(frozen=True)
class Epanechnikov(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)

kernels = kj.KernelSet(continuous=Epanechnikov())
epan = kj.local_poly(x, y, "cv_ls", degree=1, kernels=kernels)
gauss = kj.local_poly(x, y, "cv_ls", degree=1)

print(f"Epanechnikov  h={epan.bandwidth.h[0]:.6f}  r2={epan.r_squared:.6f}")
print(f"Gaussian      h={gauss.bandwidth.h[0]:.6f}  r2={gauss.r_squared:.6f}")
print(f"ratio of bandwidths {epan.bandwidth.h[0] / gauss.bandwidth.h[0]:.3f}")
```

```text
Epanechnikov  h=0.084645  r2=0.947231
Gaussian      h=0.036019  r2=0.947953
ratio of bandwidths 2.350
```

The selected bandwidth is more than double what the Gaussian default gives on the same data,
because the two kernels carry different scales and not because either is smoothing more. The
canonical bandwidth ratio predicts $2.214$ against the $2.35$ observed, a statement about
optimal bandwidths rather than selected ones. The fits agree to three digits in $r^2$, which
is what the section linked above leads you to expect.

## What `value` receives

A continuous `value` is called once for all continuous columns. It receives `x` of shape
`(n_eval, 1, p_con)`, `y` of shape `(1, n_train, p_con)`, and a bandwidth broadcast against
them, so a univariate formula written elementwise is already a product kernel over the
columns. The library multiplies over the trailing column axis itself, so `value` must leave
one factor per column in place and must not reduce over that axis. A kernel that sums over
that axis to build a genuinely multivariate kernel raises nothing and returns a weight matrix
whose rows are identical and meaningless.

Categorical kernels are called once per column, receiving `x` of shape `(n_eval, 1)`, `y` of
shape `(1, n_train)`, a scalar `lam`, and a `levels` count that arrives as a plain Python
integer rather than a traced value, so it is safe in Python control flow in a way that `lam` is
not.

The argument order is always `value(evaluation_point, training_point)`. Nothing in the library
assumes the two are interchangeable, so an asymmetric kernel runs end to end without complaint
and quietly mirrors the estimator if you had the order backwards.

## What a kernel must satisfy

```{warning}
Four requirements are easy to miss, and only one of them fails loudly.

1. **Return no** $1/h$ **factor.** The estimator divides by the bandwidths of the continuous
   columns exactly once, so a kernel that normalizes itself is applied twice. Nothing is
   raised and every density comes out scaled by $1/h$, which at $h = 0.2$ means every value is
   five times too large. Note this governs `value` only. The optional `deriv` is a derivative
   with respect to $x$ rather than to $u$, so it does carry a chain rule factor of $1/h$.
2. **Be hashable.** The kernel travels inside a {class}`~kerneljax.KernelSet` handed to the
   compiler as a static argument, so it has to be hashable. A frozen dataclass is the
   idiomatic way. A mutable dataclass is not hashable, since `@dataclass` without
   `frozen=True` sets `__hash__` to `None`, and neither is a frozen dataclass holding a
   `jax.Array` field. The failure reads `Non-hashable static arguments are not supported`, and
   it fires from {func}`~kerneljax.local_poly` and {func}`~kerneljax.cdf` even at a fixed
   bandwidth, not only during selection. A plain class is hashable and will run, but its
   instances compare unequal, so every fresh instance is a compilation cache miss.
3. **Be elementwise.** `value` receives arrays and must broadcast. Use `jnp.where` rather
   than a Python `if`, which raises `The truth value of an array with more than one element
   is ambiguous` on a traced value.
4. **Be differentiable in the smoothing parameter.** Selection differentiates the criterion
   through your kernel, and a NaN gradient is not reported as such. What you get instead is a
   bandwidth of `nan`, the full iteration budget spent, and `converged` set to `False`.
```

The last requirement is the hard one, because `jnp.where` evaluates *both* branches and
differentiates both. An unsafe expression in the branch that is not taken poisons the gradient
while leaving the value correct. A sinc kernel is the natural illustration, since $u = 0$
occurs on the diagonal of any fit evaluated at its own training points, so the guarded branch
is always reached.

```python
def unsafe(u):
    return jnp.where(u == 0.0, 1.0, jnp.sin(u) / u)

def safe(u):
    nonzero_u = jnp.where(u == 0.0, 1.0, u)
    return jnp.where(u == 0.0, 1.0, jnp.sin(nonzero_u) / nonzero_u)

unsafe_value, unsafe_grad = jax.value_and_grad(unsafe)(0.0)
safe_value, safe_grad = jax.value_and_grad(safe)(0.0)

print(f"unsafe  value={unsafe_value:.4f}  d/du={unsafe_grad:.4f}")
print(f"safe    value={safe_value:.4f}  d/du={safe_grad:.4f}")
```

```text
unsafe  value=1.0000  d/du=nan
safe    value=1.0000  d/du=0.0000
```

The two agree on the value and disagree on the gradient, so a forward-only check will not
catch it. Guard the denominator, not just the branch. Built into a kernel, the unsafe version
produces a result rather than an exception.

```python
@dataclasses.dataclass(frozen=True)
class Sinc(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(u == 0.0, 1.0, jnp.sin(u) / u)

sinc_kernels = kj.KernelSet(continuous=Sinc())
result = kj.select_bandwidth(x, kj.cv_ls_regression, y=y, kernels=sinc_kernels, n_starts=1)

print(f"h={result.bandwidth.h[0]:.6f}  n_iter={result.n_iter}  "
      f"converged={result.converged}")
```

```text
h=nan  n_iter=200  converged=False
```

Reading `converged` is the habit that catches this. A run that exhausts its iteration budget
and reports `False` has not selected anything.

## Categorical kernels

A categorical kernel owes one thing more than a continuous one, a second method
`upper_bound(levels)`. It is abstract on both categorical base classes, so leaving it out
fails at instantiation rather than silently. It answers one question. At what value of
$\lambda$ does this kernel weight every level equally, so that the column stops influencing
the estimate at all?

That value is a property of the parameterization and not of the data. The Aitchison-Aitken
kernel reaches it at $(c-1)/c$, as the [background page](../background/mixed-data.md#unordered-categories)
derives, while the unnormalized variant, which is $1$ on a match and $\lambda$ otherwise,
reaches it at $\lambda = 1$.

```python
@dataclasses.dataclass(frozen=True)
class Plain(kj.UnorderedKernel):
    """The unnormalized variant, 1 on a match and lam otherwise."""

    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, lam)

    def upper_bound(self, levels):
        return 1.0

rng = np.random.default_rng(0)
exper = rng.uniform(0, 30, 200)
region = rng.integers(0, 4, 200)
wage = 2.0 + 0.1 * exper + region + rng.normal(0, 0.5, 200)

data = kj.MixedData.from_blocks(continuous=exper, unordered=region,
                                unordered_levels=4, names=("exper", "region"))

plain = Plain()
aitchison = kj.AitchisonAitken()

custom = kj.local_poly(data, wage, "cv_ls", degree=1,
                       kernels=kj.KernelSet(unordered=plain))
shipped = kj.local_poly(data, wage, "cv_ls", degree=1)

print(f"Plain            lam={custom.bandwidth.lam_uno[0]:.6f}  "
      f"bound={plain.upper_bound(4):.2f}  r2={custom.r_squared:.6f}")
print(f"AitchisonAitken  lam={shipped.bandwidth.lam_uno[0]:.6f}  "
      f"bound={aitchison.upper_bound(4):.2f}  r2={shipped.r_squared:.6f}")
```

```text
Plain            lam=0.001238  bound=1.00  r2=0.897111
AitchisonAitken  lam=0.003699  bound=0.75  r2=0.897111
```

The two fits agree to six digits, because the normalization the shipped kernel carries is a
factor common to every level and cancels from a ratio estimator. The smoothing parameters
differ, because $\lambda$ means a different thing in each. This is why `upper_bound` cannot be
inherited from anything. Return a bound that is too high and the search walks past complete
pooling into a region where a *matching* level receives less weight than a non-matching one,
reporting `converged=True` throughout. Return zero and the search box collapses, selection
returns `nan`, and again nothing is raised.

The bound also fixes where selection begins and where it stops, since `normal_reference`
starts every categorical parameter at half of it. That ceiling shows itself on a column
carrying no signal, where the criterion pushes the parameter as far toward complete pooling as
it can reach. Against a covariate unrelated to the response, `Plain` selects $0.999998$
against its bound of $1$ and Aitchison-Aitken selects $0.749999$ against its bound of $0.75$,
neither quite arriving because the search box is open at the top. Both are saying the same
thing in their own units, which is the reading that
[Bandwidth selection](../background/selection.md#what-cross-validation-buys) sets out.

## Optional methods

`value` alone is enough for local polynomial regression at any degree, including
`gradient=True`, for likelihood cross validation, for `normal_reference`, and for
{func}`~kerneljax.summary`. The rest of the interface is optional, one capability apiece, and
each raises `NotImplementedError` naming the kernel and the method, so these are among the few
failures that announce themselves.

| Method | Needed for | Notes |
| --- | --- | --- |
| `conv` | `density(..., "cv_ls")`, and `local_poly(..., se=True)` | The self-convolution of `value` |
| `cdf` | {func}`~kerneljax.cdf` and its criterion | Fires at a fixed bandwidth too, not only in selection |
| `deriv` | derivative weight tensors, reached through the `op=` argument of {func}`~kerneljax.kweights` | Not needed by any estimator |

`local_poly(..., se=True)` needs `conv` even though nothing in a standard error looks like a
density, since the variance carries $R(k) = \int k^2$, the self-convolution at zero. The call
asks only the continuous kernel for it. And `conv` must genuinely be the self-convolution of
`value`, which the library never checks. For a kernel supported on $|u| \le 1$ the
self-convolution is supported on $|u| \le 2$, so truncating it to the kernel's own support is
an easy mistake that produces a plausible bandwidth and no warning.

## Where a valid kernel can still fail

Two properties of a kernel are not requirements of the interface but do constrain how
selection behaves with it.

Likelihood cross validation takes the log of a leave-one-out density, so it needs that density
to be strictly positive at every training point. A compactly supported kernel makes a zero
easy to reach, and in single precision even the Gaussian underflows to exactly zero far enough
into the tail, so an isolated observation can produce `nan` and a bandwidth that never moves.
The same applies to a higher-order kernel, which takes negative values by construction and can
drive the leave-one-out density below zero. Least squares cross validation has no logarithm
and handles both cases, which is the reason to reach for `cv_ls` rather than `cv_ml` with an
unusual kernel.

Second, nothing in the package special-cases the Gaussian, but `normal_reference` does read an
`order` attribute off the continuous kernel and falls back to `2` when there is none. A
fourth-order kernel that does not declare `order = 4` is therefore started at the second-order
rate. That moves only the starting point, not any kernel evaluation, but on a criterion with
more than one local minimum the starting point is not nothing.
