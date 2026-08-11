# Custom kernels

Every KernelJax estimator accepts a `kernels` argument. It is a {class}`~kerneljax.KernelSet` containing the kernels used for continuous, unordered, and ordered variables.

By default, KernelJax uses a second-order Gaussian kernel for continuous variables, Aitchison-Aitken for unordered variables, and Li-Racine for ordered variables. You can replace any of them without changing the estimator itself.

A custom kernel does not need to be registered anywhere. Define it, place it in a `KernelSet`, and pass it directly to an estimator.

This page builds an Epanechnikov kernel first, uses it in regression and density estimation, and then works through the small interface a custom kernel must satisfy to work throughout the library. For the surrounding estimator API, see the [Quickstart](../quickstart.md). For the statistical role of the kernel itself, see [Kernel smoothing](../background/smoothing.md).

## Your first custom kernel

A continuous kernel subclasses {class}`~kerneljax.ContinuousKernel` and implements `value`.

The Epanechnikov kernel is

$$
k(u) = \frac{3}{4}(1-u^2)\mathbf{1}(|u|\le1).
$$

Here is the same kernel in KernelJax.

```python
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import kerneljax as kj

@dataclasses.dataclass(frozen=True)
class Epanechnikov(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
```

We can use it immediately.

```python
rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)
kernels = kj.KernelSet(continuous=Epanechnikov())
epan = kj.local_poly(x, y, "cv_ls", degree=1, kernels=kernels)
gauss = kj.local_poly(x, y, "cv_ls", degree=1)
print(f"Epanechnikov  h={epan.bandwidth.h[0]:.6f}  r2={epan.r_squared:.6f}")
print(f"Gaussian      h={gauss.bandwidth.h[0]:.6f}  r2={gauss.r_squared:.6f}")
```

```text
Epanechnikov  h=0.084645  r2=0.947231
Gaussian      h=0.036019  r2=0.947953
```

The numerical bandwidths are different because bandwidth is measured on the scale of the kernel. A larger Epanechnikov bandwidth therefore does not mean that the resulting regression is necessarily smoother.

The two fitted regressions are nearly identical here, which is what we would expect. For ordinary second-order kernels, choosing the bandwidth well generally matters much more than choosing between reasonable kernel shapes.

## What a kernel must satisfy

The `Epanechnikov` class above is already enough for local polynomial regression. More generally, a custom continuous kernel should satisfy a few rules.

* `value(x, y, h)` should operate elementwise and preserve the broadcast shape of its inputs.
* `value` should return the kernel in standardized $u$-space and should not include its own $1/h$ normalization.
* The kernel object must be hashable so JAX can use it as a static argument.
* Kernel values and bandwidth gradients must remain finite wherever KernelJax may evaluate them.
* When the kernel is used for density or distribution estimation, `value` must integrate to one in $u$-space.

Categorical kernels follow the same general pattern but also define `upper_bound(levels)`.

Additional methods such as `conv`, `cdf`, and `deriv` are optional. You only need to implement them when you use functionality that depends on them.

The rest of this page works through those rules one at a time.

## How `value` is called

For a continuous kernel, `value` is called across all continuous columns at once. Its inputs broadcast as

```text
x:  (n_eval, 1,       p_con)
y:  (1,      n_train, p_con)
h:  broadcast against both
```

where `p_con` is the number of continuous variables.

The important part is simpler than the shapes make it look.

```{tip}
Write `value` elementwise and leave one kernel factor per continuous column. KernelJax multiplies those factors across columns for you.
```

This version is correct.

```python
@dataclasses.dataclass(frozen=True)
class Epanechnikov(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h

        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
```

This one is not.

```python
@dataclasses.dataclass(frozen=True)
class Reducing(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h

        return jnp.sum(jnp.exp(-0.5 * u * u), axis=-1, keepdims=True)
```

Run it on data with two continuous columns and KernelJax rejects the reduced output.

```python
two_columns = kj.MixedData.continuous(np.column_stack([x, x**2]))
wide = kj.Bandwidth(h=jnp.array([0.2, 0.2]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))
kj.local_poly(two_columns, y, wide, degree=1, kernels=kj.KernelSet(continuous=Reducing()))
```

```text
ValueError: Reducing.value returned shape (1, 200, 1), expected (1, 200, 2).
A kernel is applied elementwise, so it must broadcast against its inputs and
must not reduce over any axis.
```

```{important}
Do not sum or multiply across variables inside `value`.
```

### Use JAX control flow

Because `value` receives JAX arrays, array-dependent branching should use JAX operations such as `jnp.where` rather than a Python `if`.

This will fail.

```python
@dataclasses.dataclass(frozen=True)
class Branching(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h

        if jnp.abs(u) <= 1.0:
            return 0.75 * (1.0 - u * u)

        return jnp.zeros_like(u)
```

```python
Branching().value(x[:3], 0.0, 0.2)
```

```text
ValueError: The truth value of an array with more than one element is ambiguous.
Use a.any() or a.all()
```

The elementwise `jnp.where` version handles both broadcasting and JAX transformations correctly. How many evaluation points arrive at once is the estimator's concern. A well-written kernel should not need to inspect or manipulate the leading axes directly.

## Why kernels are frozen dataclasses

The `frozen=True` in the first example is functional rather than stylistic. A `KernelSet` is passed to compiled JAX functions as a static argument, so its kernels must be hashable. A frozen dataclass is the simplest way to get predictable equality and hashing behavior. This class is mutable and therefore not hashable.

```python
@dataclasses.dataclass
class Mutable(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h

        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)

kj.KernelSet(continuous=Mutable())
```

```text
TypeError: continuous kernel Mutable is not hashable, so it cannot be a static
argument. Decorate it with @dataclasses.dataclass(frozen=True), and hold any
array field as a tuple of floats rather than as a jax.Array.
```

A plain Python class is technically hashable by identity, but two otherwise identical instances then compare as different objects. Creating a fresh instance can therefore cause an unnecessary JAX compilation cache miss. For custom kernels with configuration, prefer immutable Python values such as floats, integers, strings, and tuples rather than `jax.Array` fields.

## Using a custom kernel for density estimation

The Epanechnikov kernel already works for local polynomial regression. Density estimation adds an important requirement. A regression estimator is a ratio of kernel-weighted quantities, so a common multiplicative constant cancels. Density estimation does not have that property.

For a continuous kernel used in a density, two things must hold.

1. `value` must integrate to one in standardized $u$-space.
2. `value` must not include its own $1/h$ factor.

KernelJax applies the bandwidth normalization itself. Our Epanechnikov implementation satisfies both conditions.

```python
dens = kj.density(x, "cv_ml", kernels=kernels)
print(f"h={dens.bandwidth.h[0]:.6f}")
```

```text
h=0.040642
```

### KernelJax handles the `1/h`

Notice that `value` was written as

```python
def value(self, x, y, h):
    u = (x - y) / h
    return 0.75 * (1.0 - u * u)
```

rather than

```python
def value(self, x, y, h):
    u = (x - y) / h
    return 0.75 * (1.0 - u * u) / h
```

KernelJax divides by the continuous bandwidths exactly once when the estimator requires it.

The same convention applies to `value`, `conv`, and `cdf`. The exception is `deriv`, which differentiates with respect to the original evaluation coordinate and therefore carries the corresponding chain-rule factor.

### Normalization is checked when it matters

A regression fit cannot detect a common scaling error because that scale cancels from the ratio. A kernel scaled to half its mass makes the point.

```python
@dataclasses.dataclass(frozen=True)
class HalfMass(kj.ContinuousKernel):
    def value(self, x, y, h):
        return 0.5 * Epanechnikov().value(x, y, h)
```

This kernel can still produce a regression fit, but its total mass is only one half. Density estimation catches the problem.

```python
kj.density(x, "cv_ml", kernels=kj.KernelSet(continuous=HalfMass()))
```

```text
ValueError: HalfMass.value integrates to 0.5000 in u units rather than one, so
every density it produces is scaled by that factor. A regression fit is a ratio
and cancels the constant, which is why this only fires from a density.
```

KernelJax also checks for a kernel that applies its own bandwidth normalization.

```python
@dataclasses.dataclass(frozen=True)
class SelfNormalizing(kj.ContinuousKernel):
    def value(self, x, y, h):
        return Epanechnikov().value(x, y, h) / h
```

```python
kj.density(x, "cv_ml", kernels=kj.KernelSet(continuous=SelfNormalizing()))
```

```text
ValueError: SelfNormalizing.value integrates to one at h=1 but to 0.5000 at
h=2, the signature of a kernel carrying its own 1/h factor. Return the kernel
in u units with no normalization by h, since the estimator divides by h
exactly once itself.
```

These checks run when a feature first requires the corresponding property rather than imposing every possible requirement on every regression kernel.

## Categorical kernels

Categorical kernels use the same general extension pattern and add one method,
`upper_bound(levels)`. Its meaning is important.

`upper_bound` returns the value of the smoothing parameter $\lambda$ at which every category receives the same weight. At that point, the variable no longer influences the estimate. That value belongs to the kernel's parameterization, not to the data.

### Unordered categories

For the default Aitchison-Aitken kernel with $c$ categories, complete pooling occurs at

$$
\lambda = \frac{c-1}{c}.
$$

Suppose instead we use the simpler parameterization that assigns weight 1 to a match and $\lambda$ otherwise.

```python
@dataclasses.dataclass(frozen=True)
class Plain(kj.UnorderedKernel):
    """Weight 1 on a match and lam otherwise."""

    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, lam)

    def upper_bound(self, levels):
        return 1.0
```

At $\lambda = 1$, every category receives weight 1, so the variable is completely smoothed out. We can compare it with Aitchison-Aitken.

```python
rng = np.random.default_rng(0)
exper = rng.uniform(0, 30, 200)
region = rng.integers(0, 4, 200)
wage = 2.0 + 0.1 * exper + region + rng.normal(0, 0.5, 200)

data = kj.MixedData.from_blocks(
    continuous=exper, unordered=region, unordered_levels=4, names=("exper", "region")
)

plain = Plain()
aitchison = kj.AitchisonAitken()
custom = kj.local_poly(data, wage, "cv_ls", degree=1, kernels=kj.KernelSet(unordered=plain))
shipped = kj.local_poly(data, wage, "cv_ls", degree=1)

print(
    f"Plain            "
    f"lam={custom.bandwidth.lam_uno[0]:.6f}  "
    f"bound={plain.upper_bound(4):.2f}  "
    f"r2={custom.r_squared:.6f}"
)

print(
    f"AitchisonAitken  "
    f"lam={shipped.bandwidth.lam_uno[0]:.6f}  "
    f"bound={aitchison.upper_bound(4):.2f}  "
    f"r2={shipped.r_squared:.6f}"
)
```

```text
Plain            lam=0.001234  bound=1.00  r2=0.897111
AitchisonAitken  lam=0.003673  bound=0.75  r2=0.897111
```

The fitted regressions agree, while the numerical values of $\lambda$ differ because the two kernels parameterize smoothing differently. That is why each categorical kernel defines its own `upper_bound`.

Returning the wrong bound can change the search space itself. If it is too large, the optimizer can move beyond complete pooling into a region where matching categories receive less weight than nonmatching ones. If it is too small, valid smoothing levels become inaccessible.

### Ordered categories

Ordered kernels have the same interface, except that the distance between category codes carries information. For example, weight can decay geometrically with level distance.

```python
@dataclasses.dataclass(frozen=True)
class Geometric(kj.OrderedKernel):
    """Weight decays geometrically with level distance."""

    def value(self, x, y, lam, levels):
        return lam ** jnp.abs(x - y)

    def upper_bound(self, levels):
        return 1.0
```

When $\lambda = 0$, only exact matches receive weight. As $\lambda$ approaches 1, increasingly distant levels are pooled together. At $\lambda = 1$, every level receives equal weight.

### What categorical kernels receive

Categorical columns are stored as contiguous, zero-based integer codes.

```text
0, 1, ..., levels - 1
```

A custom kernel sees those codes rather than the original labels.

For ordered variables, code order must therefore match category order. The expression

```python
jnp.abs(x - y)
```

then represents the number of levels separating two categories.

{func}`~kerneljax.MixedData.from_blocks` validates the codes against the declared number of levels and rejects degenerate categorical columns. The `levels` argument arrives as a regular Python integer, so it can safely participate in Python control flow. The smoothing parameter `lam`, by contrast, may be traced by JAX and should be handled with JAX operations.

`value` always receives the evaluation point first and the training point second.
KernelJax does not require kernels to be symmetric. If you intentionally implement an asymmetric kernel, that argument order therefore matters.

## Add capabilities when you need them

`value` is the core interface. By itself it supports the following.

* local polynomial regression at any degree
* regression gradients through `gradient=True`
* likelihood cross validation
* `normal_reference`
* {func}`~kerneljax.summary`

Other methods enable additional features.

| Method  | Needed for                                                    | Meaning                                              |
| ------- | ------------------------------------------------------------- | ---------------------------------------------------- |
| `conv`  | `density(..., "cv_ls")` and `local_poly(..., se=True)`        | Self-convolution of `value`                          |
| `cdf`   | {func}`~kerneljax.cdf` and its selection criterion            | Integrated kernel                                    |
| `deriv` | derivative weight tensors through {func}`~kerneljax.kweights` | Derivative with respect to the evaluation coordinate |

If a requested feature needs a method your kernel does not implement, KernelJax raises `NotImplementedError` and names the missing method.

### Adding `conv`

For example, our first Epanechnikov implementation does not provide `conv`. Requesting regression standard errors names the missing method.

```python
kj.local_poly(x, y, "cv_ls", degree=1, kernels=kernels, se=True, n_starts=1)
```

```text
NotImplementedError: Epanechnikov does not implement conv
```

For Epanechnikov, the self-convolution has the closed form

$$
(k * k)(u) = \frac{3}{160}(2 - |u|)^3(u^2 + 6|u| + 4), \quad |u| \le 2.
$$

We can add it in a subclass.

```python
@dataclasses.dataclass(frozen=True)
class EpanechnikovConv(Epanechnikov):
    def conv(self, x, y, h):
        u = jnp.abs(x - y) / h

        piece = 3.0 / 160.0 * (2.0 - u) ** 3 * (u * u + 6.0 * u + 4.0)

        return jnp.where(u <= 2.0, piece, 0.0)
```

Now both least squares density bandwidth selection and regression standard errors are available.

```python
conv_kernels = kj.KernelSet(continuous=EpanechnikovConv())
dens = kj.density(x, "cv_ls", kernels=conv_kernels, n_starts=1)
fit = kj.local_poly(x, y, "cv_ls", degree=1, kernels=conv_kernels, se=True, n_starts=1)
print(f"density     h={dens.bandwidth.h[0]:.6f}")
print(f"regression  mean se={fit.se.mean():.6f}")
```

```text
density     h=0.203672
regression  mean se=0.043252
```

One detail is easy to miss. Convolution expands support. If `value` is supported on $|u| \le 1$, its self-convolution is supported on $|u| \le 2$. KernelJax validates `conv` against `value` when a feature first requires it, so truncating the convolution at the original kernel support is caught automatically.

For regression standard errors, only the continuous kernel's convolution is required. In
particular, $(k * k)(0) = \int k(u)^2 \, du = R(k)$, which is the kernel quantity entering
the variance calculation.

## Bandwidth selection and gradients

Optimization-based bandwidth selectors differentiate their criterion through the kernel. That means a custom `value` can return perfectly reasonable numbers and still fail during selection if its gradient contains a `nan`. A common source is `jnp.where`. Consider the sinc function.

```python
def unsafe(u):
    return jnp.where(u == 0.0, 1.0, jnp.sin(u) / u)
```

The value at zero looks safe, but the unused branch still contains division by zero. Compare it with a version that guards the argument itself.

```python
def safe(u):
    nonzero_u = jnp.where(u == 0.0, 1.0, u)

    return jnp.where(u == 0.0, 1.0, jnp.sin(nonzero_u) / nonzero_u)
```

```python
unsafe_value, unsafe_grad = jax.value_and_grad(unsafe)(0.0)
safe_value, safe_grad = jax.value_and_grad(safe)(0.0)
print(f"unsafe  value={unsafe_value:.4f}  d/du={unsafe_grad:.4f}")
print(f"safe    value={safe_value:.4f}  d/du={safe_grad:.4f}")
```

```text
unsafe  value=1.0000  d/du=nan
safe    value=1.0000  d/du=0.0000
```

The forward values agree, but only one implementation has a valid gradient.

```{important}
Guard the unsafe expression itself, not only the branch that returns it.
```

KernelJax probes custom kernels before estimator-driven bandwidth selection and rejects non-finite bandwidth gradients early. For example, wrap the sinc into a kernel and hand it to selection.

```python
@dataclasses.dataclass(frozen=True)
class Sinc(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h

        return jnp.where(u == 0.0, 1.0, jnp.sin(u) / u)
```

```python
sinc_kernels = kj.KernelSet(continuous=Sinc())
kj.local_poly(x, y, "cv_ls", degree=1, kernels=sinc_kernels)
```

```text
ValueError: Sinc.value has a non-finite bandwidth gradient at |x - y| = 0, a
separation any sample can contain. jnp.where differentiates both branches, so
guard the argument inside the untaken branch, not just the branch.
```

Nondifferentiability by itself is not necessarily a problem. Compactly supported kernels can contain kinks. What matters for optimization is that the values and gradients encountered by the selector remain finite.

## Choose a compatible bandwidth selector

A kernel can satisfy the KernelJax interface and still be a poor match for a particular bandwidth-selection criterion.

### Likelihood cross validation needs positive densities

Likelihood cross validation contains a term of the form

$$
\log \hat f_{-i}(X_i).
$$

The leave-one-out density therefore needs to remain strictly positive at every training observation. A compactly supported kernel can easily assign zero density to an isolated observation. Even a Gaussian kernel can underflow to exactly zero far enough into its tail in single precision.

Higher-order kernels introduce another problem. Because they can take negative values, a leave-one-out density estimate can itself become nonpositive.

In those situations, least squares cross validation does not take a logarithm and is
generally the safer criterion, so select with `"cv_ls"` rather than `"cv_ml"`. The
distinction is statistical rather than an interface restriction.

### Treat `normal_reference` cautiously

`normal_reference` is also worth treating carefully with unusual custom kernels. For continuous kernels it reads an `order` attribute when one is available and otherwise assumes order 2. Cross-validation methods mainly use the resulting rule as an initialization. With

```python
bw = "normal_reference"
```

the plug-in bandwidth itself is the final answer.

The reference-rule constants are based on the Gaussian rather than recalibrated for an arbitrary custom kernel. For that reason, `normal_reference` is best treated as a quick starting point or rough benchmark when using a substantially different kernel rather than as a kernel-specific optimal bandwidth.

## Common mistakes

Most custom-kernel problems come from a small set of interface mismatches, and each row of this table is demonstrated earlier on the page with the error it produces.

| Mistake                              | Symptom                                            | Fix                                                     |
| ------------------------------------ | -------------------------------------------------- | ------------------------------------------------------- |
| Reducing across columns in `value`   | `returned shape (1, 200, 1), expected (1, 200, 2)` | Return one factor per column and let KernelJax multiply |
| Including `1/h` in `value`           | `integrates to one at h=1 but to 0.5000 at h=2`    | Return the kernel in standardized $u$-space             |
| Python `if` on array values          | `The truth value of an array ... is ambiguous`     | Branch with `jnp.where`                                 |
| Guarding only the visible branch     | `non-finite bandwidth gradient` at zero separation | Guard the argument inside the untaken branch            |
| A wrong categorical `upper_bound`    | The search misses or overshoots complete pooling   | Return the $\lambda$ where all levels weigh equally     |
| Mutable or array-valued state        | `not hashable, so it cannot be a static argument`  | Freeze the dataclass and hold plain Python values       |

## Interface at a glance

A custom kernel only needs to implement the parts of the interface required by the estimators you intend to use.

| Requirement                                          | Applies to                                             | When it matters                              |
| ---------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------- |
| `value` is elementwise and preserves broadcast shape | all kernels                                            | every estimator                              |
| kernel object is hashable                            | all kernels                                            | `KernelSet` construction and JAX compilation |
| finite values and bandwidth gradients                | all kernels used in selection                          | optimization-based bandwidth selection       |
| no `1/h` inside `value`                              | continuous kernels                                     | density and distribution estimation          |
| `value` integrates to one in $u$-space               | continuous kernels                                     | density and distribution estimation          |
| `upper_bound(levels)`                                | categorical kernels                                    | categorical bandwidth selection              |
| `conv`                                               | kernels used where convolution is required             | density LS-CV and regression standard errors |
| `cdf`                                                | continuous and ordered kernels used for CDF estimation | `cdf` and its criterion                      |
| `deriv`                                              | kernels used for derivative weight tensors             | `kweights(..., op=...)`                      |

The shortest useful custom continuous kernel is therefore still just the following.

```python
@dataclasses.dataclass(frozen=True)
class MyKernel(kj.ContinuousKernel):
    def value(self, x, y, h): ...
```

Start there. Add `conv`, `cdf`, or `deriv` only when the estimator you are building needs them.
