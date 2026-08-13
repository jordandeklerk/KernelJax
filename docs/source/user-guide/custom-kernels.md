# Custom kernels

Every KernelJax estimator accepts a `kernels` argument. It is a {class}`~kerneljax.KernelSet` containing the kernel families used for continuous, unordered, and ordered variables. By default, KernelJax uses a second-order Gaussian kernel for continuous variables, Aitchison-Aitken for unordered variables, and Li-Racine for ordered variables. Any of those can be replaced without changing the estimator itself.

A custom kernel does not need to be registered with KernelJax. Define it, place it in a `KernelSet`, and pass it directly to an estimator.

This page starts with an Epanechnikov kernel, uses it in regression and density estimation, and then develops the interface required for custom kernels to work with the rest of the library. For the statistical role of a kernel, see [Kernel smoothing](../background/smoothing.md).

## Your first custom kernel

A continuous kernel subclasses {class}`~kerneljax.ContinuousKernel` and implements `value`.

The Epanechnikov kernel is

$$
k(u) = \frac{3}{4}\,(1-u^2)\,\mathbf{1}_{|u|\leq 1}.
$$

The corresponding KernelJax implementation is short.

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
        return jnp.where(
            jnp.abs(u) <= 1.0,
            0.75 * (1.0 - u * u),
            0.0,
        )
```

It can be used immediately in a regression.

```python
rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)

kernels = kj.KernelSet(continuous=Epanechnikov())

epan = kj.local_poly(
    x,
    y,
    "cv_ls",
    degree=1,
    kernels=kernels,
)

gauss = kj.local_poly(
    x,
    y,
    "cv_ls",
    degree=1,
)

print(f"Epanechnikov  h={epan.bandwidth.h[0]:.6f}  r2={epan.r_squared:.6f}")
print(f"Gaussian      h={gauss.bandwidth.h[0]:.6f}  r2={gauss.r_squared:.6f}")
```

```text
Epanechnikov  h=0.084645  r2=0.947231
Gaussian      h=0.036019  r2=0.947953
```

The numerical bandwidths are not directly comparable because bandwidth is measured on the scale of the kernel. A larger Epanechnikov bandwidth therefore does not by itself imply a smoother fitted regression. The two fits are very similar here. For ordinary second-order kernels, bandwidth choice generally has a much larger effect on the estimate than choosing between reasonable kernel shapes.

## What `value` must do

The `Epanechnikov` class above is already enough for ordinary local polynomial regression. More generally, a continuous `value` implementation should follow a few rules.

* It must operate elementwise and preserve the broadcast shape of its inputs.
* It returns the kernel in standardized $u$-space, with no `1/h` normalization of its own.
* Its object must be hashable because kernels are static JAX configuration.
* Its values must remain finite wherever the estimator uses them.
* If bandwidth selection differentiates through `value`, the corresponding bandwidth derivatives must also remain finite.
* For density estimation, a continuous `value` must integrate to one in $u$-space.

Categorical kernels follow the same elementwise pattern and additionally implement `upper_bound(levels)`. Methods such as `conv`, `cdf`, and `deriv` add capabilities beyond the core `value` operation. They only need to be implemented when the computation you request actually uses them.

## How `value` is called

For a continuous kernel, `value` is called across all continuous columns at once. Its inputs broadcast approximately as

```
x:  (n_eval,  1,       p_con)
y:  (1,       n_train, p_con)
h:  broadcast against both
```

where `p_con` is the number of continuous variables.

The important rule is simpler than the shapes make it look.

```{tip}
Write `value` elementwise and return one kernel factor per continuous column. KernelJax multiplies those factors across columns for you.
```

The Epanechnikov implementation does exactly that.

```python
@dataclasses.dataclass(frozen=True)
class Epanechnikov(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(
            jnp.abs(u) <= 1.0,
            0.75 * (1.0 - u * u),
            0.0,
        )
```

This implementation does not.

```python
@dataclasses.dataclass(frozen=True)
class Reducing(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.sum(
            jnp.exp(-0.5 * u * u),
            axis=-1,
            keepdims=True,
        )
```

On a sample with two continuous columns, it collapses the column axis before KernelJax has a chance to form the product kernel.

```python
two_columns = kj.MixedData.continuous(
    np.column_stack([x, x**2])
)

wide = kj.Bandwidth(
    h=jnp.array([0.2, 0.2]),
    lam_uno=jnp.zeros(0),
    lam_ord=jnp.zeros(0),
)

kj.local_poly(
    two_columns,
    y,
    wide,
    degree=1,
    kernels=kj.KernelSet(continuous=Reducing()),
)
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

Because `value` receives JAX arrays, array-dependent branching should use JAX operations rather than a Python `if`.

This implementation will fail on an array.

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

The elementwise `jnp.where` version works under broadcasting and JAX transformations without needing to know how many evaluation or training points happen to arrive at once.

## Keep kernels static

The `frozen=True` in the first example is functional rather than stylistic. A `KernelSet` is passed into compiled JAX functions as static configuration, so the kernel objects it contains must be hashable.

A mutable dataclass is not.

```python
@dataclasses.dataclass
class Mutable(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(
            jnp.abs(u) <= 1.0,
            0.75 * (1.0 - u * u),
            0.0,
        )


kj.KernelSet(continuous=Mutable())
```

```text
TypeError: continuous kernel Mutable is not hashable, so it cannot be a static
argument. Decorate it with @dataclasses.dataclass(frozen=True), and hold any
array field as a tuple of floats rather than as a jax.Array.
```

A plain Python object may be hashable by identity, but two otherwise identical instances then count as different static objects. Constructing fresh instances can therefore produce unnecessary compilation-cache misses. For configurable kernels, prefer immutable Python fields such as floats, integers, strings, and tuples. Avoid storing `jax.Array` values as static dataclass fields.

## Normalization matters when the weights do not cancel

The Epanechnikov kernel already works for regression. Density estimation imposes a stronger requirement because it uses the kernel weights without dividing by another sum of the same weights.

Suppose `value` returns some function $g(u)$ with $u=(x-y)/h$. For one continuous variable, KernelJax forms the density

$$
\hat f(x)
=
\frac{1}{n h}
\sum_{i=1}^n
g\!\left(
\frac{x - X_i}{h}
\right).
$$

KernelJax supplies the `1/h` itself. Changing variables from $x$ to $u$ shows that the total mass is determined entirely by `value`.

$$
\int \hat f(x) \, dx = \int g(u) \, du.
$$

A valid continuous density kernel therefore needs $\int g(u)\,du = 1$ and must not include another `1/h` inside `value`.

Regression is different. Multiplying all of its local weights by the same constant changes neither the numerator nor denominator of a local-constant estimate and, more generally, leaves the weighted least-squares coefficients unchanged. A normalization mistake can therefore be invisible to regression while making a density incorrect.

Our Epanechnikov implementation already has unit mass.

```python
dens = kj.density(
    x,
    "cv_ml",
    kernels=kernels,
)

print(f"h={dens.bandwidth.h[0]:.6f}")
```

```text
h=0.040642
```

### KernelJax supplies the `1/h`

The correct implementation returns only the kernel in standardized units.

```python
def value(self, x, y, h):
    u = (x - y) / h
    return jnp.where(
        jnp.abs(u) <= 1.0,
        0.75 * (1.0 - u * u),
        0.0,
    )
```

It should not return

```python
def value(self, x, y, h):
    u = (x - y) / h
    return jnp.where(
        jnp.abs(u) <= 1.0,
        0.75 * (1.0 - u * u) / h,
        0.0,
    )
```

because the density estimator would then divide by the bandwidth twice.

The same standardized-coordinate convention applies to `value`, `cdf`, and `conv`. The exception is `deriv`, which is defined as a derivative with respect to the original evaluation coordinate.

If $u = (x - y)/h$, the four continuous operators follow

$$
\begin{aligned}
\texttt{value} & = k(u), \\\\
\texttt{cdf} & = \int_{-\infty}^u k(t)\,dt, \\\\
\texttt{conv} & = (k * k)(u), \\\\
\texttt{deriv} & = \frac{1}{h}k'(u).
\end{aligned}
$$

Only `deriv` carries the additional bandwidth factor because $\partial u/\partial x = 1/h$.

### Normalization is checked when it matters

A regression fit cannot detect a common scaling error because the scale cancels. A half-mass kernel makes that visible.

```python
@dataclasses.dataclass(frozen=True)
class HalfMass(kj.ContinuousKernel):
    def value(self, x, y, h):
        return 0.5 * Epanechnikov().value(x, y, h)
```

It can still produce a regression fit, but density estimation rejects it.

```python
kj.density(
    x,
    "cv_ml",
    kernels=kj.KernelSet(continuous=HalfMass()),
)
```

```text
ValueError: HalfMass.value integrates to 0.5000 in u units rather than one, so
every density it produces is scaled by that factor. A regression fit is a ratio
and cancels the constant, which is why this only fires from a density.
```

KernelJax also detects the signature of a kernel that includes its own bandwidth normalization.

```python
@dataclasses.dataclass(frozen=True)
class SelfNormalizing(kj.ContinuousKernel):
    def value(self, x, y, h):
        return Epanechnikov().value(x, y, h) / h
```

```python
kj.density(
    x,
    "cv_ml",
    kernels=kj.KernelSet(continuous=SelfNormalizing()),
)
```

```text
ValueError: SelfNormalizing.value integrates to one at h=1 but to 0.5000 at
h=2, the signature of a kernel carrying its own 1/h factor. Return the kernel
in u units with no normalization by h, since the estimator divides by h
exactly once itself.
```

Those checks run when density estimation first requires unit mass rather than imposing density-specific restrictions on every regression kernel.

### Distribution estimation uses `cdf`

A distribution estimator does not normalize `value` by `1/h` and then integrate it numerically. It asks the kernel for its cumulative operator directly.

For a continuous kernel, `cdf(x, y, h)` should return

$$
F_k(u) = \int_{-\infty}^u k(t)\,dt, \qquad u = \frac{x - y}{h}.
$$

A valid cumulative operator therefore approaches zero as $u \to -\infty$ and one as $u \to \infty$. KernelJax checks those limits when the unconditional distribution estimator first needs them. That check does not establish every possible relationship between `cdf` and `value`. A custom kernel is still responsible for making `cdf` the actual cumulative form of the `value` it represents.

## Categorical kernels

Categorical kernels follow the same elementwise design but replace a continuous bandwidth with a smoothing parameter $\lambda$. They also implement `upper_bound(levels)`. This gives the upper end of the smoothing range, or complete-pooling limit, where distinctions among category levels disappear. Some parameterizations reach that state at a usable finite value, while others approach it only as $\lambda$ reaches the edge of its admissible interval. The bound belongs to the kernel parameterization, not to the observed data.

### Unordered categories

For Aitchison-Aitken with $c$ categories, the complete-pooling value is $\lambda=(c-1)/c$.

A simpler regression kernel could instead give weight one to a match and $\lambda$ to a mismatch.

```python
@dataclasses.dataclass(frozen=True)
class Plain(kj.UnorderedKernel):
    """Weight 1 on a match and lam otherwise."""

    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, lam)

    def upper_bound(self, levels):
        return 1.0
```

At $\lambda = 1$, every category receives the same weight, so the column no longer affects a regression fit.

```python
rng = np.random.default_rng(0)
exper = rng.uniform(0, 30, 200)
region = rng.integers(0, 4, 200)
wage = 2.0 + 0.1 * exper + region + rng.normal(0, 0.5, 200)

data = kj.MixedData.from_blocks(
    continuous=exper,
    unordered=region,
    unordered_levels=4,
    names=("exper", "region"),
)

plain = Plain()
aitchison = kj.AitchisonAitken()

custom = kj.local_poly(
    data,
    wage,
    "cv_ls",
    degree=1,
    kernels=kj.KernelSet(unordered=plain),
)

shipped = kj.local_poly(
    data,
    wage,
    "cv_ls",
    degree=1,
)

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

The fitted regressions agree even though the numerical smoothing parameters differ. The kernels use different parameterizations of the same basic idea, which is why every categorical kernel defines its own `upper_bound`.

`Plain` is intentionally sufficient for this regression example, but it is **not normalized as a categorical probability kernel**. Summing its weights over the $c$ possible evaluation levels gives $1 + (c-1)\lambda$, not one. That common factor cancels from regression, but it would matter in a density. If a custom unordered kernel is intended for density estimation as well, its mass should sum to one over its declared levels.

Returning the wrong upper bound creates a separate problem. A bound that is too low makes valid smoothing levels unreachable. A bound that extends beyond complete pooling can let the optimizer enter a region where the kernel reverses its intended notion of similarity.

### Ordered categories

Ordered kernels use the same interface, except that distance between level codes carries statistical meaning.

For a simple regression-only example,

```python
@dataclasses.dataclass(frozen=True)
class Geometric(kj.OrderedKernel):
    """Weight decays geometrically with level distance."""

    def value(self, x, y, lam, levels):
        return lam ** jnp.abs(x - y)

    def upper_bound(self, levels):
        return 1.0
```

When $\lambda = 0$, only exact matches receive weight. As $\lambda$ approaches one, more distant levels receive increasingly similar weights. At the complete-pooling boundary $\lambda=1$, every level receives the same value.

Like `Plain`, this example omits a normalization constant because a common multiplicative factor is irrelevant to regression. A version intended to define a proper density or mass function should be normalized over the support on which the kernel is defined.

### What categorical kernels receive

Categorical columns are represented internally by contiguous, zero-based integer codes,

```
0, 1, ..., levels - 1
```

and a custom kernel receives those codes rather than the original labels.

For an ordered variable, the code order must therefore agree with the category order. An expression such as

```python
jnp.abs(x - y)
```

then measures the number of coded levels separating two categories.

{func}`~kerneljax.MixedData.from_blocks` validates the codes against the declared level counts. The `levels` argument passed to a kernel is an ordinary Python integer and can safely participate in static Python control flow. The smoothing parameter `lam` may be traced by JAX, so calculations depending on its value should use JAX operations.

`value` always receives the evaluation value first and the training value second. KernelJax does not require a kernel's `value` operation to be symmetric, so that argument order matters for intentionally asymmetric constructions.

## Add capabilities when you need them

`value` is the core operator, but different estimators require different additions.

| Method        | Used by                                                                                               | Meaning                                              |
| ------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `value`       | regression, density, conditional density, conditional mode, and their value-based criteria            | Pointwise kernel value                               |
| `conv`        | density LS-CV, response side of conditional-density LS-CV, and continuous regression standard errors  | Self-convolution                                     |
| `cdf`         | unconditional CDFs, conditional distributions, conditional quantiles, and their distribution criteria | Cumulative kernel                                    |
| `deriv`       | derivative weight operations such as {func}`~kerneljax.kweights_grad`                                 | Derivative with respect to the evaluation coordinate |
| `upper_bound` | categorical bandwidth transformations and search                                                      | Complete-pooling boundary or limit                   |

One distinction is easy to miss. `local_poly(..., gradient=True)` does **not** require the kernel's `deriv` method. Local polynomial slopes come from the fitted polynomial coefficients. `deriv` is needed when you explicitly differentiate the kernel weights themselves.

If a requested operation calls a method your kernel does not implement, KernelJax raises `NotImplementedError` and identifies the missing method.

## Adding `conv`

Our first Epanechnikov kernel does not implement `conv`. Regression standard errors therefore cannot yet obtain the kernel roughness they need.

```python
kj.local_poly(
    x,
    y,
    "cv_ls",
    degree=1,
    kernels=kernels,
    se=True,
    n_starts=1,
)
```

```text
NotImplementedError: Epanechnikov does not implement conv
```

For the Epanechnikov kernel, the self-convolution is

$$
(k*k)(u) =
\frac{3}{160}
(2 - |u|)^3
\left(u^2 + 6|u| + 4\right),
\qquad |u| \leq 2,
$$

and zero outside that support.

```python
@dataclasses.dataclass(frozen=True)
class EpanechnikovConv(Epanechnikov):
    def conv(self, x, y, h):
        u = jnp.abs(x - y) / h
        piece = (
            3.0
            / 160.0
            * (2.0 - u) ** 3
            * (u * u + 6.0 * u + 4.0)
        )

        return jnp.where(u <= 2.0, piece, 0.0)
```

Now density least-squares cross-validation and regression standard errors can use the kernel.

```python
conv_kernels = kj.KernelSet(
    continuous=EpanechnikovConv()
)

dens = kj.density(
    x,
    "cv_ls",
    kernels=conv_kernels,
    n_starts=1,
)

fit = kj.local_poly(
    x,
    y,
    "cv_ls",
    degree=1,
    kernels=conv_kernels,
    se=True,
    n_starts=1,
)

print(f"density     h={dens.bandwidth.h[0]:.6f}")
print(f"regression  mean se={fit.se.mean():.6f}")
```

```text
density     h=0.203672
regression  mean se=0.043252
```

Convolution expands support. A kernel supported on $|u| \leq 1$ has a self-convolution supported on $|u| \leq 2$.

KernelJax checks this to different degrees depending on what the caller needs. Density least-squares cross-validation compares the continuous `conv` against the numerical self-convolution over a range of offsets, so truncating it at the original support is detected. Regression standard errors only require `conv` at zero and therefore verify the identity

$$
(k * k)(0) = \int k(u)^2 \, du = R(k).
$$

That narrower check is enough because $R(k)$ is the only convolution quantity used by the standard-error calculation.

### Why standard errors need `conv`

Write $w_i(x)$ for the product-kernel weight of training observation $i$ at evaluation point $x$. KernelJax first estimates a local response variance

$$
\hat\sigma^2(x)
=
\frac{\sum_i w_i(x) Y_i^2}{\sum_i w_i(x)}
-
\left[
\frac{\sum_i w_i(x) Y_i}{\sum_i w_i(x)}
\right]^2.
$$

If there are $p$ continuous columns using the same kernel family, the roughness contribution is $R(k)^p$, and KernelJax reports

$$
\widehat{\operatorname{se}}\!\left(\hat m(x)\right)
=
\sqrt{
\frac{
\hat\sigma^2(x) R(k)^p
}{
\sum_i w_i(x)
}
}.
$$

The local variance is formed from the constant basis row regardless of the polynomial degree being fitted. The earlier [Local polynomial regression](regression.md#pointwise-standard-errors) page discusses the consequences of that approximation in more detail.

In the one-dimensional example above, $\sum_i w_i(x)$ behaves like $n h f(x)$, so this has the familiar leading form proportional to $R(k)\sigma^2(x) / (nhf(x))$.

## Bandwidth selection needs usable gradients

Optimization-based bandwidth selectors differentiate their criteria through the kernel operations they use. A custom method can therefore produce perfectly finite forward values but still fail during bandwidth selection if its derivative with respect to the bandwidth becomes non-finite.

A common source is an unsafe expression hidden inside `jnp.where`.

```python
def unsafe(u):
    return jnp.where(
        u == 0.0,
        1.0,
        jnp.sin(u) / u,
    )
```

The selected value at zero is finite, but the other branch still contains a division by zero that can contaminate automatic differentiation.

Guard the unsafe expression itself.

```python
def safe(u):
    nonzero_u = jnp.where(
        u == 0.0,
        1.0,
        u,
    )

    return jnp.where(
        u == 0.0,
        1.0,
        jnp.sin(nonzero_u) / nonzero_u,
    )
```

```python
unsafe_value, unsafe_grad = jax.value_and_grad(unsafe)(0.0)
safe_value, safe_grad = jax.value_and_grad(safe)(0.0)

print(
    f"unsafe  value={unsafe_value:.4f}  "
    f"d/du={unsafe_grad:.4f}"
)

print(
    f"safe    value={safe_value:.4f}  "
    f"d/du={safe_grad:.4f}"
)
```

```text
unsafe  value=1.0000  d/du=nan
safe    value=1.0000  d/du=0.0000
```

The forward values agree, but only the second implementation has a finite derivative.

```{important}
Guard the unsafe expression itself, not only the branch that selects it.
```

KernelJax proactively probes the bandwidth derivative of a continuous `value` at several representative separations before estimator-driven optimization that uses it.

```python
@dataclasses.dataclass(frozen=True)
class Sinc(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(
            u == 0.0,
            1.0,
            jnp.sin(u) / u,
        )
```

```python
sinc_kernels = kj.KernelSet(
    continuous=Sinc()
)

kj.local_poly(
    x,
    y,
    "cv_ls",
    degree=1,
    kernels=sinc_kernels,
)
```

```text
ValueError: Sinc.value has a non-finite bandwidth gradient at |x - y| = 0, a
separation any sample can contain. jnp.where differentiates both branches, so
guard the argument inside the untaken branch, not just the branch.
```

Similar validation is applied to the continuous `cdf` when unconditional distribution bandwidth selection uses it. These probes are guards against common failures, not a proof that every derivative throughout an optimization path will be well behaved. A non-finite derivative elsewhere can still make the criterion fail during the solve.

A kink by itself is not necessarily a problem. Compact-support kernels such as Epanechnikov are not smooth everywhere. What matters computationally is that the objective values and derivatives actually encountered by the selector remain finite and useful.

## Choose a compatible bandwidth selector

A kernel can satisfy the interface and still be a poor match for a particular selection criterion.

### Likelihood cross-validation needs positive density

Likelihood cross-validation contains terms of the form $\log\hat f_{-i}(X_i)$. Every held-out density entering the logarithm must therefore be strictly positive. A compactly supported kernel can assign exactly zero density to an isolated observation if no remaining point lies inside its support. A floating-point calculation can also underflow sufficiently small positive kernel values to zero. Signed higher-order kernels create another issue because a density estimate assembled from them can become nonpositive.

In those settings, least-squares cross-validation is usually the more natural numerical choice because it does not take a logarithm. Use `"cv_ls"` rather than `"cv_ml"` when positivity cannot be guaranteed.

### Treat `normal_reference` as a reference rule

`normal_reference` also deserves care with custom continuous kernels. It reads an `order` attribute when the kernel provides one and otherwise assumes order two.

For optimization-based selectors, the reference rule primarily supplies a starting scale. With

```python
bw = "normal_reference"
```

there is no optimization, so that reference bandwidth becomes the final answer.

The constants used by KernelJax are Gaussian reference constants rather than constants recalibrated for every custom kernel family. A substantially different kernel can therefore make `normal_reference` a useful rough benchmark without making it a kernel-specific optimal bandwidth. If the order of a custom continuous kernel is meaningful and differs from two, exposing it explicitly helps KernelJax choose the appropriate reference rate.

## Common mistakes

| Mistake                                           | Symptom                                          | Fix                                                       |
| ------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------- |
| Reducing across columns in `value`                | Wrong broadcast shape                            | Return one factor per column                              |
| Including `1/h` in `value`                        | Density mass changes with bandwidth              | Return the kernel in standardized $u$-space               |
| Python `if` on array values                       | Ambiguous truth-value or tracing error           | Use JAX control flow                                      |
| Guarding only the selected branch                 | Non-finite autodiff gradient                     | Guard the unsafe expression itself                        |
| Wrong categorical `upper_bound`                   | Search misses or moves beyond complete pooling   | Return the correct pooling boundary or limit              |
| Unnormalized categorical kernel used as a density | Total categorical mass is not one                | Normalize over the appropriate support                    |
| Mutable or array-valued static state              | Kernel is unhashable or recompiles unnecessarily | Use immutable Python configuration                        |
| Inconsistent `cdf` or `conv`                      | Distribution or LS-CV no longer matches `value`  | Implement the corresponding operator from the same kernel |

## Interface at a glance

A custom kernel only needs the pieces required by the estimators you intend to use.

| Requirement                          | Applies to                                               | When it matters                                             |
| ------------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------- |
| `value` preserves broadcast shape    | all kernels                                              | every value-based estimator                                 |
| kernel object is hashable            | all kernels                                              | `KernelSet` and JAX compilation                             |
| finite bandwidth derivatives         | operators used during optimized selection                | gradient-based bandwidth search                             |
| continuous `value` has no `1/h`      | continuous kernels                                       | density normalization                                       |
| continuous `value` integrates to one | continuous kernels                                       | density estimation                                          |
| categorical mass is normalized       | categorical kernels                                      | density or probability interpretation                       |
| `upper_bound(levels)`                | categorical kernels                                      | categorical bandwidth search                                |
| `conv`                               | kernels used by convolution-based computations           | density LS-CV, conditional-density LS-CV, regression SEs    |
| `cdf`                                | continuous and ordered cumulative kernels                | CDFs, conditional distributions, quantiles, distribution CV |
| `deriv`                              | continuous kernels used for derivative weight operations | `kweights_grad` and explicit derivative operators           |

The shortest useful custom continuous kernel is still only

```python
@dataclasses.dataclass(frozen=True)
class MyKernel(kj.ContinuousKernel):
    def value(self, x, y, h):
        ...
```

Start there. Add `conv`, `cdf`, or `deriv` when a computation actually needs them.
