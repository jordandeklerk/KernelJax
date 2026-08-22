# Custom kernels

Every KernelJax estimator accepts a `kernels` argument. It is a {class}`~kerneljax.KernelSet` containing the kernel families used for continuous, unordered, and ordered variables. By default, KernelJax uses a second-order Gaussian kernel for continuous variables, Aitchison-Aitken for unordered variables, and Li-Racine for ordered variables. Any of those can be replaced without changing the estimator itself.

A custom kernel does not need to be registered with KernelJax or JAX. Define it, place it in a `KernelSet`, and pass it directly to an estimator.

This page starts with the smallest useful continuous kernel, develops it into a complete Epanechnikov implementation, and then applies the same contracts to categorical kernels. It assumes familiarity with [data and column types](data.md), [bandwidth selection](selection.md), and the distinction between estimators and lower-level [kernel primitives](primitives.md). For the statistical role of a kernel, see [Kernel smoothing](../background/smoothing.md).

## Interface at a glance

Every custom kernel implements `value`. The other methods are task-specific.

| Requirement                          | When it is needed                                                                                     |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `value`                              | Regression, density, conditional density, conditional mode, and their value-based criteria           |
| Elementwise broadcast shape          | Every implemented operator                                                                           |
| Hashable kernel configuration        | Every kernel placed in a `KernelSet`                                                                  |
| Unit-mass continuous `value`          | Density estimation                                                                                    |
| Normalized categorical `value`       | Density or probability interpretation                                                                 |
| `conv`                               | Density LS-CV, response-side conditional-density LS-CV, and continuous regression standard errors    |
| `cdf`                                | Unconditional CDFs, conditional distributions, conditional quantiles, and distribution criteria      |
| `deriv`                              | Explicit derivative-weight operations through {func}`~kerneljax.kweights` with `op="deriv"`           |
| `upper_bound(levels)`                | Categorical bandwidth transformation and search                                                       |
| Finite bandwidth derivatives         | Every operator differentiated during optimized bandwidth selection                                   |

One distinction is easy to miss. `local_poly(..., gradient=True)` does **not** require the kernel's `deriv` method. Local polynomial slopes come from the fitted polynomial coefficients. `deriv` is needed when the kernel weights themselves are explicitly differentiated.

```{important}
A `KernelSet` contains one family per column kind, not one family per column. Its continuous kernel is applied to every continuous column, and the same unordered or ordered instance is reused for every column of that kind. Conditional estimators also use the same `KernelSet` for both the conditioning and response blocks.
```

The public interface therefore cannot assign different continuous kernel families to separate continuous columns, or one continuous family to `x` and another to `y`. Bandwidths and categorical level counts can still differ by column.

## Build a continuous kernel

### Implement `value`

A continuous kernel subclasses {class}`~kerneljax.ContinuousKernel` and implements `value`.

The Epanechnikov kernel is

$$
k(u) = \frac{3}{4}\,(1-u^2)\,\mathbf{1}_{|u|\leq 1}.
$$

The smallest corresponding KernelJax implementation is short.

```python
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import kerneljax as kj


@dataclasses.dataclass(frozen=True)
class EpanechnikovValue(kj.ContinuousKernel):
    """A value-only Epanechnikov kernel."""

    def value(self, x, y, h):
        u = (x - y) / h

        return jnp.where(jnp.abs(u) < 1.0, 0.75 * (1.0 - u * u), 0.0)
```

The same implementation handles scalars and arrays.

```python
kernel = EpanechnikovValue()
print(kernel.value(jnp.array([-0.5, 0.0, 1.5]), 0.0, 1.0))
```

```text
[0.5625 0.75   0.    ]
```

### Try a fixed bandwidth first

A fixed bandwidth separates the kernel interface from the additional requirements of an optimizer.

```python
rng = np.random.default_rng(1)
x = rng.uniform(size=200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 200)

fixed = kj.Bandwidth(h=jnp.array([0.1]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

value_kernels = kj.KernelSet(continuous=kernel)
fit = kj.local_poly(x, y, fixed, degree=1, kernels=value_kernels)

print(fit.mean.shape)
```

```text
(200,)
```

The value-only class is sufficient for ordinary local polynomial regression at a fixed bandwidth. Later sections add optional operators and then cover the separate requirements of optimized selection.

## Follow the core kernel contract

### Preserve the broadcast shape

Every kernel operator is elementwise. For a continuous kernel, KernelJax calls an operator across all continuous columns at once. Its inputs broadcast approximately as

```
x:  (n_eval,  1,       p_con)
y:  (1,       n_train, p_con)
h:  broadcast against both
```

where `p_con` is the number of continuous variables. The returned array must have the broadcast shape of `x`, `y`, and `h`, with one factor per continuous column. KernelJax multiplies those factors across columns.

The same shape-preservation rule applies to every implemented `value`, `cdf`, `conv`, or `deriv`. Categorical shapes are described with their input contract later on this page.

This implementation is wrong because it reduces the continuous-column axis itself.

```python
@dataclasses.dataclass(frozen=True)
class Reducing(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h

        return jnp.sum(jnp.exp(-0.5 * u * u), axis=-1, keepdims=True)
```

On a sample with two continuous columns, it collapses the column axis before KernelJax has a chance to form the product kernel.

```python
two_columns = kj.MixedData.continuous(np.column_stack([x, x**2]))

wide = kj.Bandwidth(h=jnp.array([0.2, 0.2]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

kj.local_poly(
    two_columns, y, wide, degree=1, kernels=kj.KernelSet(continuous=Reducing())
)
```

```text
ValueError: Reducing.value returned shape (1, 200, 1), expected (1, 200, 2).
A kernel is applied elementwise, so it must broadcast against its inputs and
must not reduce over any axis.
```

```{important}
Do not sum or multiply across variables inside a kernel operator.
```

`x` is always the evaluation value and `y` is the training value. Value-based computations do not require a symmetric kernel, so that order matters for intentionally asymmetric constructions. Convolution-based computations must additionally integrate the two `value` calls in their evaluation-first orientation, as described later on this page.

### Use JAX operations and control flow

Kernel methods receive JAX arrays. Array-dependent branching should therefore use JAX operations rather than a Python `if`.

This implementation is wrong.

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

The error's `.any()` and `.all()` suggestion does not solve the kernel problem. Those operations reduce an array, and data-dependent Python branching still fails when the value is traced under `jit`. Use `jnp.where` for elementwise selection or {func}`jax.lax.cond` for a scalar traced condition.

Inside a kernel method, use `jax.numpy` rather than NumPy or SciPy for calculations that depend on `x`, `y`, `h`, or `lam`. The sample-building code outside the method may still use ordinary NumPy.

### Keep kernel configuration static

The `frozen=True` in the first example is functional rather than stylistic. A `KernelSet` is compiled as static configuration, so the kernel objects it contains must be hashable.

A mutable dataclass is not.

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

A plain Python object may be hashable only by identity. Two otherwise identical instances then count as different static objects. Besides unnecessary compilation-cache misses, this can produce a kernel-mismatch error when a fresh instance is passed alongside a bandwidth selection or fit carrying the original instance.

For configurable kernels, use a frozen dataclass with immutable Python fields such as floats, integers, strings, and tuples. Avoid `jax.Array` fields in static kernel configuration. Value-based dataclass equality lets equivalent configurations compare and hash equally.

## Support density estimation

### Return values in standardized coordinates

The value-only Epanechnikov kernel already works for regression. Density estimation imposes a stronger requirement because it uses the kernel weights without dividing by another sum of the same weights.

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

KernelJax supplies the `1/h` itself. Changing variables from $x$ to $u$ shows that the total mass is determined by `value`.

$$
\int \hat f(x) \, dx = \int g(u) \, du.
$$

A valid continuous density kernel therefore needs $\int g(u)\,du=1$ and must not include another `1/h` inside `value`.

Regression is different. A common multiplicative constant cancels from weighted least squares, so a normalization mistake can be invisible to regression while making a density incorrect.

The value-only Epanechnikov implementation already has unit mass and can be used in a fixed-bandwidth density.

```python
dens = kj.density(x, fixed, kernels=value_kernels)
print(dens.value.shape)
```

```text
(200,)
```

`EpanechnikovValue` follows this rule by returning only the standardized kernel. The standardized-coordinate convention also applies to `cdf` and `conv`. The exception is `deriv`, covered with the derivative-weight operator later on this page.

### Normalization is checked when it matters

A half-mass kernel can still produce a regression fit, but the unconditional density entry point rejects it.

```python
@dataclasses.dataclass(frozen=True)
class HalfMass(kj.ContinuousKernel):
    def value(self, x, y, h):
        return 0.5 * EpanechnikovValue().value(x, y, h)
```

```python
kj.density(x, fixed, kernels=kj.KernelSet(continuous=HalfMass()))
```

```text
ValueError: HalfMass.value integrates to 0.5000 in u units rather than one, so
every density it produces is scaled by that factor. A regression fit is a ratio
and cancels the constant, which is why this only fires from a density.
```

KernelJax also recognizes the signature of a kernel carrying its own bandwidth normalization.

```python
@dataclasses.dataclass(frozen=True)
class SelfNormalizing(kj.ContinuousKernel):
    def value(self, x, y, h):
        return EpanechnikovValue().value(x, y, h) / h
```

```python
kj.density(x, fixed, kernels=kj.KernelSet(continuous=SelfNormalizing()))
```

```text
ValueError: SelfNormalizing.value integrates to one at h=1 but to 0.5000 at
h=2, the signature of a kernel carrying its own 1/h factor. Return the kernel
in u units with no normalization by h, since the estimator divides by h
exactly once itself.
```

These are targeted checks from the unconditional density wrapper. The [validation section](#validate-a-custom-kernel) describes their exact scope and what remains the kernel author's responsibility.

## Add continuous operators as needed

The value-only kernel raises `NotImplementedError` when a requested computation needs an optional operator. For example, continuous regression standard errors require `conv` at zero.

```python
kj.local_poly(x, y, fixed, degree=1, kernels=value_kernels, se=True)
```

```text
NotImplementedError: EpanechnikovValue does not implement conv
```

If $u=(x-y)/h$, the continuous operators follow these conventions.

$$
\begin{aligned}
\texttt{value} & = k(u), \\
\texttt{cdf} & = \int_{-\infty}^u k(t)\,dt, \\
\texttt{conv} & = (k*k)(u), \\
\texttt{deriv} & = \frac{1}{h}k'(u).
\end{aligned}
$$

The `conv` expression above is the familiar form for a translation-invariant symmetric kernel. More generally, the density LS-CV operator represents the integrated product `value(s, x, h) * value(s, y, h)`. An asymmetric kernel must preserve that evaluation-first orientation rather than assume the two argument orders are interchangeable.

For Epanechnikov, the cumulative kernel is

$$
F_k(u)=
\begin{cases}
0, & u \leq -1, \\
\frac{1}{2}+\frac{3}{4}u-\frac{1}{4}u^3, & -1 < u < 1, \\
1, & u \geq 1,
\end{cases}
$$

and the self-convolution is

$$
(k*k)(u) =
\frac{3}{160}
(2 - |u|)^3
\left(u^2 + 6|u| + 4\right),
\qquad |u| \leq 2,
$$

with zero outside that support. Combining those expressions gives one complete, copyable implementation.

```python
@dataclasses.dataclass(frozen=True)
class Epanechnikov(kj.ContinuousKernel):
    """A complete second-order Epanechnikov kernel."""

    order: int = dataclasses.field(default=2, init=False)

    def value(self, x, y, h):
        u = (x - y) / h
        inside = jnp.abs(u) < 1.0
        safe_u = jnp.where(inside, u, 0.0)

        return jnp.where(inside, 0.75 * (1.0 - safe_u * safe_u), 0.0)

    def cdf(self, x, y, h):
        u = (x - y) / h
        z = jnp.clip(u, -1.0, 1.0)

        return 0.5 + 0.75 * z - 0.25 * z**3

    def conv(self, x, y, h):
        u = jnp.abs(x - y) / h
        z = jnp.minimum(u, 2.0)

        return 3.0 / 160.0 * (2.0 - z) ** 3 * (z * z + 6.0 * z + 4.0)

    def deriv(self, x, y, h):
        u = (x - y) / h
        inside = jnp.abs(u) < 1.0
        safe_u = jnp.where(inside, u, 0.0)

        return -1.5 * safe_u / h
```

At the two support boundaries, the Epanechnikov derivative is not unique. This implementation chooses the outside value zero and locks `order` at two.

### Use `cdf` for distributions

A distribution estimator asks the kernel for its cumulative operator directly. It does not numerically integrate `value` for each estimate.

```python
complete_kernels = kj.KernelSet(continuous=Epanechnikov())
distribution = kj.cdf(x, fixed, kernels=complete_kernels)
```

A cumulative operator must be monotone, lie between zero and one, approach the correct limits, and represent the actual integral of `value`. KernelJax checks the limiting values for an unconditional continuous CDF and would reject a kernel that plateaus at, for example, `0.98`. Those limits alone cannot establish monotonicity or consistency with `value`, so both still need direct tests.

### Use `conv` for LS-CV and standard errors

Convolution expands support. A kernel supported on $|u|\leq1$ has a self-convolution supported on $|u|\leq2$. The `density(..., "cv_ls")` wrapper checks continuous `conv` over a range of offsets. Continuous regression standard errors need only

$$
(k*k)(0) = \int k(u)^2\,du = R(k),
$$

so `local_poly(..., se=True)` performs the narrower check at zero. The [regression guide](regression.md#pointwise-standard-errors) gives the full standard-error expression and its qualifications.

```python
with_se = kj.local_poly(x, y, fixed, degree=1, kernels=complete_kernels, se=True)
```

### Use `deriv` for derivative weights

`deriv(x, y, h)` differentiates `value` with respect to the original evaluation coordinate `x`. It includes `1/h` from the chain rule, unlike `value`, `cdf`, and `conv`.

The example below has one continuous column. With several continuous columns, `op="deriv"` replaces every continuous factor with `deriv`. It does not construct a tensor with one factor differentiated at a time.

```python
train = kj.MixedData.continuous(x)
weight_derivatives = kj.kweights(train, fixed, kernels=complete_kernels, op="deriv")

print(
    f"cdf={distribution.value.shape}  "
    f"se={with_se.se.shape}  "
    f"derivative weights={weight_derivatives.shape}"
)
```

```text
cdf=(200,)  se=(200,)  derivative weights=(200, 200)
```

## Select bandwidths safely

Once a kernel works with a fixed bandwidth, an optimized selector adds two further requirements. The operations used by its criterion must have usable bandwidth derivatives, and the criterion must be compatible with the kernel's values.

### Keep bandwidth derivatives finite

A custom method can produce finite forward values but still fail during bandwidth selection. A common source is an unsafe expression hidden inside `jnp.where`.

```python
def unsafe(u):
    return jnp.where(u == 0.0, 1.0, jnp.sin(u) / u)
```

The forward value at zero is finite, but the other branch still contains a division by zero that contaminates automatic differentiation.

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

```{important}
Guard the unsafe expression itself, not only the branch that selects it.
```

String-based optimized calls through the unconditional `density`, `cdf`, and `local_poly` wrappers probe a few representative separations and reject a non-finite continuous bandwidth derivative before starting the solve. Conditional estimators and direct calls to `select_bandwidth` do not run those probes. Even where they run, the probes catch common failures rather than proving that every derivative along an optimization path is well behaved. A kink by itself is not necessarily a problem. Compact-support kernels such as Epanechnikov are not smooth everywhere. What matters computationally is that the values and derivatives encountered by the selector remain finite and useful.

### Match the selector to the operators

Regression least-squares CV scores held-out predictions and needs only `value`. Density least-squares CV contains an integrated squared-density term and therefore needs `conv`. That is why the value-only Epanechnikov can use `local_poly(..., "cv_ls")` but not `density(..., "cv_ls")`.

A common wrong implementation truncates the convolution at the original kernel support.

```python
@dataclasses.dataclass(frozen=True)
class TruncatedConv(Epanechnikov):
    def conv(self, x, y, h):
        u = jnp.abs(x - y) / h
        piece = 3.0 / 160.0 * (2.0 - u) ** 3 * (u * u + 6.0 * u + 4.0)

        return jnp.where(u <= 1.0, piece, 0.0)
```

The density wrapper detects that mistake before optimization begins.

```python
kj.density(x, "cv_ls", kernels=kj.KernelSet(continuous=TruncatedConv()), n_starts=1)
```

```text
ValueError: TruncatedConv.conv is zero at u = -1.5 where the self-convolution
of TruncatedConv.value is not. A self-convolution doubles the support, so for a
kernel on |u| <= 1 conv reaches |u| <= 2, and truncating it to the kernel's own
support is the usual cause.
```

The complete kernel supports both.

```python
epan = kj.local_poly(x, y, "cv_ls", degree=1, kernels=complete_kernels)
gauss = kj.local_poly(x, y, "cv_ls", degree=1)
dens_ls = kj.density(x, "cv_ls", kernels=complete_kernels, n_starts=1)

print(f"Epanechnikov  h={epan.bandwidth.h[0]:.6f}  r2={epan.r_squared:.6f}")
print(f"Gaussian      h={gauss.bandwidth.h[0]:.6f}  r2={gauss.r_squared:.6f}")
print(f"density       h={dens_ls.bandwidth.h[0]:.6f}")
```

```text
Epanechnikov  h=0.084645  r2=0.947231
Gaussian      h=0.036019  r2=0.947953
density       h=0.203672
```

The numerical bandwidths are not directly comparable because bandwidth is measured on the scale of the kernel. A larger Epanechnikov bandwidth therefore does not by itself imply a smoother fit. For ordinary second-order kernels, bandwidth choice generally has a larger effect on the estimate than choosing between reasonable kernel shapes.

The examples that specify `n_starts=1` do so only to keep their execution brief. That argument is part of bandwidth optimization, not the custom-kernel contract.

### Likelihood CV works best with positive density

Likelihood cross-validation evaluates held-out log densities. A compactly supported kernel can assign zero density to an isolated observation, and a signed higher-order kernel can produce a nonpositive estimate.

The unconditional density criterion floors zero and negative held-out values at the smallest positive value for their floating-point dtype. This keeps the criterion finite but gives those observations a saturated, very large penalty. Conditional likelihood currently takes the logarithm directly, so a nonpositive conditional density can make that criterion non-finite.

Strict positivity therefore remains important for both likelihood interpretation and stable, informative optimization. Prefer `"cv_ls"` when positivity cannot be guaranteed and the required `conv` operators are available.

### Treat `normal_reference` as a reference rule

`normal_reference` reads an `order` attribute when the continuous kernel provides one and otherwise assumes order two.

For optimization-based selectors, the reference rule supplies a starting scale. Passing `"normal_reference"` as the bandwidth runs no optimization, so that reference bandwidth becomes the final answer.

KernelJax uses Gaussian reference constants rather than constants recalibrated for every custom family. A substantially different kernel can therefore make `normal_reference` a useful rough benchmark without making it a kernel-specific optimal bandwidth. If a custom continuous kernel has a meaningful order other than two, expose it explicitly so KernelJax can use the corresponding reference rate.

## Build categorical kernels

Categorical kernels follow the same elementwise design but replace `h` with a smoothing parameter $\lambda$. They subclass {class}`~kerneljax.UnorderedKernel` or {class}`~kerneljax.OrderedKernel` and implement both `value` and `upper_bound(levels)`.

Every categorical smoothing parameter uses the interval

$$
0 \leq \lambda \leq \operatorname{upper\_bound}(\mathrm{levels}).
$$

The lower endpoint is fixed by KernelJax, so zero must be a usable point in the custom parameterization. Conventionally, $\lambda=0$ means exact matching or no categorical smoothing. `upper_bound(levels)` gives the upper end of the range, normally the complete-pooling boundary or limit where distinctions among category levels disappear.

The bound should be a deterministic, finite, positive Python scalar for every valid level count. Optimized searches map the interior through a scaled sigmoid and begin at half the bound. `normal_reference`, by contrast, returns exactly zero for every categorical parameter.

KernelJax checks a fixed categorical bandwidth for finiteness and nonnegativity, but does not check it against the custom kernel's upper bound. A caller constructing {class}`~kerneljax.Bandwidth` directly remains responsible for staying within the kernel-specific range.

### Understand the categorical inputs

Categorical columns are represented internally by contiguous, zero-based integer codes,

```
0, 1, ..., levels - 1
```

and a custom kernel receives those codes rather than the original labels. For an ordered variable, the code order must agree with the category order. An expression such as `jnp.abs(x - y)` then measures the number of coded levels separating two categories.

Categorical operators are called one column at a time and receive shapes approximately like

```
x:       (n_eval,  1)
y:       (1,       n_train)
lam:     scalar
levels:  Python integer
```

{func}`~kerneljax.MixedData.from_blocks` validates codes against the declared level counts when its inputs are concrete. Under `jit` the codes are traced and cannot be checked, so pass explicit level counts and keep the codes in range yourself. The `levels` argument is an ordinary Python integer and can participate in static Python control flow. The smoothing parameter `lam` may be traced by JAX, so calculations depending on its value must use JAX operations.

Make every implemented operator finite at zero and throughout the admissible interval. When optimized selection will differentiate an operator, test its smoothing-parameter derivative at the boundaries and representative interior values as well as its forward value.

### Unordered categories

A simple regression kernel can give weight one to a match and $\lambda$ to a mismatch.

```python
@dataclasses.dataclass(frozen=True)
class RegressionOnlyPlain(kj.UnorderedKernel):
    """An unnormalized match-or-mismatch regression kernel."""

    def value(self, x, y, lam, levels):
        return jnp.where(x == y, 1.0, lam)

    def upper_bound(self, levels):
        return 1.0
```

At $\lambda=1$, every category receives the same weight, so the column no longer affects a regression fit.

`RegressionOnlyPlain` is deliberately incomplete. Summing its weights over $c$ evaluation levels gives $1+(c-1)\lambda$, not one.

```python
plain = RegressionOnlyPlain()
candidates = jnp.arange(4)
plain_mass = jnp.sum(plain.value(candidates, 0, 0.2, levels=4))
print(f"mass={plain_mass:.1f}")
```

```text
mass=1.6
```

As with continuous kernels, the common factor is invisible to regression but not to a density. Dividing by it produces a probability kernel. Adding the corresponding finite-support convolution makes the kernel usable by density LS-CV as well.

```python
@dataclasses.dataclass(frozen=True)
class NormalizedPlain(kj.UnorderedKernel):
    """A normalized match-or-mismatch kernel on the declared levels."""

    def value(self, x, y, lam, levels):
        scale = 1.0 + (levels - 1) * lam

        return jnp.where(x == y, 1.0, lam) / scale

    def conv(self, x, y, lam, levels):
        scale_sq = (1.0 + (levels - 1) * lam) ** 2
        same = (1.0 + (levels - 1) * lam**2) / scale_sq
        different = (2.0 * lam + (levels - 2) * lam**2) / scale_sq

        return jnp.where(x == y, same, different)

    def upper_bound(self, levels):
        return 1.0
```

The total mass is now one, and both fixed-bandwidth density estimation and density LS-CV can use it.

```python
rng = np.random.default_rng(0)
region = rng.integers(0, 4, 200)
regions = kj.MixedData.from_blocks(unordered=region, unordered_levels=4)
all_regions = kj.MixedData.from_blocks(unordered=np.arange(4), unordered_levels=4)
categorical_bw = kj.Bandwidth(
    h=jnp.zeros(0), lam_uno=jnp.array([0.2]), lam_ord=jnp.zeros(0)
)
normalized_kernels = kj.KernelSet(unordered=NormalizedPlain())

categorical_density = kj.density(
    regions, categorical_bw, at=all_regions, kernels=normalized_kernels
)
selected_categories = kj.density(
    regions, "cv_ls", kernels=normalized_kernels, n_starts=1
)

print(f"mass={jnp.sum(categorical_density.value):.6f}")
print(f"selected lam={selected_categories.bandwidth.lam_uno[0]:.6f}")

support = jnp.arange(4)
weights = NormalizedPlain().value(support[:, None], support[None, :], 0.2, levels=4)
convolved = NormalizedPlain().conv(support[:, None], support[None, :], 0.2, levels=4)

np.testing.assert_allclose(jnp.sum(weights, axis=0), jnp.ones(4))
np.testing.assert_allclose(convolved, weights.T @ weights)
```

```text
mass=1.000000
selected lam=0.820800
```

KernelJax does not automatically check either the categorical mass or categorical convolution. The validation remains the kernel author's responsibility.

Returning the wrong upper bound is a separate failure. A bound that is too low makes valid smoothing levels unreachable, while a bound beyond complete pooling can let the optimizer enter a region where similarity reverses. For a finite complete-pooling boundary, test that `value(jnp.arange(levels), y, upper_bound, levels)` is constant for every valid training code `y`.

### Ordered categories

Ordered kernels use the same interface, except that distance between level codes carries statistical meaning.

This deliberately regression-only example gives geometrically decreasing weight to more distant levels.

```python
@dataclasses.dataclass(frozen=True)
class RegressionOnlyGeometric(kj.OrderedKernel):
    """Geometric weights without probability normalization."""

    def value(self, x, y, lam, levels):
        return lam ** jnp.abs(x - y)

    def upper_bound(self, levels):
        return 1.0
```

When $\lambda=0$, only exact matches receive weight. As $\lambda$ approaches one, increasingly distant levels receive similar weights. At $\lambda=1$, every level receives the same value.

Like the first unordered example, this class omits its probability normalization because a common factor is irrelevant to its intended regression use. Test the lower endpoint and its gradients directly before adapting any power-based parameterization to an optimized criterion.

### Keep categorical operators on one support

Write $L(x,y,\lambda,c)$ for `value(x, y, lam, levels)`, with $c=\texttt{levels}$. A categorical probability kernel must normalize over the same support used by its other operators.

For an unordered kernel, that support is the declared finite set $\mathcal S_c=\{0,\ldots,c-1\}$. The required identities are

$$
\sum_{s=0}^{c-1}L(s,y,\lambda,c)=1
$$

and

$$
\operatorname{conv}(x,y,\lambda,c)
=
\sum_{s=0}^{c-1}
L(s,x,\lambda,c)L(s,y,\lambda,c).
$$

Unordered variables have no natural cumulative order, so the unconditional `cdf` estimator does not accept them.

The built-in ordered kernels define their cumulative and convolution operators over the entire integer lattice,

$$
\operatorname{cdf}(x,y,\lambda,c)
=
\sum_{\substack{s\in\mathbb Z\\s\leq x}}
L(s,y,\lambda,c),
$$

$$
\operatorname{conv}(x,y,\lambda,c)
=
\sum_{s\in\mathbb Z}
L(s,x,\lambda,c)L(s,y,\lambda,c).
$$

A custom ordered kernel may instead be supported only on the declared codes. In that case, its `value` is conceptually zero outside `0, ..., levels - 1`, and `cdf` and `conv` must use that same finite support. Whichever support is chosen, `cdf` should be monotone and span zero to one, and `conv` must be the actual self-convolution of `value`.

## Validate a custom kernel

Every kernel operator must return the broadcast shape KernelJax supplies. Product-kernel calls enforce that shape directly, and a missing optional method raises `NotImplementedError`. A few task-specific probes call only the scalar value they need, such as `conv(0)` for regression standard errors, so the author must still test the full broadcast contract for every method.

Its numerical conformance checks are deliberately narrower.

| Check                                             | Where it runs automatically                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Continuous `value` has unit mass and no `1/h`     | Unconditional {func}`~kerneljax.density`                                                               |
| Continuous `conv` matches numerical convolution   | Unconditional `density(..., "cv_ls")`                                                                 |
| Continuous `conv(0)` equals $R(k)$                | `local_poly(..., se=True)`                                                                              |
| Continuous `cdf` reaches zero and one             | Unconditional {func}`~kerneljax.cdf`                                                                   |
| Representative continuous bandwidth gradients    | String-based optimized calls through `density`, `cdf`, and `local_poly`                               |
| Categorical mass, `cdf`, and `conv` identities    | Not checked numerically                                                                                 |

These smoke tests do not run when a built-in criterion is passed directly to {func}`~kerneljax.select_bandwidth`, or from conditional estimators and conditional bandwidth selection. In those paths, the author must verify the required contracts directly. A continuous response kernel used by {func}`~kerneljax.cdensity` still needs unit mass, and a response kernel used by {func}`~kerneljax.cdist` still needs a valid cumulative operator.

Before relying on a custom kernel, test at least the following.

* Every implemented operator returns the exact broadcast shape for one and several columns.
* `jax.jit` accepts the method, and bandwidth gradients are finite at coincident points, support boundaries, and representative separations.
* Continuous density kernels integrate to one for several bandwidths without carrying `1/h`.
* Categorical density kernels sum to one for several training codes, level counts, and smoothing parameters.
* `cdf` is monotone, stays in $[0,1]$, has the correct limits, and agrees with `value`.
* `conv` agrees with a numerical sum or integral over the same support as `value`.
* Equal configurations compare and hash equally, and a selected bandwidth can be reused with an equivalent kernel instance.
* Every intended estimator and selector is exercised, including conditional paths when they matter.

For example, these compact checks cover the identities most often gotten wrong, unit mass across bandwidths, the convolution against its numerical integral, and the derivative against autodiff.

```python
candidate = Epanechnikov()
u_grid = jnp.linspace(-3.0, 3.0, 12_001)

for h in (1.0, 2.0):
    values = candidate.value(h * u_grid, 0.0, h)
    np.testing.assert_allclose(jnp.trapezoid(values, u_grid), 1.0, atol=1e-3)

    for offset in (0.0, 0.5, 1.5, 2.5):
        product = values * candidate.value(h * (offset - u_grid), 0.0, h)
        numeric = jnp.trapezoid(product, u_grid)
        np.testing.assert_allclose(candidate.conv(h * offset, 0.0, h), numeric, atol=1e-3)

point, h = jnp.asarray(0.25), jnp.asarray(0.4)
autodiff = jax.grad(lambda z: candidate.value(z, 0.0, h))(point)
np.testing.assert_allclose(candidate.deriv(point, 0.0, h), autodiff)
```

The same pattern extends to the remaining checklist items, monotonicity and limits for `cdf` and finite bandwidth gradients at representative separations.

Runtime probes are useful guardrails, not substitutes for tests covering the kernel's full domain. Start with an elementwise, fixed-bandwidth `value`, then add normalization, optional operators, and optimized selection only when the intended workflow needs them.
