# Custom kernels

Every estimator takes a `kernels` argument, a {class}`~kerneljax.KernelSet` holding one kernel per
column kind. The defaults are a second-order Gaussian for continuous columns,
Aitchison-Aitken for unordered ones and Li-Racine for ordered ones, and swapping any of them
changes the smoothing without touching the estimator.

That substitution is the point of the design, since a kernel travels as an argument rather
than through a registry, and one you write is held to five requirements, each
checked at the first call that depends on it. This page writes one kernel and takes it
through the library, meeting each requirement where it binds, and
[the requirements at a glance](#the-requirements-at-a-glance) collects them at the end. The
[Quickstart](../quickstart.md) covers the rest of the API, and
[Kernel smoothing](../background/smoothing.md) covers what a kernel is doing.

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
```

```python
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

The `frozen=True` on the class is load-bearing rather than style. The kernel travels inside a
{class}`~kerneljax.KernelSet` handed to the compiler as a static argument, so it has to be
hashable, and an unfrozen `@dataclass` sets `__hash__` to `None`, while a frozen one holding
a `jax.Array` field fails the moment that field is hashed. Write one and it is refused at
construction.

```python
@dataclasses.dataclass
class Mutable(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(jnp.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)

kj.KernelSet(continuous=Mutable())
```

```text
TypeError: continuous kernel Mutable is not hashable, so it cannot be a static argument.
Decorate it with @dataclasses.dataclass(frozen=True), and hold any array field as a tuple of
floats rather than as a jax.Array.
```

A plain class without the decorator is hashable and will run, but its instances compare
unequal, so every fresh instance is a compilation cache miss.

## What `value` receives

A continuous `value` is called once for all continuous columns. It receives `x` broadcasting as
`(n_eval, 1, p_con)` against `y` of shape `(1, n_train, p_con)`, with a bandwidth broadcast
against both, so a univariate formula written elementwise is already a product kernel over the
columns. The library multiplies over the trailing column axis itself, so `value` must leave
one factor per column in place. A kernel that reduces over that axis, here on data carrying
two continuous columns, is rejected on the spot.

```python
@dataclasses.dataclass(frozen=True)
class Reducing(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.sum(jnp.exp(-0.5 * u * u), axis=-1, keepdims=True)

two_columns = kj.MixedData.continuous(np.column_stack([x, x ** 2]))
wide = kj.Bandwidth(h=jnp.array([0.2, 0.2]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

kj.local_poly(two_columns, y, wide, degree=1, kernels=kj.KernelSet(continuous=Reducing()))
```

```text
ValueError: Reducing.value returned shape (1, 200, 1), expected (1, 200, 2). A kernel is applied
elementwise, so it must broadcast against its inputs and must not reduce over any axis.
```

Branching inside `value` goes through `jnp.where` rather than a Python `if`, because an
`if` asks a whole array which way to go.

```python
@dataclasses.dataclass(frozen=True)
class Branching(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        if jnp.abs(u) <= 1.0:
            return 0.75 * (1.0 - u * u)
        return jnp.zeros_like(u)

Branching().value(x[:3], 0.0, 0.2)
```

```text
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or
a.all()
```

How many evaluation points arrive at once is the estimator's business, not yours. Written
elementwise a kernel never notices the difference, which is the reason to write it that way
rather than to reach for the leading axis.

Categorical kernels are called once per column, receiving `x` and `y` shaped the same way with
the column axis dropped, a scalar `lam`, and a `levels` count that arrives as a plain Python
integer rather than a traced value, so it is safe in Python control flow in a way that `lam` is
not.

The categorical entries are `int32` codes, contiguous and zero based, and for an ordered
column the code order is the level order, which is what makes `jnp.abs(x - y)` a count of
levels between two categories. A kernel sees codes, never labels, and
{func}`~kerneljax.MixedData.from_blocks` rejects codes outside the declared level count, so
encode labels as `0, ..., levels - 1` before building the data rather than letting stray
integers pass as codes. It also rejects a degenerate column, so `levels` is always at least
two and `value` and `upper_bound` may divide by `levels - 1` without guarding.

The argument order is always `value(evaluation_point, training_point)`. Nothing in the library
assumes the two are interchangeable, so an asymmetric kernel runs end to end without complaint
and quietly mirrors the estimator if you had the order backwards.

## Selecting with it

The block in [Writing one](#writing-one) already selected a bandwidth, so the kernel has been
differentiated. Selection drives the criterion's gradient through `value`, and keeping that
gradient finite is the hardest requirement to see, because `jnp.where` evaluates *both*
branches and differentiates both. An unsafe expression in the branch that is not taken
poisons the gradient while leaving the value correct. A sinc kernel is the natural
illustration, since $u = 0$ occurs on the diagonal of any fit evaluated at its own training
points, so the guarded branch is always reached.

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
catch it. Guard the denominator, not just the branch. A kink is fine, the triangular kernel
selects without complaint, and only a NaN poisons. Built into a kernel and handed to an
estimator, the unsafe version is refused before the search starts, probed both on the
diagonal and beyond the support edge, where a clamped square root hides the same mistake.

```python
@dataclasses.dataclass(frozen=True)
class Sinc(kj.ContinuousKernel):
    def value(self, x, y, h):
        u = (x - y) / h
        return jnp.where(u == 0.0, 1.0, jnp.sin(u) / u)

sinc_kernels = kj.KernelSet(continuous=Sinc())

kj.local_poly(x, y, "cv_ls", degree=1, kernels=sinc_kernels)
```

```text
ValueError: Sinc.value has a non-finite bandwidth gradient at |x - y| = 0, a separation any
sample can contain. jnp.where differentiates both branches, so guard the argument inside the
untaken branch, not just the branch.
```

The selector itself does not probe, so calling it directly shows what the probe exists to
prevent, a search that runs its full budget and produces nothing.

```python
result = kj.select_bandwidth(x, kj.cv_ls_regression, y=y, kernels=sinc_kernels, n_starts=1)

print(f"h={result.bandwidth.h[0]:.6f}  n_iter={result.n_iter}  "
      f"converged={result.converged}")
```

```text
h=nan  n_iter=200  converged=False
```

Reading `converged` is the habit that catches this. A run that exhausts its iteration budget
and reports `False` has not selected anything, and an estimator handed its bandwidth refuses
to fit.

```python
kj.local_poly(x, y, result.bandwidth)
```

```text
ValueError: every continuous bandwidth must be finite and positive, got h=[nan]. A kernel
divides by h, so a non-positive or non-finite value produces numbers rather than an error. If
this came from select_bandwidth, its converged flag will be False.
```

## Taking it to a density

Nothing so far pinned down the kernel's normalization, neither that it integrates to one
nor that it carries no $1/h$ of its own, and that is not an oversight. A regression fit is a
ratio, so a constant factor cancels, and a kernel carrying its own $1/h$ cancels there too. The estimator divides by the bandwidths of the continuous columns exactly
once itself, which is why `value` is written in $u$ units with no $1/h$ of its own, a
convention that also governs `conv` and `cdf`. `deriv` is the one exception, differentiating
with respect to $x$ rather than $u$ and so carrying the chain rule factor.

A density puts real weight on both conventions, and the kernel that just selected for a
regression selects for a density unchanged.

```python
dens = kj.density(x, "cv_ml", kernels=kernels)

print(f"h={dens.bandwidth.h[0]:.6f}")
```

```text
h=0.040642
```

The first density call is also where normalization is enforced, by integrating `value` at
two bandwidths, and the first distribution call runs the same two-bandwidth check on `cdf`.
A kernel returning half the mass draws one error,

```python
@dataclasses.dataclass(frozen=True)
class HalfMass(kj.ContinuousKernel):
    def value(self, x, y, h):
        return 0.5 * Epanechnikov().value(x, y, h)

kj.density(x, "cv_ml", kernels=kj.KernelSet(continuous=HalfMass()))
```

```text
ValueError: HalfMass.value integrates to 0.5000 in u units rather than one, so every density it
produces is scaled by that factor. A regression fit is a ratio and cancels the constant, which
is why this only fires from a density.
```

and a kernel dividing by `h` draws the other, since it is right at one bandwidth and wrong at
the second.

```python
@dataclasses.dataclass(frozen=True)
class SelfNormalizing(kj.ContinuousKernel):
    def value(self, x, y, h):
        return Epanechnikov().value(x, y, h) / h

kj.density(x, "cv_ml", kernels=kj.KernelSet(continuous=SelfNormalizing()))
```

```text
ValueError: SelfNormalizing.value integrates to one at h=1 but to 0.5000 at h=2, the signature
of a kernel carrying its own 1/h factor. Return the kernel in u units with no normalization by
h, since the estimator divides by h exactly once itself.
```

Neither mistake is detectable where you would first look. The half-mass kernel also shifts
the `cv_ml` criterion by a constant, which moves no minimum, so the checks you would
naturally run both pass and the density call runs the one that decides it.

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
```

```python
rng = np.random.default_rng(0)
exper = rng.uniform(0, 30, 200)
region = rng.integers(0, 4, 200)
wage = 2.0 + 0.1 * exper + region + rng.normal(0, 0.5, 200)

data = kj.MixedData.from_blocks(continuous=exper, unordered=region,
                                unordered_levels=4, names=("exper", "region"))
```

```python
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
Plain            lam=0.001234  bound=1.00  r2=0.897111
AitchisonAitken  lam=0.003673  bound=0.75  r2=0.897111
```

The two fits agree to six digits, because the normalization the shipped kernel carries is a
factor common to every level and cancels from a ratio estimator. The smoothing parameters
differ, because $\lambda$ means a different thing in each. This is why `upper_bound` cannot be
inherited from anything. Return a bound that is too high and the search walks past complete
pooling into a region where a *matching* level receives less weight than a non-matching one,
reporting `converged=True` throughout. Return zero and the search box collapses,
{func}`~kerneljax.select_bandwidth` returns `nan`, and an estimator refuses to fit at it.

The bound also fixes where selection begins and where it moves, since a search starts every
categorical parameter at half of it and the box runs from zero to the bound. The rule of thumb
{func}`~kerneljax.normal_reference` returns for a categorical column is zero, matching np, but
zero sits in the flat tail of the transform where a gradient cannot move, so the search starts
inside the box instead. Two kernels with different bounds are therefore reporting the same smoothing in
different units, the reading that
[Bandwidth selection](../background/selection.md#what-cross-validation-buys) sets out.

An ordered kernel is written the same way, except that the number of levels between `x` and
`y` carries the information rather than a match alone. Here is the geometric family, `lam` to
the power of the level distance, beside the shipped Li-Racine on an education column that
moves wage.

```python
@dataclasses.dataclass(frozen=True)
class Geometric(kj.OrderedKernel):
    """The unnormalized variant, lam to the number of levels between x and y."""

    def value(self, x, y, lam, levels):
        return lam ** jnp.abs(x - y)

    def upper_bound(self, levels):
        return 1.0
```

```python
educ = rng.integers(0, 6, 200)
wage = wage + 0.25 * educ

data = kj.MixedData.from_blocks(continuous=exper, unordered=region, ordered=educ,
                                unordered_levels=4, ordered_levels=6,
                                names=("exper", "region", "educ"))
```

```python
geometric = Geometric()
liracine = kj.LiRacine()

custom = kj.local_poly(data, wage, "cv_ls", degree=1,
                       kernels=kj.KernelSet(ordered=geometric))
shipped = kj.local_poly(data, wage, "cv_ls", degree=1)

print(f"Geometric  lam_ord={custom.bandwidth.lam_ord[0]:.6f}  "
      f"bound={geometric.upper_bound(6):.2f}  r2={custom.r_squared:.6f}")
print(f"LiRacine   lam_ord={shipped.bandwidth.lam_ord[0]:.6f}  "
      f"bound={liracine.upper_bound(6):.2f}  r2={shipped.r_squared:.6f}")
```

```text
Geometric  lam_ord=0.297765  bound=1.00  r2=0.917434
LiRacine   lam_ord=0.297599  bound=1.00  r2=0.917440
```

The two land within a whisker of each other, since both are the same geometric family. The
shipped kernel computes the power through a guarded form whose gradient survives `lam = 0`,
the trap [Selecting with it](#selecting-with-it) walks through, while the bare `**` escapes
it here only because integer codes take an exact integer-power derivative.

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
| `deriv` | derivative weight tensors, reached through the `op` argument of {func}`~kerneljax.kweights` | Not needed by any estimator |

`local_poly(..., se=True)` needs `conv` even though nothing in a standard error looks like a
density, since the variance carries $R(k) = \int k^2$, the self-convolution at zero, and that
call asks only the continuous kernel. `density(..., "cv_ls")` asks every kernel in the set,
where a categorical self-convolution is the sum over levels of $k(x, s)\,k(y, s)$.
{func}`~kerneljax.cdf` asks the continuous and ordered kernels and rejects unordered columns
outright, the ordered accumulation being the kernel summed over every integer at or below `x`.

Ask the value-only kernel for a standard error and the gap announces itself.

```python
kj.local_poly(x, y, "cv_ls", degree=1, kernels=kernels, se=True, n_starts=1)
```

```text
NotImplementedError: Epanechnikov does not implement conv
```

The Epanechnikov from [Writing one](#writing-one) has a closed-form self-convolution,

$$
\tfrac{3}{160}\,(2 - |u|)^3(u^2 + 6|u| + 4), \quad |u| \le 2.
$$

One method on a subclass supplies it, and both calls in the first table row open up.

```python
@dataclasses.dataclass(frozen=True)
class EpanechnikovConv(Epanechnikov):
    def conv(self, x, y, h):
        u = jnp.abs(x - y) / h
        piece = 3.0 / 160.0 * (2.0 - u) ** 3 * (u * u + 6.0 * u + 4.0)
        return jnp.where(u <= 2.0, piece, 0.0)

conv_kernels = kj.KernelSet(continuous=EpanechnikovConv())

dens = kj.density(x, "cv_ls", kernels=conv_kernels, n_starts=1)
fit = kj.local_poly(x, y, "cv_ls", degree=1, kernels=conv_kernels,
                    se=True, n_starts=1)

print(f"density     h={dens.bandwidth.h[0]:.6f}")
print(f"regression  mean se={fit.se.mean():.6f}")
```

```text
density     h=0.203672
regression  mean se=0.043252
```

`conv` at zero is $3/5$, the $R(k)$ a standard error consumes, and the `u <= 2.0` support in
the code is the doubled support the next paragraph warns about truncating.

Each of those calls also checks that `conv` genuinely is the self-convolution of `value`,
`se=True` at zero against $R(k)$ computed from your own `value`, and `density(..., "cv_ls")`
pointwise across offsets. A self-convolution doubles the support, so for a kernel on
$|u| \le 1$ `conv` reaches $|u| \le 2$, and truncating it to the kernel's own support is
caught whatever the kernel's smoothness.

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

Second, nothing in the package special-cases the Gaussian, but `normal_reference` does read
an `order` attribute off the continuous kernel and falls back to `2` when there is none, and
the order sits in the exponent, $n^{-1/(2r + p)}$ for a density. Under a cross-validation
method a missing `order = 4` moves only the starting point. Under `bw="normal_reference"` it
is the answer, sixty percent of the bandwidth for a triangular kernel on the data above. And
the rule of thumb's constants are the Gaussian ones, never rescaled for your kernel, so it
hands that triangular kernel the same number as the default even though cross validation puts
the two a factor apart. `normal_reference` is a starting point for an unusual kernel, not a
bandwidth for one.

## The requirements at a glance

Five requirements, each shown above where it binds.

1. Return no $1/h$ factor, in `value`, `conv` and `cdf` alike, with `deriv` the one
   exception. Enforced at the first density or distribution call.
2. Be hashable, a frozen dataclass with no array fields. Enforced by
   {class}`~kerneljax.KernelSet` at construction.
3. Be elementwise, broadcasting and never reducing. Rejected at the first call that builds
   kernel weights.
4. Carry no NaN into the bandwidth gradient. Probed before any selection an estimator runs,
   with {func}`~kerneljax.select_bandwidth` called directly the one unprobed path.
5. Integrate to one in $u$ units. Enforced by the same integral that catches the $1/h$
   factor.
