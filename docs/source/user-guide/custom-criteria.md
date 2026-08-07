# Custom bandwidth selection

Writing a kernel changes how observations are weighted. Writing a criterion changes what the
library is aiming for when it picks a bandwidth. The two are independent, and both are open.

{func}`~kerneljax.select_bandwidth` takes the criterion as an argument and minimizes whatever
it returns, so the shipped rules have no privileged status and there is no base class to
inherit from, since {class}`~kerneljax.selection.criteria.Criterion` is a `Protocol`. This
page writes a criterion and takes it through selection, meeting each requirement where it
binds, and [the requirements at a glance](#the-requirements-at-a-glance) collects them at the end.
[Custom kernels](custom-kernels.md) covers the other extension point, and
[Bandwidth selection](../background/selection.md) covers what the shipped criteria are doing.

## The signature

There is one line of the library worth reading before writing a criterion, and it is the call
in `select_bandwidth`.

```python
def objective(z):
    bandwidth = transform.from_unconstrained(z)
    return criterion(train, bandwidth, **extra, kernels=kernels, chunk=chunk)
```

A criterion is therefore any callable taking a `MixedData`, a `Bandwidth`, whatever you passed
through `y` or `criterion_kwargs`, and the keywords `kernels` and `chunk`, passed whether or
not you use them, returning one number. The optimizer differentiates that number with respect
to the bandwidth through an unconstrained reparameterization, so beyond that signature the
requirement is that the whole thing be ordinary JAX. The signature is enforced before any
search runs, shown at the end of [Writing one](#writing-one).

## Writing one

Cross validation with squared error asks which bandwidth predicts held-out observations best in
a least squares sense. Absolute deviation asks the same question under a different loss, and
writing it is the whole exercise.

```python
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import kerneljax as kj

n = 150
rng = np.random.default_rng(1)
x = rng.uniform(size=n)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
outliers = rng.choice(n, 8, replace=False)
y[outliers] += 6.0
train = kj.MixedData.continuous(x)
```

```python
@dataclasses.dataclass(frozen=True)
class AbsoluteDeviation:
    degree: int = 1

    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        leave_one_out = jnp.arange(train.n)
        fit = kj.local_poly(train, y, bandwidth, kernels=kernels, chunk=chunk,
                            degree=self.degree, fold=leave_one_out)
        return jnp.mean(jnp.abs(y - fit.mean))
```

The `frozen=True` matters here the way it does for a kernel. The criterion is a static
argument of the jitted {func}`~kerneljax.select_bandwidth`, so it has to be hashable, and a
mutable dataclass is refused at the jit boundary.

```python
@dataclasses.dataclass
class MutableCriterion:
    degree: int = 1

    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        fit = kj.local_poly(train, y, bandwidth, kernels=kernels, chunk=chunk,
                            degree=self.degree, fold=jnp.arange(train.n))
        return jnp.mean(jnp.abs(y - fit.mean))

kj.select_bandwidth(train, MutableCriterion(), y=y)
```

```text
ValueError: Non-hashable static arguments are not supported. An error occurred while trying to
hash an object of type <class '__main__.MutableCriterion'>, MutableCriterion(degree=1). The
error was:
TypeError: unhashable type: 'MutableCriterion'
```

Passing it to the selector is the same call the shipped criteria go through.

```python
squared = kj.select_bandwidth(train, kj.RegressionCriterion(method="cv_ls", degree=1), y=y)
absolute = kj.select_bandwidth(train, AbsoluteDeviation(degree=1), y=y)

truth = np.sin(2 * np.pi * x)
for name, result in [("squared error", squared), ("absolute deviation", absolute)]:
    fit = kj.local_poly(train, y, result)
    print(f"{name:19s} h = {result.bandwidth.h[0]:.4f}   "
          f"median |error| = {jnp.median(jnp.abs(fit.mean - truth)):.4f}")
```

```text
squared error       h = 0.0323   median |error| = 0.1223
absolute deviation  h = 0.0628   median |error| = 0.2141
```

The two bandwidths disagree by a factor of two on data carrying eight gross outliers, which is
the point of the exercise and also a warning about it. It would be easy to present absolute
deviation as the robust choice and stop there. Measured at the training points against the
function the data were generated from, it is the worse of the two, and by a wide margin. The
selection loss is robust, but the local polynomial fit underneath it is still least squares, so
a wider bandwidth spreads the contamination further rather than resisting it.

Nothing in the library will tell you that. It minimizes what you hand it, faithfully, and the
argument that the thing being minimized is worth minimizing remains yours.

The [signature](#the-signature) is enforced before any of that runs. Forget
the two keywords and the first call refuses,

```python
def forgets_the_keywords(train, bandwidth, *, y):
    fit = kj.local_poly(train, y, bandwidth, degree=1, fold=jnp.arange(train.n))
    return jnp.mean(jnp.abs(y - fit.mean))

kj.select_bandwidth(train, forgets_the_keywords, y=y)
```

```text
TypeError: forgets_the_keywords() got an unexpected keyword argument 'kernels'
```

and hand back a vector rather than a scalar and the first gradient does.

```python
def returns_a_vector(train, bandwidth, *, y, kernels=None, chunk=None):
    fit = kj.local_poly(train, y, bandwidth, kernels=kernels, chunk=chunk, degree=1,
                        fold=jnp.arange(train.n))
    return jnp.abs(y - fit.mean)

kj.select_bandwidth(train, returns_a_vector, y=y)
```

```text
TypeError: Gradient only defined for scalar-output functions. Output had shape: (150,).
```

## A density criterion

Everything above selects for a regression, and a density criterion is smaller. There is no
`y` to accept, {func}`~kerneljax.density` takes the same `fold` array, and its fit reports
`value` rather than `mean`, so a five-fold cousin of the shipped leave-one-out `cv_ml` is a
few lines.

```python
@dataclasses.dataclass(frozen=True)
class KFoldLikelihood:
    n_folds: int = 5

    def __call__(self, train, bandwidth, *, kernels=None, chunk=None):
        fold = jnp.arange(train.n) % self.n_folds
        fit = kj.density(train, bandwidth, kernels=kernels, chunk=chunk, fold=fold)
        return -jnp.sum(jnp.log(fit.value))
```

```python
five_fold = kj.select_bandwidth(train, KFoldLikelihood(n_folds=5))
cv_ml = kj.select_bandwidth(train, kj.DensityCriterion(method="cv_ml"))

for name, result in [("five-fold likelihood", five_fold), ("leave-one-out cv_ml", cv_ml)]:
    print(f"{name:20s}  h = {result.bandwidth.h[0]:.4f}")
```

```text
five-fold likelihood  h = 0.0655
leave-one-out cv_ml   h = 0.0533
```

Five-fold holds thirty points out of every fit rather than one, so each held-out density is
built from a smaller sample and the selected bandwidth comes out wider than the leave-one-out
choice.

## Reading a failed solve

A criterion can fail in two ways that no exception reports, and from the outside they look
nothing alike. A NaN in the gradient is the same trap
[Selecting with it](custom-kernels.md#selecting-with-it) walks through for a kernel, an
unguarded branch differentiated even though it is never taken. A NaN in the value is worse,
since nothing the solver tries is ever usable and the solve never moves.

```python
@dataclasses.dataclass(frozen=True)
class NanGradient(AbsoluteDeviation):
    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        loss = super().__call__(train, bandwidth, y=y, kernels=kernels, chunk=chunk)
        zero = loss - loss
        return loss + jnp.where(zero == 0.0, 0.0, jnp.sqrt(zero))

@dataclasses.dataclass(frozen=True)
class NanValue(AbsoluteDeviation):
    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        return super().__call__(train, bandwidth, y=y, kernels=kernels, chunk=chunk) * jnp.nan
```

```python
start = kj.normal_reference(train, kj.KernelSet())
print(f"the search starts at h = {start.h[0]:.4f}")

for name, criterion in [("nan gradient", NanGradient()), ("nan value", NanValue())]:
    result = kj.select_bandwidth(train, criterion, y=y)
    print(f"{name:12s}  h = {result.bandwidth.h[0]:<7.4f} value = {result.value:<7.4f} "
          f"n_iter = {result.n_iter}  converged = {result.converged}")
```

```text
the search starts at h = 0.1081
nan gradient  h = nan     value = 0.9693  n_iter = 200  converged = False
nan value     h = 0.1081  value = nan     n_iter = 200  converged = False
```

The gradient failure walks the iterate to a `nan` bandwidth while the criterion value stays
believable. The value failure hands back a bandwidth that looks entirely ordinary, and it is
the untouched starting point, printed above it. Read `value`, `n_iter` and `converged`
together, because the bandwidth alone will not tell you and both failures spend the full
budget, and an estimator handed the `nan` bandwidth refuses to fit at it.

## Where settings live

The split between the criterion object and its keyword arguments is not stylistic, and getting
it wrong produces an error that does not explain itself.

Array data goes through `y` or `criterion_kwargs`, and reaches the criterion as traced
values. Anything that fixes the *shape* of the computation has to live on the criterion
instead, because the criterion is a static argument and its attributes stay concrete while the
search runs. The local polynomial degree is the usual case, and routing it through
`criterion_kwargs` fails deep inside the fit, because the degree is a static argument of the
estimator underneath.

```python
def with_degree_argument(train, bandwidth, *, y, degree, kernels=None, chunk=None):
    fit = kj.local_poly(train, y, bandwidth, kernels=kernels, chunk=chunk, degree=degree,
                        fold=jnp.arange(train.n))
    return jnp.mean(jnp.abs(y - fit.mean))

kj.select_bandwidth(train, with_degree_argument, y=y, criterion_kwargs={"degree": 1})
```

```text
ValueError: Non-hashable static arguments are not supported. An error occurred while trying to
hash an object of type <class 'kerneljax.basis.LocalPolyBasis'>,
LocalPolyBasis(degree=JitTracer(~int32[])). The error was:
TypeError: unhashable type: 'DynamicJaxprTracer'
```

Declaring `degree` as a field of the criterion, as above, is the supported way.

That field earns its keep a second time. Passing a `SelectionResult` back into an estimator
reuses the settings it was selected under, and `local_poly` finds the degree by reading it off
the criterion the result carries. The kernels travel the same way, so a bandwidth selected
under a kernel you wrote is refitted under that kernel without naming it again, and naming a
different one is refused rather than quietly preferred.

```python
fit = kj.local_poly(train, y, absolute)

print(f"degree read off the criterion: {fit.degree}")
```

```text
degree read off the criterion: 1
```

The lookup is by name. `local_poly` reads `getattr(criterion, "degree", None)` and falls back
to 0 when there is none, so what saves you is a field called exactly `degree`, not the fact
that the criterion is a dataclass. A frozen dataclass whose field is named `poly_degree` fits a
local constant under a bandwidth chosen for a local linear one, exactly as silently as a plain
function does. Passing `degree` explicitly overrides the lookup, and a value contradicting
the one the criterion carries is refused rather than silently preferred.

```python
kj.local_poly(train, y, absolute, degree=2)
```

```text
ValueError: degree=2 contradicts the degree 1 that bw was selected under
```

The criterion is an ordinary callable, and nothing reserves it for the selector. Build a
{class}`~kerneljax.Bandwidth` directly, sweep a few values of `h`, and the valley the search
descends is there to look at.

```python
criterion = AbsoluteDeviation(degree=1)
bandwidth = kj.Bandwidth(h=jnp.array([0.02]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

for h in [0.02, 0.06, 0.15, 0.4]:
    value = criterion(train, bandwidth.replace(h=jnp.array([h])), y=y)
    print(f"h = {h:.2f}   criterion = {value:.4f}")
```

```text
h = 0.02   criterion = 0.6511
h = 0.06   criterion = 0.6414
h = 0.15   criterion = 0.6799
h = 0.40   criterion = 0.7403
```

The minimum sits at $0.06$, where the selector landed at $0.0628$ earlier on the page.

## Holding observations out

Leave-one-out is not a flag. {func}`~kerneljax.ksum`, {func}`~kerneljax.local_poly` and
{func}`~kerneljax.density` all take a `fold` array giving each observation a label, and drop a
pair wherever the evaluation and training labels agree, so `jnp.arange(n)` gives
leave-one-out, which is what the criterion above passes, and any other labeling gives k-fold.
It is why a criterion needs no Python loop over held-out points, and `density` adjusts its
normalizer for the dropped pairs so a density criterion does not have to.

{func}`~kerneljax.cdf` is the exception and takes no `fold`, because a cumulative estimate
holds a point out by removing its own weight from the row total, which is how
`cv_cdf_distribution` works internally. `aic_c_regression` holds nothing out either, paying
for its full-sample fit with a hat-trace penalty, so those are the two shipped criteria with
no `fold` to look for.

## Where the search starts

Selection does not run once. `select_bandwidth` takes `n_starts`, defaulting to `3`, and the
first candidate is the {func}`~kerneljax.normal_reference` rule of thumb for the continuous
columns, with each categorical parameter placed at half its bound rather than at the zero the
rule returns, since zero has no gradient to follow. The rest are that same point shifted by a
constant offset in every unconstrained coordinate. Each candidate gets a
full solve, and the best finite one wins, so `bandwidth`, `value`, `n_iter` and `converged` all
describe the winning start and none of them describe the others.

A criterion whose minimum sits far from that start looks like a solver failure when it is
really a starting-point problem, and the fix is more starts rather than a different solver.
`n_starts=1` isolates a single trajectory for debugging, which is what the comparison below
does.

## Swapping the solver

The same argument applies one level up. `select_bandwidth` takes a `solver` argument, defaulting to the
built-in {func}`~kerneljax.lbfgs`, and calls it as `solver(objective, start)`, expecting the
solved coordinates, the value there, an iteration count and a convergence flag.

```python
def gradient_descent(objective, start, *, steps=300, rate=0.05):
    def step(z, _):
        return z - rate * jax.grad(objective)(z), None

    z, _ = jax.lax.scan(step, start, length=steps)
    converged = jnp.all(jnp.isfinite(z))
    return z, objective(z), jnp.asarray(steps), converged
```

```python
criterion = kj.RegressionCriterion(method="cv_ls", degree=1)

descent = kj.select_bandwidth(train, criterion, y=y, solver=gradient_descent, n_starts=1)
lbfgs = kj.select_bandwidth(train, criterion, y=y, n_starts=1)

for name, result in [("gradient descent", descent), ("L-BFGS", lbfgs)]:
    print(f"{name:16s} value = {result.value:.4f}  h = {result.bandwidth.h[0]:.4f}  "
          f"steps = {result.n_iter:d}")
```

```text
gradient descent value = 1.9194  h = 0.1336  steps = 300
L-BFGS           value = 1.9185  h = 0.1606  steps = 5
```

It runs, and it also shows why the default is not plain gradient descent. Five L-BFGS
iterations beat three hundred fixed-rate steps by five hundredths of a percent on the
criterion, and that margin separates bandwidths a fifth apart. The flag it returns is
whatever the solver says, a message from the solver rather than a verdict from the library.
Both numbers are also worse than what either solver reaches at the default `n_starts=3`,
where the two agree exactly on $h = 0.0323$, so on a criterion this flat the starting point
decides more than the solver does, and a better solver is the second thing to try.

## The requirements at a glance

Five requirements, none of which the type checker will catch, each shown above where it
binds.

1. Accept `kernels` and `chunk`, passed on every call. Refused at the first call otherwise.
2. Return a scalar. Refused by the first gradient.
3. Be hashable, a frozen dataclass, the way a kernel must be. Refused at the jit boundary.
4. Keep the bandwidth gradient finite. A NaN walks the solve to a `nan` bandwidth with
   `converged=False`.
5. Return a finite value. A NaN discards everything the solver tries and hands back the
   starting point, so read `value`, `n_iter` and `converged` rather than the bandwidth alone.
