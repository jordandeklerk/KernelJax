# Custom criteria

Writing a kernel changes how observations are weighted. Writing a criterion changes what the
library is aiming for when it picks a bandwidth. The two are independent, and both are open.

{func}`~kerneljax.select_bandwidth` takes the criterion as an argument and minimizes whatever
it returns, so the shipped rules have no privileged status. `cv_ls` and `aic` are two functions
passed to the same entry point you can pass your own to, and
{class}`~kerneljax.selection.criteria.Criterion` is a `Protocol` rather than a base class you are
required to inherit from. This page covers the contract and the places it bites. The
[Custom kernels](custom-kernels.md) page covers the other extension point, and
[Bandwidth selection](../background/selection.md) covers what the shipped criteria are doing
and why.

## The contract

There is one line of the library worth reading before writing a criterion, and it is the call
in `select_bandwidth`.

```python
def objective(z):
    bandwidth = transform.from_unconstrained(z)
    return criterion(train, bandwidth, **extra, kernels=kernels, chunk=chunk)
```

A criterion is therefore any callable taking a `MixedData`, a `Bandwidth`, whatever you passed
through `y=` or `criterion_kwargs=`, and the keywords `kernels` and `chunk`, returning one
number. The optimizer differentiates that number with respect to the bandwidth through an
unconstrained reparameterization, so beyond that signature the requirement is that the whole
thing be ordinary JAX.

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

@dataclasses.dataclass(frozen=True)
class AbsoluteDeviation:
    degree: int = 1

    def __call__(self, train, bandwidth, *, y, kernels=None, chunk=None):
        leave_one_out = jnp.arange(train.n)
        fit = kj.local_poly(train, y, bandwidth, kernels=kernels, chunk=chunk,
                            degree=self.degree, fold=leave_one_out)
        return jnp.mean(jnp.abs(y - fit.mean))
```

Passing it to the selector is the same call the shipped criteria go through.

```python
squared = kj.select_bandwidth(train, kj.RegressionCriterion(method="cv_ls", degree=1), y=y)
absolute = kj.select_bandwidth(train, AbsoluteDeviation(degree=1), y=y)

print(f"squared error       h = {squared.bandwidth.h[0]:.4f}")
print(f"absolute deviation  h = {absolute.bandwidth.h[0]:.4f}")
```

```text
squared error       h = 0.0323
absolute deviation  h = 0.0628
```

The two disagree by a factor of two on data carrying eight gross outliers, which is the point
of the exercise and also a warning about it. It would be easy to present absolute deviation as
the robust choice and stop there. Measured against the function the data were generated from,
it is the worse of the two here, with a median absolute error of $0.197$ against $0.105$. The
selection loss is robust, but the local polynomial fit underneath it is still least squares, so
a wider bandwidth spreads the contamination further rather than resisting it.

Nothing in the library will tell you that. It minimizes what you hand it, faithfully, and the
argument that the thing being minimized is worth minimizing remains yours.

## What a criterion must satisfy

```{warning}
Four requirements, none of which the type checker will catch for you.

1. **Accept** `kernels` **and** `chunk`. Both are passed on every call, whether or not you use
   them. A criterion that omits them fails with
   `TypeError: <name>() got an unexpected keyword argument 'kernels'`.
2. **Return a scalar.** The optimizer differentiates the return value, so anything else raises
   `Gradient only defined for scalar-output functions`.
3. **Be hashable.** The criterion is a static argument of the jitted `select_bandwidth`, so a
   mutable dataclass raises `Non-hashable static arguments are not supported`. Use
   `frozen=True`, the same rule that applies to a {class}`~kerneljax.KernelSet`.
4. **Be differentiable in the bandwidth.** The default solver is gradient based, so the same
   care the [custom kernels](custom-kernels.md#what-a-kernel-must-satisfy) page describes
   applies here. A NaN gradient produces a bandwidth of `nan` and `converged=False` rather
   than an exception.
```

## Where settings live

The split between the criterion object and its keyword arguments is not stylistic, and getting
it wrong produces an error that does not explain itself.

Array data goes through `y=` or `criterion_kwargs=`, and reaches the criterion as traced
values. Anything that fixes the *shape* of the computation has to live on the criterion
instead, because the criterion is a static argument and its attributes stay concrete while the
search runs. The local polynomial degree is the usual case. Passing
`criterion_kwargs={"degree": 1}` traces it, and it then fails deep inside the fit with
`TypeError: unhashable type: 'DynamicJaxprTracer'`, because the degree is a static argument of
the estimator underneath. Declaring `degree` as a field of the criterion, as above, is the
supported way.

That field earns its keep a second time. Passing a `SelectionResult` back into an estimator
reuses the settings it was selected under, and `local_poly` finds the degree by reading it off
the criterion the result carries.

```python
fit = kj.local_poly(train, y, absolute)
print(f"degree read off the criterion: {fit.degree}")
```

```text
degree read off the criterion: 1
```

A criterion written as a plain function has no such attribute, so the same call silently falls
back to degree 0 and fits a local constant under a bandwidth chosen for a local linear one.
That is the strongest reason to write a criterion as a frozen dataclass even when it holds no
state worth naming.

## Holding observations out

Leave-one-out is not a flag. Both {func}`~kerneljax.ksum` and {func}`~kerneljax.local_poly`
take a `fold` array giving each observation a label, and drop a pair wherever the evaluation
and training labels agree, so `jnp.arange(n)` gives leave-one-out, which is what the criterion
above passes, and any other labeling gives k-fold. That is the mechanism the shipped criteria
use, and it is why a criterion needs no Python loop over held-out points.

## Swapping the solver

The same argument applies one level up. `select_bandwidth` takes `solver=`, defaulting to the
built-in {func}`~kerneljax.lbfgs`, and calls it as `solver(objective, start)`, expecting the
solved coordinates, the value there, an iteration count and a convergence flag.

```python
def gradient_descent(objective, start, *, steps=300, rate=0.05):
    def step(z, _):
        return z - rate * jax.grad(objective)(z), None

    z, _ = jax.lax.scan(step, start, length=steps)
    converged = jnp.all(jnp.isfinite(z))
    return z, objective(z), jnp.asarray(steps), converged

criterion = kj.RegressionCriterion(method="cv_ls", degree=1)
descent = kj.select_bandwidth(train, criterion, y=y, solver=gradient_descent, n_starts=1)
lbfgs = kj.select_bandwidth(train, criterion, y=y, n_starts=1)

for name, result in [("gradient descent", descent), ("L-BFGS", lbfgs)]:
    print(f"{name:16s} value = {result.value:.4f}  h = {result.bandwidth.h[0]:.4f}  "
          f"steps = {result.n_iter:d}")
```

```text
gradient descent value = 1.9194  h = 0.1336  steps = 300
L-BFGS           value = 1.9185  h = 0.1606  steps = 6
```

It runs, and it also shows why the default is not plain gradient descent. Six L-BFGS
iterations beat three hundred fixed-rate steps by five hundredths of a percent on the
criterion, and that margin separates bandwidths a fifth apart. The valley is that shallow, and
a solver carrying no curvature information has correspondingly little gradient to work with.
The flag it returns is whatever the solver says, so `result.converged` here reports only that
the iterate is finite. It is a message from the solver, not a verdict from the library.
