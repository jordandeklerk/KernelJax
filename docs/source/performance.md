# Performance

Every KernelJax estimator and selector is compiled with XLA and runs on whichever device JAX is configured for. Compilation is what makes the library fast, and it is also what makes performance behave differently than it does in eager NumPy. Work is compiled once and reused, results arrive asynchronously, and a GPU will quietly change the arithmetic of a matrix product unless told otherwise. This page walks through each of these mechanics and shows how to measure the library honestly.

```python
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import kerneljax as kj
```

```python
rng = np.random.default_rng(7)
x = rng.normal(size=(100, 1))

train = kj.MixedData.continuous(x)
```

## Compilation and the jit cache

The first call to an estimator or selector traces the computation and compiles it for the current configuration of static arguments. Every later call reuses the compiled program, provided each static argument compares equal to a cached one. The data itself is traced rather than baked in, so new samples of the same shape never trigger a recompile. The converse deserves equal weight. Shapes are part of the compiled program, so every new sample size is a new compile, and a workload that sweeps over many sizes, bootstrap resamples for instance, does better padding its samples to a common length.

Tracing runs our Python code, which means we can watch it happen. A counter placed inside a custom criterion records how many times the criterion is traced, and that makes the cache visible directly. One compilation traces the criterion several times, once per transform that consumes it, and a cached call adds none.

```python
calls = []

def counting_criterion(train, bandwidth, *, kernels=None, chunk=None):
    calls.append(1)
    return kj.cv_ls_density(train, bandwidth, kernels=kernels, chunk=chunk)

kj.select_bandwidth(train, counting_criterion)
after_first = len(calls)

kj.select_bandwidth(train, counting_criterion)
print(f"traces after first call {after_first}, after second {len(calls)}")
```

```text
traces after first call 5, after second 5
```

The second selection found its program in the cache and never touched our Python again. It is tempting to configure a criterion inline with `functools.partial`, and this is exactly where the cache breaks, because two partials never compare equal even when they wrap the same function.

```python
kj.select_bandwidth(train, partial(counting_criterion))
kj.select_bandwidth(train, partial(counting_criterion))
print(f"traces after two partial calls {len(calls)}")
```

```text
traces after two partial calls 15
```

Each fresh partial is a new static value, and each one pays the full trace and compile again. The remedy is to bind configuration once, either at module level or on a frozen dataclass with a `__call__` method as shown in [Keep static settings on the criterion](user-guide/custom-criteria.md#keep-static-settings-on-the-criterion). The same reasoning covers the `solver` argument and custom kernels, which compare by configuration only when written as frozen dataclasses.

When a recompile surprises you anyway, JAX will name the cause. Setting `jax.config.update("jax_explain_cache_misses", True)` logs an explanation for every cache miss. And setting the `JAX_COMPILATION_CACHE_DIR` environment variable enables a persistent disk cache that carries compiled programs across Python sessions, which is worth knowing in a Colab runtime that restarts often. The disk cache skips compilations shorter than one second by default, so lower `jax_persistent_cache_min_compile_time_secs` if smaller programs should persist too.

## Timing

The natural instinct is to wrap a call in a timer and read off the difference, and under JAX that number is wrong in two directions at once. The first call includes compilation, which later calls never pay, so the measurement runs too large. Dispatch is asynchronous, so control returns to Python while the device is still working, and the measurement runs too small. The honest pattern warms the computation up once, then times a second call and blocks on its result, and repeating the timed call guards the number against noise.

One more cost hides in plain sight. An array that begins as NumPy data is copied to the device inside the call, so moving the sample over beforehand keeps transfer cost out of the compute measurement. Block rather than read, too, since converting a result to NumPy inside the timed region adds a device to host transfer on top of the work being measured.

```python
bandwidth = kj.normal_reference(train, kj.KernelSet())

fit = kj.density(train, bandwidth)
jax.block_until_ready(fit.value)

start = time.perf_counter()
fit = kj.density(train, bandwidth)
jax.block_until_ready(fit.value)
elapsed = time.perf_counter() - start
```

The first call above pays tracing and compilation once for this configuration of static arguments. The second call reuses the compiled computation, and its time is what a fit costs in steady state.

Two cautions keep comparisons fair. Dispatch overhead dominates small problems, and a kernel fit on a few hundred points is such a problem, so a small timing measures bookkeeping rather than arithmetic and says little about behavior at scale. And NumPy computes in float64 unless told otherwise while JAX defaults to float32, so a comparison against a NumPy baseline should match dtypes before it compares clocks.

## Bounding memory with chunk

Kernel computations are pairwise, so an unchunked evaluation materializes weight matrices that grow with the square of the sample size. Long before compute time becomes a problem, memory does. The `chunk` argument on the estimators, the primitives, and the criteria processes blocks of rows through a checkpointed scan instead, which bounds the temporaries of the value and of its gradient at the price of some recomputation on the backward pass.

We do not need to run anything to see the effect, because a compiled program will report its temporary allocations.

```python
big = kj.MixedData.continuous(jnp.zeros((1600, 1)))
wide = jnp.ones((1600, 4))
h = kj.Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

def temp_bytes(chunk):
    call = jax.jit(lambda t, b: kj.ksum(t, b, wide, chunk=chunk))
    return call.lower(big, h).compile().memory_analysis().temp_size_in_bytes

bounded = temp_bytes((64, 64)) < 0.1 * temp_bytes(None)
print(f"chunked temporaries stay under a tenth of unchunked {bounded}")
```

```text
chunked temporaries stay under a tenth of unchunked True
```

On a GPU the bound matters at allocation time. XLA reserves most of the device memory when it starts, and an intermediate that exceeds the reservation fails outright rather than degrading gradually. The chunk size trades memory against launch overhead, so the largest chunk that fits is usually the right one.

The reservation itself is adjustable. `XLA_CLIENT_MEM_FRACTION` changes the preallocated share from its default of three quarters, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` allocates on demand instead, the usual setting when another process shares the device, at some cost in fragmentation. For diagnosing memory the compile time report above is often sharper than a runtime profile, because a jitted program is opaque to the device memory profiler and attributes everything it allocates to one call.

## Matmul precision on GPU

There is one way a GPU will quietly compute something different from a CPU. Recent NVIDIA hardware executes 32 bit matrix products with TF32 arithmetic by default, which keeps about ten bits of mantissa. That is a large concession for a cross-validation criterion whose optimum is found by comparing nearby values. KernelJax pins its own matrix products to full float32 precision, so estimates and criteria agree with CPU results on every device, and the pin costs nothing on CPU.

The pin is part of the compiled program itself, so we can see it in the trace of any public entry point.

```python
jaxpr = jax.make_jaxpr(lambda t, b: kj.ksum(t, b, wide))(big, h)
dots = [eqn.params["precision"] for eqn in jaxpr.eqns if eqn.primitive.name == "dot_general"]
print(dots[0])
```

```text
(Precision.HIGHEST, Precision.HIGHEST)
```

Custom criteria and kernels that introduce their own matrix products do not inherit the pin. The JAX precision context applies the same rule to everything inside it that has not chosen its own, and since it sets a default rather than an override, it cannot relax the library's pin either. The pinned precision also travels through differentiation, which is how the criterion gradients inside selection stay in agreement. On TPU the story repeats with bfloat16 in place of TF32, so the pin protects those results even more, and under 64 bit mode the machinery goes quiet, since float64 products always run at full precision.

```python
with jax.default_matmul_precision("highest"):
    kj.cv_ls_density(train, bandwidth)
```
