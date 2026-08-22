# Performance

Every KernelJax estimator and selector is compiled with XLA and runs on the device JAX is
configured to use. This execution model is central to performance, but it also changes how
caching, timing, memory use, and numerical precision behave compared with eager NumPy.
This page explains those differences and shows how to measure performance reliably.

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

## Compilation and cache reuse

The first call to an estimator or selector traces and compiles the computation for the input
pytree structure, array shapes and dtypes, and current static arguments. Later calls can
reuse that program when those properties remain unchanged. Array values are traced rather
than baked in, so a new sample with the same structure, shape, and dtype can reuse an
existing compilation. A previously unseen sample size changes the shape and therefore
requires another compiled program.

Tracing executes our Python code, which means we can watch it happen. A counter inside a
custom criterion records each trace and makes cache reuse visible. One compilation may
trace the criterion several times as different JAX transformations consume it; the exact
count is an implementation detail. The useful invariant is that a cached call adds no new
traces.

```python
calls = []

def counting_criterion(train, bandwidth, *, kernels=None, chunk=None):
    calls.append(1)
    return kj.cv_ls_density(train, bandwidth, kernels=kernels, chunk=chunk)

kj.select_bandwidth(train, counting_criterion)
after_first = len(calls)

kj.select_bandwidth(train, counting_criterion)
print(f"second call added {len(calls) - after_first} traces")
```

```text
second call added 0 traces
```

The second selection reused the cached compilation without retracing the criterion. Creating
a fresh `functools.partial` for each call prevents that reuse because distinct partial
objects do not compare equal, even when they wrap the same function.

```python
kj.select_bandwidth(train, partial(counting_criterion))
after_first_partial = len(calls)

kj.select_bandwidth(train, partial(counting_criterion))
print(f"second fresh partial retraced: {len(calls) > after_first_partial}")
```

```text
second fresh partial retraced: True
```

Each fresh partial is a new static value, so each one causes another cache miss. Bind the
configuration once instead, either at module level or on a frozen dataclass with a
`__call__` method as shown in
[Keep static settings on the criterion](user-guide/custom-criteria.md#keep-static-settings-on-the-criterion).
The same guidance applies to the `solver` argument and custom kernels: their equality and
hashing need to represent their configuration for equivalent instances to share a
compilation.

### Diagnosing cache misses

Setting `jax.config.update("jax_explain_cache_misses", True)` asks JAX to log explanations
for cache misses when it can identify the cause.

Separately, the `JAX_COMPILATION_CACHE_DIR` environment variable enables a persistent disk
cache that carries compiled programs across Python sessions, which is useful in a Colab
runtime that restarts often. The disk cache skips compilations shorter than one second by
default, so lower `jax_persistent_cache_min_compile_time_secs` if smaller programs should
persist too.

## Timing

A wall-clock measurement around one JAX call can be misleading in two directions. The first
call includes compilation and reads high relative to later calls. Asynchronous dispatch can
return control to Python before the device finishes and make the same measurement read low.
The right method depends on the question. First-call latency includes compilation, an
end-to-end measurement may include data transfer, and steady-state device execution excludes
both. The example below measures steady-state device execution by placing the data first,
warming up once, timing repeated calls, and blocking until each result is ready.

An array that begins as NumPy data must be transferred to the device. Explicit placement
before the warmup keeps that transfer outside the timed region. Blocking on the result
without converting it to NumPy also avoids adding a device-to-host transfer to the
measurement.

```python
train = jax.device_put(train)
jax.block_until_ready(train)

bandwidth = kj.normal_reference(train, kj.KernelSet())

fit = kj.density(train, bandwidth)
jax.block_until_ready(fit.value)

elapsed = []
for _ in range(5):
    start = time.perf_counter()
    fit = kj.density(train, bandwidth)
    jax.block_until_ready(fit.value)
    elapsed.append(time.perf_counter() - start)

steady_state = float(np.median(elapsed))
```

The warmup pays tracing and compilation once for this static configuration. The timed calls
reuse the compiled computation, and their median estimates the steady-state cost of a fit.

Two cautions keep comparisons fair. Dispatch overhead can dominate small workloads,
including many kernel fits on a few hundred points. Their timings may measure more
bookkeeping than arithmetic and say little about behavior at scale. NumPy also computes in
`float64` unless told otherwise while JAX defaults to `float32`, so match dtypes before
comparing runtimes.

## Bounding memory with chunk

Kernel computations are pairwise. Evaluating at the training sample can therefore
materialize weight matrices that grow with the square of the sample size. Long before
compute time becomes a problem, memory does. The `chunk` argument processes smaller blocks
through a checkpointed scan, bounding temporary memory for both the value and its gradient.
This trades memory for some recomputation during the backward pass.

XLA's memory analysis shows the effect without executing the compiled computation.

```python
big = kj.MixedData.continuous(jnp.zeros((1600, 1)))
wide = jnp.ones((1600, 4))
h = kj.Bandwidth(h=jnp.array([0.5]), lam_uno=jnp.zeros(0), lam_ord=jnp.zeros(0))

def temp_bytes(chunk):
    call = jax.jit(lambda t, b: kj.ksum(t, b, wide, chunk=chunk))
    return call.lower(big, h).compile().memory_analysis().temp_size_in_bytes

bounded = temp_bytes((64, 64)) < 0.1 * temp_bytes(None)
print(f"chunked temporaries under 10% of unchunked in this example {bounded}")
```

```text
chunked temporaries under 10% of unchunked in this example True
```

On a GPU the bound matters at allocation time. An intermediate that exceeds the available
allocation can fail with an out-of-memory error rather than slow down gradually. Chunk size
trades memory against launch overhead and recomputation, so benchmark several sizes and
choose the fastest one that stays within the memory limit.

JAX preallocates 75% of GPU memory when the first JAX operation runs. The reservation is
adjustable: `XLA_PYTHON_CLIENT_MEM_FRACTION` changes the preallocated share, while
`XLA_PYTHON_CLIENT_PREALLOCATE=false` allocates on demand. Allocating on demand can help
when another process shares the device, at some cost in fragmentation. For diagnosis, the
compile-time report above is often more useful than a runtime profile because a
`jit`-compiled program appears as one opaque call in the device memory profiler, so its
internal allocations are not broken out.

## Matrix multiplication precision on accelerators

One important source of differences between CPU and accelerator results is matrix
multiplication precision. Recent NVIDIA GPUs can use TF32 arithmetic for `float32` matrix
products by default, retaining about ten bits of mantissa. That accuracy tradeoff can
matter for a cross-validation criterion whose optimum is found by comparing nearby values.
KernelJax explicitly requests JAX's highest precision setting for its own `float32` matrix
products. This prevents reduced-precision matrix multiplication from becoming an additional
source of disagreement and does not change CPU behavior.

The precision request is part of the compiled program, so we can inspect it in the jaxpr of
a public entry point.

```python
values = jnp.ones((train.n, 4))
jaxpr = jax.make_jaxpr(lambda t, b: kj.ksum(t, b, values))(train, bandwidth)
dots = [eqn.params["precision"] for eqn in jaxpr.eqns if eqn.primitive.name == "dot_general"]
print(dots[0])
```

```text
(Precision.HIGHEST, Precision.HIGHEST)
```

Custom criteria and kernels that introduce their own matrix products need to choose their
own precision. Wrap the call in JAX's precision context to supply a default for operations
that do not set one explicitly.

```python
with jax.default_matmul_precision("highest"):
    kj.cv_ls_density(train, bandwidth)
```

The context does not override KernelJax's explicit precision request, and differentiation
preserves that request in the compiled gradient. On TPU, the same setting controls whether
`float32` products may use reduced-precision `bfloat16` arithmetic. It has no effect on
`float64` matrix products.

In practice, stable performance comes from reusing input structures and static
configuration, separating warmup from timed execution, choosing `chunk` from both memory
and timing measurements, and setting precision explicitly in custom matrix products.
