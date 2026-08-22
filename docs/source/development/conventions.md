# Code conventions

KernelJax code follows a small set of conventions covering testing, style, tooling, and the constraints JAX places on numerical code. The [development process](contributing.md) describes how a change travels from a branch to a merge.

## Testing

Tests should assert on behavior a caller can observe rather than on implementation details. A useful test fails before the change and passes afterward.

Structural assertions rarely meet that bar. The test below passes whether or not bandwidth selection works, because a selection that returns its starting point untouched still has the right shape.

```python
def test_selection_returns_a_bandwidth(probe_x, probe_y):
    criterion = RegressionCriterion(method="cv_ls", degree=1)
    result = select_bandwidth(probe_x, criterion, y=probe_y)

    assert result.bandwidth.h.shape == (1,)
```

Assert instead on the promise the feature makes. Selection promises to improve the criterion it minimizes, so compare the selected bandwidth against the reference starting point on that criterion.

```python
def test_selection_improves_on_its_starting_point(probe_x, probe_y):
    criterion = RegressionCriterion(method="cv_ls", degree=1)
    result = select_bandwidth(probe_x, criterion, y=probe_y)

    start = normal_reference(MixedData.continuous(probe_x), KernelSet())
    at_start = cv_ls_regression(MixedData.continuous(probe_x), start, y=probe_y, degree=1)

    assert float(result.value) < float(at_start)
```

The first version once let a bandwidth that was never updated survive the entire test suite. The second fails the moment selection stops selecting.

When the same behavior should hold across several configurations, parameterize over them rather than writing near-identical copies.

```python
@pytest.mark.parametrize("low, high", [(100, 200), (-1, 1), (7, 0)])
def test_fold_labels_are_arbitrary_integers(density_data, density_bandwidth, low, high):
    half = density_data.n // 2
    canonical = jnp.where(jnp.arange(density_data.n) < half, 0, 1)
    relabeled = jnp.where(jnp.arange(density_data.n) < half, low, high)

    reference = density(density_data, density_bandwidth, fold=canonical)
    got = density(density_data, density_bandwidth, fold=relabeled)

    np.testing.assert_allclose(np.asarray(got.value), np.asarray(reference.value), rtol=1e-6)
```

Shared fixtures live in `tests/conftest.py`. Put reusable setup there rather than duplicating it across test files.

Public functions and classes use numpydoc docstrings, which are also used to generate the API reference. Include an `Examples` section when a caller would benefit from seeing the function used, and a `References` section when the implementation comes from the literature.

Numerical agreement with the R package [np](https://cran.r-project.org/package=np) is the reference standard for KernelJax estimators. If you change an estimator, criterion, or kernel, compare the resulting numbers against `np` rather than treating the previous KernelJax output as ground truth.

## Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a line length of 120.

Use the standard import conventions.

```python
import jax.numpy as jnp
import numpy as np
```

Keep imports grouped as standard library, third party, then local. Absolute imports are enforced by a hook.

Prefer descriptive names over abbreviations unless the shorter name is conventional in the surrounding mathematics or statistics.

## Code quality tools

[ruff](https://docs.astral.sh/ruff/) handles linting, formatting, and import sorting. Its configuration lives in `pyproject.toml` and uses the NumPy docstring convention.

```bash
ruff check kerneljax tests        # lint
ruff format kerneljax tests       # format
ruff check --fix kerneljax tests  # auto-fix
```

Type checking runs both [mypy](https://mypy-lang.org/) and [ty](https://github.com/astral-sh/ty) over `kerneljax`.

```bash
pixi run typecheck
```

### Pre-commit hooks

We use [prek](https://github.com/j178/prek) to manage pre-commit hooks. Install them once after cloning.

```bash
prek install
```

The hooks then run on every commit. The command below runs them over the entire repository.

```bash
prek run --all-files
```

This is also what `pixi run lint` runs. Alongside ruff, the hooks prevent commits directly to protected branches, reject stray `print` statements, and check for private keys, merge conflicts, and large files.

## JAX conventions

KernelJax relies on JAX transformations throughout the library, so a few constraints show up repeatedly in new code.

Static values need to remain hashable. Code that may be differentiated has to stay differentiable along every branch JAX traces, not only the branch that happens to execute. Settings that determine the structure of a computation need to remain concrete rather than arriving as traced array values.

The user guides work through these cases with running examples. [Custom kernels](../user-guide/custom-kernels.md#interface-at-a-glance) covers the first two, and [custom bandwidth selection](../user-guide/custom-criteria.md#keep-static-settings-on-the-criterion) covers the third.

### Keep static configuration out of pytree data

When a container is registered as a pytree, each field is either dynamic data or static metadata. Putting a structural setting in `data_fields` turns it into a tracer as soon as the object crosses a `jit` boundary.

That often causes the failure much later, when the traced value is eventually used somewhere that requires a concrete shape.

```python
@partial(
    jax.tree_util.register_dataclass,
    data_fields=["values", "degree"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True)
class Wrong:
    values: jax.Array
    degree: int
```

```text
TypeError: Shapes must be 1D sequences of concrete values of integer type
```

`degree` belongs in `meta_fields`. KernelJax uses the same distinction for fields such as the `ColumnSpec` carried by `MixedData` and the `h_axis` carried by `Bandwidth`.

A useful check is the following.

```python
jax.tree_util.tree_leaves(obj)
```

For these containers, the leaves should be the array-valued data rather than structural configuration.

### Normalize weakly typed inputs

An array created directly from a Python scalar can carry JAX's `weak_type` flag. Two values that are otherwise identical but differ in weak typing can produce different cache keys, causing a function to retrace when it could have reused an existing compilation.

```python
weak = jnp.full((3,), 3.0)        # from a Python float, weak_type=True
strong = weak.astype(weak.dtype)  # same values, concrete dtype
```

That final `astype` is why `from_blocks` normalizes its continuous input. The same pattern is worth using anywhere user-provided values feed into a cached computation.
