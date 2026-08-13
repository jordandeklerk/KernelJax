# Contributing

Welcome to KernelJax. Whether you are fixing a bug, adding an estimator, improving the documentation, or reviewing code, contributions are welcome.

If you have a question or run into trouble, open an issue on [GitHub](https://github.com/jordandeklerk/KernelJax).

## Development process

### 1. Set up your environment

We use [pixi](https://pixi.sh/) to manage development environments. It keeps the Python, conda, and PyPI dependencies in one lockfile so contributors work from the same setup.

[Install pixi](https://pixi.sh/latest/#installation) if you do not already have it, fork the repository on GitHub, then clone your fork.

```bash
git clone https://github.com/your-username/KernelJax.git
cd KernelJax
git remote add upstream https://github.com/jordandeklerk/KernelJax.git
pixi install -e dev
```

The repository has five environments. `dev` runs the tests, `docs` builds the documentation, `check` runs the linters, `build` packages the project, and `gpu` installs the CUDA build of JAX.

If you would rather not use pixi, a virtual environment works too.

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[test,dev]"
```

### 2. Develop your contribution

Branch from the current upstream `main` rather than from whatever state your fork happens to be in.

```bash
git fetch upstream
git checkout -b fix-aic-penalty-barrier upstream/main
```

Make your changes, add tests for new behavior, and commit as you go.

Commit messages start with a capitalized prefix describing the kind of change, followed by a short imperative description. `FEAT` marks new behavior, `REF` covers refactoring and maintenance, and `DOC` covers documentation changes. Pull request titles use the same convention.

```text
FEAT: add standard errors to local polynomial regression
REF: fold the smoother diagonal into the Cholesky pass
DOC: derive the boundary constant on the regression page
```

A commit message should make sense without reading the diff. Say what changed and, where useful, why. If the change closes an issue, reference it with `Closes #123` in the commit body or pull request.

If upstream moves while you work, rebase your branch rather than merging `main` into it. This keeps the branch as a clean sequence of the commits that belong to your change.

```bash
git fetch upstream
git rebase upstream/main
```

### 3. Validate your changes

Run the relevant checks before opening a pull request.

```bash
pixi run -e dev tests      # test suite
pixi run lint              # style and hooks, via prek
pixi run typecheck         # mypy and ty
pixi run docs              # documentation build
```

Code changes should come with tests for the behavior they introduce or modify. Documentation changes should build without warnings.

### 4. Submit your contribution

Push your branch to your fork and open a pull request with a clear description of what changed and why.

```bash
git push origin fix-aic-penalty-barrier
```

If the change affects behavior that a user can observe, mention it in the pull request description. The [changelog](../release-notes.md) is assembled from those descriptions, so user-visible changes that are not called out there are easy to miss at release time.

### 5. Review

Reviewers may leave both inline and general comments. Every change is reviewed, including a maintainer's own changes. The goal is to improve the result, not to evaluate the person who wrote it.

Respond to review by committing and pushing to the same branch. The pull request updates automatically.

CI must pass and a maintainer must approve the change before it can merge. If a week goes by without a response, a comment on the pull request is a reasonable nudge.

## Reporting an issue

Open a [bug report](https://github.com/jordandeklerk/KernelJax/issues/new/choose) with the smallest example you can find that reproduces the problem from a fresh interpreter.

For a numerical library, a few details make a large difference. Include the JAX and KernelJax versions, whether `jax_enable_x64` was enabled, and the value you expected to get along with why you expected it.

The most useful reproducer is small and self-contained. If the problem currently depends on private data or a long pipeline, try to reduce it to the smallest example that still shows the behavior.

## Code and testing conventions

Tests should assert on behavior a caller can observe rather than on implementation details. A useful test fails before the change and passes afterward.

For example, a bandwidth that is never updated can still satisfy a test that checks only the shape of the returned array. If the behavior being tested is numerical, assert on the number that comes back.

Shared fixtures live in `tests/conftest.py`. Put reusable setup there rather than duplicating it across test files.

Public functions and classes use numpydoc docstrings, which are also used to generate the API reference. Include an `Examples` section when a caller would benefit from seeing the function used, and a `References` section when the implementation comes from the literature.

Numerical agreement with the R package [np](https://cran.r-project.org/package=np) is the reference standard for KernelJax estimators. If you change an estimator, criterion, or kernel, compare the resulting numbers against `np` rather than treating the previous KernelJax output as ground truth.

## Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a line length of 120.

Use the standard import conventions:

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

The hooks then run on every commit. To run them over the entire repository, use:

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

A useful check is:

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

## Building documentation

The documentation is built with Sphinx and lives under [docs](https://github.com/jordandeklerk/KernelJax/tree/main/docs).

```bash
pixi run docs         # build
pixi run docs-open    # build, then open in a browser
pixi run docs-serve   # build, then serve on localhost:8765
pixi run docs-clean   # remove the build and doctree
```

The standard build runs:

```text
sphinx-build -W --keep-going
```

Warnings therefore fail the build, including broken cross-references. Running a plain `sphinx-build` from an arbitrary virtual environment is not equivalent because it does not necessarily enable `-W` or use the pinned documentation dependencies.

The example pages under `docs/source` are ordinary Markdown files with their output pasted into the page. Runnable notebook copies of the same examples live in `docs/notebooks`, which is gitignored.

If you change an estimator, run the corresponding notebooks and check whether any printed results have changed. Update the documentation when the values shown on the page no longer match what the library produces.

## Continuous integration

Every pull request and push to `main` runs the `test` workflow across Python 3.11 through 3.14 on Ubuntu. The Python 3.12 job uploads coverage. `codeql` runs static analysis and also runs on Mondays.

Both checks block a merge. If a pull request fails, `test` is usually the first workflow to inspect.

Several other workflows run independently of pull requests. `nightly` runs on Sundays against nightly scientific Python wheels. A failure there usually signals an upcoming upstream compatibility problem rather than a regression introduced by a particular pull request.

`publish` uploads releases to PyPI through Trusted Publishing when a `v*` tag is created, and `post-release` regenerates the changelog after a release is published.

Two differences between CI and a local environment explain many failures that are difficult to reproduce. CI runs on Ubuntu while development is often done on macOS, and floating-point behavior can differ slightly across platforms.

JAX also uses 32-bit floating point by default. If a numerical comparison passes locally but fails under a tighter tolerance, check the configured precision. For comparisons that require 64-bit values, enable them before the relevant JAX computation:

```python
jax.config.update("jax_enable_x64", True)
```

## Dependency management

The core dependencies are `jax`, `jaxlib`, and `jaxtyping`, with minimum supported versions declared in `pyproject.toml`.

A core dependency is installed for every KernelJax user, so packages needed only for tests, documentation, or development tooling belong in the corresponding extras instead.

Those development dependencies are represented in two places. `pixi.toml` controls what commands such as `pixi run docs` resolve, either as conda dependencies or under `pypi-dependencies` when a package is not available from conda-forge. `pyproject.toml` controls what environments such as Read the Docs install.

When adding or changing a dependency, make sure the two declarations remain consistent. A dependency that exists in one environment but not the other can make the documentation build locally and fail elsewhere.

KernelJax supports Python 3.11 and above, so code that relies on a newer Python feature needs an appropriate version guard or compatible alternative.
