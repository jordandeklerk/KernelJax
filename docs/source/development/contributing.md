# Contributing

Welcome to KernelJax. Whether you are fixing a bug, adding an estimator, improving the
documentation or reviewing code, contributions are welcome.

If you have a question or run into trouble, open an issue on
[GitHub](https://github.com/jordandeklerk/KernelJax).

## Development process

**1. Set up your environment.**

We use [pixi](https://pixi.sh/) to manage development environments. Pixi handles Python,
conda and PyPI dependencies in one lockfile, so every contributor gets an identical setup.

[Install pixi](https://pixi.sh/latest/#installation) if you do not have it, fork the
repository on GitHub, then clone your fork.

```bash
git clone https://github.com/your-username/KernelJax.git
cd KernelJax
git remote add upstream https://github.com/jordandeklerk/KernelJax.git
pixi install -e dev
```

There are five environments. `dev` runs the tests, `docs` builds the documentation, `check`
runs the linters, `build` packages the project, and `gpu` installs the CUDA build of JAX.

If you would rather not use pixi, a virtual environment works too.

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[test,dev]"
```

**2. Develop your contribution.**

Branch from the upstream main rather than from whatever your fork happens to be at, so you
start from current work. Make your changes with tests for anything new, and commit as you go.

```bash
git fetch upstream
git checkout -b fix-aic-penalty-barrier upstream/main
```

Commit messages start with a capitalized prefix saying what kind of change it is, then a
short description in the imperative. For example, `FEAT` marks new behavior, `REF` indicates
refactoring and maintenance, and `DOC` refers to documentation changes. Pull request titles take the same prefix.

```text
FEAT: add standard errors to local polynomial regression
REF: fold the smoother diagonal into the Cholesky pass
DOC: derive the boundary constant on the regression page
```

A message should be understandable without the diff, so say what changed and why rather than
"fix another one". Reference an issue with `Closes #123` in the body.

If upstream moves while you work, rebase rather than merging, which keeps the branch a clean
sequence of your own commits.

```bash
git fetch upstream
git rebase upstream/main
```

**3. Validate your changes.**

```bash
pixi run -e dev tests      # the test suite
pixi run lint              # style, via prek
pixi run typecheck         # mypy and ty
pixi run docs              # build the documentation
```

**4. Submit your contribution.**

Push to your fork and open a pull request with a clear description of what the change does
and why.

```bash
git push origin fix-aic-penalty-barrier
```

If the change alters behavior a user can see, say so in the pull request description. Those
notes are what the [release notes](../release-notes.md) are assembled from, so a change that
goes unmentioned there tends to go unmentioned at release.

**5. Review.**

Reviewers leave inline and general comments. Every change gets reviewed, including a
maintainer's own, and the aim is the quality of the result rather than a judgment of the
author. Update the pull request by committing and pushing to the same branch, and it updates
itself. CI has to pass and a maintainer has to approve before a merge. If a week goes by with
no response, a comment on the pull request is a fair nudge.

## Reporting an issue

Open a [bug report](https://github.com/jordandeklerk/KernelJax/issues/new/choose) with a
snippet that reproduces the problem from a fresh interpreter. For a numerical library, three
details decide whether a report is actionable. Include the JAX and KernelJax versions, whether
`jax_enable_x64` was set, and what you expected the number to be and why. A reproducer that
depends on private data or on a fifty-line pipeline is usually not one anybody can act on.

## Guidelines

Every code change should come with tests, and the useful standard is that they fail before
your change and pass after it. Write them to assert on behavior a caller can observe rather
than on the shape of the implementation. A bandwidth that is never updated will still satisfy
a test that only checks the returned array's shape, so assert on the number that came back.

Public functions and classes carry numpydoc docstrings, which is what the API reference is
generated from. Include an `Examples` section wherever a caller would benefit from seeing the
call, and a `References` section for anything with a source in the literature.

Numerical agreement with the R package [np](https://cran.r-project.org/package=np) is the
standard KernelJax holds its estimators to. If you change an estimator, a criterion or a
kernel, check the numbers against np rather than against the previous KernelJax output.

## Stylistic guidelines

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/), with a line length of 120.

Use the standard import conventions, `import jax.numpy as jnp` and `import numpy as np`, and
keep imports grouped as standard library, third party, then local. Absolute imports are
enforced by a hook.

Prefer descriptive names to short ones. Code is read more often than it is written.

## Code quality tools

[ruff](https://docs.astral.sh/ruff/) handles linting, formatting and import sorting in one
tool, configured in `pyproject.toml` with the numpy docstring convention.

```bash
ruff check kerneljax tests        # lint
ruff format kerneljax tests       # format
ruff check --fix kerneljax tests  # auto-fix
```

Type checking runs both [mypy](https://mypy-lang.org/) and
[ty](https://github.com/astral-sh/ty) over `kerneljax`, which is what `pixi run typecheck`
does.

### Pre-commit hooks

We use [prek](https://github.com/j178/prek) to manage pre-commit hooks. Install them once
after cloning.

```bash
prek install
```

They then run on every commit. To run them over the whole tree, use `prek run --all-files`,
which is what `pixi run lint` does. Alongside ruff, the hooks block committing directly to a
protected branch, reject stray `print` statements, and check for private keys, merge
conflicts and large files.

## JAX conventions

KernelJax is a JAX library, and a few conventions follow from that.

Three rules bind anything you write here, and the user guides already work them through with
running examples rather than assertions. Anything handed to the compiler as a static argument
has to be hashable, so kernels and criteria are frozen dataclasses. Code that will be
differentiated has to be differentiable on every branch it traces, not only the one that runs,
which is why `jnp.where` needs its denominator guarded and not just its branch. And a setting
that fixes the shape of a computation has to stay concrete rather than arrive as array data.
[Custom kernels](../user-guide/custom-kernels.md#the-requirements-at-a-glance) demonstrates the
first two, [Custom bandwidth selection](../user-guide/custom-criteria.md#where-settings-live) the third.

When you register a container as a pytree, a field is either data or metadata, and putting a
static setting in `data_fields` makes it a tracer the moment the container crosses a `jit`
boundary. It then fails wherever that value is used as a shape, far from the registration that
caused it.

```python
@partial(jax.tree_util.register_dataclass, data_fields=["values", "degree"], meta_fields=[])
@dataclasses.dataclass(frozen=True)
class Wrong:
    values: jax.Array
    degree: int
```

```text
TypeError: Shapes must be 1D sequences of concrete values of integer type
```

`degree` belongs in `meta_fields`, which is where `MixedData` keeps its `ColumnSpec` and
`Bandwidth` its `h_axis`. The quick check is that `jax.tree_util.tree_leaves` should return
only arrays.

An array built from a Python scalar is weakly typed, and two otherwise identical values
differing only in that flag are different cache keys, so a function retraces when it should
have hit the cache.

```python
weak = jnp.full((3,), 3.0)        # from a Python float, weak_type=True
strong = weak.astype(weak.dtype)  # same values, concrete dtype
```

That single `astype` is why `from_blocks` normalizes its continuous block, and it is worth
doing wherever user input reaches a cached call.

## Test coverage

Pull requests that change code should include tests covering the new behavior, including edge
cases.

```bash
pixi run -e dev tests
```

Shared fixtures live in `tests/conftest.py`. Put new fixtures there rather than duplicating
setup across files.

## Building documentation

The documentation is built with Sphinx and lives in
[docs](https://github.com/jordandeklerk/KernelJax/tree/main/docs).

```bash
pixi run docs         # build
pixi run docs-open    # build, then open in a browser
pixi run docs-serve   # build, then serve on localhost:8765
pixi run docs-clean   # remove the build and doctree
```

The build runs `sphinx-build -W --keep-going`, so warnings are errors and a broken
cross-reference fails the build rather than slipping through. A plain `sphinx-build` from a
virtual environment is not the same check, since it neither enforces `-W` nor resolves the
pinned documentation dependencies.

The example pages under `docs/source` are ordinary markdown with their output pasted in.
Runnable notebook copies of the same examples live in `docs/notebooks`, which is gitignored,
so you can execute them locally to confirm that the numbers a page reports are still what the
library produces. If you change an estimator, run those notebooks and update any page whose
printed output moved.

## Continuous integration

Every pull request and push to `main` runs `test` across Python 3.11 to 3.14 on Ubuntu,
uploading coverage from the 3.12 job, and `codeql` for static analysis, which also runs on
Monday. Both block a merge, and `test` is the one to check first when a pull request fails.

Three more run on their own schedule. `nightly` runs on Sunday against nightly
scientific-python wheels, and a failure there flags upcoming upstream breakage rather than
anything wrong with your change. `publish` uploads to PyPI through Trusted Publishing on a
`v*` tag, and `post-release` regenerates the changelog once a release is published.

Two differences from your machine account for most failures that reproduce nowhere else. CI
runs on Ubuntu while you are likely on macOS, and floating point differs slightly between
platforms. JAX also defaults to 32-bit floats, which is the usual reason a comparison passes
locally and fails on a tighter tolerance, so set
`jax.config.update("jax_enable_x64", True)` before importing when you are comparing numbers.

## Dependency management

The core is `jax`, `jaxlib` and `jaxtyping`, pinned to minimum versions in `pyproject.toml`.
Every addition there is one that every user carries, so anything needed only for the tests, the
documentation or the linters goes in an extra instead.

Those extras are declared twice. `pixi.toml` is what `pixi run docs` resolves, as a conda
dependency or under `pypi-dependencies` when the package is not on conda-forge, and
`pyproject.toml` is what Read the Docs installs. A dependency added to one and not the other
builds in one place and fails in the other.

KernelJax supports Python 3.11 and above, so a feature newer than that needs a version gate.
