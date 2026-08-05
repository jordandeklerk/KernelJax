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

Create a branch with a descriptive name, make your changes with tests for anything new, and
commit as you go.

```bash
git checkout -b fix-aic-penalty-barrier
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

**5. Review.**

Reviewers will give feedback on the pull request. Update it by committing and pushing to the
same branch, and the pull request updates itself. CI must pass before a merge.

## Guidelines

Every code change should come with tests that verify the new behavior, and those tests should
assert on behavior a caller can observe rather than on the shape of the implementation. A
bandwidth that is never updated will still satisfy a test that only checks the returned
array's shape, so assert on the number that came back.

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

KernelJax is a JAX library, and a few conventions follow from that rather than from taste.

Anything handed to the compiler as a static argument has to be hashable. Kernels and criteria
travel that way, so they are frozen dataclasses. A mutable dataclass fails with
`Non-hashable static arguments are not supported`, and the failure can surface far from the
change that caused it.

Code that will be differentiated has to be differentiable everywhere it is traced, not only on
the branch that runs. `jnp.where` evaluates and differentiates *both* branches, so an unguarded
expression in the untaken branch poisons the gradient while leaving the value correct. Guard
the denominator rather than only the branch, and test the gradient rather than only the value,
since a forward-only check will not catch it.

Prefer `jnp.where` to Python control flow on traced values, and keep static settings such as a
polynomial degree on the object rather than passing them through as array data.

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

Every pull request and push to `main` triggers GitHub Actions.

The `test` workflow runs the suite across Python 3.11, 3.12, 3.13 and 3.14 on Ubuntu, and
uploads coverage to Codecov from the 3.12 job. This is the job to check first when a pull
request fails.

The `nightly` workflow runs on Sunday at 03:00 UTC against nightly scientific-python wheels.
Failures there are informational, flagging upcoming upstream breakage rather than blocking a
pull request.

The `codeql` workflow runs static analysis weekly and on pushes to `main`, `publish` builds
and uploads to PyPI on a `v*` tag through Trusted Publishing, and `post-release` regenerates
the changelog after a release.

CI runs on Ubuntu while you may be developing on macOS, and floating point behavior differs
slightly between platforms. JAX also defaults to 32-bit floats, which is usually the reason a
comparison that passes locally fails on a tighter tolerance. Enable double precision with
`jax.config.update("jax_enable_x64", True)` before importing when you are comparing numbers
rather than exploring.

## Dependency management

The core dependencies are `jax`, `jaxlib` and `jaxtyping`, pinned to minimum versions in
`pyproject.toml`. Keep that set small. KernelJax is a low-level library, and every dependency
added to the core is one that every user carries.

Optional dependencies are grouped as extras in `pyproject.toml` and mirrored as features in
`pixi.toml`. Anything needed only for the tests, the documentation or the linters belongs in
one of those groups rather than in the core.

A documentation dependency has to be added in **both** places, `pixi.toml` for the local build
and `pyproject.toml` for Read the Docs, which installs the `doc` extra. Adding it to only one
gives you a build that works in one place and fails in the other.

KernelJax supports Python 3.11 and above. Do not use language features that require a newer
version without gating them behind a check.
