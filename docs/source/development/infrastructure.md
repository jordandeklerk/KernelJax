# Infrastructure

Continuous integration runs around every contribution, and one set of dependency declarations keeps every environment consistent.

## Continuous integration

Every pull request and push to `main` runs the `test` workflow across Python 3.11 through 3.14 on Ubuntu. The Python 3.12 job uploads coverage. `codeql` runs static analysis and also runs on Mondays.

Both checks block a merge. If a pull request fails, `test` is usually the first workflow to inspect.

Several other workflows run independently of pull requests. `nightly` runs on Sundays against nightly scientific Python wheels. A failure there usually signals an upcoming upstream compatibility problem rather than a regression introduced by a particular pull request.

`publish` uploads releases to PyPI through Trusted Publishing when a `v*` tag is created, and `post-release` regenerates the changelog after a release is published.

Two differences between CI and a local environment explain many failures that are difficult to reproduce. CI runs on Ubuntu while development is often done on macOS, and floating-point behavior can differ slightly across platforms.

JAX also uses 32-bit floating point by default. If a numerical comparison passes locally but fails under a tighter tolerance, check the configured precision. For comparisons that require 64-bit values, enable them before the relevant JAX computation.

```python
jax.config.update("jax_enable_x64", True)
```

## Dependency management

The core dependencies are `jax`, `jaxlib`, and `jaxtyping`, with minimum supported versions declared in `pyproject.toml`.

A core dependency is installed for every KernelJax user, so packages needed only for tests, documentation, or development tooling belong in the corresponding extras instead.

Those development dependencies are represented in two places. `pixi.toml` controls what commands such as `pixi run docs` resolve, either as conda dependencies or under `pypi-dependencies` when a package is not available from conda-forge. `pyproject.toml` controls what environments such as Read the Docs install.

When adding or changing a dependency, make sure the two declarations remain consistent. A dependency that exists in one environment but not the other can make the documentation build locally and fail elsewhere.

KernelJax supports Python 3.11 and above, so code that relies on a newer Python feature needs an appropriate version guard or compatible alternative.
