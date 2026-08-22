# Development process

Every change to KernelJax travels the same path from a fork to a merge. The steps below cover that path, and the final section covers reporting a problem instead.

## 1. Set up your environment

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

## 2. Develop your contribution

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

## 3. Validate your changes

Run the relevant checks before opening a pull request.

```bash
pixi run -e dev tests      # test suite
pixi run lint              # style and hooks, via prek
pixi run typecheck         # mypy and ty
pixi run docs              # documentation build
```

Code changes should come with tests for the behavior they introduce or modify. Documentation changes should build without warnings.

## 4. Submit your contribution

Push your branch to your fork and open a pull request with a clear description of what changed and why.

```bash
git push origin fix-aic-penalty-barrier
```

If the change affects behavior that a user can observe, mention it in the pull request description. The [changelog](../release-notes.md) is assembled from those descriptions, so user-visible changes that are not called out there are easy to miss at release time.

## 5. Review

Reviewers may leave both inline and general comments. Every change is reviewed, including a maintainer's own changes. The goal is to improve the result, not to evaluate the person who wrote it.

Respond to review by committing and pushing to the same branch. The pull request updates automatically.

CI must pass and a maintainer must approve the change before it can merge. If a week goes by without a response, a comment on the pull request is a reasonable nudge.

## Reporting an issue

Open a [bug report](https://github.com/jordandeklerk/KernelJax/issues/new/choose) with the smallest example you can find that reproduces the problem from a fresh interpreter.

For a numerical library, a few details make a large difference. Include the JAX and KernelJax versions, whether `jax_enable_x64` was enabled, and the value you expected to get along with why you expected it.

The most useful reproducer is small and self-contained. If the problem currently depends on private data or a long pipeline, try to reduce it to the smallest example that still shows the behavior.
