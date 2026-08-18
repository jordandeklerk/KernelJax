# Building documentation

The documentation is built with Sphinx and lives under [docs](https://github.com/jordandeklerk/KernelJax/tree/main/docs).

```bash
pixi run docs         # build
pixi run docs-open    # build, then open in a browser
pixi run docs-serve   # build, then serve on localhost:8765
pixi run docs-clean   # remove the build and doctree
```

The standard build runs the following.

```text
sphinx-build -W --keep-going
```

Warnings therefore fail the build, including broken cross-references. Running a plain `sphinx-build` from an arbitrary virtual environment is not equivalent because it does not necessarily enable `-W` or use the pinned documentation dependencies.

The example pages under `docs/source` are ordinary Markdown files with their output pasted into the page. Runnable notebook copies of the same examples live in `docs/notebooks`, which is gitignored.

If you change an estimator, run the corresponding notebooks and check whether any printed results have changed. Update the documentation when the values shown on the page no longer match what the library produces.
