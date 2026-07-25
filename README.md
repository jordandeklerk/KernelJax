<div style="text-align: center;" align="center">

<img alt="KernelJAX" src="docs/source/_static/kerneljax-logo.png" width="650">

<br>

<p>
  <em>A low-level JAX interface for nonparametric kernel smoothing of mixed-type data.</em>
</p>

<p>
  <a href="https://kerneljax.readthedocs.io/en/latest/" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://kerneljax.readthedocs.io/en/latest/api/index.html" target="_blank"><strong>API Reference</strong></a> ·
  <a href="https://kerneljax.readthedocs.io/en/latest/user_guide/index.html" target="_blank"><strong>Tutorials</strong></a> ·
  <a href="https://github.com/jordandeklerk/KernelJax/blob/main/CHANGELOG.md" target="_blank"><strong>Changelog</strong></a>
</p>

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jordandeklerk/KernelJax/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![Documentation](https://readthedocs.org/projects/kerneljax/badge/?version=latest)](https://kerneljax.readthedocs.io/en/latest/)
[![Python version](https://img.shields.io/badge/3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/KernelJax)](https://github.com/jordandeklerk/KernelJax/graphs/commit-activity)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/KernelJax)](https://github.com/jordandeklerk/KernelJax/graphs/commit-activity)

</div>

__KernelJAX__ is a low-level JAX library for nonparametric kernel smoothing of
mixed-type data, aimed at researchers who want to develop new methodology from
composable, differentiable building blocks. Each kernel, bandwidth selector, and
smoother is designed to be understood, modified, and recombined, not just
called. Through JAX and XLA, estimators built from these pieces run natively on
CPUs, GPUs, and TPUs with automatic differentiation and vectorized operations,
and fit naturally alongside the broader JAX ecosystem.

<br>

> [!WARNING]
> KernelJAX is in early development and has not been released yet. The API is
> unstable, and installation instructions, documentation, and examples are
> coming soon.
