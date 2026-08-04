# Installation

## Development version

KernelJax has not been released to PyPI yet, so install it from the repository:

```bash
uv pip install git+https://github.com/jordandeklerk/KernelJax.git
```

```{tip}
Check the install with `python -c 'import kerneljax; print(kerneljax.__version__)'`.
```

## GPU and TPU support

JAX is not pinned, so install the build that matches your hardware before KernelJax. The
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) covers the CPU,
CUDA and TPU wheels.

```bash
uv pip install --upgrade "jax[cuda12]"
```

Everything in KernelJax runs on whichever device JAX is configured for, with no change to
the calling code.

## Double precision

JAX defaults to 32 bit floats. Bandwidth selection and the cross-validation criteria are
sensitive to that near an optimum, and agreement with established implementations is
pinned in double precision, so enable 64 bit before importing if you are comparing
numbers rather than exploring:

```python
import jax

jax.config.update("jax_enable_x64", True)
```
