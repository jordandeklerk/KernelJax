"""Tests for the shared typing aliases."""

import jax.numpy as jnp


def test_aliases_exist_and_annotate():
    from kerneljax.typing import Array, FloatArray, IntArray, ScalarFloat

    assert Array is not None
    assert FloatArray is not None
    assert IntArray is not None
    assert ScalarFloat is not None


def test_jaxtyping_annotation_accepts_real_array():
    from jaxtyping import Float

    from kerneljax.typing import Array

    def f(x: Float[Array, "n p"]) -> Float[Array, " n"]:
        return x.sum(axis=-1)

    out = f(jnp.ones((3, 2)))
    assert out.shape == (3,)
