"""Shared type aliases for KernelJax."""

from jax import Array
from jaxtyping import Bool, Float, Int

__all__ = ["Array", "BoolArray", "FloatArray", "IntArray", "ScalarFloat", "ScalarInt"]

FloatArray = Float[Array, "..."]
IntArray = Int[Array, "..."]
BoolArray = Bool[Array, "..."]
ScalarFloat = Float[Array, ""]
ScalarInt = Int[Array, ""]
