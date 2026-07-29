"""MRST ``discreteDivergence`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.operators import (
    discrete_divergence,
    divergence_with_accumulation,
)

discreteDivergence = divergence_with_accumulation

__all__ = [
    "discrete_divergence",
    "divergence_with_accumulation",
    "discreteDivergence",
]
