"""MRST-shaped gravity namespace."""

from pymrst.params.gravity.gravity import (
    NORM_GRAVITY, current_gravity, gravity, gravity_or_zero, gravity_vector,
    isolated_gravity,
)

__all__ = [
    "NORM_GRAVITY", "current_gravity", "gravity", "gravity_or_zero",
    "gravity_vector", "isolated_gravity",
]
