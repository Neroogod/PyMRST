"""Gravity handling, replacing MRST's global gravity() state.

PyMRST has no global gravity toggle: solvers take an explicit `gravity`
vector argument (None = no gravity). Use gravity_vector() to get the
standard downward vector, matching MRST's `gravity reset; gravity on`.
"""

from typing import Optional

import numpy as np

NORM_GRAVITY: float = 9.80665  # m/s^2, standard gravity (MRST's norm(gravity))


def gravity_vector(griddim: int = 3) -> np.ndarray:
    """Standard gravity vector: NORM_GRAVITY along the last axis (depth)."""
    g = np.zeros(griddim)
    g[-1] = NORM_GRAVITY
    return g


def gravity_or_zero(gravity: Optional[np.ndarray], griddim: int) -> np.ndarray:
    """Normalize a solver's gravity argument: None becomes the zero vector."""
    if gravity is None:
        return np.zeros(griddim)
    gravity = np.asarray(gravity, dtype=np.float64).ravel()
    assert gravity.size >= griddim, "gravity vector shorter than griddim"
    return gravity[:griddim]


if __name__ == "__main__":
    def test_gravity_vector():
        """Vector points down the last axis with standard magnitude."""
        assert np.allclose(gravity_vector(3), [0.0, 0.0, 9.80665])
        assert np.allclose(gravity_vector(2), [0.0, 9.80665])
        print("ok test_gravity_vector")

    def test_gravity_or_zero():
        """None disables gravity; longer vectors are truncated to griddim."""
        assert np.allclose(gravity_or_zero(None, 3), 0.0)
        assert np.allclose(gravity_or_zero([0, 0, 9.81], 2), [0.0, 0.0])
        assert np.allclose(gravity_or_zero(gravity_vector(3), 3)[2], 9.80665)
        print("ok test_gravity_or_zero")

    test_gravity_vector()
    test_gravity_or_zero()
    print("\nAll tests passed!")
