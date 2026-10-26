"""MRST ``diagnostics/utils/computeLorenz.m``: the Lorenz coefficient, twice
the area between the F-Phi curve and the diagonal - 0 for a homogeneous
displacement, towards 1 for a strongly heterogeneous one. With one curve
per column, one coefficient per column.

`lorenz_coefficient`, below, is PyMRST's earlier single-curve version, kept
for its callers.
"""

from typing import Union

import numpy as np


def compute_lorenz(F: np.ndarray, Phi: np.ndarray) -> Union[float, np.ndarray]:
    """MRST `Lc = computeLorenz(F, Phi)`."""
    F, Phi = np.asarray(F, dtype=float), np.asarray(Phi, dtype=float)
    v = np.diff(Phi, axis=0)
    Lc = 2*(np.sum((F[:-1] + F[1:])/2*v, axis=0) - 0.5)
    return float(Lc) if np.ndim(Lc) == 0 else Lc


computeLorenz = compute_lorenz


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

def lorenz_coefficient(f: np.ndarray, phi: np.ndarray) -> float:
    """Return Lorenz coefficient from flow/storage capacity curves."""
    dv = np.diff(phi)
    return float(2.0 * (np.sum(0.5 * (f[:-1] + f[1:]) * dv) - 0.5))


__all__ = ["computeLorenz", "compute_lorenz", "lorenz_coefficient"]


if __name__ == "__main__":
    def test_the_diagonal_has_no_heterogeneity():
        x = np.linspace(0, 1, 11)
        assert abs(compute_lorenz(x, x)) < 1e-15
        print("ok test_the_diagonal_has_no_heterogeneity")

    def test_a_step_curve_is_fully_heterogeneous():
        assert np.isclose(compute_lorenz(np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0])), 1.0)
        print("ok test_a_step_curve_is_fully_heterogeneous")

    def test_one_coefficient_per_column():
        x = np.linspace(0, 1, 11)
        Lc = compute_lorenz(np.column_stack([x, np.sqrt(x)]), np.column_stack([x, x]))
        assert Lc.shape == (2,) and abs(Lc[0]) < 1e-15 and Lc[1] > 0
        assert np.isclose(Lc[1], lorenz_coefficient(np.sqrt(x), x))
        print("ok test_one_coefficient_per_column")

    test_the_diagonal_has_no_heterogeneity()
    test_a_step_curve_is_fully_heterogeneous()
    test_one_coefficient_per_column()
    print("\nAll tests passed!")
