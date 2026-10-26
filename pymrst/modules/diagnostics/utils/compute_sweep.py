"""MRST ``diagnostics/utils/computeSweep.m``: volumetric sweep efficiency `Ev`
against dimensionless time `tD` (pore volumes injected) from an F-Phi curve.
Points where `F` does not increase by more than sqrt(eps) are dropped first.
As in MATLAB, a matrix input is read in column-major (linear) order and the
result is a single curve.

`compute_sweep_ad`, below, is PyMRST's earlier single-curve version, kept
for its callers.
"""

from typing import Tuple

import numpy as np


def compute_sweep(F: np.ndarray, Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[Ev, tD] = computeSweep(F, Phi)`."""
    F = np.asarray(F, dtype=float).ravel(order="F")
    Phi = np.asarray(Phi, dtype=float).ravel(order="F")
    hit = np.ones(F.size, dtype=bool)
    hit[1:] = F[:-1] <= F[1:] - np.sqrt(np.finfo(float).eps)
    F, Phi = F[hit], Phi[hit]
    with np.errstate(divide="ignore", invalid="ignore"):
        tD = np.concatenate([[0.0], (Phi[1:] - Phi[:-1])/(F[1:] - F[:-1])])
    Ev = Phi + (1 - F)*tD
    return Ev, tD


computeSweep = compute_sweep


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

def compute_sweep_ad(f: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep efficiency ``Ev`` against dimensionless time ``tD``."""
    keep = np.ones(f.size, dtype=bool)
    keep[1:] = f[:-1] <= f[1:] - np.sqrt(np.finfo(float).eps)
    f, phi = f[keep], phi[keep]
    td = np.concatenate([[0.0], np.diff(phi) / np.diff(f)])
    ev = phi + (1.0 - f) * td
    return ev, td


__all__ = ["computeSweep", "compute_sweep", "compute_sweep_ad"]


if __name__ == "__main__":
    def test_homogeneous_sweep_is_complete_at_one_pore_volume():
        x = np.linspace(0, 1, 11)
        Ev, tD = compute_sweep(x, x)
        assert np.allclose(tD[1:], 1.0) and np.allclose(Ev[1:], 1.0)
        print("ok test_homogeneous_sweep_is_complete_at_one_pore_volume")

    def test_flat_stretches_of_f_are_dropped():
        F = np.array([0.0, 0.5, 0.5, 1.0])
        Phi = np.array([0.0, 0.2, 0.4, 1.0])
        Ev, tD = compute_sweep(F, Phi)
        assert Ev.size == 3 and np.allclose(tD, [0.0, 0.4, 1.6])
        print("ok test_flat_stretches_of_f_are_dropped")

    def test_a_matrix_is_read_column_major():
        """MATLAB's `F(1:end-1) <= F(2:end)` on a matrix runs down the columns in
        turn: only the step from the end of column 1 back to 0 is dropped."""
        x = np.linspace(0, 1, 5)
        Ev, tD = compute_sweep(np.column_stack([x, x]), np.column_stack([x, x]))
        assert Ev.size == 9 and np.allclose(Ev[:5], compute_sweep_ad(x, x)[0])
        print("ok test_a_matrix_is_read_column_major")

    test_homogeneous_sweep_is_complete_at_one_pore_volume()
    test_flat_stretches_of_f_are_dropped()
    test_a_matrix_is_read_column_major()
    print("\nAll tests passed!")
