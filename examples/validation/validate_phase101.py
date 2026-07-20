"""Validate MPSA-W style cell-centered stress recovery."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_mechanics import (
    mpsaw_cell_gradient,
    mpsaw_recover,
    mpsaw_strain,
)


def _check_affine_2d():
    g = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    A = np.array([[2.0e-4, -1.0e-4], [3.0e-4, 4.0e-4]])
    b = np.array([0.02, -0.01])
    u = g.cells.centroids[:, :2] @ A.T + b
    grad = mpsaw_cell_gradient(g, u)
    assert np.allclose(grad, A[None, :, :], atol=1e-14)
    strain = mpsaw_strain(grad)
    assert np.allclose(strain[:, 0], A[0, 0])
    assert np.allclose(strain[:, 1], A[1, 1])
    assert np.allclose(strain[:, 2], A[0, 1] + A[1, 0])
    out = mpsaw_recover(g, u, young=1.0e9, poisson=0.25)
    assert out.face_tractions.shape == (g.faces.num, 2)
    assert out.residual.shape == (g.cells.num, 2)
    assert np.all(np.isfinite(out.stress))
    print("  ok 2D affine displacement gradient/stress recovery")


def _check_rigid_rotation_zero_stress():
    g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    x = g.cells.centroids[:, :2]
    omega = 1.0e-3
    u = np.column_stack([-omega * x[:, 1], omega * x[:, 0]])
    out = mpsaw_recover(g, u, young=1.0e9, poisson=0.25)
    assert np.max(np.abs(out.strain)) < 1e-14
    assert np.max(np.abs(out.stress)) < 1e-5
    assert np.max(np.abs(out.residual)) < 1e-5
    print("  ok rigid rotation gives zero strain/stress/residual")


def _check_affine_3d():
    g = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
    A = np.array([
        [1.0e-4, 2.0e-4, -1.0e-4],
        [0.5e-4, -2.0e-4, 0.7e-4],
        [-0.3e-4, 0.9e-4, 1.5e-4],
    ])
    u = g.cells.centroids[:, :3] @ A.T
    grad = mpsaw_cell_gradient(g, u)
    assert np.allclose(grad, A[None, :, :], atol=1e-14)
    out = mpsaw_recover(g, u, young=1.0e9, poisson=0.25)
    assert out.strain.shape == (g.cells.num, 6)
    assert out.face_tractions.shape == (g.faces.num, 3)
    assert np.all(np.isfinite(out.residual))
    print("  ok 3D affine displacement gradient/stress recovery")


def main():
    print("Phase 101: MPSA-W style stress approximation utilities")
    _check_affine_2d()
    _check_rigid_rotation_zero_stress()
    _check_affine_3d()
    print("Phase 101 validation passed: MPSA-W recovery invariants hold.")


if __name__ == "__main__":
    main()
