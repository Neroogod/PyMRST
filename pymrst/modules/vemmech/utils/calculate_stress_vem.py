"""MRST ``calculateStressVEM.m`` stress and strain recovery."""

from typing import Optional

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.utils.c2d import c2d
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc
from pymrst.modules.vemmech.utils.calculate_qf import calculate_qf
from pymrst.modules.vemmech.vem import _cell_projections, _nlin


def vem_strain_operator(g: Grid) -> sp.csr_array:
    """Map node displacements to cell strain in Voigt form."""
    qc_all = calculate_qc(g)[0] if g.griddim == 3 else calculate_qf(g)[0]
    nlin = _nlin(g)
    dim = g.griddim
    rows, cols, vals = [], [], []
    for cell in range(g.cells.num):
        _, _, WC, _, dofs = _cell_projections(g, cell, qc_all)
        r = nlin * cell + np.arange(nlin)
        rows.append(np.repeat(r, dofs.size))
        cols.append(np.tile(dofs, nlin))
        vals.append(WC.T.ravel())
    return sp.csr_array(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(nlin * g.cells.num, dim * g.nodes.num),
    )


def vem_stress_operator(g: Grid, C: np.ndarray) -> sp.csr_array:
    """Map node displacements to cell effective stress in Voigt form."""
    nlin = 3 if np.asarray(C).shape[1] == 9 else 6
    D_all = c2d(C).reshape(-1, nlin, nlin)
    blocks = sp.block_diag([D_all[c] for c in range(g.cells.num)], format="csr")
    return (blocks @ vem_strain_operator(g)).tocsr()


def calculate_stress_vem(
    g: Grid,
    uu: np.ndarray,
    C: Optional[np.ndarray] = None,
    strain_op: Optional[sp.spmatrix] = None,
    stress_op: Optional[sp.spmatrix] = None,
    do_patch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover cell stress and strain from VEM node displacement."""
    if do_patch:
        raise NotImplementedError("MRST patchRecovery is not ported")
    dim = g.griddim
    nlin = _nlin(g)
    u = np.asarray(uu, dtype=float)
    if u.ndim == 2:
        u = u[:, :dim].reshape(-1)
    else:
        u = u.ravel()
    if strain_op is None:
        strain_op = vem_strain_operator(g)
    if stress_op is None:
        if C is None:
            raise ValueError("C is required when stress_op is not supplied")
        stress_op = vem_stress_operator(g, C)
    strain = np.asarray(strain_op @ u, dtype=float).reshape(g.cells.num, nlin)
    stress = np.asarray(stress_op @ u, dtype=float).reshape(g.cells.num, nlin)
    if g.griddim == 3:
        stress = stress[:, [0, 1, 2, 3, 5, 4]]
        stress[:, 3:] /= 2.0
        strain = strain[:, [0, 1, 2, 3, 5, 4]]
    else:
        stress[:, 2] /= 2.0
    return stress, strain

__all__ = ["calculate_stress_vem", "vem_strain_operator", "vem_stress_operator"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import create_augmented_grid

    def _grid(dims=(3, 2), extent=(3.0, 2.0)):
        # `calculate_qf`/`calculate_qc` need `cells.nodes`, which only the
        # augmented grid carries; a plain cart_grid asserts instead.
        return create_augmented_grid(
            compute_geometry(cart_grid(list(dims), list(extent))))

    def test_rigid_translation_produces_no_strain():
        """The first patch test, and exact: moving a body cannot strain it.

        A strain operator that leaks translation is wrong in a way no
        displacement plot reveals - the field looks right and every recovered
        stress carries a constant offset.
        """
        g = _grid()
        B = vem_strain_operator(g)
        strain = B @ np.tile([0.7, -0.3], g.nodes.num)
        worst = float(np.max(np.abs(strain)))
        assert worst == 0.0, worst
        print(f"ok test_rigid_translation_produces_no_strain (max |strain| {worst:.1e})")

    def test_infinitesimal_rotation_produces_no_strain():
        """The second patch test. A rotation is not a deformation.

        This is the one that catches a symmetrisation error: using the
        displacement gradient instead of its symmetric part gives a non-zero
        shear here while leaving the stretch case below perfect.
        """
        g = _grid()
        coords = np.asarray(g.nodes.coords, dtype=float)
        theta = 1e-3
        u = np.column_stack([-coords[:, 1] * theta, coords[:, 0] * theta]).ravel()
        worst = float(np.max(np.abs(vem_strain_operator(g) @ u)))
        assert worst == 0.0, worst
        print(f"ok test_infinitesimal_rotation_produces_no_strain "
              f"(max |strain| {worst:.1e})")

    def test_uniform_stretch_is_recovered_exactly():
        """A linear displacement field is in VEM's space, so this is exact."""
        g = _grid()
        coords = np.asarray(g.nodes.coords, dtype=float)
        a = 0.01
        u = np.column_stack([coords[:, 0] * a, np.zeros(g.nodes.num)]).ravel()
        strain = (vem_strain_operator(g) @ u).reshape(g.cells.num, -1)
        assert np.allclose(strain[:, 0], a, rtol=1e-12, atol=1e-15), strain[:, 0]
        assert np.allclose(strain[:, 1:], 0.0, atol=1e-15), strain[:, 1:]
        print(f"ok test_uniform_stretch_is_recovered_exactly "
              f"(exx = {strain[0, 0]:.6g} in every cell)")

    def test_shear_lands_in_the_shear_component_only():
        """Placement matters: Voigt order is (exx, eyy, gxy) in 2D."""
        g = _grid()
        coords = np.asarray(g.nodes.coords, dtype=float)
        a = 0.004
        u = np.column_stack([coords[:, 1] * a, np.zeros(g.nodes.num)]).ravel()
        strain = (vem_strain_operator(g) @ u).reshape(g.cells.num, -1)
        assert np.allclose(strain[:, :2], 0.0, atol=1e-14), strain[:, :2]
        assert np.all(np.abs(strain[:, 2]) > 1e-6), strain[:, 2]
        print(f"ok test_shear_lands_in_the_shear_component_only "
              f"(gxy = {strain[0, 2]:.6g})")

    def test_the_operator_has_the_declared_shape():
        g = _grid()
        B = vem_strain_operator(g)
        assert B.shape == (3 * g.cells.num, 2 * g.nodes.num), B.shape
        print(f"ok test_the_operator_has_the_declared_shape {B.shape}")

    def test_patch_recovery_is_refused_rather_than_faked():
        g = _grid()
        try:
            calculate_stress_vem(g, np.zeros(2 * g.nodes.num), do_patch=True)
        except NotImplementedError as error:
            assert "patchRecovery" in str(error), error
        else:
            raise AssertionError("expected patch recovery to be refused")
        print("ok test_patch_recovery_is_refused_rather_than_faked")

    test_rigid_translation_produces_no_strain()
    test_infinitesimal_rotation_produces_no_strain()
    test_uniform_stretch_is_recovered_exactly()
    test_shear_lands_in_the_shear_component_only()
    test_the_operator_has_the_declared_shape()
    test_patch_recovery_is_refused_rather_than_faked()
    print("\nAll tests passed!")
