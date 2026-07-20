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
