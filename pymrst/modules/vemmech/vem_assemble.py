"""MRST ``VEM_assemble.m`` global VEM stiffness assembly."""

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.utils.c2d import c2d
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc
from pymrst.modules.vemmech.utils.calculate_qf import calculate_qf
from pymrst.modules.vemmech.vem import _cell_projections, _nlin


def vem_assemble(
    g: Grid,
    C: np.ndarray,
    alpha_scaling: float = 1.0,
) -> sp.csr_array:
    """Assemble the global VEM stiffness matrix for 2D/3D grids."""
    qc_all = calculate_qc(g)[0] if g.griddim == 3 else calculate_qf(g)[0]
    D_all = c2d(C)
    dim = g.griddim
    nlin = _nlin(g)

    rows, cols, vals = [], [], []
    for cell in range(g.cells.num):
        NC, NR, WC, WR, dofs = _cell_projections(g, cell, qc_all)
        PP = NR @ WR.T + NC @ WC.T
        D = D_all[cell].reshape(nlin, nlin)

        tr_d = np.trace(D)
        dnc = np.sum(NC * NC, axis=0)
        alpha = abs(tr_d * g.cells.volumes[cell] / dnc.sum()) * alpha_scaling

        I = np.eye(dofs.size)
        kh = (
            g.cells.volumes[cell] * (WC @ D @ WC.T)
            + alpha * (I - PP).T @ (I - PP)
        )

        rows.append(np.repeat(dofs, dofs.size))
        cols.append(np.tile(dofs, dofs.size))
        vals.append(kh.ravel())

    ndof = dim * g.nodes.num
    return sp.csr_array(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(ndof, ndof),
    )

__all__ = ["vem_assemble"]
