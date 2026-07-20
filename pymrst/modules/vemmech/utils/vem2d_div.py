"""MRST ``VEM2D_div.m`` discrete 2D VEM divergence."""

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.utils.calculate_qf import calculate_qf


def vem2d_div(g: Grid) -> sp.csr_array:
    """Map 2D node displacements to volume-integrated cell divergence."""
    qf, _ = calculate_qf(g)
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.nodePos))
    nodes = g.cells.nodes
    dofs = 2 * nodes[:, None] + np.arange(2)
    return sp.csr_array(
        (qf.ravel(), (np.repeat(cell_no, 2), dofs.ravel())),
        shape=(g.cells.num, 2 * g.nodes.num),
    )

__all__ = ["vem2d_div"]
