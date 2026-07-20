"""MRST ``VEM3D_div.m`` discrete 3D VEM divergence."""

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc
from pymrst.utils.gridtools import mcolon


def vem3d_div(g: Grid) -> sp.csr_array:
    """Map 3D node displacements to volume-integrated cell divergence."""
    faces = g.cells.faces
    cell_no = g.cell_no()
    nfl = np.diff(g.faces.nodePos)[faces]
    inodes = mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
    nodes = g.faces.nodes[inodes]
    sgn = np.where(g.faces.neighbors[faces, 0] == cell_no, 1.0, -1.0)
    nvec = g.faces.normals[faces] * (sgn / g.faces.areas[faces])[:, None]
    _, qf, _ = calculate_qc(g)
    matel = np.repeat(nvec, nfl, axis=0) * qf[inodes][:, None]
    jind = 3 * np.repeat(nodes, 3) + np.tile(np.arange(3), nodes.size)
    iind = np.repeat(np.repeat(cell_no, nfl), 3)
    return sp.csr_array(
        (matel.ravel(), (iind, jind)),
        shape=(g.cells.num, 3 * g.nodes.num),
    )

__all__ = ["vem3d_div"]
