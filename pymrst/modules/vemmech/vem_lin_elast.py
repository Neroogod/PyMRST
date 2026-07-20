"""MRST ``VEM_linElast.m`` linear-elastic VEM solve."""

from typing import Callable, Optional

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc
from pymrst.modules.vemmech.utils.calculate_qf import calculate_qf
from pymrst.modules.vemmech.utils.vem_div import vem_div
from pymrst.modules.vemmech.vem import ElasticBC
from pymrst.modules.vemmech.vem_assemble import vem_assemble
from pymrst.utils.gridtools import mcolon


def vem_lin_elast(
    g: Grid,
    C: np.ndarray,
    el_bc: ElasticBC,
    load: Callable[[np.ndarray], np.ndarray],
    pressure: Optional[np.ndarray] = None,
    alpha_scaling: float = 1.0,
):
    """Solve linear elasticity with VEM and return displacement plus operators."""
    dim = g.griddim
    S = vem_assemble(g, C, alpha_scaling=alpha_scaling)
    if dim == 3:
        qc_all, qf_all, _ = calculate_qc(g)
        q_cell, q_force = qc_all, qf_all
    else:
        q_cell, _ = calculate_qf(g)
        q_force = np.repeat(g.faces.areas / 2.0, np.diff(g.faces.nodePos))

    bcnodes, first_ix = np.unique(el_bc.disp_nodes, return_index=True)
    u_bc = el_bc.disp_vals[first_ix]
    mask = np.zeros((bcnodes.size, dim), dtype=bool)
    inv = np.searchsorted(bcnodes, el_bc.disp_nodes)
    np.logical_or.at(mask, inv, el_bc.disp_mask.astype(bool))

    ndof = dim * g.nodes.num
    dirdofs = (dim * bcnodes[:, None] + np.arange(dim))[mask]
    u_dir = u_bc[mask]
    isdir = np.zeros(ndof, dtype=bool)
    isdir[dirdofs] = True

    v_dir = np.zeros(ndof)
    v_dir[dirdofs] = u_dir
    rhs = -S @ v_dir

    cells_nodes = g.cells.nodes
    nlc = np.diff(g.cells.nodePos)
    xc = np.repeat(g.cells.centroids[:, :dim], nlc, axis=0)
    rel_vec = -(xc - g.nodes.coords[cells_nodes, :dim])
    ll = load(xc) * (q_cell[:, :dim] * rel_vec)
    dofs = (dim * cells_nodes[:, None] + np.arange(dim)).ravel()
    rhs += np.bincount(dofs, weights=ll.ravel(), minlength=ndof)

    div = vem_div(g)
    if pressure is not None:
        rhs += div.T @ pressure

    if el_bc.force_faces is not None and el_bc.force_faces.size > 0:
        faces = el_bc.force_faces
        assert np.all(np.sum(g.faces.neighbors[faces] == -1, axis=1) == 1)
        lnn = np.diff(g.faces.nodePos)[faces]
        inodes = mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
        nodes = g.faces.nodes[inodes]
        force = np.repeat(el_bc.force[:, :dim], lnn, axis=0) * q_force[inodes][:, None]
        fdofs = (dim * nodes[:, None] + np.arange(dim)).ravel()
        rhs += np.bincount(fdofs, weights=force.ravel(), minlength=ndof)

    free = ~isdir
    A = S[free][:, free]
    A = (A + A.T) / 2.0
    x = spla.spsolve(A.tocsc(), rhs[free])

    u = np.zeros(ndof)
    u[isdir] = v_dir[isdir]
    u[free] = x
    uu = u.reshape(-1, dim)

    extra = {
        "S": S,
        "A": A,
        "isdirdofs": isdir,
        "V_dir": v_dir,
        "div": div,
        "rhs": rhs[free],
    }
    return uu, extra

__all__ = ["vem_lin_elast"]
