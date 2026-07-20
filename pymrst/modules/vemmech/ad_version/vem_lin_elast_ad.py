"""MRST ``VEM_linElast_AD.m`` AD-version linear-elastic solve."""

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.modules.vemmech.ad_version import (
    _bc_vector_from_tensor,
    _force_bc_vector,
    _load_vector_dual_grad,
    compute_ax_derivs,
)
from pymrst.modules.vemmech.ad_version.vem_assemble_ad import vem_assemble_ad
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc
from pymrst.modules.vemmech.utils.calculate_qf import calculate_qf
from pymrst.modules.vemmech.utils.vem_div import vem_div
from pymrst.utils.adi import ADI


def vem_lin_elast_ad(
    g,
    E,
    nu,
    el_bc,
    load,
    linsolve=None,
    alpha_scaling: float = 1.0,
    pressure=None,
    extra=None,
    background_forces=None,
    no_solve: bool = False,
):
    """Numeric subset of MRST ``VEM_linElast_AD``."""
    dim = g.griddim
    S, op = vem_assemble_ad(g, E, nu, extra=extra,
                            alpha_scaling=alpha_scaling)
    if dim == 2:
        qc_all, _ = calculate_qf(g)
        qf_all = g.faces.areas / 2.0
    else:
        qc_all, qf_all, _ = calculate_qc(g)

    u_bc, isdir = _bc_vector_from_tensor(g, el_bc.disp_bc)
    rhs = -(S @ u_bc)
    rhs += _load_vector_dual_grad(g, load, qc_all)
    if pressure is not None and np.asarray(pressure).size:
        rhs += vem_div(g).T @ np.asarray(pressure, dtype=float).ravel()
    if background_forces is not None and np.asarray(background_forces).size:
        rhs += np.asarray(background_forces, dtype=float).ravel()
    if getattr(el_bc, "force_bc", None) is not None:
        rhs += _force_bc_vector(g, el_bc.force_bc, qf_all)

    free = ~isdir
    A = S[free][:, free]
    A = (A + A.T) / 2.0
    red_rhs = rhs[free]
    if no_solve:
        x = np.full(np.count_nonzero(free), np.nan)
    else:
        solver = linsolve or (lambda AA, bb: spla.spsolve(AA.tocsc(), bb))
        x = solver(A, red_rhs)

    u = np.full(dim * g.nodes.num, np.nan)
    u[isdir] = u_bc[isdir]
    u[free] = x
    uu = u.reshape(g.nodes.num, dim)
    vdiv = vem_div(g)
    op.A = A
    op.S = S
    op.rhs = red_rhs
    if isinstance(E, ADI) or isinstance(nu, ADI):
        op.Ax_derivs = compute_ax_derivs(g, u, op, E, nu, alpha_scaling, isdir)
    else:
        op.Ax_derivs = 0
    op.disc = {
        "isdirdofs": isdir,
        "rhs": red_rhs,
        "V_dir": u_bc,
        "ovol_div": vdiv,
        "gradP": vdiv[:, free].T,
        "div": vdiv[:, free],
        "divrhs": vdiv @ u_bc,
    }
    return uu, op

__all__ = ["vem_lin_elast_ad"]
