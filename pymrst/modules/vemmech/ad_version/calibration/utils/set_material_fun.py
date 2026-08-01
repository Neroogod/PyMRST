"""MRST ``set_material_fun.m`` calibration material-function builder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.vemmech.ad_version import SparseMultiArray
from pymrst.params.gravity_const import gravity_vector
from pymrst.utils.adi import ADI, ad_matmul, value


def _enu_to_kg(E, nu):
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 + 2.0 * nu)
    return K, G


def _kg_to_enu(K, G):
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return E, nu


def _interp_range(par, ranges):
    ranges = np.asarray(ranges, dtype=float)
    if isinstance(par, ADI):
        if len(par) == 0:
            return np.zeros(0)
        return (1.0 - par) * ranges[:, 0] + par * ranges[:, 1]
    par = np.asarray(par, dtype=float).ravel()
    if par.size == 0:
        return np.zeros(0)
    return (1.0 - par) * ranges[:, 0] + par * ranges[:, 1]


def _distribute_layers(g, controls, layer_ixs):
    controls = np.asarray(controls, dtype=np.int64)
    layer_ixs = np.asarray(layer_ixs, dtype=np.int64)
    cell_layer = np.zeros(g.cells.num, dtype=np.int64)
    if controls.size == 0:
        return cell_layer
    layer_size = int(np.prod(g.cartDims[:2]))
    for local, ctrl in enumerate(controls, start=1):
        start_layer = layer_ixs[ctrl]
        end_layer = layer_ixs[ctrl + 1] - 1
        start = layer_size * (start_layer - 1)
        end = layer_size * end_layer
        cell_layer[start:end] = local
    return cell_layer[np.asarray(g.cells.indexMap, dtype=np.int64)]


def _control_matrix(cell_controls, nctrl):
    mat = np.zeros((cell_controls.size, nctrl))
    keep = cell_controls != 0
    if nctrl:
        mat[np.flatnonzero(keep), cell_controls[keep] - 1] = 1.0
    return mat


def _cell_node_load_tensor(g, rho):
    nlc = np.diff(g.cells.nodePos)
    rho_cn = np.repeat(np.asarray(value(rho), dtype=float).ravel(), nlc)
    return SparseMultiArray(rho_cn, "cn") * SparseMultiArray(
        gravity_vector(g.griddim), "d"
    )


def _cell_node_load_jac(g, rho_jac):
    from pymrst.modules.vemmech.vem import calculate_qc, calculate_qf
    from pymrst.utils.mcolon import mcolon

    rho_jac = np.asarray(rho_jac, dtype=float)
    if rho_jac.ndim == 1:
        rho_jac = rho_jac[:, None]
    if rho_jac.size == 0:
        return sp.csr_array((g.griddim * g.nodes.num, rho_jac.shape[1]))
    qc_all = calculate_qf(g)[0] if g.griddim == 2 else calculate_qc(g)[0]
    cells = np.arange(g.cells.num)
    inodes = mcolon(g.cells.nodePos[cells], g.cells.nodePos[cells + 1])
    nodes = g.cells.nodes[inodes]
    nlc = np.diff(g.cells.nodePos)
    X = np.repeat(g.cells.centroids[cells, :g.griddim], nlc, axis=0)
    rel_vec = -(X - g.nodes.coords[nodes, :g.griddim])
    weights = qc_all[:, :g.griddim] * rel_vec * gravity_vector(g.griddim)
    cell_no = np.repeat(cells, nlc)
    local = rho_jac[cell_no, :]
    ndof = g.griddim * g.nodes.num
    out = np.zeros((ndof, rho_jac.shape[1]))
    dofs = (g.griddim * nodes[:, None] + np.arange(g.griddim)).ravel()
    vals = (weights[:, :, None] * local[:, None, :]).reshape(-1, rho_jac.shape[1])
    for k in range(rho_jac.shape[1]):
        out[:, k] = np.bincount(dofs, weights=vals[:, k], minlength=ndof)
    return sp.csr_array(out)


def set_material_fun(g, props, matcond=None, u_offset: int = 0):
    """Return calibration material/load functions and control counts."""
    young = np.asarray(props.young, dtype=float).ravel()
    poisson = np.asarray(props.poisson, dtype=float).ravel()
    rho = np.asarray(props.rho, dtype=float).ravel()
    if matcond is None or np.asarray(matcond).size == 0:
        funs = SimpleNamespace(
            efun=lambda u: young,
            nufun=lambda u: poisson,
            loadfun=lambda u: _cell_node_load_tensor(g, rho),
            load_jac=lambda u: sp.csr_array((g.griddim * g.nodes.num,
                                             np.asarray(u, dtype=float).size)),
            num_ctrls=np.array([0, 0, 0], dtype=np.int64),
        )
        return funs, np.array([0, 0, 0], dtype=np.int64)

    matcond = np.asarray(matcond, dtype=float)
    K_ctrls = np.flatnonzero(matcond[:, 1] > 0)
    G_ctrls = np.flatnonzero(matcond[:, 2] > 0)
    R_ctrls = np.flatnonzero(matcond[:, 3] > 0)
    num_ctrls = np.array([K_ctrls.size, G_ctrls.size, R_ctrls.size],
                         dtype=np.int64)
    layer_ixs = np.concatenate([
        matcond[:, 0].astype(np.int64),
        [int(g.cartDims[2]) + 1],
    ])
    cell_K = _distribute_layers(g, K_ctrls, layer_ixs)
    cell_G = _distribute_layers(g, G_ctrls, layer_ixs)
    cell_R = _distribute_layers(g, R_ctrls, layer_ixs)
    k_mat = _control_matrix(cell_K, K_ctrls.size)
    g_mat = _control_matrix(cell_G, G_ctrls.size)
    r_mat = _control_matrix(cell_R, R_ctrls.size)
    k_fixed = 1.0 - k_mat @ np.ones(K_ctrls.size)
    g_fixed = 1.0 - g_mat @ np.ones(G_ctrls.size)
    r_fixed = 1.0 - r_mat @ np.ones(R_ctrls.size)
    K0, G0 = _enu_to_kg(young, poisson)
    K_ranges = np.column_stack([np.ones(K_ctrls.size),
                                1.0 + matcond[K_ctrls, 1]])
    G_ranges = np.column_stack([np.ones(G_ctrls.size),
                                1.0 + matcond[G_ctrls, 2]])
    R_ranges = np.column_stack([np.ones(R_ctrls.size),
                                1.0 + matcond[R_ctrls, 3]])

    def _matvec(mat, x):
        return ad_matmul(sp.csr_array(mat), x) if isinstance(x, ADI) else mat @ x

    def _split(u):
        if isinstance(u, ADI):
            uu = u[u_offset:]
        else:
            uu = np.asarray(u, dtype=float).ravel()[u_offset:]
        nk, ng, nr = num_ctrls
        return uu[:nk], uu[nk:nk + ng], uu[nk + ng:nk + ng + nr]

    def _factors(u):
        uk, ug, ur = _split(u)
        kfac = _matvec(k_mat, _interp_range(uk, K_ranges)) + k_fixed
        gfac = _matvec(g_mat, _interp_range(ug, G_ranges)) + g_fixed
        rfac = _matvec(r_mat, _interp_range(ur, R_ranges)) + r_fixed
        return kfac, gfac, rfac

    def efun(u):
        kfac, gfac, _ = _factors(u)
        E, _ = _kg_to_enu(K0 * kfac, G0 * gfac)
        return E

    def nufun(u):
        kfac, gfac, _ = _factors(u)
        _, nu = _kg_to_enu(K0 * kfac, G0 * gfac)
        return nu

    def loadfun(u):
        _, _, rfac = _factors(u)
        return _cell_node_load_tensor(g, rho * rfac)

    def load_jac(u):
        uu = np.asarray(value(u), dtype=float).ravel()
        ncols = uu.size
        nr = int(num_ctrls[2])
        if nr == 0:
            return sp.csr_array((g.griddim * g.nodes.num, ncols))
        slopes = matcond[R_ctrls, 3]
        rho_local = rho[:, None] * (r_mat @ np.diag(slopes))
        full = np.zeros((g.cells.num, ncols))
        start = u_offset + int(num_ctrls[0] + num_ctrls[1])
        full[:, start:start + nr] = rho_local
        return _cell_node_load_jac(g, full)

    return SimpleNamespace(efun=efun, nufun=nufun, loadfun=loadfun,
                           load_jac=load_jac, num_ctrls=num_ctrls), num_ctrls


__all__ = ["set_material_fun"]
