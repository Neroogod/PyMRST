"""Calibration helpers from MRST ``vemmech/ad_version/calibration``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.vemmech.ad_version import (
    SparseMultiArray,
    compute_ax_derivs,
    vem_lin_elast_ad,
    vem_reduced_system_residual,
)
from pymrst.modules.vemmech.vem import (
    calculate_stress_vem,
    enu2c,
    enu2c_2d,
    vem_stress_operator,
)
from pymrst.params.gravity_const import gravity_vector
from pymrst.utils.adi import (
    ADI,
    ADI2,
    ad2_atan2,
    ad2_cos,
    ad2_linsolve,
    ad2_matmul,
    ad2_scatter,
    ad2_sin,
    ad2_sum,
    ad2_vertcat,
    ad_matmul,
    init_variables_adi2,
    value,
)


def sigma_to_smax_smin_theta(o_xx, o_yy, o_xy):
    """MRST ``Sigma2SmaxSminTheta`` for compressive stress convention."""
    o_xx = np.asarray(o_xx, dtype=float)
    o_yy = np.asarray(o_yy, dtype=float)
    o_xy = np.asarray(o_xy, dtype=float)
    sgn = -1.0
    s = o_xx + o_yy
    d = o_xx - o_yy
    g = 2.0 * o_xy
    a = np.sqrt(d * d + g * g)
    Smax = 0.5 * (s + sgn * a)
    Smin = 0.5 * (s - sgn * a)

    d = sgn * d
    g = sgn * g
    acos_ix = np.abs(d) < np.abs(g)
    theta2 = np.zeros_like(Smax, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp = np.arccos(np.clip(d[acos_ix] / a[acos_ix], -1.0, 1.0))
    tmp_g = g[acos_ix]
    loc = np.flatnonzero(acos_ix)
    theta2[loc[tmp_g >= 0]] = tmp[tmp_g >= 0]
    theta2[loc[tmp_g < 0]] = -tmp[tmp_g < 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        tmp = np.arcsin(np.clip(g[~acos_ix] / a[~acos_ix], -1.0, 1.0))
    tmp = np.where(np.isnan(tmp), 0.0, tmp)
    tmp_d = d[~acos_ix]
    loc = np.flatnonzero(~acos_ix)
    theta2[loc[tmp_d >= 0]] = tmp[tmp_d >= 0]
    theta2[loc[tmp_d < 0]] = np.pi - tmp[tmp_d < 0]
    theta2 = np.where(theta2 < 0, theta2 + 2.0 * np.pi, theta2)
    return Smax, Smin, theta2 / 2.0


@dataclass
class BoundaryInfo:
    side_faces: list[np.ndarray]
    side_nodes: list[np.ndarray]


@dataclass
class ElasticCalibrationBC:
    disp_bc: SparseMultiArray
    force_bc: None = None
    disp_bc_jac: sp.csr_array | None = None


@dataclass
class CalibrationOptions:
    ignore_prior: bool = False
    ignore_bnd_prior: bool = False
    scaling: float | None = None
    produce_syndata_for: np.ndarray | None = None
    grad_tol: float = 1e-6
    obj_change_tol: float = 1e-7
    max_it: int = 100
    line_search_max_it: int = 8
    u_start: np.ndarray | None = None
    model_sigma: np.ndarray | None = None
    alpha_p: np.ndarray | None = None
    bounds: tuple[float, float] | None = (0.0, 1.0)
    optimizer: str = "sr1"
    delta: float = 0.1
    eta: float = 1e-3
    sr1_update_tol: float = 1e-3
    b_scale: float = 1.0
    hi: np.ndarray | None = None
    backup_file: str | None = None
    resume_backup: bool = True
    progress_callback: Callable | None = None


@dataclass
class OptimizationHistory:
    values: list[float]
    controls: list[np.ndarray]
    grad_norms: list[float]
    step_lengths: list[float]
    status: str
    gradients: list[np.ndarray] | None = None
    trust_radii: list[float] | None = None
    hessians: list[np.ndarray] | None = None
    iterations: list[int] | None = None
    ratios: list[float] | None = None


def identify_boundary_elements(g, tol: float = 1e-9) -> BoundaryInfo | None:
    """Identify xmin/xmax/ymin/ymax/zmin/zmax boundary faces and nodes."""
    coords = g.faces.centroids[:, :g.griddim]
    sides = []
    for ax in range(g.griddim):
        sides.append((ax, np.min(coords[:, ax])))
        sides.append((ax, np.max(coords[:, ax])))
    if g.griddim == 2:
        sides.append((None, None))
        sides.append((None, None))

    side_faces: list[np.ndarray] = []
    side_nodes: list[np.ndarray] = []
    boundary = np.any(g.faces.neighbors < 0, axis=1)
    for ax, val in sides[:6]:
        if ax is None:
            side_faces.append(np.zeros(0, dtype=np.int64))
            side_nodes.append(np.zeros(0, dtype=np.int64))
            continue
        keep = boundary & (np.abs(coords[:, ax] - val) <= tol)
        faces = np.flatnonzero(keep)
        if faces.size == 0:
            return None
        nodes = np.unique(np.concatenate([
            g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
            for f in faces
        ]))
        side_faces.append(faces)
        side_nodes.append(nodes)
    return BoundaryInfo(side_faces=side_faces, side_nodes=side_nodes)


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
    from pymrst.utils.gridtools import mcolon

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


def _as_1based_nodes(nodes):
    return np.asarray(nodes, dtype=np.int64).ravel() + 1


def _node_vector_tensor(values, nodes, name="n"):
    nodes = _as_1based_nodes(nodes)
    if nodes.size == 0:
        return SparseMultiArray([], np.zeros((0, 1), dtype=np.int64), [name],
                                keep_zeros=True)
    return SparseMultiArray(np.asarray(values, dtype=float), nodes[:, None],
                            [name], keep_zeros=True)


def _theta_e1_e2_grad(g, prior, sigma, zdisp=np.nan):
    prior = np.asarray(prior, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()

    def ipol(t, vmin, vmax):
        return vmin * (1.0 - t) + vmax * t

    def varval(u, ix):
        return prior[ix] + ipol(u[ix], 0.0, sigma[ix])

    def side_bc(n_ix, u):
        n_ix = np.asarray(n_ix, dtype=np.int64).ravel()
        if n_ix.size == 0:
            return SparseMultiArray([], np.zeros((0, 2), dtype=np.int64),
                                    ["d", "n"], keep_zeros=True)
        u = np.asarray(u, dtype=float).ravel()
        theta = varval(u, 0)
        e1 = varval(u, 1)
        e2 = varval(u, 2)
        de1dz = varval(u, 3)
        de2dz = varval(u, 4)

        R = SparseMultiArray(
            [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta), 1.0],
            [[1, 1], [1, 2], [2, 1], [2, 2], [3, 3]],
            ["d", "tmp1"],
            keep_zeros=True,
        )
        R2 = SparseMultiArray(
            [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta), 1.0],
            [[1, 1], [1, 2], [2, 1], [2, 2], [3, 3]],
            ["d2", "tmp2"],
            keep_zeros=True,
        )
        E0 = SparseMultiArray([e1, e2, 1.0],
                              [[1, 1], [2, 2], [3, 3]],
                              ["tmp1", "tmp2"], keep_zeros=True)
        E1 = SparseMultiArray([de1dz, de2dz, 1.0],
                              [[1, 1], [2, 2], [3, 3]],
                              ["tmp1", "tmp2"], keep_zeros=True)
        M0 = R * E0 * R2
        M1 = R * E1 * R2

        ncoords = np.asarray(g.nodes.coords[n_ix, :3], dtype=float).copy()
        ncoords[:, 2] = zdisp
        local = np.arange(1, n_ix.size + 1, dtype=np.int64)
        nodes = n_ix + 1
        coords0 = SparseMultiArray(ncoords, ["n_loc", "d2"],
                                   keep_zeros=True)
        map_n = SparseMultiArray(None, np.column_stack([local, nodes]),
                                 ["n_loc", "n"], keep_zeros=True)
        map_z = SparseMultiArray(None, np.column_stack([local, nodes]),
                                 ["n_loc", "z"], keep_zeros=True)
        coords1 = SparseMultiArray(g.nodes.coords[n_ix, 2],
                                   np.column_stack([nodes, nodes]),
                                   ["z", "n"], keep_zeros=True)
        return (M0 * coords0 * map_n) + (M1 * coords0 * map_z * coords1)

    return side_bc


def _theta_dense_and_jac(g, nodes, pars, sigma, active_u, zdisp, nctrl):
    nodes = np.asarray(nodes, dtype=np.int64).ravel()
    vals = np.zeros((nodes.size, g.griddim))
    jac = sp.csr_array((g.griddim * g.nodes.num, nctrl))
    if nodes.size == 0:
        return vals, jac

    theta, e1, e2, de1dz, de2dz = np.asarray(pars, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    dR = np.array([[-s, -c], [c, -s]])
    D0 = np.diag([e1, e2])
    D1 = np.diag([de1dz, de2dz])
    M0 = R @ D0 @ R.T
    M1 = R @ D1 @ R.T

    xy = g.nodes.coords[nodes, :2]
    z = g.nodes.coords[nodes, 2] if g.griddim == 3 else np.zeros(nodes.size)
    vals[:, :2] = xy @ M0.T + (xy @ M1.T) * z[:, None]
    if g.griddim == 3:
        vals[:, 2] = zdisp * (1.0 + z)

    dM0 = [
        dR @ D0 @ R.T + R @ D0 @ dR.T,
        R @ np.diag([1.0, 0.0]) @ R.T,
        R @ np.diag([0.0, 1.0]) @ R.T,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    ]
    dM1 = [
        dR @ D1 @ R.T + R @ D1 @ dR.T,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        R @ np.diag([1.0, 0.0]) @ R.T,
        R @ np.diag([0.0, 1.0]) @ R.T,
    ]

    rows, cols, data = [], [], []
    for local_col, par_ix in enumerate(active_u):
        scale = sigma[par_ix]
        if scale == 0:
            continue
        deriv_xy = (xy @ dM0[par_ix].T + (xy @ dM1[par_ix].T) * z[:, None]) * scale
        dofs = g.griddim * nodes[:, None] + np.arange(2)
        rows.append(dofs.ravel())
        cols.append(np.full(2 * nodes.size, local_col, dtype=np.int64))
        data.append(deriv_xy.ravel())
    if rows:
        jac = sp.csr_array(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(g.griddim * g.nodes.num, nctrl),
        )
    return vals, jac


def set_bcfun(g, binfo: BoundaryInfo, bcond):
    """Numeric counterpart of MRST ``set_bcfun``.

    ``bcond`` has rows ``theta, e1, e2`` and optionally ``de1dz, de2dz``;
    columns are prior and control half-width. The returned callable maps a
    control vector to an ``ElasticCalibrationBC`` with a named displacement
    tensor.
    """
    bcond = np.asarray(bcond, dtype=float)
    if bcond.shape[0] not in (3, 5) or bcond.shape[1] != 2:
        raise ValueError("bcond must have shape (3, 2) or (5, 2)")
    if bcond.shape[0] == 3:
        bcond = np.vstack([bcond, np.zeros((2, 2))])

    bottom_nodes = np.asarray(binfo.side_nodes[5], dtype=np.int64)
    side_nodes = np.unique(np.concatenate([
        np.asarray(x, dtype=np.int64) for x in binfo.side_nodes[:4]
    ]))
    bottom_edge_nodes = np.intersect1d(bottom_nodes, side_nodes)
    side_nodes = np.setdiff1d(side_nodes, bottom_edge_nodes)
    bottom_nodes = np.setdiff1d(bottom_nodes, bottom_edge_nodes)

    active_u = np.flatnonzero(bcond[:, 1] > 0)
    num_bc_ctrls = active_u.size

    def full_u(u):
        out = np.zeros(5, dtype=float)
        uu = np.asarray(u, dtype=float).ravel()
        out[active_u] = uu[:num_bc_ctrls]
        return out

    bottom_1 = _as_1based_nodes(bottom_nodes)
    roller_bottom = SparseMultiArray(
        0.0, bottom_1[:, None], ["n"], keep_zeros=True
    ) * SparseMultiArray([np.nan, np.nan, 1.0], "d", keep_zeros=True)

    disp_bc_fun = _theta_e1_e2_grad(g, bcond[:, 0], bcond[:, 1])
    disp_bc_fun_zlock = _theta_e1_e2_grad(g, bcond[:, 0], bcond[:, 1],
                                          zdisp=0.0)

    def bcfun(u):
        uu = full_u(u)
        disp_bc = (
            roller_bottom
            + disp_bc_fun(side_nodes, uu)
            + disp_bc_fun_zlock(bottom_edge_nodes, uu)
        )
        pars = bcond[:, 0] + bcond[:, 1] * uu
        _, side_jac = _theta_dense_and_jac(
            g, side_nodes, pars, bcond[:, 1], active_u, np.nan, num_bc_ctrls
        )
        _, edge_jac = _theta_dense_and_jac(
            g, bottom_edge_nodes, pars, bcond[:, 1], active_u, 0.0,
            num_bc_ctrls
        )
        return ElasticCalibrationBC(
            disp_bc=disp_bc,
            force_bc=None,
            disp_bc_jac=(side_jac + edge_jac).tocsr(),
        )

    bcfun._pymrst_calibration_bc_data = SimpleNamespace(
        bcond=bcond.copy(),
        active_u=active_u.copy(),
        num_bc_ctrls=num_bc_ctrls,
        side_nodes=side_nodes.copy(),
        bottom_nodes=bottom_nodes.copy(),
        bottom_edge_nodes=bottom_edge_nodes.copy(),
    )
    return bcfun, num_bc_ctrls


def _dirichlet_values_from_bc(g, bc: ElasticCalibrationBC):
    dim = g.griddim
    names = bc.disp_bc.index_names()
    d_col = names.index("d")
    n_col = names.index("n")
    ixs = np.asarray(bc.disp_bc.ixs, dtype=np.int64)
    vals = np.asarray(bc.disp_bc.coefs, dtype=float)
    dofs = (ixs[:, d_col] - 1) + dim * (ixs[:, n_col] - 1)
    keep = ~np.isnan(vals)
    return dofs[keep], vals[keep]


def calculate_stress(u, dd, g, bcfun, efun, nufun, op=None):
    """Numeric calibration counterpart of MRST ``calculate_stress``.

    The MRST function builds AD dependencies with respect to calibration
    controls and mechanical unknowns. This Python helper returns the numeric
    stress value using the same displacement replacement and stress convention:
    imposed displacement BC entries are taken from ``bcfun(u)``, the full
    displacement vector is pushed through the VEM stress operator, shear terms
    are halved, and the 3D xz/yz components are swapped to match
    ``calculateStressVEM``.
    """
    dim = g.griddim
    uu = np.asarray(dd, dtype=float)
    if uu.ndim == 2:
        full = uu[:, :dim].reshape(-1).copy()
    else:
        full = uu.ravel().copy()
    if full.size != dim * g.nodes.num:
        raise ValueError("dd must contain one displacement vector per node")

    bc = bcfun(u)
    dirdofs, dirvals = _dirichlet_values_from_bc(g, bc)
    full[dirdofs] = dirvals

    if op is not None and isinstance(op, dict) and "stress_op" in op:
        stress_op = op["stress_op"]
        C = None
    else:
        stress_op = None
        C = enu2c(efun(u), nufun(u))
    stress, _ = calculate_stress_vem(g, full, C=C, stress_op=stress_op)
    return stress.reshape(-1)


def theta_diff(theta1, theta2):
    """MRST ``thetadiff``: orientation mismatch modulo pi and pi/2."""
    dth = np.abs(np.asarray(theta1, dtype=float) - np.asarray(theta2, dtype=float))
    while np.any(dth > np.pi):
        dth = np.where(dth > np.pi, dth - np.pi, dth)
    return np.where(dth > np.pi / 2.0, np.pi - dth, dth)


def _data_array(data, name):
    val = getattr(data, name, None) if not isinstance(data, dict) else data.get(name)
    if val is None:
        return np.zeros((0, 3), dtype=float)
    arr = np.asarray(val, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    return arr.reshape(-1, 3)


def _objective_data(data):
    orientation = _data_array(data, "orientation")
    smax = _data_array(data, "Smax")
    smin = _data_array(data, "Smin")
    sv = _data_array(data, "Sv")
    d_cell = np.concatenate([
        orientation[:, 0], smax[:, 0], smin[:, 0], sv[:, 0]
    ]).astype(np.int64) - 1
    d_obs = np.concatenate([
        orientation[:, 1], smax[:, 1], smin[:, 1], sv[:, 1]
    ])
    d_sigma = np.concatenate([
        orientation[:, 2], smax[:, 2], smin[:, 2], sv[:, 2]
    ])
    d_type = np.concatenate([
        np.ones(orientation.shape[0], dtype=np.int64),
        2 * np.ones(smax.shape[0], dtype=np.int64),
        3 * np.ones(smin.shape[0], dtype=np.int64),
        4 * np.ones(sv.shape[0], dtype=np.int64),
    ])
    return d_cell, d_obs, d_sigma, d_type


def _objective_match_value_from_stress(
    stress,
    data,
    model_sigma,
    alpha_p=None,
    scaling=None,
):
    stress = np.asarray(stress, dtype=float).reshape(-1, 6)
    d_cell, d_obs, d_sigma, d_type = _objective_data(data)
    model_sigma = np.asarray(model_sigma, dtype=float).ravel()
    if model_sigma.size == 0:
        model_sigma = np.zeros(4)
    Sv = stress[:, 2]
    Smax, Smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3]
    )
    if alpha_p is not None and np.asarray(alpha_p).size:
        ap = np.asarray(alpha_p, dtype=float).ravel()
        Smax = Smax - ap
        Smin = Smin - ap
        Sv = Sv - ap

    theta_ixs = np.flatnonzero(d_type == 1)
    smax_ixs = np.flatnonzero(d_type == 2)
    smin_ixs = np.flatnonzero(d_type == 3)
    sv_ixs = np.flatnonzero(d_type == 4)
    theta_cells = d_cell[theta_ixs]
    smax_cells = d_cell[smax_ixs]
    smin_cells = d_cell[smin_ixs]
    sv_cells = d_cell[sv_ixs]

    threshold_theta = 5e-2
    theta_sigma = np.sqrt(d_sigma[theta_ixs] ** 2 + model_sigma[0] ** 2)
    r_theta = theta_diff(d_obs[theta_ixs], theta[theta_cells]) / theta_sigma
    if theta_ixs.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            theta_rel_ds = np.abs(Smax[theta_cells] - Smin[theta_cells]) / (
                threshold_theta * np.abs(Smax[theta_cells])
            )
        theta_iso = theta_rel_ds < 1.0
        if np.any(theta_iso):
            r_iso = (np.pi / 2.0) / theta_sigma[theta_iso]
            scale = 0.5 * (1.0 - np.cos(theta_rel_ds[theta_iso] * np.pi))
            r_theta[theta_iso] = (
                scale * r_theta[theta_iso] + (1.0 - scale) * r_iso
            )

    smax_sigma = np.sqrt((d_sigma[smax_ixs] * d_obs[smax_ixs]) ** 2
                         + model_sigma[1] ** 2)
    r_smax = (d_obs[smax_ixs] - Smax[smax_cells]) / smax_sigma
    smin_sigma = np.sqrt((d_sigma[smin_ixs] * d_obs[smin_ixs]) ** 2
                         + model_sigma[2] ** 2)
    r_smin = (d_obs[smin_ixs] - Smin[smin_cells]) / smin_sigma
    sv_sigma = np.sqrt((d_sigma[sv_ixs] * d_obs[sv_ixs]) ** 2
                       + model_sigma[3] ** 2)
    r_sv = (d_obs[sv_ixs] - Sv[sv_cells]) / sv_sigma
    if scaling is None or np.asarray(scaling).size == 0:
        scaling = 1.0
    return (
        np.sum(r_theta ** 2) + np.sum(r_smin ** 2)
        + np.sum(r_smax ** 2) + np.sum(r_sv ** 2)
    ) / (d_obs.size * float(scaling))


def _theta_diff_with_derivative(theta1, theta2):
    """Return MRST ``thetadiff`` and derivative with respect to ``theta2``."""
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    dth = np.abs(theta1 - theta2)
    deriv = np.sign(theta2 - theta1)
    while np.any(dth > np.pi):
        ixs = dth > np.pi
        dth = np.where(ixs, dth - np.pi, dth)
    ixs = dth > np.pi / 2.0
    deriv = np.where(ixs, -deriv, deriv)
    dth = np.where(ixs, np.pi - dth, dth)
    return dth, deriv


def _principal_stress_derivatives(stress):
    stress = np.asarray(stress, dtype=float).reshape(-1, 6)
    o_xx = stress[:, 0]
    o_yy = stress[:, 1]
    o_xy = stress[:, 3]
    d = o_xx - o_yy
    g = 2.0 * o_xy
    a2 = d * d + g * g
    a = np.sqrt(a2)
    safe_a = np.where(a > 0.0, a, 1.0)
    safe_a2 = np.where(a2 > 0.0, a2, 1.0)

    dSmax = np.zeros_like(stress)
    dSmin = np.zeros_like(stress)
    dtheta = np.zeros_like(stress)
    active = a > 0.0

    dSmax[:, 0] = 0.5 * (1.0 - d / safe_a)
    dSmax[:, 1] = 0.5 * (1.0 + d / safe_a)
    dSmax[:, 3] = -g / safe_a
    dSmin[:, 0] = 0.5 * (1.0 + d / safe_a)
    dSmin[:, 1] = 0.5 * (1.0 - d / safe_a)
    dSmin[:, 3] = g / safe_a
    dtheta[:, 0] = -0.5 * g / safe_a2
    dtheta[:, 1] = 0.5 * g / safe_a2
    dtheta[:, 3] = d / safe_a2
    dSmax[~active, :] = 0.0
    dSmin[~active, :] = 0.0
    dtheta[~active, :] = 0.0
    return dSmax, dSmin, dtheta


def objective_stress_gradient(
    stress,
    data,
    model_sigma,
    alpha_p=None,
    scaling=None,
):
    """Gradient of the Gaussian mismatch value with respect to stress entries."""
    stress = np.asarray(stress, dtype=float).reshape(-1, 6)
    d_cell, d_obs, d_sigma, d_type = _objective_data(data)
    model_sigma = np.asarray(model_sigma, dtype=float).ravel()
    if model_sigma.size == 0:
        model_sigma = np.zeros(4)
    if model_sigma.size != 4:
        raise ValueError("model_sigma must have four entries")
    if scaling is None or np.asarray(scaling).size == 0:
        scaling = 1.0
    denom = d_obs.size * float(scaling)

    grad = np.zeros_like(stress)
    Sv = stress[:, 2]
    Smax, Smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3]
    )
    if alpha_p is not None and np.asarray(alpha_p).size:
        ap = np.asarray(alpha_p, dtype=float).ravel()
        Smax = Smax - ap
        Smin = Smin - ap
        Sv = Sv - ap

    dSmax, dSmin, dtheta = _principal_stress_derivatives(stress)
    theta_ixs = np.flatnonzero(d_type == 1)
    smax_ixs = np.flatnonzero(d_type == 2)
    smin_ixs = np.flatnonzero(d_type == 3)
    sv_ixs = np.flatnonzero(d_type == 4)

    threshold_theta = 5e-2
    for ix in theta_ixs:
        cell = d_cell[ix]
        sigma = np.sqrt(d_sigma[ix] ** 2 + model_sigma[0] ** 2)
        dth, d_dth_dtheta = _theta_diff_with_derivative(d_obs[ix], theta[cell])
        r = float(dth / sigma)
        dr = (float(d_dth_dtheta) / sigma) * dtheta[cell]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = abs(Smax[cell] - Smin[cell]) / (
                threshold_theta * abs(Smax[cell])
            )
        if np.isfinite(rel) and rel < 1.0:
            r_iso = (np.pi / 2.0) / sigma
            scale = 0.5 * (1.0 - np.cos(rel * np.pi))
            d_delta = dSmax[cell] - dSmin[cell]
            delta = Smax[cell] - Smin[cell]
            denom_rel = threshold_theta * abs(Smax[cell])
            if denom_rel > 0.0:
                drel = (
                    np.sign(delta) * d_delta * denom_rel
                    - abs(delta) * threshold_theta * np.sign(Smax[cell])
                    * dSmax[cell]
                ) / (denom_rel * denom_rel)
            else:
                drel = np.zeros(6)
            dscale = 0.5 * np.pi * np.sin(rel * np.pi) * drel
            dr = scale * dr + (r - r_iso) * dscale
            r = scale * r + (1.0 - scale) * r_iso
        grad[cell] += 2.0 * r * dr / denom

    for ix in smax_ixs:
        cell = d_cell[ix]
        sigma = np.sqrt((d_sigma[ix] * d_obs[ix]) ** 2 + model_sigma[1] ** 2)
        r = (d_obs[ix] - Smax[cell]) / sigma
        grad[cell] += 2.0 * r * (-dSmax[cell] / sigma) / denom
    for ix in smin_ixs:
        cell = d_cell[ix]
        sigma = np.sqrt((d_sigma[ix] * d_obs[ix]) ** 2 + model_sigma[2] ** 2)
        r = (d_obs[ix] - Smin[cell]) / sigma
        grad[cell] += 2.0 * r * (-dSmin[cell] / sigma) / denom
    for ix in sv_ixs:
        cell = d_cell[ix]
        sigma = np.sqrt((d_sigma[ix] * d_obs[ix]) ** 2 + model_sigma[3] ** 2)
        r = (d_obs[ix] - Sv[cell]) / sigma
        grad[cell, 2] += 2.0 * r * (-1.0 / sigma) / denom
    return grad.reshape(-1)


def _calibration_stress_operator(g, E, nu):
    C = enu2c_2d(E, nu) if g.griddim == 2 else enu2c(E, nu)
    op = vem_stress_operator(g, C)
    rows = []
    scales = []
    nlin = 3 if g.griddim == 2 else 6
    for c in range(g.cells.num):
        base = nlin * c
        if g.griddim == 2:
            rows.extend([base, base + 1, base + 2])
            scales.extend([1.0, 1.0, 0.5])
        else:
            rows.extend([base, base + 1, base + 2, base + 3, base + 5, base + 4])
            scales.extend([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    pick = sp.csr_array(
        (scales, (np.arange(len(rows)), rows)),
        shape=(len(rows), op.shape[0]),
    )
    return (pick @ op).tocsr()


def _objective_stress_and_gradient(
    u,
    xx,
    op,
    g,
    data,
    bcfun,
    efun,
    nufun,
    model_sigma,
    alpha_p=None,
    scaling=None,
):
    stress = calculate_stress(
        u, xx, g, bcfun, efun, nufun, op
    ).reshape(-1, 6)
    grad_stress = objective_stress_gradient(
        stress, data, model_sigma, alpha_p, scaling
    )
    return stress.reshape(-1), grad_stress


def _objective_prior_gradient(
    u,
    data,
    num_bc_ctrls,
    mat_facs,
    u_prior,
    u_sigma,
    scaling=None,
    ignore_prior=False,
):
    if ignore_prior:
        return np.zeros_like(np.asarray(u, dtype=float).ravel())
    u = np.asarray(u, dtype=float).ravel()
    u_prior = np.asarray(u_prior, dtype=float).ravel()
    u_sigma = np.asarray(u_sigma, dtype=float).ravel()
    if u_prior.size == 0:
        return np.zeros_like(u)
    if scaling is None or np.asarray(scaling).size == 0:
        scaling = 1.0
    _, d_obs, _, _ = _objective_data(data)
    denom = d_obs.size * float(scaling)
    raw = (u - u_prior) / u_sigma
    r_prior = raw.copy()
    dr_du = 1.0 / u_sigma
    mat_ind = np.zeros(u.size, dtype=bool)
    mat_ind[num_bc_ctrls:] = True
    mat_facs = np.asarray(mat_facs, dtype=float).ravel()
    if np.any(mat_ind):
        facs = np.resize(mat_facs, np.count_nonzero(mat_ind))
        raw_mat = raw[mat_ind].copy()
        scaled = facs != 0.0
        rmat = raw_mat.copy()
        drmat = 1.0 / u_sigma[mat_ind]
        if np.any(scaled):
            norm = np.sqrt(np.log1p(facs[scaled] ** 2))
            rmat[scaled] = np.log1p(raw_mat[scaled] * facs[scaled]) / norm
            drmat[scaled] = (
                facs[scaled]
                / (u_sigma[mat_ind][scaled]
                   * (1.0 + raw_mat[scaled] * facs[scaled]) * norm)
            )
        r_prior[mat_ind] = rmat
        dr_du[mat_ind] = drmat
    return 2.0 * r_prior * dr_du / denom


def objective_control_gradient(
    u,
    xx,
    op,
    g,
    data,
    bcfun,
    efun,
    nufun,
    num_bc_ctrls,
    mat_facs,
    u_prior,
    u_sigma,
    model_sigma,
    scaling=None,
    ignore_prior=False,
    alpha_p=None,
    num_material_ctrls=None,
):
    """Partial objective gradient with respect to controls at fixed displacement."""
    u = np.asarray(u, dtype=float).ravel()
    full = np.asarray(xx, dtype=float).ravel().copy()
    stress, grad_stress = _objective_stress_and_gradient(
        u, full, op, g, data, bcfun, efun, nufun, model_sigma, alpha_p, scaling
    )
    out = _objective_prior_gradient(
        u, data, num_bc_ctrls, mat_facs, u_prior, u_sigma, scaling,
        ignore_prior,
    )
    stress_op = _calibration_stress_operator(g, efun(u), nufun(u))
    bc = bcfun(u)
    if bc.disp_bc_jac is not None and num_bc_ctrls:
        out[:num_bc_ctrls] += np.asarray(
            bc.disp_bc_jac[:, :num_bc_ctrls].T @ (stress_op.T @ grad_stress),
            dtype=float,
        ).ravel()

    if num_material_ctrls is None:
        material_stiff_cols = u.size - num_bc_ctrls
    else:
        material_stiff_cols = int(np.sum(np.asarray(num_material_ctrls)[:2]))
    start = num_bc_ctrls
    stop = min(u.size, start + material_stiff_cols)
    for i in range(start, stop):
        step = 1e-6 * max(abs(u[i]), 1.0)
        up = u.copy()
        um = u.copy()
        up[i] += step
        um[i] -= step
        spv = calculate_stress(up, full, g, bcfun, efun, nufun, op)
        smv = calculate_stress(um, full, g, bcfun, efun, nufun, op)
        out[i] += float(grad_stress @ ((spv - smv) / (2.0 * step)))
    return out


def objective_displacement_gradient(
    u,
    xx,
    op,
    g,
    data,
    bcfun,
    efun,
    nufun,
    model_sigma,
    alpha_p=None,
    scaling=None,
):
    """Gradient of the Gaussian mismatch value with respect to displacements."""
    u = np.asarray(u, dtype=float).ravel()
    xx = np.asarray(xx, dtype=float).ravel()
    full = xx.copy()
    bc = bcfun(u)
    dirdofs, dirvals = _dirichlet_values_from_bc(g, bc)
    full[dirdofs] = dirvals
    _, grad_stress = _objective_stress_and_gradient(
        u, full, op, g, data, bcfun, efun, nufun, model_sigma, alpha_p, scaling
    )
    stress_op = _calibration_stress_operator(g, efun(u), nufun(u))
    return np.asarray(stress_op.T @ grad_stress, dtype=float).ravel()


def gaussian_objfun(
    u,
    xx,
    op,
    g,
    data,
    bcfun,
    num_bc_ctrls,
    mat_facs,
    alpha_p,
    efun,
    nufun,
    u_prior,
    u_sigma,
    model_sigma,
    scaling=None,
    ignore_prior=False,
):
    """Numeric value counterpart of MRST ``gaussian_objfun``.

    Returns ``(value, terms, simvals, datapts, all_simvals)``. AD derivatives
    and adjoint Hessian outputs are intentionally not part of this numeric
    slice.
    """
    u = np.asarray(u, dtype=float).ravel()
    d_cell, d_obs, d_sigma, d_type = _objective_data(data)
    model_sigma = np.asarray(model_sigma, dtype=float).ravel()
    if model_sigma.size == 0:
        model_sigma = np.zeros(4)
    if model_sigma.size != 4:
        raise ValueError("model_sigma must have four entries")

    stress = calculate_stress(u, xx, g, bcfun, efun, nufun, op).reshape(-1, 6)
    Sv = stress[:, 2]
    Smax, Smin, theta = sigma_to_smax_smin_theta(stress[:, 0],
                                                 stress[:, 1],
                                                 stress[:, 3])
    if alpha_p is not None and np.asarray(alpha_p).size:
        ap = np.asarray(alpha_p, dtype=float).ravel()
        Smax = Smax - ap
        Smin = Smin - ap
        Sv = Sv - ap

    theta_ixs = np.flatnonzero(d_type == 1)
    smax_ixs = np.flatnonzero(d_type == 2)
    smin_ixs = np.flatnonzero(d_type == 3)
    sv_ixs = np.flatnonzero(d_type == 4)
    theta_cells = d_cell[theta_ixs]
    smax_cells = d_cell[smax_ixs]
    smin_cells = d_cell[smin_ixs]
    sv_cells = d_cell[sv_ixs]

    threshold_theta = 5e-2
    theta_sigma = np.sqrt(d_sigma[theta_ixs] ** 2 + model_sigma[0] ** 2)
    r_theta = theta_diff(d_obs[theta_ixs], theta[theta_cells]) / theta_sigma
    if theta_cells.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            theta_rel_ds = np.abs(Smax[theta_cells] - Smin[theta_cells]) / (
                threshold_theta * np.abs(Smax[theta_cells])
            )
        theta_iso = theta_rel_ds < 1.0
        if np.any(theta_iso):
            r_theta_iso = (np.pi / 2.0) / theta_sigma[theta_iso]
            scaling_fun = 0.5 * (1.0 - np.cos(theta_rel_ds[theta_iso] * np.pi))
            r_theta[theta_iso] = (
                scaling_fun * r_theta[theta_iso]
                + (1.0 - scaling_fun) * r_theta_iso
            )

    smax_sigma = np.sqrt((d_sigma[smax_ixs] * d_obs[smax_ixs]) ** 2
                         + model_sigma[1] ** 2)
    r_smax = (d_obs[smax_ixs] - Smax[smax_cells]) / smax_sigma

    smin_sigma = np.sqrt((d_sigma[smin_ixs] * d_obs[smin_ixs]) ** 2
                         + model_sigma[2] ** 2)
    r_smin = (d_obs[smin_ixs] - Smin[smin_cells]) / smin_sigma

    sv_sigma = np.sqrt((d_sigma[sv_ixs] * d_obs[sv_ixs]) ** 2
                       + model_sigma[3] ** 2)
    r_sv = (d_obs[sv_ixs] - Sv[sv_cells]) / sv_sigma

    match_term = (
        np.sum(r_theta ** 2) + np.sum(r_smin ** 2)
        + np.sum(r_smax ** 2) + np.sum(r_sv ** 2)
    )
    num_terms = d_obs.size
    u_prior = np.asarray(u_prior, dtype=float).ravel()
    u_sigma = np.asarray(u_sigma, dtype=float).ravel()
    if u_prior.size:
        r_prior = (u - u_prior) / u_sigma
    else:
        r_prior = np.zeros_like(u)

    mat_ind = np.zeros(u.size, dtype=bool)
    mat_ind[num_bc_ctrls:] = True
    mat_facs = np.asarray(mat_facs, dtype=float).ravel()
    if np.any(mat_ind):
        facs = np.resize(mat_facs, np.count_nonzero(mat_ind))
        rmat = r_prior[mat_ind].copy()
        scaled = facs != 0.0
        rmat[scaled] = np.log1p(rmat[scaled] * facs[scaled]) / np.sqrt(
            np.log1p(facs[scaled] ** 2)
        )
        r_prior[mat_ind] = rmat
    if not ignore_prior:
        match_term += np.sum(r_prior ** 2)
    if scaling is None or np.asarray(scaling).size == 0:
        scaling = 1.0
    value = match_term / (num_terms * float(scaling))

    terms = SimpleNamespace(
        r_theta=r_theta, r_Smax=r_smax, r_Smin=r_smin, r_Sv=r_sv,
        r_prior=r_prior,
    )
    simvals = SimpleNamespace(
        Smax=Smax[smax_cells], Smin=Smin[smin_cells], Sv=Sv[sv_cells],
        theta=theta[theta_cells],
    )
    datapts = SimpleNamespace(
        theta=np.column_stack([d_cell[theta_ixs] + 1, d_obs[theta_ixs],
                               theta_sigma]) if theta_ixs.size else
        np.zeros((0, 3)),
        Smax=np.column_stack([d_cell[smax_ixs] + 1, d_obs[smax_ixs],
                              smax_sigma]) if smax_ixs.size else
        np.zeros((0, 3)),
        Smin=np.column_stack([smin_cells + 1, d_obs[smin_ixs],
                              smin_sigma]) if smin_ixs.size else
        np.zeros((0, 3)),
        Sv=np.column_stack([sv_cells + 1, d_obs[sv_ixs], sv_sigma])
        if sv_ixs.size else np.zeros((0, 3)),
    )
    all_simvals = SimpleNamespace(theta=theta, Smax=Smax, Smin=Smin, Sv=Sv)
    return value, terms, simvals, datapts, all_simvals


def _ad2_bc_vector(g, bcfun, u_ad: ADI2):
    """Replay `set_bcfun` boundary controls as a full ADI2 DOF vector."""
    meta = getattr(bcfun, "_pymrst_calibration_bc_data", None)
    if meta is None:
        raise NotImplementedError("ADI2 Hessian requires BCs from set_bcfun")
    bcond = np.asarray(meta.bcond, dtype=float)
    active = np.asarray(meta.active_u, dtype=np.int64)
    nctrl = int(meta.num_bc_ctrls)
    if len(u_ad) < nctrl:
        raise ValueError("control vector is shorter than BC control count")

    pars = []
    for par_ix in range(5):
        loc = np.flatnonzero(active == par_ix)
        if loc.size:
            pars.append(bcond[par_ix, 0] + bcond[par_ix, 1] * u_ad[int(loc[0])])
        else:
            pars.append(bcond[par_ix, 0])

    ndof = g.griddim * g.nodes.num
    dofs = []
    vals = []

    def add_side(nodes, zdisp):
        nodes = np.asarray(nodes, dtype=np.int64).ravel()
        if nodes.size == 0:
            return
        theta, e1, e2, de1dz, de2dz = pars
        c = ad2_cos(theta)
        s = ad2_sin(theta)
        m00 = e1 * c * c + e2 * s * s
        m01 = (e1 - e2) * c * s
        m11 = e1 * s * s + e2 * c * c
        n00 = de1dz * c * c + de2dz * s * s
        n01 = (de1dz - de2dz) * c * s
        n11 = de1dz * s * s + de2dz * c * c
        xy = np.asarray(g.nodes.coords[nodes, :2], dtype=float)
        z = (np.asarray(g.nodes.coords[nodes, 2], dtype=float)
             if g.griddim == 3 else np.zeros(nodes.size))
        for node, (x, y), zz in zip(nodes, xy, z):
            ux = m00 * x + m01 * y + (n00 * x + n01 * y) * zz
            uy = m01 * x + m11 * y + (n01 * x + n11 * y) * zz
            dofs.extend([g.griddim * node, g.griddim * node + 1])
            vals.extend([ux, uy])
            if g.griddim == 3 and np.isfinite(zdisp):
                dofs.append(g.griddim * node + 2)
                vals.append(float(zdisp) * (1.0 + zz))

    add_side(meta.side_nodes, np.nan)
    add_side(meta.bottom_edge_nodes, 0.0)
    bottom_nodes = np.asarray(meta.bottom_nodes, dtype=np.int64).ravel()
    if g.griddim == 3 and bottom_nodes.size:
        dofs.extend((g.griddim * bottom_nodes + 2).tolist())
        vals.extend([0.0] * bottom_nodes.size)

    if not vals:
        return ad2_scatter([], np.zeros(0), ndof)
    return ad2_scatter(dofs, ad2_vertcat(*vals), ndof)


def _ad2_principal_values(stress_ad: ADI2, cell: int):
    base = 6 * int(cell)
    o_xx = stress_ad[base]
    o_yy = stress_ad[base + 1]
    o_xy = stress_ad[base + 3]
    s = o_xx + o_yy
    d = o_xx - o_yy
    gxy = 2.0 * o_xy
    a = (d * d + gxy * gxy) ** 0.5
    smax = 0.5 * (s - a)
    smin = 0.5 * (s + a)
    theta = 0.5 * ad2_atan2(gxy, d)
    smax_ref, smin_ref, theta_ref = sigma_to_smax_smin_theta(
        np.array([stress_ad.val[base]]),
        np.array([stress_ad.val[base + 1]]),
        np.array([stress_ad.val[base + 3]]),
    )
    theta = theta + (theta_ref[0] - theta.val[0])
    sv = stress_ad[base + 2]
    return smax, smin, theta, sv


def _ad2_theta_residual(obs, theta_ad: ADI2):
    """Local smooth branch of MRST theta_diff as a signed residual."""
    shifted = float(obs)
    diff = shifted - theta_ad.val[0]
    while diff > np.pi / 2.0:
        shifted -= np.pi
        diff = shifted - theta_ad.val[0]
    while diff < -np.pi / 2.0:
        shifted += np.pi
        diff = shifted - theta_ad.val[0]
    return shifted - theta_ad


def _ad2_objective_from_stress(
    u_ad: ADI2,
    stress_ad: ADI2,
    data,
    u_prior,
    u_sigma,
    model_sigma,
    scaling=None,
    ignore_prior=False,
    alpha_p=None,
):
    d_cell, d_obs, d_sigma, d_type = _objective_data(data)
    model_sigma = np.asarray(model_sigma, dtype=float).ravel()
    if model_sigma.size == 0:
        model_sigma = np.zeros(4)
    if scaling is None or np.asarray(scaling).size == 0:
        scaling = 1.0
    denom = max(d_obs.size, 1) * float(scaling)
    terms = []
    ap = np.asarray(alpha_p, dtype=float).ravel() if alpha_p is not None else np.array([])
    for ix, cell in enumerate(d_cell):
        smax, smin, theta, sv = _ad2_principal_values(stress_ad, int(cell))
        if ap.size:
            smax = smax - ap[int(cell)]
            smin = smin - ap[int(cell)]
            sv = sv - ap[int(cell)]
        typ = int(d_type[ix])
        if typ == 1:
            sigma = np.sqrt(d_sigma[ix] ** 2 + model_sigma[0] ** 2)
            r = _ad2_theta_residual(d_obs[ix], theta) / sigma
        elif typ == 2:
            sigma = np.sqrt((d_sigma[ix] * d_obs[ix]) ** 2 + model_sigma[1] ** 2)
            r = (d_obs[ix] - smax) / sigma
        elif typ == 3:
            sigma = np.sqrt((d_sigma[ix] * d_obs[ix]) ** 2 + model_sigma[2] ** 2)
            r = (d_obs[ix] - smin) / sigma
        elif typ == 4:
            sigma = np.sqrt((d_sigma[ix] * d_obs[ix]) ** 2 + model_sigma[3] ** 2)
            r = (d_obs[ix] - sv) / sigma
        else:
            continue
        terms.append(r * r)

    if not ignore_prior:
        u_prior = np.asarray(u_prior, dtype=float).ravel()
        u_sigma = np.asarray(u_sigma, dtype=float).ravel()
        for i in range(min(len(u_ad), u_prior.size)):
            r = (u_ad[i] - u_prior[i]) / u_sigma[i]
            terms.append(r * r)
    if not terms:
        return ad2_sum(u_ad * 0.0)
    return ad2_sum(ad2_vertcat(*terms)) / denom


def setup_objective_fun(
    g,
    data,
    bcfun,
    mfuns,
    num_bc_ctrls,
    mat_facs,
    u_init,
    u_sigma,
    model_sigma,
    scaling=None,
    ignore_prior=False,
    solver=None,
    alpha_p=None,
):
    """Numeric setup counterpart of MRST ``setup_objective_fun``.

    The returned namespace provides ``ofun`` and ``matchfun`` callables with
    the MRST objective value. ``objfun`` eliminates displacement dependence by
    solving the AD VEM mechanics problem and uses the MRST adjoint structure
    for gradients. ``objfun_with_hessian`` propagates boundary-control
    eliminated mechanics end-to-end through ADI2 for constant-material
    calibration cases and uses the validated eliminated-gradient path for
    material-control Hessians.
    """
    alpha_p_arg = np.array([]) if alpha_p is None else alpha_p

    def ofun(u, xx, op=None):
        return gaussian_objfun(
            u, xx, op, g, data, bcfun, num_bc_ctrls, mat_facs, alpha_p,
            mfuns.efun, mfuns.nufun, u_init, u_sigma, model_sigma, scaling,
            ignore_prior,
        )

    def matchfun(u, xx, op=None):
        return gaussian_objfun(
            u, xx, op, g, data, bcfun, num_bc_ctrls, mat_facs, alpha_p,
            mfuns.efun, mfuns.nufun, u_init, u_sigma, model_sigma, scaling,
            True,
        )

    def solve_value(u, extra0=None):
        uu, extra = vem_lin_elast_ad(
            g,
            mfuns.efun(u),
            mfuns.nufun(u),
            bcfun(u),
            mfuns.loadfun(u),
            linsolve=solver,
            pressure=alpha_p_arg,
            extra=extra0,
        )
        val = ofun(u, uu.reshape(-1), extra)[0]
        return float(val), uu, extra

    def _eliminated_gradient(u):
        u = np.asarray(u, dtype=float).ravel()
        val, uu, extra = solve_value(u, op_0)
        full = uu.reshape(-1)
        free = ~extra.disc["isdirdofs"]
        oval_du = objective_control_gradient(
            u, full, extra, g, data, bcfun, mfuns.efun, mfuns.nufun,
            num_bc_ctrls, mat_facs, u_init, u_sigma, model_sigma, scaling,
            ignore_prior, alpha_p, getattr(mfuns, "num_ctrls", None),
        )

        oval_dd_full = objective_displacement_gradient(
            u, full, extra, g, data, bcfun, mfuns.efun, mfuns.nufun,
            model_sigma, alpha_p, scaling,
        )
        oval_dd = oval_dd_full[free]

        dsys_du = np.zeros((oval_dd.size, u.size))
        bc = bcfun(u)
        bc_ad_cols = np.zeros(u.size, dtype=bool)
        if bc.disp_bc_jac is not None and num_bc_ctrls:
            dsys_du[:, :num_bc_ctrls] = (
                extra.S @ bc.disp_bc_jac
            )[free, :num_bc_ctrls].toarray()
            bc_ad_cols[:num_bc_ctrls] = True
        material_ad_cols = np.zeros(u.size, dtype=bool)
        material_stiff_cols = int(np.sum(getattr(mfuns, "num_ctrls", [0, 0, 0])[:2]))
        if material_stiff_cols:
            from pymrst.utils.adi import init_variables_adi

            (u_ad,) = init_variables_adi(u)
            ax = compute_ax_derivs(
                g,
                full,
                extra,
                mfuns.efun(u_ad),
                mfuns.nufun(u_ad),
                dirdofs=extra.disc["isdirdofs"],
            )
            cols = slice(num_bc_ctrls, num_bc_ctrls + material_stiff_cols)
            dsys_du[:, cols] = ax.toarray()[:, cols]
            material_ad_cols[cols] = True
        load_ad_cols = np.zeros(u.size, dtype=bool)
        if hasattr(mfuns, "load_jac"):
            ljac = mfuns.load_jac(u)
            if ljac.shape[1]:
                dsys_du -= ljac[free, :].toarray()
                load_ad_cols[np.flatnonzero(np.asarray(np.abs(ljac).sum(axis=0)).ravel() > 0)] = True
        for i in range(u.size):
            if bc_ad_cols[i] or material_ad_cols[i] or load_ad_cols[i]:
                continue
            step = 1e-6 * max(abs(u[i]), 1.0)
            up = u.copy()
            um = u.copy()
            up[i] += step
            um[i] -= step
            rp = vem_reduced_system_residual(
                g, mfuns.efun(up), mfuns.nufun(up), bcfun(up),
                mfuns.loadfun(up), full, pressure=alpha_p_arg, extra=op_0,
            )[0]
            rm = vem_reduced_system_residual(
                g, mfuns.efun(um), mfuns.nufun(um), bcfun(um),
                mfuns.loadfun(um), full, pressure=alpha_p_arg, extra=op_0,
            )[0]
            dsys_du[:, i] = (rp - rm) / (2.0 * step)

        solve = solver or (lambda AA, bb: spla.spsolve(AA.tocsc(), bb))
        lam = -solve(extra.A, oval_dd)
        grad = oval_du + dsys_du.T @ lam
        return val, grad

    def _eliminated_objective_ad2(u):
        u = np.asarray(u, dtype=float).ravel()
        u_ad = init_variables_adi2(u)
        bc_ad_vec = _ad2_bc_vector(g, bcfun, u_ad)
        val0, uu0, extra = solve_value(u, op_0)
        free = ~extra.disc["isdirdofs"]
        bc_val = bc_ad_vec.val
        load_free = extra.rhs + (extra.S[free, :] @ bc_val)
        rhs_ad = ad2_matmul(-extra.S[free, :], bc_ad_vec) + load_free
        free_ad = ad2_linsolve(extra.A, rhs_ad)
        dir_ad = bc_ad_vec
        free_scattered = ad2_scatter(np.flatnonzero(free), free_ad,
                                     g.griddim * g.nodes.num)
        mask_dir = np.zeros(g.griddim * g.nodes.num)
        mask_dir[extra.disc["isdirdofs"]] = 1.0
        full_ad = dir_ad * mask_dir + free_scattered
        stress_op = _calibration_stress_operator(
            g, mfuns.efun(u), mfuns.nufun(u)
        )
        stress_ad = ad2_matmul(stress_op, full_ad)
        obj_ad = _ad2_objective_from_stress(
            u_ad, stress_ad, data, u_init, u_sigma, model_sigma, scaling,
            ignore_prior, alpha_p,
        )
        return val0, obj_ad

    def _gradient_difference_hessian(u, grad0=None):
        """Derivative-backed Hessian for material-control mechanics paths.

        The boundary-control-only path above propagates the reduced mechanics
        solve through ADI2. Material controls change the realized VEM stiffness,
        stress operator, and load vector, which currently live as numeric
        sparse matrices. This fallback differentiates the validated eliminated
        gradient, so the Hessian includes material stiffness and density-load
        effects without dropping those columns.
        """
        u = np.asarray(u, dtype=float).ravel()
        hess = np.zeros((u.size, u.size))
        for i in range(u.size):
            step = 1e-5 * max(abs(u[i]), 1.0)
            up = u.copy()
            um = u.copy()
            up[i] += step
            um[i] -= step
            gp = _eliminated_gradient(up)[1]
            gm = _eliminated_gradient(um)[1]
            hess[:, i] = (gp - gm) / (2.0 * step)
        hess = 0.5 * (hess + hess.T)
        if grad0 is not None and not np.all(np.isfinite(hess)):
            raise FloatingPointError("non-finite calibration Hessian entries")
        return hess

    def objfun(u):
        val, grad = _eliminated_gradient(u)
        return val, grad, None, None

    def objfun_with_hessian(u):
        u = np.asarray(u, dtype=float).ravel()
        material_ctrls = int(np.sum(getattr(mfuns, "num_ctrls", [0, 0, 0])))
        if u.size == num_bc_ctrls and material_ctrls == 0:
            _, obj_ad = _eliminated_objective_ad2(u)
            val = float(obj_ad.val[0])
            grad = obj_ad.jac.toarray().ravel()
            qhess = obj_ad.hess[0].toarray()
            qhess = 0.5 * (qhess + qhess.T)
        else:
            val, grad = _eliminated_gradient(u)
            qhess = _gradient_difference_hessian(u, grad)

        def qhess_dir(direction):
            return qhess @ np.asarray(direction, dtype=float).ravel()

        return val, grad, qhess, qhess_dir

    _, op_0 = vem_lin_elast_ad(
        g,
        mfuns.efun(u_init),
        mfuns.nufun(u_init),
        bcfun(u_init),
        mfuns.loadfun(u_init),
        pressure=alpha_p_arg,
        no_solve=True,
    )

    return SimpleNamespace(objfun=objfun, objfun_with_hessian=objfun_with_hessian,
                           op_0=op_0, matchfun=matchfun, ofun=ofun)


def set_material_fun(g, props, matcond=None, u_offset: int = 0):
    """Numeric counterpart of MRST ``set_material_fun``.

    Returns ``(funs, num_ctrls)`` where ``funs`` has ``efun``, ``nufun`` and
    ``loadfun`` callables.
    """
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


def _project_controls(u, bounds):
    u = np.asarray(u, dtype=float).ravel()
    if bounds is None:
        return u
    lo, hi = bounds
    return np.clip(u, lo, hi)


def _gradient_minimize(
    fun: Callable[[np.ndarray], tuple[float, np.ndarray, object, object]],
    u0,
    grad_tol: float = 1e-6,
    obj_change_tol: float = 1e-7,
    max_it: int = 100,
    line_search_max_it: int = 8,
    bounds: tuple[float, float] | None = (0.0, 1.0),
):
    """Small projected backtracking minimizer for calibration workflows."""
    u = _project_controls(u0, bounds)
    val, grad, _, _ = fun(u)
    val = float(val)
    grad = np.asarray(grad, dtype=float).ravel()
    values = [val]
    controls = [u.copy()]
    grad_norms = [float(np.linalg.norm(grad, ord=np.inf))]
    step_lengths: list[float] = []
    status = "max_it"
    for _ in range(max_it):
        gnorm = float(np.linalg.norm(grad, ord=np.inf))
        if gnorm <= grad_tol:
            status = "grad_tol"
            break
        direction = -grad
        accepted = False
        step = 1.0
        for _ls in range(line_search_max_it):
            cand = _project_controls(u + step * direction, bounds)
            if np.allclose(cand, u):
                step *= 0.5
                continue
            cval, cgrad, _, _ = fun(cand)
            cval = float(cval)
            if cval <= val - 1e-4 * step * float(grad @ (u - cand)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = "line_search"
            break
        delta = abs(val - cval)
        u = cand
        val = cval
        grad = np.asarray(cgrad, dtype=float).ravel()
        values.append(val)
        controls.append(u.copy())
        grad_norms.append(float(np.linalg.norm(grad, ord=np.inf)))
        step_lengths.append(step)
        if delta <= obj_change_tol:
            status = "obj_change_tol"
            break
    history = OptimizationHistory(
        values=values,
        controls=controls,
        grad_norms=grad_norms,
        step_lengths=step_lengths,
        status=status,
    )
    return val, u, history


def _sr1_determine_subspace(g, B):
    Bstar = np.asarray(B, dtype=float).copy()
    eigvals = np.linalg.eigvalsh(Bstar)
    lmin = float(np.min(eigvals))
    if lmin < 0.0:
        Bstar = Bstar + 1.5 * abs(lmin) * np.eye(Bstar.shape[0])
    g = np.asarray(g, dtype=float).ravel()
    try:
        v2 = np.linalg.solve(Bstar, g)
    except np.linalg.LinAlgError:
        v2 = g.copy()
    if not np.all(np.isfinite(v2)):
        v2 = g.copy()
    basis = np.column_stack([g, v2])
    q, r = np.linalg.qr(basis)
    keep = np.abs(np.diag(r)) > 1e-12
    q = q[:, keep]
    e1 = q[:, 0]
    e2 = q[:, 1] if q.shape[1] > 1 else None
    return e1, e2, Bstar


def _sr1_compute_step(g, B, delta):
    g = np.asarray(g, dtype=float).ravel()
    if np.linalg.norm(g) == 0.0:
        return np.zeros_like(g), np.asarray(B, dtype=float)
    e1, e2, Bstar = _sr1_determine_subspace(g, B)
    p_int = None
    try:
        s = -np.linalg.solve(Bstar, g)
    except np.linalg.LinAlgError:
        s = -g / np.linalg.norm(g) * delta
    if np.linalg.norm(s) < delta:
        if np.min(np.linalg.eigvalsh(B)) > 0.0:
            return s, Bstar
        p_int = s.copy()
    if e2 is None:
        return -g / np.linalg.norm(g) * delta, Bstar

    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    c = np.cos(theta)
    sines = np.sin(theta)
    pts = delta * (np.outer(c, e1) + np.outer(sines, e2))
    vals = pts @ g + 0.5 * np.einsum("ij,jk,ik->i", pts, Bstar, pts)
    best = pts[int(np.argmin(vals))]
    if p_int is not None:
        pint_val = float(g @ p_int + 0.5 * p_int @ Bstar @ p_int)
        if pint_val < float(np.min(vals)):
            best = p_int
    return best, Bstar


def _sr1_history(
    values,
    controls,
    grad_norms,
    step_lengths,
    status,
    grads,
    trust_radii,
    hessians,
    iterations,
    ratios,
):
    return OptimizationHistory(
        values=values,
        controls=controls,
        grad_norms=grad_norms,
        step_lengths=step_lengths,
        status=status,
        gradients=grads,
        trust_radii=trust_radii,
        hessians=hessians,
        iterations=iterations,
        ratios=ratios,
    )


def _save_sr1_backup(path, state):
    if path is None:
        return
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    with pth.open("wb") as handle:
        pickle.dump(state, handle)


def _load_sr1_backup(path):
    if path is None:
        return None
    pth = Path(path)
    if not pth.exists():
        return None
    with pth.open("rb") as handle:
        return pickle.load(handle)


def optimize_sr1(
    fun: Callable[[np.ndarray], tuple[float, np.ndarray, object, object]],
    u0,
    *,
    grad_tol: float = 1e-8,
    obj_change_tol: float = 1e-8,
    max_it: int = 100,
    bounds: tuple[float, float] | None = (0.0, 1.0),
    delta: float = 1.0,
    eta: float = 1e-3,
    update_tol: float = 1e-3,
    b_scale: float = 1.0,
    b_init=None,
    backup_file: str | None = None,
    resume_backup: bool = True,
    progress_callback: Callable | None = None,
):
    """MRST-style SR1 trust-region minimizer for calibration controls."""
    loaded = _load_sr1_backup(backup_file) if resume_backup else None
    if loaded is not None:
        u = np.asarray(loaded["u"], dtype=float).ravel()
        val = float(loaded["val"])
        grad = np.asarray(loaded["grad"], dtype=float).ravel()
        B = np.asarray(loaded["B"], dtype=float)
        delta = float(loaded["delta"])
        mini_iter = int(loaded.get("mini_iter", 1))
        skipped_last_update = bool(loaded.get("skipped_last_update", False))
        values = list(loaded["history"].values)
        controls = [np.asarray(x, dtype=float).copy()
                    for x in loaded["history"].controls]
        grads = [np.asarray(x, dtype=float).copy()
                 for x in (loaded["history"].gradients or [grad])]
        grad_norms = list(loaded["history"].grad_norms)
        step_lengths = list(loaded["history"].step_lengths)
        trust_radii = list(loaded["history"].trust_radii or [float(delta)])
        hessians = [np.asarray(x, dtype=float).copy()
                    for x in (loaded["history"].hessians or [B])]
        iterations = list(loaded["history"].iterations or [0])
        ratios = list(loaded["history"].ratios or [])
        start_iter = int(loaded.get("iteration", max(len(values) - 1, 0)))
    else:
        u = _project_controls(u0, bounds)
        val, grad, _, _ = fun(u)
        val = float(val)
        grad = np.asarray(grad, dtype=float).ravel()
        n = u.size
        B = np.asarray(b_init, dtype=float).copy() if b_init is not None else (
            float(b_scale) * np.eye(n)
        )
        values = [val]
        controls = [u.copy()]
        grads = [grad.copy()]
        grad_norms = [float(np.linalg.norm(grad, ord=np.inf))]
        step_lengths: list[float] = []
        trust_radii = [float(delta)]
        hessians = [B.copy()]
        iterations = [0]
        ratios: list[float] = []
        mini_iter = 1
        skipped_last_update = False
        start_iter = 0
    status = "max_it"
    if progress_callback is not None:
        progress_callback(_sr1_history(
            values, controls, grad_norms, step_lengths, "start", grads,
            trust_radii, hessians, iterations, ratios,
        ))
    for it in range(start_iter, max_it):
        if np.linalg.norm(grad) <= grad_tol:
            status = "grad_tol"
            break
        step, Bstar = _sr1_compute_step(grad, B, delta)
        cand = _project_controls(u + step, bounds)
        actual_step = cand - u
        if np.linalg.norm(actual_step) == 0.0:
            status = "stalled"
            break
        cval, cgrad, _, _ = fun(cand)
        cval = float(cval)
        cgrad = np.asarray(cgrad, dtype=float).ravel()
        dg = cgrad - grad
        act_red = val - cval
        pre_red = -(grad @ actual_step + 0.5 * actual_step @ Bstar @ actual_step)
        ratio = act_red / pre_red if pre_red != 0.0 else -np.inf
        accepted = ratio > eta
        if accepted:
            old_val = val
            u = cand
            val = cval
            grad = cgrad
            values.append(val)
            controls.append(u.copy())
            grads.append(grad.copy())
            grad_norms.append(float(np.linalg.norm(grad, ord=np.inf)))
            step_lengths.append(float(np.linalg.norm(actual_step)))
            trust_radii.append(float(delta))
            hessians.append(B.copy())
            iterations.append(it + 1)
            ratios.append(float(ratio))
            if abs(old_val - val) < obj_change_tol:
                status = "obj_change_tol"
            if progress_callback is not None:
                progress_callback(_sr1_history(
                    values, controls, grad_norms, step_lengths, status,
                    grads, trust_radii, hessians, iterations, ratios,
                ))
            _save_sr1_backup(backup_file, {
                "u": u.copy(),
                "val": val,
                "grad": grad.copy(),
                "B": B.copy(),
                "delta": float(delta),
                "mini_iter": mini_iter,
                "skipped_last_update": skipped_last_update,
                "iteration": it + 1,
                "history": _sr1_history(
                    values, controls, grad_norms, step_lengths, status,
                    grads, trust_radii, hessians, iterations, ratios,
                ),
            })
            if status == "obj_change_tol":
                break
            mini_iter = 1
        else:
            if abs(act_red) < obj_change_tol and mini_iter > 4:
                status = "no_progress"
                break
            mini_iter += 1

        if ratio > 0.75:
            if np.linalg.norm(actual_step) > delta * 0.8:
                delta *= 2.0
        elif ratio < 0.1:
            if np.linalg.norm(actual_step) < delta and skipped_last_update:
                delta = max(np.linalg.norm(actual_step) / 2.0, 1e-12)
            else:
                delta = max(delta / 2.0, 1e-12)

        tmp = dg - B @ actual_step
        denom = float(actual_step @ tmp)
        if abs(denom) >= update_tol * np.linalg.norm(actual_step) * np.linalg.norm(tmp):
            B = B + np.outer(tmp, tmp) / denom
            skipped_last_update = False
        else:
            skipped_last_update = True
    history = _sr1_history(
        values, controls, grad_norms, step_lengths, status, grads,
        trust_radii, hessians, iterations, ratios,
    )
    if backup_file is not None:
        _save_sr1_backup(backup_file, {
            "u": u.copy(),
            "val": val,
            "grad": grad.copy(),
            "B": B.copy(),
            "delta": float(delta),
            "mini_iter": mini_iter,
            "skipped_last_update": skipped_last_update,
            "iteration": max(iterations) if iterations else 0,
            "history": history,
        })
    if progress_callback is not None:
        progress_callback(history)
    return val, u, history


def _props_to_young_poisson(g, props):
    if hasattr(props, "young") and hasattr(props, "poisson"):
        return props
    if not (hasattr(props, "K") and hasattr(props, "G")):
        return props
    K = np.asarray(props.K, dtype=float).ravel()
    Gs = np.asarray(props.G, dtype=float).ravel()
    E, nu = _kg_to_enu(K, Gs)
    if E.size == 1:
        E = np.full(g.cells.num, float(E[0]))
    if nu.size == 1:
        nu = np.full(g.cells.num, float(nu[0]))
    rho = np.asarray(props.rho, dtype=float).ravel()
    if rho.size == 1:
        rho = np.full(g.cells.num, float(rho[0]))
    return SimpleNamespace(young=E, poisson=nu, rho=rho)


def produce_syndata(g, bcfun, mfuns, u, solver=None, alpha_p=None):
    """MRST ``produce_syndata`` helper: solve mechanics for a control vector."""
    alpha_p_arg = np.array([]) if alpha_p is None else alpha_p
    uu, extra = vem_lin_elast_ad(
        g,
        mfuns.efun(u),
        mfuns.nufun(u),
        bcfun(u),
        mfuns.loadfun(u),
        linsolve=solver,
        pressure=alpha_p_arg,
    )
    return SimpleNamespace(bcfun=bcfun, mfuns=mfuns, uu=uu, extra=extra)


def optimize_mech(
    u,
    g,
    bcfun,
    efun,
    nufun,
    loadfun,
    obj_fun,
    *,
    extra=None,
    background_forces=None,
    grad_tol: float = 1e-9,
    obj_change_tol: float = 1e-10,
    max_it: int = 100,
    line_search_max_it: int = 8,
    bounds: tuple[float, float] | None = (0.0, 1.0),
    optimizer: str = "sr1",
    delta: float = 1.0,
    eta: float = 1e-3,
    sr1_update_tol: float = 1e-3,
    b_scale: float = 1.0,
    hi=None,
    backup_file: str | None = None,
    resume_backup: bool = True,
    progress_callback: Callable | None = None,
    solver=None,
):
    """Runnable PyMRST counterpart of MRST ``optimize_mech``.

    ``obj_fun`` has the MRST contract ``obj_fun(u, dd, extra)`` and returns
    value plus partial derivatives with respect to controls and displacements.
    """
    alpha_p_arg = np.array([])

    def wrapped(uu_ctrl):
        dd, ex = vem_lin_elast_ad(
            g,
            efun(uu_ctrl),
            nufun(uu_ctrl),
            bcfun(uu_ctrl),
            loadfun(uu_ctrl),
            linsolve=solver,
            extra=extra,
            background_forces=background_forces,
        )
        val, oval_du, oval_dd = obj_fun(uu_ctrl, dd.reshape(-1), ex)[:3]
        full = dd.reshape(-1)
        free = ~ex.disc["isdirdofs"]
        rhs_grad = np.asarray(oval_dd, dtype=float).ravel()
        if rhs_grad.size != np.count_nonzero(free):
            rhs_grad = rhs_grad[free]
        solve = solver or (lambda AA, bb: spla.spsolve(AA.tocsc(), bb))
        lam = -solve(ex.A, rhs_grad)
        dsys_du = np.zeros((rhs_grad.size, np.asarray(uu_ctrl).size))
        for i in range(np.asarray(uu_ctrl).size):
            step = 1e-6 * max(abs(uu_ctrl[i]), 1.0)
            up = np.asarray(uu_ctrl, dtype=float).copy()
            um = up.copy()
            up[i] += step
            um[i] -= step
            rp = vem_reduced_system_residual(
                g, efun(up), nufun(up), bcfun(up), loadfun(up), full,
                pressure=alpha_p_arg, extra=extra,
                background_forces=background_forces,
            )[0]
            rm = vem_reduced_system_residual(
                g, efun(um), nufun(um), bcfun(um), loadfun(um), full,
                pressure=alpha_p_arg, extra=extra,
                background_forces=background_forces,
            )[0]
            dsys_du[:, i] = (rp - rm) / (2.0 * step)
        grad = np.asarray(oval_du, dtype=float).ravel() + dsys_du.T @ lam
        return float(val), grad, None, None

    if optimizer.lower() == "sr1":
        foptval, uopt, history = optimize_sr1(
            wrapped,
            u,
            grad_tol=grad_tol,
            obj_change_tol=obj_change_tol,
            max_it=max_it,
            bounds=bounds,
            delta=delta,
            eta=eta,
            update_tol=sr1_update_tol,
            b_scale=b_scale,
            b_init=hi,
            backup_file=backup_file,
            resume_backup=resume_backup,
            progress_callback=progress_callback,
        )
    else:
        foptval, uopt, history = _gradient_minimize(
            wrapped,
            u,
            grad_tol=grad_tol,
            obj_change_tol=obj_change_tol,
            max_it=max_it,
            line_search_max_it=line_search_max_it,
            bounds=bounds,
        )
    uu_opt, opt_extra = vem_lin_elast_ad(
        g,
        efun(uopt),
        nufun(uopt),
        bcfun(uopt),
        loadfun(uopt),
        linsolve=solver,
        extra=extra,
        background_forces=background_forces,
    )
    return foptval, uopt, history, uu_opt, opt_extra


def calibrate_model(g, props, bcond, matcond, indata, **kwargs):
    """Calibrate VEM mechanics controls from stress/orientation observations."""
    opt = CalibrationOptions(**kwargs)
    props = _props_to_young_poisson(g, props)
    bcond = np.zeros((3, 2)) if bcond is None or np.asarray(bcond).size == 0 else bcond
    matcond = np.zeros((0, 4)) if matcond is None or np.asarray(matcond).size == 0 else matcond

    binfo = identify_boundary_elements(g)
    if binfo is None:
        raise ValueError("could not identify all calibration boundary elements")
    bcfun, num_bc_ctrls = set_bcfun(g, binfo, bcond)
    mfuns, num_mctrls = set_material_fun(g, props, matcond, num_bc_ctrls)
    mat_facs = np.asarray(matcond, dtype=float)[:, 1:].reshape(-1)

    if opt.produce_syndata_for is not None:
        return produce_syndata(
            g, bcfun, mfuns, opt.produce_syndata_for, alpha_p=opt.alpha_p
        )

    num_controls = int(num_bc_ctrls + np.sum(num_mctrls))
    u_init = np.zeros(num_controls)
    u_sigma = np.ones(num_controls)
    if opt.ignore_bnd_prior and num_bc_ctrls:
        u_sigma[:num_bc_ctrls] = np.inf
    u_start = u_init if opt.u_start is None else np.asarray(opt.u_start, dtype=float)
    model_sigma = np.zeros(4) if opt.model_sigma is None else opt.model_sigma

    setup = setup_objective_fun(
        g,
        indata,
        bcfun,
        mfuns,
        num_bc_ctrls,
        mat_facs,
        u_init,
        u_sigma,
        model_sigma,
        opt.scaling,
        opt.ignore_prior,
        None,
        opt.alpha_p,
    )
    if opt.optimizer.lower() == "sr1":
        foptval, uopt, history = optimize_sr1(
            setup.objfun,
            u_start,
            grad_tol=opt.grad_tol,
            obj_change_tol=opt.obj_change_tol,
            max_it=opt.max_it,
            bounds=opt.bounds,
            delta=opt.delta,
            eta=opt.eta,
            update_tol=opt.sr1_update_tol,
            b_scale=opt.b_scale,
            b_init=opt.hi,
            backup_file=opt.backup_file,
            resume_backup=opt.resume_backup,
            progress_callback=opt.progress_callback,
        )
    else:
        foptval, uopt, history = _gradient_minimize(
            setup.objfun,
            u_start,
            grad_tol=opt.grad_tol,
            obj_change_tol=opt.obj_change_tol,
            max_it=opt.max_it,
            line_search_max_it=opt.line_search_max_it,
            bounds=opt.bounds,
        )

    alpha_p_arg = np.array([]) if opt.alpha_p is None else opt.alpha_p
    uu_init, extra_init = vem_lin_elast_ad(
        g, mfuns.efun(u_init), mfuns.nufun(u_init), bcfun(u_init),
        mfuns.loadfun(u_init), pressure=alpha_p_arg, extra=setup.op_0,
    )
    uu_opt, extra_opt = vem_lin_elast_ad(
        g, mfuns.efun(uopt), mfuns.nufun(uopt), bcfun(uopt),
        mfuns.loadfun(uopt), pressure=alpha_p_arg, extra=setup.op_0,
    )
    return SimpleNamespace(
        foptval=foptval,
        history=history,
        u_opt=uopt,
        uu_opt=uu_opt,
        extra=extra_opt,
        bcfun=bcfun,
        mfuns=mfuns,
        u_init=u_init,
        uu_init=uu_init,
        extra_init=extra_init,
        obj_fun=setup.objfun,
        matchfun=setup.matchfun,
        indata=indata,
        num_bc_ctrls=num_bc_ctrls,
        num_mctrls=num_mctrls,
    )


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import create_augmented_grid

    def test_sigma_principal():
        smax, smin, th = sigma_to_smax_smin_theta([-3], [-1], [0])
        assert np.allclose(smax, [-3])
        assert np.allclose(smin, [-1])
        assert np.allclose(th, [0])
        print("ok test_sigma_principal")

    def test_boundary_elements():
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 2.0, 3.0]))
        b = identify_boundary_elements(g)
        assert b is not None
        assert all(f.size == 1 for f in b.side_faces)
        print("ok test_boundary_elements")

    def test_material_no_controls():
        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        )
        props = SimpleNamespace(young=np.array([1e9]),
                                poisson=np.array([0.25]),
                                rho=np.array([1000.0]))
        funs, n = set_material_fun(g, props, None)
        assert np.array_equal(n, [0, 0, 0])
        assert np.allclose(funs.efun([]), [1e9])
        assert funs.loadfun([]).as_matrix([["cn"], ["d"]]).shape == (8, 3)
        print("ok test_material_no_controls")

    def test_set_bcfun():
        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        )
        binfo = identify_boundary_elements(g)
        bcond = np.array([[0.1, 0.02], [0.01, 0.0], [-0.02, 0.03]])
        bcfun, n = set_bcfun(g, binfo, bcond)
        bc = bcfun(np.linspace(0.2, 0.8, n))
        mat = bc.disp_bc.as_matrix([["n"], ["d"]], True,
                                   [g.nodes.num, g.griddim]).toarray()
        assert n == 2
        assert mat.shape == (g.nodes.num, g.griddim)
        assert np.isnan(mat).any()
        print("ok test_set_bcfun")

    def test_calculate_stress():
        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        )
        binfo = identify_boundary_elements(g)
        bcfun, n = set_bcfun(
            g, binfo,
            np.array([[0.1, 0.02], [0.01, 0.0], [-0.02, 0.03]])
        )
        stress = calculate_stress(
            np.linspace(0.2, 0.8, n),
            np.zeros((g.nodes.num, g.griddim)),
            g,
            bcfun,
            lambda u: np.full(g.cells.num, 1e9),
            lambda u: np.full(g.cells.num, 0.25),
        )
        assert stress.shape == (6 * g.cells.num,)
        assert np.all(np.isfinite(stress))
        print("ok test_calculate_stress")

    def test_gaussian_objfun():
        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        )
        binfo = identify_boundary_elements(g)
        bcfun, n = set_bcfun(
            g, binfo,
            np.array([[0.1, 0.02], [0.01, 0.0], [-0.02, 0.03]])
        )
        u = np.linspace(0.2, 0.8, n)
        efun = lambda uu: np.full(g.cells.num, 1e9)
        nufun = lambda uu: np.full(g.cells.num, 0.25)
        stress = calculate_stress(
            u, np.zeros((g.nodes.num, g.griddim)), g, bcfun, efun, nufun
        ).reshape(-1, 6)
        smax, smin, theta = sigma_to_smax_smin_theta(
            stress[:, 0], stress[:, 1], stress[:, 3]
        )
        data = SimpleNamespace(
            orientation=np.array([[1, theta[0] + 0.01, 0.1]]),
            Smax=np.array([[1, 1.01 * smax[0], 0.1]]),
            Smin=np.array([[1, 0.99 * smin[0], 0.1]]),
            Sv=np.array([[1, stress[0, 2], 0.1]]),
        )
        val, terms, *_ = gaussian_objfun(
            u, np.zeros((g.nodes.num, g.griddim)), None, g, data, bcfun,
            n, [], None, efun, nufun, np.zeros(n), np.ones(n),
            np.zeros(4), ignore_prior=True,
        )
        assert np.isfinite(val)
        assert terms.r_theta.size == 1
        print("ok test_gaussian_objfun")

    test_sigma_principal()
    test_boundary_elements()
    test_material_no_controls()
    test_set_bcfun()
    test_calculate_stress()
    test_gaussian_objfun()
    print("\nAll tests passed!")
