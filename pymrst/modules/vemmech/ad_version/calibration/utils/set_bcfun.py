"""MRST ``set_bcfun.m`` calibration boundary-condition builder."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.vemmech.ad_version import SparseMultiArray
from pymrst.utils.adi import ADI


@dataclass
class ElasticCalibrationBC:
    disp_bc: SparseMultiArray
    force_bc: None = None
    disp_bc_jac: sp.csr_array | None = None


def _as_1based_nodes(nodes):
    return np.asarray(nodes, dtype=np.int64).ravel() + 1


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


def set_bcfun(g, binfo, bcond):
    """Return the mechanics calibration boundary-condition function."""
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
        uu = np.asarray(u.val if isinstance(u, ADI) else u, dtype=float).ravel()
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


__all__ = ["ElasticCalibrationBC", "set_bcfun"]
