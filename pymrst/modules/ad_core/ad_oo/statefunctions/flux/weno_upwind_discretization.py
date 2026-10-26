"""MRST ``ad-core/statefunctions/flux/upwind/WENOUpwindDiscretization.m``.

Weighted essentially non-oscillatory face values: in place of the upstream
cell's value, a reconstruction at the face centroid from linear interpolants
around the upstream cell, blended by smoothness so that steep interpolants
count for little.

- 1D: the two-point slopes to the left and right neighbour, each weighted by
  `(dx*sigma^2 + 1e-7)^-2`.
- 2D and 3D: around each cell, a Delaunay triangulation of the cell centroid
  and its face neighbours' centroids (boundary faces give reflected ghost
  points, `getBoundaryFacesMerged`), in a frame from the SVD of the cell's
  face centroids scaled by the singular values. Each simplex touching the cell
  gives a gradient; the face value is the cell value plus the
  `(|grad|^2 + 1e-12)^-2 * volume`-weighted average of `grad . dx`. With
  `cap`, a face value that is not positive falls back to the cell value.

The weights are differentiated, as in MRST. Options are MRST's:
`includeBoundary`, `interpolateReference`, `useMatrixProducts`,
`compactStencil` (one face per direction tag on tensor and corner-point
grids) and `cap`. `getSingleWeights` and `getSubsetDiscretization` read
`disc.G` and `disc.N`, which MRST's class does not define, so both fail in
MRST unless those are set; the port keeps that.

`WENOUpwindDiscretization` in `ad_core/statefunctions/flux/upwind/weno.py` is
PyMRST's earlier version for the legacy models, kept for its callers.
"""

from __future__ import annotations

import copy
import time
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.get_boundary_faces_merged import (
    get_boundary_faces_merged,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.upwind_discretization import (
    UpwindDiscretization,
)
from pymrst.utils.adi import value
from pymrst.utils.rldecode import rldecode

SMOOTHNESS_EPS_1D = 1e-7
SMOOTHNESS_EPS_TRI = 1e-12
SMOOTHNESS_POWER = 2
PROGRESS_EVERY = 100
OPTIONS = ("includeBoundary", "interpolateReference", "useMatrixProducts",
           "compactStencil", "cap")


class WENOUpwindDiscretization(UpwindDiscretization):
    """MRST `WENOUpwindDiscretization`."""

    def __init__(self, model: Any, dim: Optional[int] = None, **options: Any):
        super().__init__()
        self.dim = model.G.griddim if dim is None else int(dim)
        self.includeBoundary = True
        self.interpolateReference = True
        self.useMatrixProducts = True
        self.compactStencil = True
        self.cap = True
        self.interp_setup = SimpleNamespace()
        for key, val in options.items():
            if key not in OPTIONS:
                raise ValueError(f"Unknown option {key}")
            setattr(self, key, val)
        self.setup_interpolators(model)

    def face_upstream(self, model: Any, state: Any, flag: Any, cellvalue: Any) -> Any:
        if self.dim == 1:
            return self.get_face_value_1d(model, state, flag, cellvalue)
        if self.dim in (2, 3):
            return self.get_face_value_tri(model, state, flag, cellvalue)[0]
        raise ValueError(f"Dimension {self.dim} too high")

    def setup_interpolators(self, model: Any) -> "WENOUpwindDiscretization":
        if self.dim == 1:
            return self.setup_interpolators_1d(model)
        if self.dim in (2, 3):
            return self.setup_interpolators_tri(model)
        raise ValueError("Dimension too high")

    def get_face_value_tri(self, model: Any, state: Any, flag: Any, cellvalue: Any):
        N = np.asarray(model.operators.N)
        flag = np.asarray(flag, dtype=bool)
        cells = np.where(flag, N[:, 0], N[:, 1])
        internal = np.asarray(model.operators.internalConn)
        fc = np.asarray(model.G.faces.centroids, dtype=float)[internal, :self.dim]
        vq, faces, w = self.interpolate_tri_weno(model, state, cells, fc, cellvalue)
        return vq, faces, w, state

    def interpolate_tri_weno(self, model: Any, state: Any, cells: np.ndarray,
                             points: np.ndarray, q: Any):
        """`(q_interp, pointNo, w)` at `points`, each from its cell in `cells`."""
        setup = self.interp_setup
        cells = np.asarray(cells, dtype=int).ravel()
        nf = points.shape[0]
        tri_ix = _concat([setup.cell_support[c] for c in cells], int)
        linear_weights = _concat([setup.linear_weights[c] for c in cells], float)
        sup_count = setup.cell_support_count[cells]
        centers = rldecode(cells, sup_count)
        point_no = rldecode(np.arange(nf), sup_count)
        scaling = setup.scaling
        scale = scaling.scale[centers, :self.dim]
        points = points[point_no, :]
        dx = points - np.asarray(model.G.cells.centroids, dtype=float)[centers, :self.dim]
        if scaling.mapping is not None and np.size(scaling.mapping):
            dx0 = dx
            dx = np.column_stack([np.sum(scaling.mapping[centers, :, d]*dx0, axis=1)
                                  for d in range(self.dim)])
        dx = dx/scale
        nw = point_no.size
        M = sp.csr_array((np.ones(nw), (point_no, np.arange(nw))), shape=(nf, nw))
        if self.useMatrixProducts:
            sigma = [_matvec(m, q) for m in setup.tri_basis_matrices]
            q_c = q[cells]
            ns = setup.tri_basis_matrices[0].shape[0]
            ss = _vertcat(sigma)
            I = np.repeat(np.arange(nw)[:, None], self.dim, axis=1)
            J = tri_ix[:, None] + np.arange(self.dim)[None, :]*ns
            TF = sp.csr_array((dx.ravel(), (I.ravel(), J.ravel())), shape=(nw, ns*self.dim))
            ws = _matvec(TF, ss)
        else:
            sigma = []
            for d in range(self.dim):
                b = setup.tri_basis[d]
                s = 0
                for l in range(self.dim + 1):
                    ccl = setup.tri_cells[setup.C[:, l]]
                    s = s + b[:, l]*q[ccl]
                sigma.append(s)
            q_c = q[cells]
            ws = 0
            for d in range(self.dim):
                ws = ws + sigma[d][tri_ix]*dx[:, d]
        sigma_norm = 0
        for d in range(self.dim):
            sigma_norm = sigma_norm + sigma[d]**2
        if self.useMatrixProducts:
            Q_tri = sp.csr_array((np.ones(tri_ix.size), (np.arange(tri_ix.size), tri_ix)),
                                 shape=(tri_ix.size, setup.C.shape[0]))
            sigma_norm_tri = _matvec(Q_tri, sigma_norm)
        else:
            sigma_norm_tri = sigma_norm[tri_ix]
        beta = (sigma_norm_tri + SMOOTHNESS_EPS_TRI)**(-SMOOTHNESS_POWER)
        beta = beta*linear_weights
        beta_tot = _matvec(M, beta)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = beta/beta_tot[point_no]
        w = _assign_zero(w, ~np.isfinite(np.asarray(value(w), dtype=float)))
        dvq = _matvec(M, ws*w)
        q_interp = q_c + dvq
        if self.cap:
            bad = np.asarray(value(q_interp), dtype=float) <= 0
            if np.any(bad):
                q_interp = _assign(q_interp, bad, q_c[np.flatnonzero(bad)])
        return q_interp, point_no, w

    def get_face_value_1d(self, model: Any, state: Any, flag: Any, cellvalue: Any) -> Any:
        G = model.G
        x = np.asarray(G.cells.centroids, dtype=float)[:, 0]
        internal = np.asarray(model.operators.internalConn)
        xq = np.asarray(G.faces.centroids, dtype=float)[internal, 0]
        N = np.asarray(model.operators.N)
        mid = np.where(np.asarray(flag, dtype=bool), N[:, 0], N[:, 1])
        left = self.interp_setup.cellNeighbors[mid, 0]
        right = self.interp_setup.cellNeighbors[mid, 1]
        q_L, B_L = self.interpolate_1d(left, mid, x, cellvalue, xq)
        q_R, B_R = self.interpolate_1d(mid, right, x, cellvalue, xq)
        B_T = B_L + B_R
        w_L = B_L/B_T
        w_R = B_R/B_T
        return w_L*q_L + w_R*q_R

    def setup_interpolators_tri(self, model: Any) -> "WENOUpwindDiscretization":
        print("Getting supports.")
        tic = time.perf_counter()
        C, pts, cells, basis, supports, linear_weights, scaling = self.get_triangulation(model)
        print(f"Elapsed time is {time.perf_counter() - tic:.6f} seconds.")
        print("Inverting systems...")
        setup = SimpleNamespace(tri_cells=cells, tri_basis=basis, tri_points=pts,
                                linear_weights=linear_weights, cell_support=supports,
                                scaling=scaling, C=C)
        setup.cell_support_count = np.array([s.size for s in supports], dtype=int)
        n_tri, d_tri = C.shape
        I = np.repeat(np.arange(n_tri)[:, None], d_tri, axis=1)
        J = setup.tri_cells[setup.C]
        setup.tri_basis_matrices = [
            sp.csr_array((basis[d].ravel(), (I.ravel(), J.ravel())),
                         shape=(n_tri, int(model.G.cells.num)))
            for d in range(self.dim)]
        self.interp_setup = setup
        return self

    def get_triangulation(self, model: Any):
        """`(C, pts, cells, grad_basis, supports, linear_weights, scaling)`."""
        from scipy.spatial import Delaunay
        G = model.G
        nc = int(G.cells.num)
        dim = self.dim
        cell_centroids = np.asarray(G.cells.centroids, dtype=float)
        face_centroids = np.asarray(G.faces.centroids, dtype=float)
        pts = cell_centroids[:, :dim]
        if self.includeBoundary:
            bfaces, bcells, bf_cent = get_boundary_faces_merged(G, dim)
            pts = np.vstack([pts, bf_cent])
        else:
            bfaces = bcells = np.zeros(0, dtype=int)
        is_boundary_face = np.zeros(int(G.faces.num), dtype=bool)
        is_boundary_face[bfaces] = True
        N_global = np.asarray(G.faces.neighbors, dtype=int).copy()
        l_out, r_out = N_global[:, 0] < 0, N_global[:, 1] < 0
        N_global[l_out, 0] = N_global[l_out, 1]
        N_global[r_out, 1] = N_global[r_out, 0]
        bnd_map = np.full(int(G.faces.num), -1, dtype=int)
        bnd_map[bfaces] = np.arange(bfaces.size)
        cells = np.concatenate([np.arange(nc), bcells])
        supports: List[np.ndarray] = []
        linear_weights: List[np.ndarray] = []
        coord_mapping = np.zeros((nc, dim, dim))
        min_values = np.zeros((nc, dim))
        ranges = np.ones((nc, dim))
        tri: List[np.ndarray] = []
        gbases: List[List[np.ndarray]] = []
        compact = (getattr(G.cells, "tags", None) is not None and np.size(G.cells.tags) > 0
                   and any(t in ("tensorGrid", "processGRDECL") for t in G.type)
                   and self.compactStencil)
        face_pos = np.asarray(G.cells.facePos, dtype=int)
        cell_faces = np.asarray(G.cells.faces, dtype=int)
        offset = 0
        for i in range(nc):
            if (i + 1) % PROGRESS_EVERY == 0:
                print(f"{i + 1} of {nc}...")
            fpos = np.arange(face_pos[i], face_pos[i + 1])
            local_faces = cell_faces[fpos]
            if compact:
                local_faces = _compact_faces(local_faces, np.asarray(G.cells.tags)[fpos],
                                             np.asarray(G.faces.areas, dtype=float))
            L = N_global[local_faces, 0]
            R = N_global[local_faces, 1]
            neighbors = np.where(L != i, L, R)
            global_cells = np.concatenate([[i], neighbors])
            is_bf_local = np.flatnonzero(is_boundary_face[local_faces]) + 1
            global_cells[is_bf_local] = bnd_map[local_faces[is_bf_local - 1]] + nc
            lpts = pts[global_cells, :]
            fc = face_centroids[local_faces, :]
            min_pts = cell_centroids[i, :]
            _, S, Vt = np.linalg.svd(fc - min_pts)
            rng = S
            M = np.linalg.inv(Vt.T)
            lpts_scaled = project_points(lpts, min_pts, M, rng)
            for d in range(dim):
                coord_mapping[i, :, d] = M[d, :]
            if global_cells.size < dim + 1:
                raise AssertionError(f"cell {i} has too few stencil points")
            gpts = pts[global_cells[1:], :]
            centers = project_points(gpts, min_pts, M, rng)
            fpoints = np.vstack([np.zeros((1, dim)), centers])
            t = Delaunay(fpoints, qhull_options="Qt Qbb Qc").simplices
            T = global_cells[t]
            keep = np.any(T == i, axis=1)
            T, t = T[keep], t[keep]
            if self.interpolateReference:
                ranges[i, :] = rng
                min_values[i, :] = min_pts[:dim]
                pts_basis = lpts_scaled
            else:
                pts_basis = lpts
            gbases.append(self.get_gradient_basis(t, pts_basis)[0])
            vol, _ = self.get_tri_volume(t, pts_basis)
            linear_weights.append(vol)
            tri.append(T)
            supports.append(np.arange(offset, offset + T.shape[0]))
            offset += T.shape[0]
        C = np.vstack(tri)
        grad_basis = [np.vstack([g[d] for g in gbases]) for d in range(dim)]
        scaling = SimpleNamespace(scale=ranges, shift=-min_values, mapping=coord_mapping,
                                  inverses=None)
        return C, pts, cells, grad_basis, supports, linear_weights, scaling

    def setup_interpolators_1d(self, model: Any) -> "WENOUpwindDiscretization":
        G = model.G
        x = np.asarray(G.cells.centroids, dtype=float)[:, 0]
        N = np.asarray(model.operators.N)
        neighbors = np.full((int(G.cells.num), 2), -1, dtype=int)
        internal = np.asarray(model.operators.internalConn)
        xf = np.asarray(G.faces.centroids, dtype=float)[internal, 0]
        L, R = N[:, 0], N[:, 1]
        smaller = xf < x[R]
        neighbors[R[smaller], 0] = L[smaller]
        neighbors[R[~smaller], 1] = L[~smaller]
        smaller = xf < x[L]
        neighbors[L[smaller], 0] = R[smaller]
        neighbors[L[~smaller], 1] = R[~smaller]
        self.interp_setup.cellNeighbors = neighbors
        return self

    def interpolate_1d(self, l: np.ndarray, r: np.ndarray, x: np.ndarray, v: Any,
                       xq: np.ndarray) -> Tuple[Any, Any]:
        """Value at `xq` from the line through cells `l` and `r`, and its
        smoothness. A missing neighbour (-1) takes the other cell."""
        l, r = l.copy(), r.copy()
        l[l == -1] = r[l == -1]
        r[r == -1] = l[r == -1]
        x_r, x_l = x[r], x[l]
        v_l, v_r = v[l], v[r]
        dx = x_r - x_l
        bad = dx == 0
        dx[bad] = 1
        sigma = (v_r - v_l)/dx
        sigma = _assign_zero(sigma, bad)
        vq = v_l + sigma*(xq - x_l)
        smoothness = (dx*sigma**2 + SMOOTHNESS_EPS_1D)**(-SMOOTHNESS_POWER)
        return vq, smoothness

    def get_single_weights(self, tri_num: int, center: int, point: np.ndarray):
        s = self.interp_setup
        tri_pt_num = s.C[tri_num, :]
        cells = s.tri_cells[tri_pt_num]
        dx = np.asarray(self.G.cells.centroids)[center, :] - point
        w = np.zeros(cells.size)
        for i in range(cells.size):
            for d in range(self.dim):
                w[i] += s.tri_basis[d][tri_num, i]*dx[d]
        return w, cells

    def get_gradient_basis(self, C: np.ndarray, pts: np.ndarray):
        """Per simplex, the gradient of each vertex's linear basis function."""
        n_tri, d_tri = C.shape
        basis = [np.zeros((n_tri, d_tri)) for _ in range(self.dim)]
        C_inv: List[Any] = [None]*n_tri
        for i in range(n_tri):
            if np.all(C[i, :] == C[i, 0]):
                bi = np.zeros((self.dim, self.dim + 1))
            else:
                bi, C_inv[i] = get_basis(pts[C[i, :], :], d_tri)
            for j in range(self.dim):
                basis[j][i, :] = bi[j, :]
        return basis, C_inv

    def get_tri_volume(self, C: np.ndarray, pts: np.ndarray):
        p1, p2, p3 = pts[C[:, 0], :], pts[C[:, 1], :], pts[C[:, 2], :]
        v1, v2 = p2 - p1, p3 - p1
        cent = p1 + p2 + p3
        if self.dim == 2:
            z = np.zeros((v1.shape[0], 1))
            vol = 0.5*np.sqrt(np.sum(np.cross(np.hstack([v1, z]), np.hstack([v2, z]))**2,
                                     axis=1))
        else:
            p4 = pts[C[:, 3], :]
            cent = cent + p4
            v3 = p4 - p1
            vol = (1/6)*np.sqrt(np.sum(np.cross(v1, v2)*v3, axis=1)**2)
        return vol, cent/(self.dim + 1)

    def get_subset_discretization(self, cells: Any) -> "WENOUpwindDiscretization":
        disc0 = self
        G = disc0.G
        cells = np.asarray(cells)
        if cells.dtype == bool:
            cells = np.flatnonzero(cells)
        dof_cells = cells
        tri = _concat([disc0.interp_setup.cell_support[c] for c in cells], int)
        supp = disc0.interp_setup.tri_cells[disc0.interp_setup.C[tri, :]]
        cells = np.unique(supp.ravel())
        disc = copy.deepcopy(disc0)
        disc.G.cells.num = cells.size
        disc.G.cells.centroids = disc.G.cells.centroids[cells, :]
        keep = np.zeros(G.cells.num, dtype=bool)
        keep[cells] = True
        renum = np.full(G.cells.num, -1, dtype=int)
        renum[cells] = np.arange(cells.size)
        keep_all = np.zeros(G.cells.num, dtype=bool)
        keep_all[dof_cells] = True
        keep_face = np.all(keep_all[disc.N], axis=1)
        disc.N = renum[disc.N[keep_face, :]]
        s = disc.interp_setup
        idx = np.flatnonzero(keep)
        s.cell_support = [s.cell_support[k] for k in idx]
        s.linear_weights = [s.linear_weights[k] for k in idx]
        s.cell_support_count = s.cell_support_count[keep]
        s.scaling.scale = s.scaling.scale[keep, :]
        s.scaling.shift = s.scaling.shift[keep, :]
        offsets = np.cumsum(np.concatenate([[0], disc0.interp_setup.cell_support_count]))
        tri_keep = np.concatenate([np.arange(offsets[c], offsets[c + 1]) for c in cells])
        tri_renum = np.full(disc0.interp_setup.C.shape[0], -1, dtype=int)
        tri_renum[tri_keep] = np.arange(tri_keep.size)
        s.cell_support = [tri_renum[x] for x in s.cell_support]
        s.C = s.C[tri_keep, :]
        s.tri_basis_matrices = [m.tocsr()[tri_keep, :][:, keep] for m in s.tri_basis_matrices]
        s.tri_basis = [b[tri_keep, :] for b in s.tri_basis]
        s.tri_cells = renum[s.tri_cells]
        disc.interp_setup = s
        return disc

    @staticmethod
    def scale_points(pts: np.ndarray, range_: Optional[np.ndarray] = None,
                     mv: Optional[np.ndarray] = None):
        if mv is None:
            mv = pts.min(axis=0)
            if range_ is None:
                range_ = pts.max(axis=0) - mv
        return (pts - mv)/range_, range_, mv

    faceUpstream = face_upstream
    setupInterpolators = setup_interpolators
    getFaceValueTri = get_face_value_tri
    interpolateTriWENO = interpolate_tri_weno
    getFaceValue1D = get_face_value_1d
    setupInterpolatorsTri = setup_interpolators_tri
    getTriangulation = get_triangulation
    setupInterpolators1D = setup_interpolators_1d
    interpolate1D = interpolate_1d
    getSingleWeights = get_single_weights
    getGradientBasis = get_gradient_basis
    getTriVolume = get_tri_volume
    getSubsetDiscretization = get_subset_discretization
    scalePoints = scale_points


def get_basis(tri_pts: np.ndarray, Tdim: int):
    """MRST's `getBasis`: gradient rows of `inv([pts'; 1])`, non-finite to 0."""
    C = np.vstack([tri_pts.T, np.ones((1, Tdim))])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            C_inv = np.full(C.shape, np.inf)
    C_inv[~np.isfinite(C_inv)] = 0
    return C_inv[:, :Tdim - 1].T, C_inv


def project_points(pts: np.ndarray, center: np.ndarray, M: np.ndarray,
                   range_: np.ndarray) -> np.ndarray:
    """MRST's `projectPoints`: shift by `center`, apply `M`, divide by `range`."""
    pts = pts - center[:pts.shape[1]]
    pts = pts @ M.T
    return pts/range_


def _compact_faces(faces: np.ndarray, tags: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """One face per tag `0..max(tags)` (PyMRST's tags are MRST's minus one),
    the largest; an absent tag takes the first face, as MATLAB's `max` of all
    zeros does."""
    act = np.zeros(faces.size, dtype=bool)
    face_areas = areas[faces]
    for j in range(int(tags.max()) + 1):
        act[np.argmax(face_areas*(tags == j))] = True
    return faces[act]


def _concat(parts: List[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)


def _matvec(A: Any, x: Any) -> Any:
    return x.__rmatmul__(A) if hasattr(x, "jac") else A @ x


def _vertcat(parts: List[Any]) -> Any:
    if any(hasattr(p, "jac") for p in parts):
        from pymrst.utils.adi import vertcat_adi
        return vertcat_adi(*parts)
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def _assign(x: Any, mask: np.ndarray, new: Any) -> Any:
    """`x(mask) = new`, as MATLAB's indexed assignment."""
    if not np.any(mask):
        return x
    if hasattr(x, "jac"):
        x = copy.copy(x)
        x.val = np.array(x.val, copy=True)
        x[np.flatnonzero(mask)] = new
        return x
    x = np.array(x, dtype=float, copy=True)
    x[mask] = value(new)
    return x


def _assign_zero(x: Any, mask: np.ndarray) -> Any:
    """`x(mask) = 0`: the value and its derivative rows zero, even where they
    were not finite."""
    if not np.any(mask):
        return x
    from pymrst.utils.adi.adi import ADI
    if isinstance(x, ADI):
        keep = np.flatnonzero(~mask)
        n = mask.size
        P = sp.csr_array((np.ones(keep.size), (keep, np.arange(keep.size))),
                         shape=(n, keep.size))
        val = np.zeros(n)
        val[keep] = x.val[keep]
        return ADI(val, [sp.csr_array(P @ sp.csr_array(j)[keep, :]) for j in x.jac])
    return _assign(x, mask, 0.0)


__all__ = ["WENOUpwindDiscretization"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.adi import init_variables_adi

    def _model(dims):
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            GenericBlackOilModel,
        )
        from pymrst.modules.ad_props import init_simple_adi_fluid
        from pymrst.params.rock import make_rock
        G = compute_geometry(cart_grid(dims, [float(n) for n in dims]))
        fluid = init_simple_adi_fluid(phases="WO", n=[2, 2], rho=[1000, 800], mu=[1, 5])
        return GenericBlackOilModel(G, make_rock(G, 1, 1), fluid, gas=False)

    def test_1d_linear_field_is_reproduced_inside():
        model = _model([8])
        weno = WENOUpwindDiscretization(model, 1)
        x = model.G.cells.centroids[:, 0]
        flag = np.ones(model.operators.N.shape[0], dtype=bool)
        fv = weno.face_upstream(model, None, flag, 0.1 + 0.05*x)
        xf = model.G.faces.centroids[model.operators.internalConn, 0]
        np.testing.assert_allclose(fv[1:-1], (0.1 + 0.05*xf)[1:-1], rtol=1e-12)
        print("ok test_1d_linear_field_is_reproduced_inside")

    def test_2d_linear_field_is_reproduced_away_from_ghosts():
        import contextlib
        import io
        model = _model([4, 3])
        with contextlib.redirect_stdout(io.StringIO()):
            weno = WENOUpwindDiscretization(model)
        c = model.G.cells.centroids
        q = 1 + 0.1*c[:, 0] + 0.2*c[:, 1]
        flag = np.zeros(model.operators.N.shape[0], dtype=bool)
        fv = weno.face_upstream(model, None, flag, q)
        fc = model.G.faces.centroids[model.operators.internalConn]
        _, bcells, _ = get_boundary_faces_merged(model.G)
        upstream = model.operators.N[:, 1]
        inner = ~np.isin(upstream, bcells)
        assert inner.sum() > 0
        np.testing.assert_allclose(fv[inner], (1 + 0.1*fc[:, 0] + 0.2*fc[:, 1])[inner],
                                   rtol=1e-12)
        print("ok test_2d_linear_field_is_reproduced_away_from_ghosts")

    def test_derivatives_match_finite_differences():
        import contextlib
        import io
        model = _model([4, 3])
        with contextlib.redirect_stdout(io.StringIO()):
            weno = WENOUpwindDiscretization(model)
        rng = np.random.default_rng(0)
        q0 = rng.uniform(0.2, 0.8, model.G.cells.num)
        flag = rng.uniform(size=model.operators.N.shape[0]) > 0.5
        (qad,) = init_variables_adi(q0)
        J = weno.face_upstream(model, None, flag, qad).jac[0].toarray()
        h = 1e-7
        for k in range(0, q0.size, 3):
            qp = q0.copy()
            qp[k] += h
            fd = (weno.face_upstream(model, None, flag, qp)
                  - weno.face_upstream(model, None, flag, q0))/h
            np.testing.assert_allclose(J[:, k], fd, atol=1e-5)
        print("ok test_derivatives_match_finite_differences")

    def test_compact_faces_takes_first_face_for_absent_tag():
        faces = np.array([10, 11, 12])
        tags = np.array([0, 2, 2])
        areas = np.zeros(20)
        areas[[10, 11, 12]] = [1.0, 2.0, 5.0]
        assert list(_compact_faces(faces, tags, areas)) == [10, 12]
        print("ok test_compact_faces_takes_first_face_for_absent_tag")

    def test_unknown_option_is_refused():
        try:
            WENOUpwindDiscretization(_model([4]), 1, bogus=True)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
        print("ok test_unknown_option_is_refused")

    test_1d_linear_field_is_reproduced_inside()
    test_2d_linear_field_is_reproduced_away_from_ghosts()
    test_derivatives_match_finite_differences()
    test_compact_faces_takes_first_face_for_absent_tag()
    test_unknown_option_is_refused()
    print("\nAll tests passed!")
