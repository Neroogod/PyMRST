"""WENO upwind discretization (MRST ad-core/statefunctions/flux/upwind).

Ports MRST's `WENOUpwindDiscretizationAD.m` - the weighted
essentially-non-oscillatory reconstruction of a cell field onto the interior
faces. The one-dimensional path comes first, the triangulated 2-D/3-D one
after it. Where single-point upwind takes the raw upstream cell value,
WENO reconstructs a higher-order face value from two linear stencils (one biased
to each side of the upstream cell) and blends them by a smoothness indicator:
in smooth regions both stencils contribute and the scheme is second order; near
a shock the oscillatory stencil is down-weighted, keeping the front monotone.

For each interior face with upstream cell ``mid`` (chosen by the upwind flag)
and its ordered left/right neighbours:

    q_L, B_L = linear reconstruction from (left, mid)   evaluated at the face
    q_R, B_R = linear reconstruction from (mid, right)  evaluated at the face
    w_L = B_L/(B_L+B_R),  w_R = B_R/(B_L+B_R)
    q_face = w_L q_L + w_R q_R

with the smoothness weight ``B = (dx * sigma^2 + E)^(-L)`` (E=1e-7, L=2), where
``sigma`` is the stencil slope. This matches MRST's `getFaceValue1D` /
`interpolate1D`, validated against a MATLAB dump in `validate_phase348.py`.

The 2D and 3D scheme is a different construction and is ported below it. There
is no directional stencil to walk on an unstructured grid, so MRST builds a
*local triangulation* around each cell instead: take the cell centroid and its
face neighbours' centroids, tetrahedralise them, and each simplex touching the
cell carries one linear reconstruction. The gradient of that reconstruction is
exact on the simplex, so evaluating it at the face centroid gives a candidate
face value, and the candidates are blended by the same idea as in 1D - a
smoothness weight, here `(|grad q|^2 + E)^(-L)`, times the simplex volume.

Three details in that construction are load-bearing and none is obvious:

* **the local coordinate system.** Each cell's points are projected through
  the SVD of its own face centroids and divided by the singular values. On a
  high-aspect-ratio cell - a metre thick and a hundred metres wide, which is
  most reservoir grids - the raw coordinates make the simplices numerically
  degenerate, and the gradient basis comes from inverting a matrix built from
  exactly those coordinates;
* **the volume weight.** Simplices are not equal witnesses. A sliver spanning
  two nearly-collinear neighbours gives a gradient dominated by rounding, and
  weighting by volume is what stops it from voting;
* **the cap.** A reconstruction can undershoot to a negative saturation, and a
  negative saturation reaches a relperm table that has no value there. MRST
  falls back to the plain upstream value cell by cell rather than clipping,
  because clipping invents a value the scheme never computed.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.statefunctions.flux.upwind.boundary_faces import (
    get_boundary_faces_merged_ad,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.upwind_discretization import (  # noqa: E501
    UpwindDiscretization,
)
from pymrst.utils.rldecode import rldecode

_E = 1e-7
_L = 2

#: The multi-dimensional smoothness floor. MRST uses 1e-12 here against 1e-7 in
#: 1D, and the difference is not a typo in either place: the 1D indicator adds
#: `dx*sigma^2` in the field's own units, the multi-D one adds `|grad q|^2` in
#: scaled coordinates where lengths are O(1). The floor only has to keep the
#: `(...)^(-2)` finite where the gradient vanishes, and it is subtracted from
#: nothing, so a smaller one is a weaker perturbation of a smooth region.
_E_TRI = 1e-12


class WENOUpwindDiscretizationAD(UpwindDiscretization):
    """WENO face reconstruction in one, two or three dimensions."""

    def __init__(self, model: Any = None, dim: Optional[int] = None,
                 include_boundary: bool = True,
                 interpolate_reference: bool = True,
                 use_matrix_products: bool = True,
                 compact_stencil: bool = True,
                 cap: bool = True):
        super().__init__(model)
        if dim is None:
            # MRST's one-argument constructor takes the dimension from the
            # grid. Defaulting to 1 instead would silently give a 3-D model a
            # scheme that reconstructs from the x-coordinate alone.
            dim = int(model.G.griddim) if model is not None else 1
        dim = int(dim)
        # A `raise`, not an `assert`. Assertions are stripped under `python -O`,
        # and this one guards *numerical correctness* rather than a programming
        # error. Verified: under `-O` an earlier assertion vanished and a 3-D
        # grid was accepted by the 1-D setup, which reconstructs faces from the
        # x-coordinate of the cell centroids alone, ignoring y and z - a
        # plausible field and a wrong one.
        if dim not in (1, 2, 3):
            raise NotImplementedError(
                f"WENO is defined for one, two or three dimensions; got "
                f"dim={dim}. MRST raises 'Dimension too high' here for the "
                f"same reason: the triangulated stencil is built by "
                f"tetrahedralising a point cloud, and there is no 4-simplex "
                f"construction behind it.")
        self.dim = dim
        self.includeBoundary = bool(include_boundary)
        self.interpolateReference = bool(interpolate_reference)
        self.useMatrixProducts = bool(use_matrix_products)
        self.compactStencil = bool(compact_stencil)
        self.cap = bool(cap)

        self._x: Optional[np.ndarray] = None
        self._xf: Optional[np.ndarray] = None
        self._neighbors: Optional[np.ndarray] = None   # [left, right] per cell
        self.interp_setup = SimpleNamespace()
        if model is not None:
            self.setup_interpolators(model)

    # -- dispatch ----------------------------------------------------------

    def setup_interpolators(self, model: Any) -> "WENOUpwindDiscretizationAD":
        """Build whichever interpolator the dimension calls for."""
        if self.dim == 1:
            self._setup_1d(model)
        else:
            self.setup_interpolators_tri(model)
        return self

    setupInterpolators = setup_interpolators

    def _setup_1d(self, model: Any) -> None:
        g = model.G
        x = np.asarray(g.cells.centroids[:, 0], dtype=float)
        n = model.operators.N
        internal = np.all(g.faces.neighbors >= 0, axis=1)
        xf = np.asarray(g.faces.centroids[internal, 0], dtype=float)

        # cellNeighbors[cell, {0:left, 1:right}] - the neighbour on each side,
        # ordered by x. -1 marks "no neighbour" (a boundary). MRST uses 0 as
        # that marker because its cell indices are 1-based; in PyMRST's 0-based
        # scheme cell 0 is a real cell, so a distinct sentinel is required.
        neighbors = np.full((g.cells.num, 2), -1, dtype=int)
        left, right = n[:, 0], n[:, 1]
        # for the right cell R of each face, its neighbour L is on the smaller
        # side if the face lies below R's centre
        smaller = xf < x[right]
        neighbors[right[smaller], 0] = left[smaller]
        neighbors[right[~smaller], 1] = left[~smaller]
        smaller = xf < x[left]
        neighbors[left[smaller], 0] = right[smaller]
        neighbors[left[~smaller], 1] = right[~smaller]

        self._x, self._xf, self._neighbors = x, xf, neighbors

    def _interpolate_1d(self, l: np.ndarray, r: np.ndarray, v: Any,
                        xq: np.ndarray):
        """Linear reconstruction from cells (l, r) at face point xq, and the
        smoothness weight. Mirrors MRST's `interpolate1D`."""
        l = l.copy()
        r = r.copy()
        # constant extrapolation at boundaries (a -1 index means "no cell")
        l[l == -1] = r[l == -1]
        r[r == -1] = l[r == -1]

        x = self._x
        x_l, x_r = x[l], x[r]
        dx = x_r - x_l
        bad = dx == 0
        dx = np.where(bad, 1.0, dx)

        v_l = v[l]
        v_r = v[r]
        sigma = (v_r - v_l) / dx
        if np.any(bad):
            sigma = sigma * (~bad)                     # AD-safe zero on bad
        vq = v_l + sigma * (xq - x_l)

        sigma_v = np.asarray(_value(sigma), dtype=float)
        smoothness = (dx * sigma_v ** 2 + _E) ** (-_L)
        return vq, smoothness

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cell_values: Any) -> Any:
        if self.dim != 1:
            return self.get_face_value_tri(model, state, flag, cell_values)
        return self.get_face_value_1d(model, state, flag, cell_values)

    faceUpstream = face_upstream

    def get_face_value_1d(self, model: Any, state: Any, flag: Any,
                          cell_values: Any) -> Any:
        n = model.operators.N
        flag = np.asarray(flag, dtype=bool)
        mid = np.where(flag, n[:, 0], n[:, 1])

        left = self._neighbors[mid, 0]
        right = self._neighbors[mid, 1]
        xq = self._xf

        q_l, b_l = self._interpolate_1d(left, mid, cell_values, xq)
        q_r, b_r = self._interpolate_1d(mid, right, cell_values, xq)

        b_t = b_l + b_r
        w_l = b_l / b_t
        w_r = b_r / b_t
        return w_l * q_l + w_r * q_r

    getFaceValue1D = get_face_value_1d
    interpolate1D = _interpolate_1d
    setupInterpolators1D = _setup_1d

    # ======================================================================
    # The triangulated scheme, 2-D and 3-D.
    # ======================================================================

    def get_face_value_tri(self, model: Any, state: Any, flag: Any,
                           cell_values: Any):
        """Reconstruct at every interior face centroid from its upstream cell."""
        n = np.asarray(model.operators.N)
        flag = np.asarray(flag, dtype=bool)
        cells = np.where(flag, n[:, 0], n[:, 1])

        internal = _internal_connections(model)
        fc = np.asarray(model.G.faces.centroids)[internal, :self.dim]
        vq, _, _ = self.interpolate_tri_weno(model, state, cells, fc,
                                             cell_values)
        return vq

    getFaceValueTri = get_face_value_tri

    def interpolate_tri_weno(self, model: Any, state: Any, cells: np.ndarray,
                             points: np.ndarray, q: Any):
        """WENO reconstruction of `q` at `points`, from each point's cell.

        Returns `(q_interp, point_no, w)`. `point_no` and `w` are the simplex
        -> point map and the blend weights, returned because
        `getSingleWeights` and the diagnostics want to see them.
        """
        setup = self.interp_setup
        cells = np.asarray(cells, dtype=int).ravel()
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n_points = points.shape[0]

        support = setup.cell_support
        tri_ix = np.concatenate([support[c] for c in cells]) \
            if cells.size else np.zeros(0, dtype=int)
        linear_weights = np.concatenate(
            [setup.linear_weights[c] for c in cells]) \
            if cells.size else np.zeros(0)
        counts = setup.cell_support_count[cells]

        # One row per (point, simplex) pair. `centers` is the cell that owns
        # the simplex, `point_no` the point being reconstructed.
        centers = rldecode(cells, counts)
        point_no = rldecode(np.arange(n_points), counts)

        dx = points[point_no] - np.asarray(
            model.G.cells.centroids)[centers, :self.dim]
        mapping = setup.scaling.mapping
        if mapping is not None:
            # Rotate into the cell's own frame before scaling. `mapping[i]` is
            # the SVD basis of cell i's face centroids, so this is the same
            # projection the basis was built in - applying the scale without
            # the rotation would divide the wrong axes.
            dx = np.einsum("ndj,nj->nd", mapping[centers], dx)
        dx = dx/setup.scaling.scale[centers, :self.dim]

        # sigma[d] is the reconstructed gradient in direction d, per simplex.
        # `q.__rmatmul__(A)` rather than `A @ q`: scipy's sparse `@` raises on
        # an operand it does not recognise instead of returning
        # NotImplemented, so Python never offers ADI the reflected operation.
        sigma = [q.__rmatmul__(setup.tri_basis_matrices[d])
                 if hasattr(q, "jac")
                 else setup.tri_basis_matrices[d] @ q
                 for d in range(self.dim)]

        q_c = q[cells]

        # The directional derivative from each simplex, evaluated at the point.
        ws = None
        for d in range(self.dim):
            term = sigma[d][tri_ix]*dx[:, d]
            ws = term if ws is None else ws + term

        # Smoothness: a steep gradient is a small weight.
        #
        # These stay differentiable. Freezing them at their current values -
        # computing the weights from `value(sigma)` - is tempting, because the
        # weights are a limiter and differentiating a limiter makes a stiffer
        # Jacobian. It also produces an analytic Jacobian that disagrees with
        # its own function: measured here at 49% relative error against a
        # finite difference, which a Newton solve then pays for in iterations
        # while the residual gives no sign anything is wrong. MRST
        # differentiates them and so does this.
        sigma_norm = None
        for d in range(self.dim):
            sq = sigma[d]*sigma[d]
            sigma_norm = sq if sigma_norm is None else sigma_norm + sq
        beta = (sigma_norm[tri_ix] + _E_TRI)**(-_L)
        beta = beta*linear_weights

        # Normalise per point: `M` sums the simplices belonging to each point.
        M = sp.csr_array((np.ones(point_no.size),
                          (point_no, np.arange(point_no.size))),
                         shape=(n_points, point_no.size))
        beta_tot = beta.__rmatmul__(M) if hasattr(beta, "jac") else M @ beta
        with np.errstate(divide="ignore", invalid="ignore"):
            w = beta/beta_tot[point_no]
        # A point whose weights all underflow gives 0/0. Masking by
        # multiplication rather than assignment keeps the Jacobian rows in
        # step with the values - `w[bad] = 0` would zero a value and leave its
        # derivative behind.
        finite = np.isfinite(np.asarray(_value(w), dtype=float)).astype(float)
        if not finite.all():
            w = w*finite

        scaled = ws*w
        dvq = scaled.__rmatmul__(M) if hasattr(scaled, "jac") else M @ scaled
        q_interp = q_c + dvq

        if self.cap:
            # A reconstruction that undershoots to <= 0 is replaced by the
            # plain upstream value, not clipped to zero: clipping invents a
            # value the scheme never produced, and zero saturation is just as
            # far outside the relperm table as a negative one.
            bad = np.asarray(_value(q_interp), dtype=float) <= 0
            if np.any(bad):
                q_interp = _where_ad(bad, q_c, q_interp)
        return q_interp, point_no, w

    interpolateTriWENO = interpolate_tri_weno

    def setup_interpolators_tri(self, model: Any) -> "WENOUpwindDiscretizationAD":
        """Build and store the per-cell triangulation and gradient bases."""
        C, pts, cells, basis, supports, linear_weights, scaling = \
            self.get_triangulation(model)

        counts = np.array([s.size for s in supports], dtype=int)
        setup = SimpleNamespace(
            tri_cells=cells, tri_basis=basis, tri_points=pts,
            linear_weights=linear_weights, cell_support=supports,
            scaling=scaling, C=C, cell_support_count=counts)

        # One sparse matrix per direction, mapping cell values straight to
        # per-simplex gradients. MRST calls this `useMatrixProducts` and it is
        # not only a speed choice: it is what keeps the operation a single
        # matrix product, so an AD vector passes through it in one step
        # instead of `dim*(dim+1)` indexed accumulations.
        n_tri, d_tri = C.shape
        rows = np.repeat(np.arange(n_tri), d_tri)
        cols = cells[C].ravel()
        setup.tri_basis_matrices = [
            sp.csr_array((basis[d].ravel(), (rows, cols)),
                         shape=(n_tri, int(model.G.cells.num)))
            for d in range(self.dim)]

        self.interp_setup = setup
        return self

    setupInterpolatorsTri = setup_interpolators_tri

    def get_triangulation(self, model: Any):
        """Per-cell local triangulation, gradient bases and volume weights.

        Returns `(C, pts, cells, grad_basis, supports, linear_weights,
        scaling)`, matching MRST's output order.

        `pts` holds one point per cell centroid followed by one *mirrored*
        centroid per boundary face. The mirror is what lets a boundary cell
        have a full simplex around it; without it the cell's point cloud is
        one-sided, the triangulation is degenerate, and the reconstruction
        there silently drops to first order - which is exactly where a
        saturation front usually is.
        """
        G = model.G
        nc = int(G.cells.num)
        dim = self.dim
        centroids = np.asarray(G.cells.centroids, dtype=float)
        pts = centroids[:, :dim].copy()

        if self.includeBoundary:
            bfaces, bcells, bfc = get_boundary_faces_merged_ad(G, dim)
            pts = np.vstack([pts, np.asarray(bfc, dtype=float)[:, :dim]])
        else:
            bfaces = np.zeros(0, dtype=int)
            bcells = np.zeros(0, dtype=int)

        is_boundary_face = np.zeros(int(G.faces.num), dtype=bool)
        is_boundary_face[bfaces] = True
        bnd_map = np.full(int(G.faces.num), -1, dtype=int)
        bnd_map[bfaces] = np.arange(bfaces.size)

        # Collapse the outside neighbour onto the interior one, so that
        # reading either column of a boundary face gives a real cell.
        neighbors = np.asarray(G.faces.neighbors, dtype=int).copy()
        outside = neighbors < 0
        neighbors[outside[:, 0], 0] = neighbors[outside[:, 0], 1]
        neighbors[outside[:, 1], 1] = neighbors[outside[:, 1], 0]

        cells = np.concatenate([np.arange(nc), np.asarray(bcells, dtype=int)])
        supports: List[np.ndarray] = []
        linear_weights: List[np.ndarray] = []
        tri: List[np.ndarray] = []
        gbases: List[List[np.ndarray]] = []
        coord_mapping = np.zeros((nc, dim, dim))
        ranges = np.ones((nc, dim))
        min_values = np.zeros((nc, dim))

        face_pos = np.asarray(G.cells.facePos, dtype=int)
        cell_faces = np.asarray(G.cells.faces, dtype=int)
        if cell_faces.ndim > 1:
            face_tags = cell_faces[:, 1] if cell_faces.shape[1] > 1 else None
            cell_faces = cell_faces[:, 0]
        else:
            face_tags = None

        offset = 0
        degenerate = 0
        for i in range(nc):
            fpos = slice(face_pos[i], face_pos[i + 1])
            local_faces = cell_faces[fpos]
            if face_tags is not None and self.compactStencil:
                local_faces = _compact_stencil(local_faces, face_tags[fpos],
                                               np.asarray(G.faces.areas))

            L = neighbors[local_faces, 0]
            R = neighbors[local_faces, 1]
            others = np.where(L != i, L, R)
            global_cells = np.concatenate([[i], others])

            # A boundary face contributes its mirrored point, not its cell -
            # which would be `i` itself and add nothing.
            local_bf = np.flatnonzero(is_boundary_face[local_faces]) + 1
            if local_bf.size:
                global_cells[local_bf] = nc + bnd_map[local_faces[local_bf - 1]]

            lpts = pts[global_cells, :]
            centre = centroids[i:i + 1, :dim]

            # The local frame: the SVD of this cell's face centroids about its
            # own centroid. `range` is the singular values, so dividing by it
            # makes the cell roughly isotropic before anything is inverted.
            fc = np.asarray(G.faces.centroids, dtype=float)[local_faces, :dim]
            _, S, Vt = np.linalg.svd(fc - centre, full_matrices=False)
            rng = S.copy()
            rng[rng == 0] = 1.0
            M = np.linalg.inv(Vt.T) if Vt.shape[0] == dim else np.eye(dim)
            lpts_scaled = project_points(lpts, centre, M, rng)

            coord_mapping[i] = M
            ranges[i, :] = rng
            min_values[i, :] = centre.ravel()

            if global_cells.size < dim + 1:
                # MRST asserts here. Too few neighbours cannot span a simplex,
                # and there is nothing to reconstruct from.
                raise ValueError(
                    f"cell {i} has {global_cells.size} stencil points, fewer "
                    f"than the {dim + 1} needed to span a {dim}-D simplex")

            t = _delaunay(lpts_scaled)
            if t.size == 0:
                degenerate += 1
                tri.append(np.zeros((0, dim + 1), dtype=int))
                gbases.append([np.zeros((0, dim + 1)) for _ in range(dim)])
                linear_weights.append(np.zeros(0))
                supports.append(np.zeros(0, dtype=int))
                continue

            T = global_cells[t]
            # Keep only simplices touching this cell. A simplex made of
            # neighbours alone describes a gradient somewhere else and would
            # blend in a reconstruction the cell never sees.
            #
            # Measured inert for this stencil: across graded, sheared and
            # rotated grids in 2-D and 3-D, with and without ghost points,
            # Qhull produced 100/64/512/216 simplices and *none* excluded the
            # centre - the cell centroid is interior to its own face
            # neighbours' hull, so every simplex touches it. It is kept for
            # MRST fidelity and because it stops being inert the moment a
            # wider stencil is used (MRST offers `neighboursByNodes`), but no
            # test can catch its removal today and none pretends to.
            keep = np.any(t == 0, axis=1)
            t, T = t[keep], T[keep]

            pts_basis = lpts_scaled if self.interpolateReference else lpts
            b = self.get_gradient_basis(t, pts_basis)
            vol, _ = self.get_tri_volume(t, pts_basis)

            gbases.append(b)
            linear_weights.append(vol)
            tri.append(T)
            supports.append(np.arange(offset, offset + T.shape[0]))
            offset += T.shape[0]

        if degenerate:
            # Loud, because the alternative is a scheme that quietly reports
            # first-order answers under a second-order name.
            warnings.warn(
                f"WENO: {degenerate} of {nc} cells produced a degenerate "
                f"triangulation and fall back to first-order upwind there. "
                f"This is usually a collapsed or extremely flat cell.",
                RuntimeWarning, stacklevel=2)

        C = (np.vstack(tri) if tri else np.zeros((0, dim + 1), dtype=int))
        grad_basis = [np.vstack([g[d] for g in gbases]) if gbases
                      else np.zeros((0, dim + 1)) for d in range(dim)]
        scaling = SimpleNamespace(scale=ranges, shift=-min_values,
                                  mapping=coord_mapping)
        return C, pts, cells, grad_basis, supports, linear_weights, scaling

    getTriangulation = get_triangulation

    def get_gradient_basis(self, C: np.ndarray, pts: np.ndarray):
        r"""Gradients of the linear basis functions on each simplex.

        For a simplex with vertices `p_l` and a linear `q(x) = g.x + c`,
        `q(p_l) = [g; c] . [p_l; 1]`, so stacking `[p_l; 1]` as the columns of
        a square `A` gives `[g; c] = A^-T q`. The first `dim` rows of that are
        the gradient, and `basis[d][i, l]` is the coefficient of `q_l` in
        direction `d` on simplex `i`.

        A degenerate simplex gives a singular `A`. MATLAB's `inv` returns Inf
        there and MRST zeroes the non-finite entries; a zero row means that
        simplex contributes no gradient, which is the right answer - its
        vertices carry no directional information.
        """
        C = np.atleast_2d(np.asarray(C, dtype=int))
        n_tri, d_tri = C.shape
        basis = [np.zeros((n_tri, d_tri)) for _ in range(self.dim)]

        for i in range(n_tri):
            if np.all(C[i, :] == C[i, 0]):
                continue                      # an "upwind triangle": no span
            bi = get_basis(pts[C[i, :], :], d_tri)
            for d in range(self.dim):
                basis[d][i, :] = bi[d, :]
        return basis

    getGradientBasis = get_gradient_basis

    def get_tri_volume(self, C: np.ndarray, pts: np.ndarray):
        """Simplex volumes (areas in 2-D) and centroids.

        The volume is the linear weight: a sliver spanning two nearly
        collinear neighbours produces a gradient made of rounding error, and
        weighting by volume is what keeps it from voting.
        """
        C = np.atleast_2d(np.asarray(C, dtype=int))
        p1, p2, p3 = pts[C[:, 0]], pts[C[:, 1]], pts[C[:, 2]]
        v1, v2 = p2 - p1, p3 - p1
        cent = p1 + p2 + p3
        if self.dim == 2:
            z = np.zeros((v1.shape[0], 1))
            vol = 0.5*np.linalg.norm(
                np.cross(np.hstack([v1, z]), np.hstack([v2, z])), axis=1)
        else:
            p4 = pts[C[:, 3]]
            cent = cent + p4
            vol = np.abs(np.sum(np.cross(v1, v2)*(p4 - p1), axis=1))/6.0
        return vol, cent/(self.dim + 1)

    getTriVolume = get_tri_volume

    def get_single_weights(self, model: Any, tri_num: int, center: int,
                           point: np.ndarray):
        """The weight each cell of one simplex contributes at one point.

        A diagnostic: it answers "where did this face value come from", which
        is otherwise invisible once everything is a matrix product.
        """
        setup = self.interp_setup
        tri_num = int(np.asarray(tri_num).ravel()[0])
        tri_pt_num = setup.C[tri_num, :]
        cells = setup.tri_cells[tri_pt_num]
        dx = (np.asarray(model.G.cells.centroids)[center, :self.dim]
              - np.asarray(point, dtype=float).ravel()[:self.dim])

        w = np.zeros(cells.size)
        for i in range(cells.size):
            for d in range(self.dim):
                w[i] += setup.tri_basis[d][tri_num, i]*dx[d]
        return w, cells

    getSingleWeights = get_single_weights

    def get_subset_discretization(self, model: Any, cells) -> "WENOUpwindDiscretizationAD":
        """A discretization restricted to `cells` and their support.

        Used by the local/adaptive solvers, which assemble on a handful of
        cells at a time. Everything is renumbered into the subset's own
        indexing, and the *support* cells are kept as well as the requested
        ones - a reconstruction on a cell reads its neighbours, so dropping
        them would change the answer rather than just the cost.
        """
        import copy

        cells = np.asarray(cells)
        if cells.dtype == bool:
            cells = np.flatnonzero(cells)
        cells = cells.astype(int)

        setup = self.interp_setup
        nc = setup.scaling.scale.shape[0]

        tri = np.concatenate([setup.cell_support[c] for c in cells]) \
            if cells.size else np.zeros(0, dtype=int)
        support_cells = np.unique(setup.tri_cells[setup.C[tri, :]].ravel())

        renum = np.full(nc, -1, dtype=int)
        renum[support_cells] = np.arange(support_cells.size)

        tri_renum = np.full(setup.C.shape[0], -1, dtype=int)
        tri_renum[tri] = np.arange(tri.size)

        new = copy.copy(self)
        new_setup = SimpleNamespace(
            tri_cells=renum[setup.tri_cells],
            tri_points=setup.tri_points,
            C=setup.C[tri, :],
            cell_support=[tri_renum[setup.cell_support[c]] for c in cells],
            linear_weights=[setup.linear_weights[c] for c in cells],
            cell_support_count=setup.cell_support_count[cells],
            scaling=SimpleNamespace(scale=setup.scaling.scale[cells, :],
                                    shift=setup.scaling.shift[cells, :],
                                    mapping=setup.scaling.mapping[cells]),
            tri_basis=[b[tri, :] for b in setup.tri_basis],
            tri_basis_matrices=[m.tocsr()[tri, :][:, support_cells]
                                for m in setup.tri_basis_matrices],
        )
        new.interp_setup = new_setup
        return new

    getSubsetDiscretization = get_subset_discretization

    @staticmethod
    def scale_points(pts: np.ndarray, rng: Optional[np.ndarray] = None,
                     mv: Optional[np.ndarray] = None):
        """Map points into the unit box (MRST `scalePoints`)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if mv is None:
            mv = pts.min(axis=0)
        if rng is None:
            rng = pts.max(axis=0) - mv
        rng = np.asarray(rng, dtype=float).copy()
        rng[rng == 0] = 1.0
        return (pts - mv)/rng, rng, mv

    scalePoints = scale_points


def _value(x: Any) -> Any:
    return x.val if hasattr(x, "val") else x


def get_basis(tri_pts: np.ndarray, t_dim: int) -> np.ndarray:
    """MRST's file-local `getBasis`: gradient coefficients on one simplex.

    `A` has one column per vertex, `[p_l; 1]`. Its inverse's first `dim`
    columns, transposed, are the gradient coefficients. A singular `A` means
    the vertices do not span the space; MATLAB's `inv` yields Inf there and
    MRST zeroes it, so a `LinAlgError` maps to the same all-zero basis.
    """
    tri_pts = np.atleast_2d(np.asarray(tri_pts, dtype=float))
    A = np.vstack([tri_pts.T, np.ones((1, t_dim))])
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.zeros((t_dim - 1, t_dim))
    A_inv[~np.isfinite(A_inv)] = 0.0
    return A_inv[:, :t_dim - 1].T


def project_points(pts: np.ndarray, center: np.ndarray, M: np.ndarray,
                   rng: np.ndarray) -> np.ndarray:
    """Shift to `center`, rotate by `M`, then divide by `rng`.

    The order matters and is MRST's: rotating after scaling would mix axes
    that have already been stretched by different amounts, and the resulting
    frame would not be the SVD frame the singular values belong to.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    shifted = pts - np.asarray(center, dtype=float).reshape(1, -1)
    return (shifted @ np.asarray(M, dtype=float).T)/np.asarray(rng, dtype=float)


def _internal_connections(model: Any) -> np.ndarray:
    """The internal-face mask, however the operators spell it."""
    ops = model.operators
    internal = getattr(ops, "internalConn", None)
    if internal is None and isinstance(ops, dict):
        internal = ops.get("internalConn")
    if internal is None:
        return np.all(np.asarray(model.G.faces.neighbors) >= 0, axis=1)
    return np.asarray(internal)


def _compact_stencil(faces: np.ndarray, tags: np.ndarray,
                     areas: np.ndarray) -> np.ndarray:
    """One face per cardinal direction, the largest by area.

    On a corner-point grid a cell can meet several cells across what is
    logically one direction. Taking all of them biases the stencil towards
    whichever side happens to be more finely gridded; MRST keeps the largest
    face per tag so each direction gets one vote.

    On a tensor grid every cell already has exactly one face per tag, so this
    is an identity there - which is why it cannot be tested on a Cartesian
    grid and is tested directly instead.
    """
    keep = []
    for tag in np.unique(tags):
        same = np.flatnonzero(tags == tag)
        keep.append(same[np.argmax(areas[faces[same]])])
    return faces[np.sort(np.asarray(keep, dtype=int))]


def _delaunay(points: np.ndarray) -> np.ndarray:
    """Delaunay simplices, or an empty array when the cloud is degenerate.

    MRST's `delaunayn` is Qhull, and so is this. The joggle/`QJ` options are
    deliberately not used: they perturb the input to force a triangulation,
    which turns "these points are coplanar" into a set of near-zero-volume
    simplices that then receive near-zero linear weights anyway. Returning
    nothing is the same answer, said out loud.
    """
    from scipy.spatial import Delaunay
    from scipy.spatial import QhullError

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[0] < points.shape[1] + 1:
        return np.zeros((0, points.shape[1] + 1), dtype=int)
    try:
        return Delaunay(points, qhull_options="Qt Qbb Qc Qz").simplices
    except (QhullError, ValueError):
        return np.zeros((0, points.shape[1] + 1), dtype=int)


def _where_ad(mask: np.ndarray, a: Any, b: Any) -> Any:
    """`np.where(mask, a, b)` that survives AD on either operand.

    Written arithmetically rather than by assignment because an ADI object's
    Jacobian rows have to be selected as well as its values, and multiplying
    by a 0/1 mask does both in one expression.
    """
    keep = (~mask).astype(float)
    take = mask.astype(float)
    return a*take + b*keep



__all__ = ["WENOUpwindDiscretizationAD"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi, value

    def _model(nx=20):
        g = compute_geometry(cart_grid([nx, 1, 1], [float(nx), 1.0, 1.0]))
        rock = make_rock(g, 1e-13, 1.0)
        ops = setup_operators_tpfa(g, rock)
        return type("M", (), {"G": g, "operators": ops})(), g

    def test_smooth_field_is_second_order_exact():
        """A linear field is reconstructed exactly on interior faces, where
        both stencils are full (the two boundary-adjacent faces have a
        one-sided stencil and are handled by the smoothness weighting, exactly
        as MRST does - see the MATLAB cross-check in validate_phase348)."""
        model, g = _model()
        weno = WENOUpwindDiscretizationAD(model, 1)
        xc = g.cells.centroids[:, 0]
        v = 3.0 + 2.0 * xc                         # linear
        n_if = model.operators.T.size
        out = np.asarray(weno.face_upstream(model, None,
                                            np.ones(n_if, bool), v))
        xf = g.faces.centroids[np.all(g.faces.neighbors >= 0, axis=1), 0]
        interior = slice(1, n_if)                  # exclude the boundary face
        assert np.allclose(out[interior], (3.0 + 2.0 * xf)[interior])
        print("ok test_smooth_field_is_second_order_exact")

    def test_weno_is_bounded_near_a_step():
        """Near a monotone step WENO does not overshoot outside [min,max]."""
        model, g = _model()
        weno = WENOUpwindDiscretizationAD(model, 1)
        xc = g.cells.centroids[:, 0]
        v = (xc > 10).astype(float)
        n_if = model.operators.T.size
        out = np.asarray(weno.face_upstream(model, None,
                                            np.ones(n_if, bool), v))
        assert out.min() >= -1e-9 and out.max() <= 1.0 + 1e-9
        print("ok test_weno_is_bounded_near_a_step")

    def test_carries_ad_jacobian():
        """WENO of an AD field propagates a Jacobian (it is a linear combo)."""
        model, g = _model(8)
        weno = WENOUpwindDiscretizationAD(model, 1)
        (v,) = init_variables_adi(np.linspace(0.0, 1.0, g.cells.num))
        n_if = model.operators.T.size
        out = weno.face_upstream(model, None, np.ones(n_if, bool), v)
        assert hasattr(out, "jac")
        assert out.jac[0].shape == (n_if, g.cells.num)
        print("ok test_carries_ad_jacobian")

    test_smooth_field_is_second_order_exact()
    test_weno_is_bounded_near_a_step()
    test_carries_ad_jacobian()
    print("\nAll tests passed!")
