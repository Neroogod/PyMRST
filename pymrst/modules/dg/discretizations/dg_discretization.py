"""Spatial DG discretization operators (MRST ``DGDiscretization``).

The framework-free core: per-cell reference-frame dimensions
(`compute_cell_dimensions`, MRST `computeCellDimensions`), the coordinate map
that scales each cell onto `[-1, 1]^dim`, and the cell inner products that turn
the basis and cubature into the DG operators - the mass matrix
`M_ij = integral psi_i psi_j` and the stiffness `S_ijd = integral psi_i d/dx_d psi_j`
that the discontinuous-Galerkin transport update is built from. (Assembling the
full residual couples this to MRST's AD / state-function layer, which sits on top.)
"""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.dg.cubatures.element_cubature import (
    TetrahedronCubature,
    TriangleCubature,
)
from pymrst.modules.dg.discretizations.dg_basis import dg_basis


def compute_cell_dimensions(g):
    """Add per-cell bounding box `xMin`/`xMax` and span `dx` (MRST computeCellDimensions).

    `dx = 2 * max(xMax - centroid, centroid - xMin)` is the reference-frame span
    (symmetric about the centroid): a cell is mapped onto `[-1, 1]^dim` by
    `(x - centroid) * 2 / dx`. Returns `g`.
    """
    dim = g.griddim
    coords = np.asarray(g.nodes.coords, dtype=float)[:, :dim]
    if g.cells.nodes is None:
        _augment_cell_nodes(g)
    node_pos = np.asarray(g.cells.nodePos, dtype=np.int64)
    cell_nodes = np.asarray(g.cells.nodes, dtype=np.int64)
    cc = np.asarray(g.cells.centroids, dtype=float)[:, :dim]
    xmin = np.zeros((g.cells.num, dim))
    xmax = np.zeros((g.cells.num, dim))
    for c in range(g.cells.num):
        pts = coords[cell_nodes[node_pos[c]:node_pos[c + 1]]]
        xmin[c] = pts.min(axis=0)
        xmax[c] = pts.max(axis=0)
    g.cells.xMin = xmin
    g.cells.xMax = xmax
    g.cells.dx = 2.0 * np.maximum(xmax - cc, cc - xmin)
    return g


def _augment_cell_nodes(g):
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    nodes, pos = [], [0]
    for c in range(g.cells.num):
        faces = cell_faces[face_pos[c]:face_pos[c + 1]]
        cn = np.unique(np.concatenate(
            [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
        nodes.append(cn)
        pos.append(pos[-1] + cn.size)
    g.cells.nodes = np.concatenate(nodes).astype(np.int64)
    g.cells.nodePos = np.asarray(pos, dtype=np.int64)


@dataclass
class DGDiscretization:
    """DG spatial discretization of a given `degree` on grid `g`.

    Holds the basis, the per-cell reference-frame scaling and a cell cubature;
    provides the coordinate transform and the cell mass/stiffness matrices.
    """

    g: object
    degree: int
    basis: object
    cell_cubature: object
    type: str = "legendre"

    @classmethod
    def build(cls, g, degree: int, type: str = "legendre") -> "DGDiscretization":
        compute_cell_dimensions(g)
        basis = dg_basis(g.griddim, degree, type)
        cub = (TriangleCubature(g, 2 * degree + 1) if g.griddim == 2
               else TetrahedronCubature(g, 2 * degree + 1))
        return cls(g, degree, basis, cub, type)

    def transform_coords(self, x: np.ndarray, cells: np.ndarray):
        """Map physical points `x` (one per `cells`) to the reference frame.

        Returns `(xhat, scaling)` with `xhat = (x - centroid) * 2/dx` in
        `[-1, 1]^dim` and `scaling = 2/dx` (needed to map gradients back).
        """
        dim = self.g.griddim
        cc = np.asarray(self.g.cells.centroids, dtype=float)[cells, :dim]
        scaling = 2.0 / np.asarray(self.g.cells.dx, dtype=float)[cells]
        return (x[:, :dim] - cc) * scaling, scaling

    def _cell_points(self, c):
        i0, i1 = self.cell_cubature.pos[c], self.cell_cubature.pos[c + 1]
        return (self.cell_cubature.points[i0:i1], self.cell_cubature.weights[i0:i1])

    def mass_matrix(self, c: int) -> np.ndarray:
        """Cell mass matrix `M_ij = integral_cell psi_i psi_j`."""
        pts, w = self._cell_points(c)
        xhat, _ = self.transform_coords(pts, np.full(pts.shape[0], c))
        V = self.basis.evaluate(xhat)                  # (nq, nDof)
        vol = float(np.asarray(self.g.cells.volumes)[c])
        return vol * (V.T * w) @ V

    def stiffness_matrix(self, c: int) -> np.ndarray:
        """Cell stiffness `S[i,j,d] = integral_cell psi_i d(psi_j)/dx_d`.

        Physical gradients come from the reference gradients times the per-cell
        scaling `2/dx`.
        """
        pts, w = self._cell_points(c)
        idx = np.full(pts.shape[0], c)
        xhat, scaling = self.transform_coords(pts, idx)
        V = self.basis.evaluate(xhat)                  # (nq, nDof)
        G = self.basis.evaluate_gradient(xhat) * scaling[:, None, :]   # physical
        vol = float(np.asarray(self.g.cells.volumes)[c])
        return vol * np.einsum("q,qi,qjd->ijd", w, V, G)

    # --- DG field: a variable is stored as (n_cells, nDof) modal coefficients. -
    def evaluate(self, dofs: np.ndarray, x: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Evaluate the DG field `dofs` at physical points `x` in `cells`.

        `p_i = sum_j dofs[cells_i, j] * psi_j(transform(x_i, cells_i))`.
        """
        dofs = np.asarray(dofs, dtype=float)
        cells = np.asarray(cells, dtype=np.int64)
        xhat, _ = self.transform_coords(np.atleast_2d(x), cells)
        V = self.basis.evaluate(xhat)                  # (n, nDof)
        return np.einsum("nj,nj->n", dofs[cells], V)

    def cell_mean(self, dofs: np.ndarray) -> np.ndarray:
        """Cell average of the DG field: `(1/|cell|) integral psi_j dofs_j`."""
        dofs = np.asarray(dofs, dtype=float)
        out = np.zeros(self.g.cells.num)
        for c in range(self.g.cells.num):
            pts, w = self._cell_points(c)
            xhat, _ = self.transform_coords(pts, np.full(pts.shape[0], c))
            out[c] = w @ (self.basis.evaluate(xhat) @ dofs[c])
        return out

    def assign_from_cell_values(self, values: np.ndarray) -> np.ndarray:
        """DG dofs for a piecewise-constant field (MRST assignDofFromState).

        For a Legendre basis `psi_0 = 1`, so the constant mode carries the cell
        value and the higher modes are zero.
        """
        dofs = np.zeros((self.g.cells.num, self.basis.n_dof))
        dofs[:, 0] = np.asarray(values, dtype=float).ravel()
        return dofs

    def project(self, f) -> np.ndarray:
        """L2-project a function `f(x) -> values` onto the DG space.

        Solves the per-cell mass system `M dof = b`, `b_j = integral_cell f psi_j`,
        so the projection is exact for polynomials of degree <= the basis degree.
        """
        dofs = np.zeros((self.g.cells.num, self.basis.n_dof))
        for c in range(self.g.cells.num):
            pts, w = self._cell_points(c)
            xhat, _ = self.transform_coords(pts, np.full(pts.shape[0], c))
            V = self.basis.evaluate(xhat)
            vol = float(np.asarray(self.g.cells.volumes)[c])
            b = vol * (V.T * w) @ np.asarray(f(pts), dtype=float).ravel()
            dofs[c] = np.linalg.solve(self.mass_matrix(c), b)
        return dofs


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_cell_dimensions_cartesian():
        g = compute_geometry(cart_grid([3, 4], [6.0, 8.0]))   # cells 2 x 2
        compute_cell_dimensions(g)
        assert np.allclose(g.cells.dx, [2.0, 2.0])
        assert np.allclose(g.cells.xMax - g.cells.xMin, [2.0, 2.0])
        print("ok test_cell_dimensions_cartesian")

    def test_transform_maps_cell_to_reference():
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        disc = DGDiscretization.build(g, 2)
        # Each cell's four corner nodes map to (+/-1, +/-1).
        cn = np.asarray(g.cells.nodes)
        npos = np.asarray(g.cells.nodePos)
        coords = np.asarray(g.nodes.coords)
        for c in range(g.cells.num):
            nds = cn[npos[c]:npos[c + 1]]
            xhat, _ = disc.transform_coords(coords[nds], np.full(nds.size, c))
            assert np.allclose(np.abs(xhat), 1.0), (c, xhat)
        print("ok test_transform_maps_cell_to_reference")

    def test_mass_matrix_diagonal_legendre():
        # On a Cartesian grid the Legendre mass matrix is diagonal with
        # M_ii = |cell| * prod_d 1/(2 k_d + 1).
        g = compute_geometry(cart_grid([2, 3], [4.0, 3.0]))   # cells 2 x 1
        disc = DGDiscretization.build(g, 3)
        vols = np.asarray(g.cells.volumes)
        for c in range(g.cells.num):
            M = disc.mass_matrix(c)
            off = M - np.diag(np.diag(M))
            assert np.max(np.abs(off)) < 1e-12, np.max(np.abs(off))
            expect = vols[c] * np.prod(1.0 / (2 * disc.basis.k + 1), axis=1)
            assert np.allclose(np.diag(M), expect, atol=1e-12)
        print("ok test_mass_matrix_diagonal_legendre")

    def test_stiffness_reproduces_gradient_integral():
        # S[i,j,d] integrates psi_i * d psi_j/dx_d; check against a fine
        # analytic value for a low-order pair on a unit-ish cell.
        g = compute_geometry(cart_grid([1, 1], [2.0, 2.0]))   # single [-1,1]^2 cell
        g.nodes.coords = np.asarray(g.nodes.coords) - 1.0
        g = compute_geometry(g)
        disc = DGDiscretization.build(g, 2)
        S = disc.stiffness_matrix(0)
        # psi: [1, x, y, ...]; integral(1 * d(x)/dx) over [-1,1]^2 = area = 4.
        # dof 0 is the constant, dof for k=(1,0) is x.
        kx = np.where((disc.basis.k == [1, 0]).all(axis=1))[0][0]
        assert abs(S[0, kx, 0] - 4.0) < 1e-12, S[0, kx, 0]
        assert abs(S[0, kx, 1]) < 1e-12               # d(x)/dy = 0
        print("ok test_stiffness_reproduces_gradient_integral")

    def test_project_reproduces_polynomials_exactly():
        # The L2 projection is exact for any polynomial up to the basis degree;
        # evaluating the projection at random points recovers it.
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        rng = np.random.default_rng(0)
        for deg in (1, 2, 3):
            disc = DGDiscretization.build(g, deg)
            # A polynomial of total degree exactly `deg`.
            f = lambda P, d=deg: (0.7 * P[:, 0] + 0.3 * P[:, 1] + 0.5) ** d
            dofs = disc.project(f)
            x = rng.random((20, 2)) * 3.0
            cells = (np.floor(x[:, 0]) + 3 * np.floor(x[:, 1])).astype(np.int64)
            err = np.max(np.abs(disc.evaluate(dofs, x, cells) - f(x)))
            assert err < 1e-10, (deg, err)
        print("ok test_project_reproduces_polynomials_exactly")

    def test_cell_mean_and_assign():
        g = compute_geometry(cart_grid([4, 2], [4.0, 2.0]))
        disc = DGDiscretization.build(g, 2)
        vals = np.arange(g.cells.num, dtype=float) + 1.0
        dofs = disc.assign_from_cell_values(vals)
        assert np.allclose(disc.cell_mean(dofs), vals)
        # A projected linear field's cell mean equals its value at the centroid.
        cc = np.asarray(g.cells.centroids)
        f = lambda P: 2.0 * P[:, 0] - P[:, 1] + 3.0
        dm = disc.project(f)
        assert np.allclose(disc.cell_mean(dm), f(cc), atol=1e-12)
        print("ok test_cell_mean_and_assign")

    def test_l2_projection_converges():
        # For a smooth non-polynomial field, the L2 error drops at order deg+1.
        f = lambda P: np.sin(P[:, 0]) * np.cos(P[:, 1])
        deg = 2
        errs = []
        for n in (4, 8):
            g = compute_geometry(cart_grid([n, n], [2 * np.pi, 2 * np.pi]))
            disc = DGDiscretization.build(g, deg)
            dofs = disc.project(f)
            # L2 error via the cell cubature.
            e2 = 0.0
            for c in range(g.cells.num):
                pts, w = disc._cell_points(c)
                vol = float(np.asarray(g.cells.volumes)[c])
                fv = disc.evaluate(dofs, pts, np.full(pts.shape[0], c))
                e2 += vol * (w @ (fv - f(pts)) ** 2)
            errs.append(np.sqrt(e2))
        rate = np.log2(errs[0] / errs[1])
        assert rate > deg + 0.7, (rate, errs)
        print(f"ok test_l2_projection_converges (order {rate:.2f} for deg {deg})")

    test_cell_dimensions_cartesian()
    test_transform_maps_cell_to_reference()
    test_mass_matrix_diagonal_legendre()
    test_stiffness_reproduces_gradient_integral()
    test_project_reproduces_polynomials_exactly()
    test_cell_mean_and_assign()
    test_l2_projection_converges()
    print("\nAll tests passed!")
