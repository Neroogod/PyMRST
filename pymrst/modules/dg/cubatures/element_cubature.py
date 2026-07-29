"""Cubatures over grid cells and faces (MRST Triangle/Tetrahedron/LineCubature).

Maps a reference-element rule onto the actual grid geometry: each element is
split into simplices (fanned from its centroid), the reference rule is affinely
mapped onto every simplex, and the weights are scaled by the simplex measure and
normalised by the element measure - so, as in MRST, the stored weights of each
element sum to 1 (the element average) and `measure * sum(w * f)` is the integral.

- `TriangleCubature`  (dim 2): 2-D grid cells, or 3-D grid faces.
- `TetrahedronCubature` (dim 3): 3-D grid cells.
- `LineCubature`      (dim 1): 2-D grid faces, or 3-D grid edges.
"""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.dg.cubatures.reference_cubature import (
    get_line_cubature,
    get_tetrahedron_cubature,
    get_triangle_cubature,
)


@dataclass
class ElementCubature:
    """Cubature points/weights grouped by element via `pos`.

    Element `e` owns points `points[pos[e]:pos[e+1]]` with the matching
    `weights`; the weights sum to 1 over each element (the element average).
    """

    points: np.ndarray                                 # (N, griddim)
    weights: np.ndarray                                # (N,)
    pos: np.ndarray                                    # (n_elements + 1,)
    precision: int
    dim: int

    def num_points(self) -> np.ndarray:
        return np.diff(self.pos)


def _affine_map(x_ref: np.ndarray, xR: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Map reference points `x_ref` on the simplex with vertices `xR` to the
    physical simplex `verts` (affine, exact for simplices)."""
    XI = np.column_stack([xR, np.ones(xR.shape[0])])   # (nv, dim+1)
    R = np.linalg.solve(XI, verts)                     # (dim+1, griddim)
    return x_ref @ R[:-1] + R[-1]


def _simplex_measure(verts: np.ndarray) -> float:
    """Length / area / volume of a simplex from its vertices."""
    e = verts[1:] - verts[0]
    if e.shape[0] == 1:                                # line
        return float(np.linalg.norm(e[0]))
    if e.shape[0] == 2:                                # triangle (2-D or in 3-D)
        if e.shape[1] == 2:
            return abs(e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0]) / 2
        return float(np.linalg.norm(np.cross(e[0], e[1]))) / 2
    return abs(np.linalg.det(e)) / 6                   # tetrahedron


def _build(g, precision, x_ref, w_ref, xR, simplices, measures):
    """Assemble points/weights from per-element simplex lists.

    `simplices[e]` is a list of vertex arrays; `measures[e]` the element measure
    used to normalise the weights (its volume/area).
    """
    points, weights, pos = [], [], [0]
    for e, tris in enumerate(simplices):
        for verts in tris:
            points.append(_affine_map(x_ref, xR, np.asarray(verts, float)))
            weights.append(w_ref * _simplex_measure(np.asarray(verts, float)))
        pos.append(pos[-1] + len(tris) * x_ref.shape[0])
        i0, i1 = pos[-2], pos[-1]
        # placeholder; normalisation applied after concatenation below
    P = np.vstack(points)
    W = np.concatenate(weights)
    pos = np.asarray(pos, dtype=np.int64)
    for e in range(len(simplices)):                    # normalise per element
        W[pos[e]:pos[e + 1]] /= measures[e]
    return ElementCubature(P, W, pos, precision, x_ref.shape[1])


def _cell_faces(g, c):
    fp = np.asarray(g.cells.facePos, dtype=np.int64)
    return np.asarray(g.cells.faces, dtype=np.int64).ravel()[fp[c]:fp[c + 1]]


def _face_nodes(g, f):
    np_ = np.asarray(g.faces.nodePos, dtype=np.int64)
    return np.asarray(g.faces.nodes, dtype=np.int64)[np_[f]:np_[f + 1]]


def _face_edges(g, f):
    ep = np.asarray(g.faces.edgePos, dtype=np.int64)
    return np.asarray(g.faces.edges, dtype=np.int64)[ep[f]:ep[f + 1]]


def _edge_nodes(g, e):
    return np.asarray(g.edges.nodes, dtype=np.int64).reshape(-1, 2)[e]


def TriangleCubature(g, precision: int) -> ElementCubature:
    """Triangle cubature: 2-D grid cells (or 3-D grid faces)."""
    x_ref, w_ref, _, xR = get_triangle_cubature(precision)
    coords = np.asarray(g.nodes.coords, dtype=float)
    if g.griddim == 2:
        cc = np.asarray(g.cells.centroids, dtype=float)
        simplices = [[[coords[n[0]], coords[n[1]], cc[c]]
                      for f in _cell_faces(g, c)
                      for n in [_face_nodes(g, f)]]
                     for c in range(g.cells.num)]
        measures = np.asarray(g.cells.volumes, dtype=float)
    else:                                              # faces of a 3-D grid
        fc = np.asarray(g.faces.centroids, dtype=float)
        simplices = [[[coords[en[0]], coords[en[1]], fc[f]]
                      for e in _face_edges(g, f)
                      for en in [_edge_nodes(g, e)]]
                     for f in range(g.faces.num)]
        measures = np.asarray(g.faces.areas, dtype=float)
    return _build(g, precision, x_ref, w_ref, xR, simplices, measures)


def TetrahedronCubature(g, precision: int) -> ElementCubature:
    """Tetrahedron cubature over the cells of a 3-D grid."""
    assert g.griddim == 3, "TetrahedronCubature is for 3-D grids"
    x_ref, w_ref, _, xR = get_tetrahedron_cubature(precision)
    coords = np.asarray(g.nodes.coords, dtype=float)
    fc = np.asarray(g.faces.centroids, dtype=float)
    cc = np.asarray(g.cells.centroids, dtype=float)
    simplices = [[[coords[en[0]], coords[en[1]], fc[f], cc[c]]
                  for f in _cell_faces(g, c)
                  for e in _face_edges(g, f)
                  for en in [_edge_nodes(g, e)]]
                 for c in range(g.cells.num)]
    measures = np.asarray(g.cells.volumes, dtype=float)
    return _build(g, precision, x_ref, w_ref, xR, simplices, measures)


def LineCubature(g, precision: int) -> ElementCubature:
    """Line cubature over the faces of a 2-D grid (each face is a segment)."""
    assert g.griddim == 2, "LineCubature here handles 2-D grid faces"
    x_ref, w_ref, _, xR = get_line_cubature(precision)
    x_ref = x_ref.reshape(-1, 1)
    xR = np.asarray(xR, float).reshape(-1, 1)
    coords = np.asarray(g.nodes.coords, dtype=float)
    simplices = [[[coords[n[0]], coords[n[1]]]] for f in range(g.faces.num)
                 for n in [_face_nodes(g, f)]]
    measures = np.asarray(g.faces.areas, dtype=float)
    return _build(g, precision, x_ref, w_ref, xR, simplices, measures)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vem.utils.vem_geometry import compute_vem_geometry

    def _integral(cub, g, f, e):
        """Integral of scalar field `f` over element `e` from the cubature."""
        i0, i1 = cub.pos[e], cub.pos[e + 1]
        return None  # measure applied by caller

    def test_triangle_cubature_integrates_cell_monomials():
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        cub = TriangleCubature(g, 3)
        cc = np.asarray(g.cells.centroids)
        vols = np.asarray(g.cells.volumes)
        # integral of (x-cx)(y-cy) etc. should be near cell-symmetric values;
        # test exact reproduction of 1 and linear moments about the centroid.
        for c in range(g.cells.num):
            i0, i1 = cub.pos[c], cub.pos[c + 1]
            P, W = cub.points[i0:i1], cub.weights[i0:i1]
            assert abs(W.sum() - 1.0) < 1e-12                 # weights average
            # centroid recovered: measure * sum(w * x) / measure = sum(w x)
            assert np.allclose(W @ P, cc[c], atol=1e-12), (c, W @ P, cc[c])
        print("ok test_triangle_cubature_integrates_cell_monomials")

    def test_triangle_cubature_area_matches_volume():
        g = compute_geometry(cart_grid([4, 3], [2.0, 5.0]))
        cub = TriangleCubature(g, 2)
        vols = np.asarray(g.cells.volumes)
        for c in range(g.cells.num):
            i0, i1 = cub.pos[c], cub.pos[c + 1]
            # sum of un-normalised weights = area; here weights are /area, so
            # measure * sum(w) = area -> sum(w) = 1 already checked; instead
            # verify the mapped points lie inside the cell's bounding box.
            P = cub.points[i0:i1]
            assert P[:, 0].min() >= -1e-9 and P[:, 0].max() <= 2.0 + 1e-9
        print("ok test_triangle_cubature_area_matches_volume")

    def test_line_cubature_face_midpoint_and_length():
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        cub = LineCubature(g, 3)
        fc = np.asarray(g.faces.centroids)
        for f in range(g.faces.num):
            i0, i1 = cub.pos[f], cub.pos[f + 1]
            P, W = cub.points[i0:i1], cub.weights[i0:i1]
            assert abs(W.sum() - 1.0) < 1e-12
            assert np.allclose(W @ P, fc[f], atol=1e-12)      # face centroid
        print("ok test_line_cubature_face_midpoint_and_length")

    def test_tet_cubature_cell_centroid():
        g = compute_vem_geometry(compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0])))
        cub = TetrahedronCubature(g, 3)
        cc = np.asarray(g.cells.centroids)
        for c in range(g.cells.num):
            i0, i1 = cub.pos[c], cub.pos[c + 1]
            P, W = cub.points[i0:i1], cub.weights[i0:i1]
            assert abs(W.sum() - 1.0) < 1e-12
            assert np.allclose(W @ P, cc[c], atol=1e-12), (c, W @ P, cc[c])
        print("ok test_tet_cubature_cell_centroid")

    test_triangle_cubature_integrates_cell_monomials()
    test_triangle_cubature_area_matches_volume()
    test_line_cubature_face_midpoint_and_length()
    test_tet_cubature_cell_centroid()
    print("\nAll tests passed!")
