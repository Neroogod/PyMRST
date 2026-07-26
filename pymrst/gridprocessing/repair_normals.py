"""Recompute consistent 2D grid geometry (MRST `repairNormals`).

The 2D grids built by `tessellationGrid` / `glue2DGrid` carry topology but no
geometry, and a face's two nodes are not guaranteed to be ordered consistently
around its cells. This recomputes cell areas ("volumes"), face lengths ("areas")
and outward-consistent normals directly from the node coordinates: each cell is
fanned into triangles from a boundary-node average, their signed area vectors are
summed for the cell area, and each face normal is the edge rotated by the average
of its neighbour cells' unit normals. After this the grid has the
`cells.normals/volumes` and `faces.normals/areas` a 2D solver expects.
"""

import numpy as np


def repair_normals(g):
    """Fill in `cells.normals/volumes` and `faces.normals/areas` for a 2D grid
    (in place); returns `g`."""
    coords = np.asarray(g.nodes.coords, dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    assert np.all(np.diff(node_pos) == 2), "repairNormals is for 2D grids"
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64).reshape(-1, 2)
    fno, fno_next = fnodes[:, 0], fnodes[:, 1]
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)

    nc = g.cells.num
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    c_no = np.repeat(np.arange(nc), np.diff(face_pos))
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()

    # orient each half-face's edge consistently around its cell
    is_pos = c_no == neigh[cell_faces, 0]
    a_node = np.where(is_pos, fno[cell_faces], fno_next[cell_faces])
    b_node = np.where(is_pos, fno_next[cell_faces], fno[cell_faces])

    # cell centroid ~ average of its boundary nodes
    counts = np.bincount(c_no, minlength=nc)
    cent = np.zeros((nc, 3))
    for d in range(3):
        cent[:, d] = np.bincount(c_no, weights=coords[a_node, d], minlength=nc)
    cent /= counts[:, None]

    a = coords[a_node]
    b = coords[b_node]
    nt = np.cross(b - a, cent[c_no] - a) / 2.0            # triangle area vectors
    cell_normals = np.zeros((nc, 3))
    for d in range(3):
        cell_normals[:, d] = np.bincount(c_no, weights=nt[:, d], minlength=nc)
    cell_volumes = np.sqrt((cell_normals ** 2).sum(axis=1))

    # face normal: edge crossed with the average unit cell-normal of its sides
    n_unit = np.vstack([np.zeros((1, 3)), cell_normals / cell_volumes[:, None]])
    neig1 = neigh + 1                                     # boundary -1 -> row 0
    n_neigh = np.sum(neig1 > 0, axis=1)
    ncf = (n_unit[neig1[:, 0]] + n_unit[neig1[:, 1]]) / n_neigh[:, None]
    fa = coords[fno]
    fb = coords[fno_next]
    face_normals = np.cross(fb - fa, ncf)
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))

    g.cells.normals = cell_normals
    g.cells.volumes = cell_volumes
    g.faces.normals = face_normals
    g.faces.areas = face_areas
    g.type = list(getattr(g, "type", [])) + ["repairNormals"]
    return g


if __name__ == "__main__":
    from pymrst.gridprocessing.glue_2d_grid import glue_2d_grid
    from pymrst.gridprocessing.tessellation_grid import tessellation_grid

    def _square(x0):
        p = np.array([[x0, 0.], [x0 + 1, 0.], [x0 + 1, 1.], [x0, 1.]])
        return tessellation_grid(p, [[0, 1, 2, 3]])

    def test_unit_square_area_is_one():
        g = repair_normals(_square(0.0))
        assert np.isclose(g.cells.volumes[0], 1.0), g.cells.volumes
        print(f"ok test_unit_square_area_is_one ({g.cells.volumes[0]:.3f})")

    def test_face_lengths_are_unit():
        """Each edge of a unit square has length 1."""
        g = repair_normals(_square(0.0))
        assert np.allclose(g.faces.areas, 1.0), g.faces.areas
        print("ok test_face_lengths_are_unit")

    def test_glued_strip_areas():
        """A 2x1 glued strip has two unit-area cells and a unit internal edge."""
        g = repair_normals(glue_2d_grid(_square(0.0), _square(1.0)))
        assert np.allclose(g.cells.volumes, 1.0), g.cells.volumes
        assert np.allclose(g.faces.areas, 1.0)
        print(f"ok test_glued_strip_areas (areas {g.cells.volumes})")

    def test_normal_magnitude_equals_face_length():
        g = repair_normals(_square(0.0))
        assert np.allclose(np.linalg.norm(g.faces.normals, axis=1), g.faces.areas)
        print("ok test_normal_magnitude_equals_face_length")

    test_unit_square_area_is_one()
    test_face_lengths_are_unit()
    test_glued_strip_areas()
    test_normal_magnitude_equals_face_length()
    print("\nAll tests passed!")
