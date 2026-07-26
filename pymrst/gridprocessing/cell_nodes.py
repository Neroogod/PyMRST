"""Corner nodes of each cell in canonical order (MRST `cellNodes`).

For a logically-Cartesian / corner-point grid, returns each cell's corner nodes
tagged by their local vertex position - west-south-down, east-south-down, ... up
to east-north-up in 3D. The trick (MRST's) is that a corner is where `dim` of a
cell's faces meet, and each face carries a direction tag (W/E/S/N/B/T); summing a
power-of-two per incident direction gives a binary key that names the corner. So
one sparse accumulation over half-face/node incidences recovers the full
cell-to-corner map with no geometry.

The direction tag is read from `cells.tags` (0-5), the 0-based form of MRST's
`cells.faces(:,2)` (1-6).
"""

import numpy as np
import scipy.sparse as sp

# direction tags (1-6: W E S N B T) meeting at each local corner
_CORNERS_3D = [[1, 3, 5], [2, 3, 5], [1, 4, 5], [2, 4, 5],
               [1, 3, 6], [2, 3, 6], [1, 4, 6], [2, 4, 6]]
_CORNERS_2D = [[1, 3], [2, 3], [1, 4], [2, 4]]


def cell_nodes(g) -> np.ndarray:
    """Return an `(m, 3)` array of `[cell, local_vertex, node]`, sorted, giving
    each cell's corner nodes in canonical local-vertex order (0-based)."""
    dim = len(g.cartDims)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)

    cell_no = np.repeat(np.arange(g.cells.num), np.diff(face_pos))
    cf_face = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    cf_tag = np.asarray(g.cells.tags, dtype=np.int64).ravel() + 1      # 1..6

    ne = np.diff(node_pos)[cf_face]                       # nodes per half-face
    a_cell = np.repeat(cell_no, ne)
    a_tag = np.repeat(cf_tag, ne)
    hf_nodes = np.concatenate([fnodes[node_pos[f]:node_pos[f + 1]]
                               for f in cf_face])

    # mat[cell, node] = sum of 2^(tag-1) over the cell's half-faces at that node
    val = 2.0 ** (a_tag - 1)
    mat = sp.coo_array((val, (a_cell, hf_nodes)),
                       shape=(g.cells.num, g.nodes.num)).tocsr()
    mat.sum_duplicates()
    coo = mat.tocoo()
    cell_number, node_number, key = coo.row, coo.col, coo.data.astype(np.int64)

    corners = _CORNERS_3D if dim == 3 else _CORNERS_2D
    keymap = {sum(2 ** (t - 1) for t in row): c for c, row in enumerate(corners)}
    local = np.array([keymap.get(int(k), -1) for k in key], dtype=np.int64)

    s = np.column_stack([cell_number, local, node_number])
    return s[np.lexsort((s[:, 2], s[:, 1], s[:, 0]))]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_each_cube_has_eight_corners_in_order():
        """Every cell of a 3D Cartesian grid gets its 8 corner nodes, tagged
        local vertex 0..7."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        s = cell_nodes(g)
        for c in range(g.cells.num):
            rows = s[s[:, 0] == c]
            assert rows.shape[0] == 8, (c, rows.shape)
            assert list(rows[:, 1]) == list(range(8)), rows[:, 1]
        assert np.all(s[:, 2] >= 0), "all corners map to a real node"
        print(f"ok test_each_cube_has_eight_corners_in_order ({s.shape[0]} rows)")

    def test_corner_coordinates_are_the_cube_extremes():
        """Local vertex 0 (west-south-down) of cell 0 is the min-corner node."""
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        s = cell_nodes(g)
        coords = np.asarray(g.nodes.coords)
        wsd = s[(s[:, 0] == 0) & (s[:, 1] == 0), 2][0]
        enu = s[(s[:, 0] == 0) & (s[:, 1] == 7), 2][0]
        assert np.allclose(coords[wsd], [0, 0, 0]), coords[wsd]
        assert np.allclose(coords[enu], [1, 1, 1]), coords[enu]
        print("ok test_corner_coordinates_are_the_cube_extremes")

    def test_2d_grid_has_four_corners_per_cell():
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        s = cell_nodes(g)
        for c in range(g.cells.num):
            assert (s[:, 0] == c).sum() == 4
        print("ok test_2d_grid_has_four_corners_per_cell")

    test_each_cube_has_eight_corners_in_order()
    test_corner_coordinates_are_the_cube_extremes()
    test_2d_grid_has_four_corners_per_cell()
    print("\nAll tests passed!")
