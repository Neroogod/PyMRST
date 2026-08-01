"""Corner-node coordinates of a cell.

Private, and deliberately not an MRST counterpart: MRST inlines this gather in
both `polygonInt` and `polyhedronInt`. Duplicating ten lines of index
arithmetic in two files is the worse option - it is exactly the kind of thing
that gets fixed in one copy and not the other - so it lives here, under a
leading underscore to mark it as PyMRST's own rather than a port.
"""

import numpy as np


def cell_node_coords(g, cell: int, dim: int = 2) -> np.ndarray:
    """A cell's corner-node coordinates, gathered from its faces."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
    nodes = np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
    return np.asarray(g.nodes.coords, dtype=float)[nodes][:, :dim]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_square_cell_has_four_corners() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        X = cell_node_coords(g, 0)
        assert X.shape == (4, 2), X.shape
        print("ok test_a_square_cell_has_four_corners")

    def test_a_cube_cell_has_eight_corners() -> None:
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        X = cell_node_coords(g, 0, dim=3)
        assert X.shape == (8, 3), X.shape
        print("ok test_a_cube_cell_has_eight_corners")

    def test_the_corners_bound_the_cell() -> None:
        g = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
        X = cell_node_coords(g, 1)
        assert np.allclose(X.min(axis=0), [1.0, 0.0]), X
        assert np.allclose(X.max(axis=0), [2.0, 1.0]), X
        print("ok test_the_corners_bound_the_cell")

    def test_nodes_are_unique() -> None:
        """Faces share nodes, so the gather must deduplicate."""
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        X = cell_node_coords(g, 0)
        assert len(np.unique(X, axis=0)) == len(X)
        print("ok test_nodes_are_unique")

    test_a_square_cell_has_four_corners()
    test_a_cube_cell_has_eight_corners()
    test_the_corners_bound_the_cell()
    test_nodes_are_unique()
    print("\nAll tests passed!")
