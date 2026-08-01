"""Faces of given cells, packed (MRST ``utils/gridtools/gridCellFaces.m``).

Returns `(faces, pos)`: the faces of the named cells concatenated, and the
offsets marking where each cell's run begins, so cell `i`'s faces are
`faces[pos[i]:pos[i+1]]`.

The packed form rather than a list of arrays, because that is what the rest of
the grid uses and what lets a caller stay vectorised. **`pos` is 0-based here**
where MRST's is 1-based; a `pos` copied straight from MATLAB code will be off
by one on both ends.
"""

from typing import Tuple

import numpy as np

from pymrst.utils.mcolon import mcolon


def grid_cell_faces(g, cells=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(faces, pos)`` for the named cells, or all of them."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cells = (np.arange(g.cells.num) if cells is None
             else np.atleast_1d(np.asarray(cells, dtype=np.int64)).ravel())

    all_faces = np.asarray(g.cells.faces, dtype=np.int64)
    all_faces = all_faces[:, 0] if all_faces.ndim > 1 else all_faces

    counts = face_pos[cells + 1] - face_pos[cells]
    faces = all_faces[mcolon(face_pos[cells], face_pos[cells + 1])]
    return faces, np.concatenate([[0], np.cumsum(counts)])


gridCellFaces = grid_cell_faces


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    G = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))

    def test_all_cells_reproduces_the_grid_s_own_list() -> None:
        faces, pos = grid_cell_faces(G)
        stored = np.asarray(G.cells.faces)
        stored = stored[:, 0] if stored.ndim > 1 else stored
        assert np.array_equal(faces, stored)
        assert np.array_equal(pos, np.asarray(G.cells.facePos))
        print("ok test_all_cells_reproduces_the_grid_s_own_list")

    def test_pos_delimits_each_cell() -> None:
        faces, pos = grid_cell_faces(G, [0, 2, 4])
        assert pos.size == 4 and pos[0] == 0 and pos[-1] == faces.size
        for i in range(3):
            assert pos[i + 1] > pos[i]
        print("ok test_pos_delimits_each_cell")

    def test_a_subset_returns_only_those_cells_faces() -> None:
        faces, pos = grid_cell_faces(G, [1])
        face_pos = np.asarray(G.cells.facePos)
        stored = np.asarray(G.cells.faces)
        stored = stored[:, 0] if stored.ndim > 1 else stored
        assert np.array_equal(faces, stored[face_pos[1]:face_pos[2]])
        print("ok test_a_subset_returns_only_those_cells_faces")

    def test_pos_is_zero_based() -> None:
        """MRST's is 1-based; copying its indexing gives an off-by-one."""
        _, pos = grid_cell_faces(G, [0])
        assert pos[0] == 0, pos
        print("ok test_pos_is_zero_based")

    def test_the_order_within_a_cell_is_preserved() -> None:
        faces, pos = grid_cell_faces(G, [2])
        face_pos = np.asarray(G.cells.facePos)
        stored = np.asarray(G.cells.faces)
        stored = stored[:, 0] if stored.ndim > 1 else stored
        assert np.array_equal(faces, stored[face_pos[2]:face_pos[3]])
        print("ok test_the_order_within_a_cell_is_preserved")

    def test_a_repeated_cell_appears_twice() -> None:
        faces, pos = grid_cell_faces(G, [1, 1])
        assert pos.size == 3
        assert np.array_equal(faces[pos[0]:pos[1]], faces[pos[1]:pos[2]])
        print("ok test_a_repeated_cell_appears_twice")

    test_all_cells_reproduces_the_grid_s_own_list()
    test_pos_delimits_each_cell()
    test_a_subset_returns_only_those_cells_faces()
    test_pos_is_zero_based()
    test_the_order_within_a_cell_is_preserved()
    test_a_repeated_cell_appears_twice()
    print("\nAll tests passed!")
