"""Cell-face pairs including NNCs (MRST ``utils/gridtools/getCellNoFaces.m``).

`grid_cell_no` paired with the cell-face list, plus the non-neighbouring
connections. Returns `(cell_no, cell_faces, is_nnc)`.

**The NNC rows are what distinguishes this from `grid_cell_no`.** A
non-neighbouring connection joins two cells the geometry does not, so it has no
face; MRST invents one, numbered past the real faces, and appends *both* cells
against it. The `is_nnc` flag exists because those synthetic face numbers index
nothing in `faces.areas` or `faces.normals` - reaching for a geometric property
of an NNC face is the error this guards against.
"""

from typing import Tuple

import numpy as np

from pymrst.utils.rldecode import rldecode


def get_cell_no_faces(g) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cell_no, cell_faces, is_nnc)`` over cell-faces and NNCs."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_no = rldecode(np.arange(g.cells.num), np.diff(face_pos))
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    is_nnc = np.zeros(cell_no.size, dtype=bool)

    nnc = getattr(g, "nnc", None)
    cells = getattr(nnc, "cells", None) if nnc is not None else None
    if cells is None:
        return cell_no, cell_faces, is_nnc

    cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
    synthetic = g.faces.num + np.arange(cells.shape[0])
    cell_no = np.concatenate([cell_no, cells[:, 0], cells[:, 1]])
    cell_faces = np.concatenate([cell_faces, synthetic, synthetic])
    is_nnc = np.concatenate([is_nnc, np.ones(2 * cells.shape[0], dtype=bool)])
    return cell_no, cell_faces, is_nnc


getCellNoFaces = get_cell_no_faces


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.grid_cell_no import grid_cell_no

    G = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))

    def test_without_nncs_it_matches_grid_cell_no() -> None:
        cell_no, cell_faces, is_nnc = get_cell_no_faces(G)
        assert np.array_equal(cell_no, grid_cell_no(G))
        assert not is_nnc.any()
        print("ok test_without_nncs_it_matches_grid_cell_no")

    def test_the_three_outputs_are_the_same_length() -> None:
        cell_no, cell_faces, is_nnc = get_cell_no_faces(G)
        assert cell_no.size == cell_faces.size == is_nnc.size
        print("ok test_the_three_outputs_are_the_same_length")

    def _with_nnc():
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        g.nnc = SimpleNamespace(cells=np.array([[0, 5]]))
        return g

    def test_an_nnc_adds_a_row_for_each_of_its_cells() -> None:
        g = _with_nnc()
        plain = get_cell_no_faces(G)[0].size
        cell_no, _, is_nnc = get_cell_no_faces(g)
        assert cell_no.size == plain + 2, (plain, cell_no.size)
        assert is_nnc.sum() == 2
        print("ok test_an_nnc_adds_a_row_for_each_of_its_cells")

    def test_both_nnc_cells_share_one_synthetic_face() -> None:
        g = _with_nnc()
        _, cell_faces, is_nnc = get_cell_no_faces(g)
        assert len(set(cell_faces[is_nnc].tolist())) == 1
        print("ok test_both_nnc_cells_share_one_synthetic_face")

    def test_the_synthetic_face_is_numbered_past_the_real_ones() -> None:
        """Which is why is_nnc exists: it indexes no geometric array."""
        g = _with_nnc()
        _, cell_faces, is_nnc = get_cell_no_faces(g)
        assert np.all(cell_faces[is_nnc] >= g.faces.num)
        assert np.all(cell_faces[~is_nnc] < g.faces.num)
        print("ok test_the_synthetic_face_is_numbered_past_the_real_ones")

    def test_the_nnc_rows_name_the_connected_cells() -> None:
        g = _with_nnc()
        cell_no, _, is_nnc = get_cell_no_faces(g)
        assert set(cell_no[is_nnc].tolist()) == {0, 5}
        print("ok test_the_nnc_rows_name_the_connected_cells")

    def test_several_nncs_get_distinct_faces() -> None:
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        g.nnc = SimpleNamespace(cells=np.array([[0, 5], [1, 4]]))
        _, cell_faces, is_nnc = get_cell_no_faces(g)
        assert len(set(cell_faces[is_nnc].tolist())) == 2
        print("ok test_several_nncs_get_distinct_faces")

    test_without_nncs_it_matches_grid_cell_no()
    test_the_three_outputs_are_the_same_length()
    test_an_nnc_adds_a_row_for_each_of_its_cells()
    test_both_nnc_cells_share_one_synthetic_face()
    test_the_synthetic_face_is_numbered_past_the_real_ones()
    test_the_nnc_rows_name_the_connected_cells()
    test_several_nncs_get_distinct_faces()
    print("\nAll tests passed!")
