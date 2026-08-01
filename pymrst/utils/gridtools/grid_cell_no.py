"""Which cell owns each cell-face (MRST ``utils/gridtools/gridCellNo.m``).

`cells.faces` is a flat list: every cell's faces laid end to end, with
`cells.facePos` marking where each cell's run starts. This returns the matching
flat list of *cell* indices, so an operation over cell-faces can be scattered
back to cells with `np.add.at(out, cell_no, values)`.

Given a subset, the numbering is **positional within the subset**, not the
original cell index - cell `cells[0]` is numbered 0 whatever its global index.
That is MRST's behaviour and it is what makes the result usable directly as a
row index into a subset-sized array; using it as a global cell index instead is
the mistake it invites.
"""

import numpy as np

from pymrst.utils.rldecode import rldecode


def grid_cell_no(g, cells=None) -> np.ndarray:
    """One cell index per cell-face, over all cells or a subset."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    if cells is None:
        return rldecode(np.arange(g.cells.num), np.diff(face_pos))

    cells = np.atleast_1d(np.asarray(cells, dtype=np.int64)).ravel()
    if cells.size == 0:
        raise ValueError("the cell subset may not be empty")
    if cells.min() < 0 or cells.max() >= g.cells.num:
        raise ValueError(
            f"cell subset must lie in [0, {g.cells.num - 1}], got "
            f"[{cells.min()}, {cells.max()}]")
    counts = face_pos[cells + 1] - face_pos[cells]
    return rldecode(np.arange(cells.size), counts)


gridCellNo = grid_cell_no


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    G = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))

    def test_there_is_one_entry_per_cell_face() -> None:
        assert grid_cell_no(G).size == np.asarray(g_faces := G.cells.faces).shape[0]
        print("ok test_there_is_one_entry_per_cell_face")

    def test_each_cell_appears_once_per_face() -> None:
        counts = np.bincount(grid_cell_no(G), minlength=G.cells.num)
        assert np.array_equal(counts, np.diff(np.asarray(G.cells.facePos)))
        print("ok test_each_cell_appears_once_per_face")

    def test_it_is_non_decreasing() -> None:
        """Cell-faces are stored grouped by cell."""
        assert np.all(np.diff(grid_cell_no(G)) >= 0)
        print("ok test_it_is_non_decreasing")

    def test_it_scatters_cell_face_values_back_to_cells() -> None:
        """The reason it exists."""
        cell_no = grid_cell_no(G)
        out = np.zeros(G.cells.num)
        np.add.at(out, cell_no, np.ones(cell_no.size))
        assert np.array_equal(out, np.diff(np.asarray(G.cells.facePos)))
        print("ok test_it_scatters_cell_face_values_back_to_cells")

    def test_a_subset_is_numbered_positionally() -> None:
        """Cell 4 becomes 1 when the subset is [2, 4] - not 4."""
        out = grid_cell_no(G, [2, 4])
        assert set(out.tolist()) == {0, 1}, out
        print("ok test_a_subset_is_numbered_positionally")

    def test_a_subset_has_that_subset_s_face_counts() -> None:
        face_pos = np.asarray(G.cells.facePos)
        out = grid_cell_no(G, [1, 3])
        expected = [face_pos[2] - face_pos[1], face_pos[4] - face_pos[3]]
        assert np.array_equal(np.bincount(out), expected), out
        print("ok test_a_subset_has_that_subset_s_face_counts")

    def test_a_single_cell_is_accepted() -> None:
        assert np.all(grid_cell_no(G, 2) == 0)
        print("ok test_a_single_cell_is_accepted")

    def test_an_out_of_range_cell_is_rejected() -> None:
        try:
            grid_cell_no(G, [0, 999])
        except ValueError as error:
            assert "must lie in" in str(error)
            print("ok test_an_out_of_range_cell_is_rejected")
            return
        raise AssertionError("an out-of-range cell must be rejected")

    def test_an_empty_subset_is_rejected() -> None:
        try:
            grid_cell_no(G, [])
        except ValueError as error:
            assert "may not be empty" in str(error)
            print("ok test_an_empty_subset_is_rejected")
            return
        raise AssertionError("an empty subset must be rejected")

    test_there_is_one_entry_per_cell_face()
    test_each_cell_appears_once_per_face()
    test_it_is_non_decreasing()
    test_it_scatters_cell_face_values_back_to_cells()
    test_a_subset_is_numbered_positionally()
    test_a_subset_has_that_subset_s_face_counts()
    test_a_single_cell_is_accepted()
    test_an_out_of_range_cell_is_rejected()
    test_an_empty_subset_is_rejected()
    print("\nAll tests passed!")
