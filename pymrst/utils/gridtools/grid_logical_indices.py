"""Cartesian (i, j, k) of each cell (MRST ``gridLogicalIndices.m``).

Recovers a cell's logical position in the underlying Cartesian box. Needed
whenever a grid derived from a box - by removing inactive cells, or by corner
point processing - has to be related back to layers, columns or an ECLIPSE
array indexed by (i, j, k).

**The map goes through `cells.indexMap`, not the cell number**, and that is the
whole point: after inactive cells are dropped, cell 7 is no longer the eighth
box cell. Using the cell index directly gives the right answer only for a grid
where nothing was removed - which is exactly the case a test on `cart_grid`
would exercise, so there is a test here on a grid with a hole.

Returned 0-based, like every other index in PyMRST; MRST's are 1-based.
"""

from typing import List

import numpy as np


def grid_logical_indices(g, cells=None) -> List[np.ndarray]:
    """Return one array of logical indices per Cartesian dimension."""
    dims = getattr(g, "cartDims", None)
    if dims is None:
        raise ValueError(
            "logical indices need the grid's cartDims; this grid has none")
    dims = np.asarray(dims, dtype=np.int64).ravel()

    index_map = getattr(g.cells, "indexMap", None)
    box = (np.arange(g.cells.num) if index_map is None
           else np.asarray(index_map, dtype=np.int64).ravel())
    if cells is not None:
        box = box[np.atleast_1d(np.asarray(cells, dtype=np.int64)).ravel()]

    return list(np.unravel_index(box, tuple(dims), order="F"))


gridLogicalIndices = grid_logical_indices


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.gridprocessing.remove_cells import remove_cells

    G = compute_geometry(cart_grid([4, 3, 2], [4.0, 3.0, 2.0]))

    def test_it_returns_one_array_per_dimension() -> None:
        ijk = grid_logical_indices(G)
        assert len(ijk) == 3 and all(a.size == G.cells.num for a in ijk)
        print("ok test_it_returns_one_array_per_dimension")

    def test_the_indices_span_the_box() -> None:
        i, j, k = grid_logical_indices(G)
        assert i.min() == 0 and i.max() == 3
        assert j.min() == 0 and j.max() == 2
        assert k.min() == 0 and k.max() == 1
        print("ok test_the_indices_span_the_box")

    def test_they_are_zero_based() -> None:
        """MRST's are 1-based."""
        assert min(a.min() for a in grid_logical_indices(G)) == 0
        print("ok test_they_are_zero_based")

    def test_the_first_cell_is_the_origin() -> None:
        i, j, k = grid_logical_indices(G, [0])
        assert (i[0], j[0], k[0]) == (0, 0, 0)
        print("ok test_the_first_cell_is_the_origin")

    def test_the_ordering_is_column_major() -> None:
        """i varies fastest, as in MATLAB's ind2sub and ECLIPSE arrays."""
        i, j, k = grid_logical_indices(G)
        assert (i[1], j[1], k[1]) == (1, 0, 0), (i[1], j[1], k[1])
        print("ok test_the_ordering_is_column_major")

    def test_a_subset_returns_only_those_cells() -> None:
        i, _, _ = grid_logical_indices(G, [0, 5])
        assert i.size == 2
        print("ok test_a_subset_returns_only_those_cells")

    def test_it_follows_index_map_after_cells_are_removed() -> None:
        """The case a cart_grid test cannot reach: cell n is not box cell n."""
        holed = compute_geometry(remove_cells(
            cart_grid([4, 3, 2], [4.0, 3.0, 2.0]), [0, 1, 2]))
        i, j, k = grid_logical_indices(holed)
        assert i.size == holed.cells.num
        # cell 0 of the holed grid is box cell 3, i.e. (3, 0, 0)
        assert (i[0], j[0], k[0]) == (3, 0, 0), (i[0], j[0], k[0])
        naive = np.unravel_index(np.arange(holed.cells.num), (4, 3, 2),
                                 order="F")
        assert not np.array_equal(i, naive[0]), "index_map made no difference"
        print("ok test_it_follows_index_map_after_cells_are_removed")

    def test_a_grid_without_cart_dims_is_rejected() -> None:
        from types import SimpleNamespace
        try:
            grid_logical_indices(SimpleNamespace(cells=SimpleNamespace(num=1)))
        except ValueError as error:
            assert "cartDims" in str(error)
            print("ok test_a_grid_without_cart_dims_is_rejected")
            return
        raise AssertionError("a grid without cartDims must be rejected")

    def test_two_dimensions_gives_two_arrays() -> None:
        flat = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        assert len(grid_logical_indices(flat)) == 2
        print("ok test_two_dimensions_gives_two_arrays")

    test_it_returns_one_array_per_dimension()
    test_the_indices_span_the_box()
    test_they_are_zero_based()
    test_the_first_cell_is_the_origin()
    test_the_ordering_is_column_major()
    test_a_subset_returns_only_those_cells()
    test_it_follows_index_map_after_cells_are_removed()
    test_a_grid_without_cart_dims_is_rejected()
    test_two_dimensions_gives_two_arrays()
    print("\nAll tests passed!")
