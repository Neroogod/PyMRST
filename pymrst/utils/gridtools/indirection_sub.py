"""Gather from a packed array (MRST ``utils/gridtools/indirectionSub.m``).

The read half of the packed `(values, pos)` layout that `grid_cell_faces` and
`grid_face_nodes` return: `indirection_sub(i, pos, values)` gives the entries
belonging to the items in `i`, concatenated.

One function for the pattern `values[mcolon(pos[i], pos[i+1])]`, which appears
throughout the grid code and is easy to write with an off-by-one - MRST's `pos`
is 1-based and inclusive at the top, PyMRST's is 0-based and exclusive.
"""

import numpy as np

from pymrst.utils.mcolon import mcolon


def indirection_sub(items, pos, values) -> np.ndarray:
    """Entries of ``values`` belonging to ``items``, concatenated."""
    items = np.atleast_1d(np.asarray(items, dtype=np.int64)).ravel()
    pos = np.asarray(pos, dtype=np.int64)
    return np.asarray(values)[mcolon(pos[items], pos[items + 1])]


indirectionSub = indirection_sub


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.grid_cell_faces import grid_cell_faces

    VALUES = np.array([10, 11, 12, 20, 21, 30, 31, 32, 33])
    POS = np.array([0, 3, 5, 9])          # three items of 3, 2 and 4 entries

    def test_it_gathers_one_item() -> None:
        assert np.array_equal(indirection_sub([1], POS, VALUES), [20, 21])
        print("ok test_it_gathers_one_item")

    def test_it_concatenates_several() -> None:
        assert np.array_equal(indirection_sub([0, 2], POS, VALUES),
                              [10, 11, 12, 30, 31, 32, 33])
        print("ok test_it_concatenates_several")

    def test_the_order_follows_the_request() -> None:
        assert np.array_equal(indirection_sub([2, 0], POS, VALUES),
                              [30, 31, 32, 33, 10, 11, 12])
        print("ok test_the_order_follows_the_request")

    def test_a_repeated_item_appears_twice() -> None:
        assert np.array_equal(indirection_sub([1, 1], POS, VALUES),
                              [20, 21, 20, 21])
        print("ok test_a_repeated_item_appears_twice")

    def test_all_items_reproduces_the_whole_array() -> None:
        assert np.array_equal(indirection_sub([0, 1, 2], POS, VALUES), VALUES)
        print("ok test_all_items_reproduces_the_whole_array")

    def test_it_reads_what_grid_cell_faces_packs() -> None:
        """The layout it exists for."""
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        faces, pos = grid_cell_faces(g)
        assert np.array_equal(indirection_sub([2], pos, faces),
                              faces[pos[2]:pos[3]])
        print("ok test_it_reads_what_grid_cell_faces_packs")

    def test_an_empty_request_gives_an_empty_result() -> None:
        assert indirection_sub([], POS, VALUES).size == 0
        print("ok test_an_empty_request_gives_an_empty_result")

    def test_a_scalar_item_is_accepted() -> None:
        assert np.array_equal(indirection_sub(1, POS, VALUES), [20, 21])
        print("ok test_a_scalar_item_is_accepted")

    test_it_gathers_one_item()
    test_it_concatenates_several()
    test_the_order_follows_the_request()
    test_a_repeated_item_appears_twice()
    test_all_items_reproduces_the_whole_array()
    test_it_reads_what_grid_cell_faces_packs()
    test_an_empty_request_gives_an_empty_result()
    test_a_scalar_item_is_accepted()
    print("\nAll tests passed!")
