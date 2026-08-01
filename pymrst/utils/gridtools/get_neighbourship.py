"""Cell-to-cell connections (MRST ``utils/gridtools/getNeighbourship.m``).

The `(n, 2)` list of cell pairs sharing a face. Two choices:

- ``include_boundary`` keeps the rows where one side is outside. PyMRST marks
  outside as **-1** where MRST uses 0, so a row copied from MATLAB code reads
  cell 0 as "outside" here and as a real cell there.
- ``topological`` adds non-neighbouring connections (NNCs) - pairs joined by
  the deck rather than by geometry, across a fault or between a well and a
  distant cell. Geometry alone misses them, and a transmissibility calculation
  that misses an NNC silently disconnects part of the model.
"""

from typing import Tuple

import numpy as np


def get_neighbourship(g, topological: bool = False,
                      include_boundary: bool = False,
                      with_nnc_flag: bool = False):
    """Return the neighbourship list, optionally with NNCs and boundary rows."""
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    if not include_boundary:
        neighbours = neighbours[np.all(neighbours >= 0, axis=1)]
    is_nnc = np.zeros(neighbours.shape[0], dtype=bool)

    nnc = getattr(g, "nnc", None)
    cells = getattr(nnc, "cells", None) if nnc is not None else None
    if topological and cells is not None:
        cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
        if cells.shape[1] != 2:
            raise ValueError(f"nnc.cells must be n-by-2, got {cells.shape}")
        neighbours = np.vstack([neighbours, cells])
        is_nnc = np.concatenate([is_nnc, np.ones(cells.shape[0], dtype=bool)])

    return (neighbours, is_nnc) if with_nnc_flag else neighbours


getNeighbourship = get_neighbourship


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry

    G = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))

    def test_interior_connections_only_by_default() -> None:
        N = get_neighbourship(G)
        assert np.all(N >= 0), "a boundary row leaked in"
        print(f"ok test_interior_connections_only_by_default ({N.shape[0]} pairs)")

    def test_including_the_boundary_adds_the_outer_faces() -> None:
        interior = get_neighbourship(G).shape[0]
        everything = get_neighbourship(G, include_boundary=True).shape[0]
        assert everything == G.faces.num, everything
        assert everything > interior
        print("ok test_including_the_boundary_adds_the_outer_faces")

    def test_outside_is_minus_one_not_zero() -> None:
        """MRST marks it 0, which here is a legitimate cell index."""
        N = get_neighbourship(G, include_boundary=True)
        assert (N == -1).any() and not (N == -2).any()
        assert 0 in N, "cell 0 is a real cell and must appear"
        print("ok test_outside_is_minus_one_not_zero")

    def test_every_interior_face_appears_once() -> None:
        neighbours = np.asarray(G.faces.neighbors)
        expected = np.count_nonzero(np.all(neighbours >= 0, axis=1))
        assert get_neighbourship(G).shape[0] == expected
        print("ok test_every_interior_face_appears_once")

    def test_non_neighbouring_connections_are_added_when_asked() -> None:
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        g.nnc = SimpleNamespace(cells=np.array([[0, 5]]))
        plain = get_neighbourship(g).shape[0]
        topo = get_neighbourship(g, topological=True).shape[0]
        assert topo == plain + 1, (plain, topo)
        print("ok test_non_neighbouring_connections_are_added_when_asked")

    def test_nncs_are_absent_unless_topological() -> None:
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        g.nnc = SimpleNamespace(cells=np.array([[0, 5]]))
        N = get_neighbourship(g)
        assert not ((N[:, 0] == 0) & (N[:, 1] == 5)).any()
        print("ok test_nncs_are_absent_unless_topological")

    def test_the_flag_marks_which_rows_are_nncs() -> None:
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        g.nnc = SimpleNamespace(cells=np.array([[0, 5], [1, 4]]))
        N, is_nnc = get_neighbourship(g, topological=True, with_nnc_flag=True)
        assert is_nnc.sum() == 2 and np.all(is_nnc[-2:])
        assert np.array_equal(N[is_nnc], [[0, 5], [1, 4]])
        print("ok test_the_flag_marks_which_rows_are_nncs")

    def test_a_grid_without_nncs_is_unaffected() -> None:
        assert np.array_equal(get_neighbourship(G, topological=True),
                              get_neighbourship(G))
        print("ok test_a_grid_without_nncs_is_unaffected")

    def test_a_malformed_nnc_list_is_rejected() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        g.nnc = SimpleNamespace(cells=np.array([[0, 1, 2]]))
        try:
            get_neighbourship(g, topological=True)
        except ValueError as error:
            assert "n-by-2" in str(error)
            print("ok test_a_malformed_nnc_list_is_rejected")
            return
        raise AssertionError("a 3-column nnc list must be rejected")

    test_interior_connections_only_by_default()
    test_including_the_boundary_adds_the_outer_faces()
    test_outside_is_minus_one_not_zero()
    test_every_interior_face_appears_once()
    test_non_neighbouring_connections_are_added_when_asked()
    test_nncs_are_absent_unless_topological()
    test_the_flag_marks_which_rows_are_nncs()
    test_a_grid_without_nncs_is_unaffected()
    test_a_malformed_nnc_list_is_rejected()
    print("\nAll tests passed!")
