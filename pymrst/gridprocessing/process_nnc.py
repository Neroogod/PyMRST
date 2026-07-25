"""Turn an explicit non-neighbour-connection list into cell pairs (MRST
`processNNC`).

A non-neighbour connection (NNC) is a flow path between two cells that do *not*
share a face - the way a deck expresses a fault juxtaposition, a pinch-out, or,
crucially for the fracture modules, a fracture-to-matrix connection. The input is
the simplified explicit NNC list a deck carries: rows of
`[i1 j1 k1  i2 j2 k2  trans]` in logical (i,j,k) coordinates. This resolves each
row to a pair of *active* cell indices and its transmissibility, dropping any row
that references an inactive cell, and - optionally - any connection that already
exists as a real face or a previously registered NNC.

This is a prerequisite for the fracture tiers, where matrix/fracture coupling is
carried entirely as NNCs.
"""

from typing import Optional

import numpy as np


def _cart_to_active(g) -> np.ndarray:
    """Map a linear Cartesian index to its active-cell index, or -1 if
    inactive (the inverse of `cells.indexMap`)."""
    n_cart = int(np.prod(g.cartDims))
    lut = np.full(n_cart, -1, dtype=np.int64)
    idx = np.asarray(g.cells.indexMap, dtype=np.int64)
    lut[idx] = np.arange(g.cells.num, dtype=np.int64)
    return lut


def process_nnc(g, nnc: np.ndarray, exclude_existing: bool = False):
    """Resolve an explicit NNC list to `{'cells': (m,2), 'trans': (m,)}`.

    `nnc` is `(n, 7)`: `[i1 j1 k1 i2 j2 k2 trans]` with 1-based logical indices,
    as in a deck. Rows touching an inactive cell are dropped. With
    `exclude_existing`, connections that are already faces (or already in
    `g.nnc`) are dropped too. Returns `None` if nothing survives.
    """
    nnc = np.atleast_2d(np.asarray(nnc, dtype=float))
    if nnc.shape[1] != 7:
        raise ValueError("processNNC supports the simplified 7-column NNC list "
                         "[i1 j1 k1 i2 j2 k2 trans]")
    dims = tuple(int(d) for d in g.cartDims)

    def s2i(ijk):
        ijk = ijk.astype(np.int64) - 1                 # 1-based -> 0-based
        return np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]),
                                    dims, order="F")

    n_cart = int(np.prod(dims))
    act = np.zeros(n_cart, dtype=bool)
    act[np.asarray(g.cells.indexMap, dtype=np.int64)] = True

    i1 = s2i(nnc[:, 0:3])
    i2 = s2i(nnc[:, 3:6])
    keep = act[i1] & act[i2]

    lut = _cart_to_active(g)
    cells = np.column_stack([lut[i1[keep]], lut[i2[keep]]])
    trans = nnc[keep, 6]

    if exclude_existing and cells.size:
        neigh = np.sort(np.asarray(g.faces.neighbors, dtype=np.int64), axis=1)
        pairs = np.sort(cells, axis=1)
        existing = set(map(tuple, neigh.tolist()))
        mask = np.array([tuple(p) not in existing for p in pairs.tolist()])
        cells, trans = cells[mask], trans[mask]

        prior = getattr(g, "nnc", None)
        if cells.size and prior is not None and "cells" in prior:
            pc = np.sort(np.asarray(prior["cells"], dtype=np.int64), axis=1)
            prior_set = set(map(tuple, pc.tolist()))
            pairs = np.sort(cells, axis=1)
            mask = np.array([tuple(p) not in prior_set for p in pairs.tolist()])
            cells, trans = cells[mask], trans[mask]

    if cells.size == 0:
        return None
    return {"cells": cells, "trans": trans}


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_connects_two_active_cells():
        """An NNC between (1,1,1) and (3,3,1) resolves to the right active-cell
        pair and carries its transmissibility."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        nnc = np.array([[1, 1, 1, 3, 3, 1, 0.5]])
        out = process_nnc(g, nnc)
        # (1,1,1)->cell 0 ; (3,3,1)->cell 8 (0-based, column-major)
        assert out["cells"].tolist() == [[0, 8]], out["cells"]
        assert np.allclose(out["trans"], [0.5])
        print("ok test_connects_two_active_cells")

    def test_inactive_cell_rows_are_dropped():
        """A row referencing a cell outside the active set is discarded; a grid
        with all cells active keeps everything, so shrink the active set."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        # fake an inactive cell 8 by trimming indexMap and cartDims stays 3x3
        g.cells.indexMap = np.arange(8, dtype=np.int64)   # cell (3,3,1) inactive
        g.cells.num = 8
        nnc = np.array([[1, 1, 1, 3, 3, 1, 0.5],     # touches inactive -> drop
                        [1, 1, 1, 2, 1, 1, 0.7]])    # both active -> keep
        out = process_nnc(g, nnc)
        assert out["cells"].shape[0] == 1, out["cells"]
        assert np.allclose(out["trans"], [0.7])
        print("ok test_inactive_cell_rows_are_dropped")

    def test_exclude_existing_drops_real_faces():
        """(1,1,1)-(2,1,1) are already face neighbours; exclude_existing drops
        that NNC but keeps a genuine non-neighbour one."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        nnc = np.array([[1, 1, 1, 2, 1, 1, 0.5],     # a real face -> drop
                        [1, 1, 1, 3, 3, 1, 0.9]])    # not a face -> keep
        out = process_nnc(g, nnc, exclude_existing=True)
        assert out["cells"].tolist() == [[0, 8]], out["cells"]
        assert np.allclose(out["trans"], [0.9])
        print("ok test_exclude_existing_drops_real_faces")

    def test_none_when_nothing_survives():
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        nnc = np.array([[1, 1, 1, 2, 1, 1, 0.5]])    # the only face, excluded
        assert process_nnc(g, nnc, exclude_existing=True) is None
        print("ok test_none_when_nothing_survives")

    test_connects_two_active_cells()
    test_inactive_cell_rows_are_dropped()
    test_exclude_existing_drops_real_faces()
    test_none_when_nothing_survives()
    print("\nAll tests passed!")
