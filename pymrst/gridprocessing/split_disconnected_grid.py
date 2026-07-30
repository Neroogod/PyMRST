"""Split a grid into its connected components (MRST `splitDisconnectedGrid`).

A grid can fall into pieces that do not communicate - after cells are removed, a
fault fully seals a region, or a fracture network has isolated clusters. A single
linear solve over a disconnected grid is singular (each component floats
independently), so a simulator must handle the pieces separately. This finds the
connected components of the cell-adjacency graph (cells linked by an interior
face or a non-neighbour connection) and returns one subgrid per component,
largest first. A connected grid is returned unchanged.
"""

from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pymrst.gridprocessing.grid_structure import extract_subgrid


def _adjacency(g) -> sp.csr_array:
    """Symmetric cell-adjacency: interior faces and NNCs, with self-loops."""
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    interior = neigh[np.all(neigh >= 0, axis=1)]
    pairs = [interior]
    nnc = getattr(g, "nnc", None)
    if nnc is not None:
        cells = nnc.get("cells") if isinstance(nnc, dict) else getattr(nnc, "cells", None)
        if cells is not None:
            pairs.append(np.asarray(cells, dtype=np.int64))
    n = np.vstack(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)

    nc = g.cells.num
    i = np.concatenate([n[:, 0], n[:, 1], np.arange(nc)])
    j = np.concatenate([n[:, 1], n[:, 0], np.arange(nc)])
    return sp.csr_array((np.ones(i.size), (i, j)), shape=(nc, nc))


def split_disconnected_grid(g, *, verbose: bool = False) -> List[object]:
    """Return the grid's connected components as subgrids, largest first.

    A connected grid returns `[g]`. Each subgrid keeps the parent's `cartDims`,
    as MRST does.
    """
    ncomp, labels = connected_components(_adjacency(g), directed=False)
    if verbose and ncomp > 1:
        print(f"Grid has {ncomp} disconnected components")
    if ncomp <= 1:
        return [g]

    subs = []
    for c in range(ncomp):
        cells = np.flatnonzero(labels == c)
        sub = extract_subgrid(g, cells)
        sub.cartDims = g.cartDims
        subs.append((cells.size, sub))

    subs.sort(key=lambda item: -item[0])            # decreasing cell count
    return [sub for _, sub in subs]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry, remove_cells

    def test_connected_grid_returns_itself():
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        parts = split_disconnected_grid(g)
        assert len(parts) == 1 and parts[0] is g
        print("ok test_connected_grid_returns_itself")

    def test_two_components_after_removing_a_separating_column():
        """Remove the middle column of a 3x1 grid -> two isolated end cells."""
        g = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        g2 = remove_cells(g, np.array([1]))          # drop the middle cell
        parts = split_disconnected_grid(g2)
        assert len(parts) == 2, len(parts)
        assert all(p.cells.num == 1 for p in parts)
        print("ok test_two_components_after_removing_a_separating_column")

    def test_components_sorted_by_decreasing_size():
        """A 5x1 strip cut in two unequal pieces returns the larger first."""
        g = compute_geometry(cart_grid([5, 1, 1], [5.0, 1.0, 1.0]))
        g2 = remove_cells(g, np.array([1]))          # split into [1 cell][3 cells]
        parts = split_disconnected_grid(g2)
        assert len(parts) == 2
        assert parts[0].cells.num >= parts[1].cells.num
        assert parts[0].cells.num == 3 and parts[1].cells.num == 1
        print("ok test_components_sorted_by_decreasing_size")

    def test_total_cells_preserved_and_cartdims_propagated():
        g = compute_geometry(cart_grid([4, 4, 1], [4.0, 4.0, 1.0]))
        # remove a full interior cross to make several islands
        g2 = remove_cells(g, np.array([1, 4, 5, 6, 9]))
        g2.cartDims = g.cartDims                      # keep the logical dims
        parts = split_disconnected_grid(g2)
        assert sum(p.cells.num for p in parts) == g2.cells.num
        # each component inherits the input grid's cartDims (MRST's behaviour)
        assert all(p.cartDims == g2.cartDims for p in parts)
        print(f"ok test_total_cells_preserved_and_cartdims_propagated "
              f"({len(parts)} components)")

    test_connected_grid_returns_itself()
    test_two_components_after_removing_a_separating_column()
    test_components_sorted_by_decreasing_size()
    test_total_cells_preserved_and_cartdims_propagated()
    print("\nAll tests passed!")
