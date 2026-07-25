"""Merge coincident boundary faces back into internal faces (MRST
`removeInternalBoundary`).

The inverse of `make_internal_boundary`: given the pairing `N` it returned - the
two boundary faces that replaced each cut face - this welds each pair back into a
single internal face, restoring communication between the two cells. It merges
the pair's coincident nodes, removes the two boundary faces, and adds one face
carrying both original cells as neighbours. Round-tripping
`make_internal_boundary` then `remove_internal_boundary` returns the grid to its
original topology.
"""

from typing import Tuple

import numpy as np

from pymrst.gridprocessing.make_internal_boundary import _copy_faces
from pymrst.gridprocessing.private.add_faces import add_faces
from pymrst.gridprocessing.private.remove_faces import remove_faces


def _unique_stable_rows(a: np.ndarray) -> np.ndarray:
    """Unique rows in first-appearance order (MRST `uniqueStable(..., 'rows')`)."""
    seen = set()
    keep = []
    for i, row in enumerate(map(tuple, a.tolist())):
        if row not in seen:
            seen.add(row)
            keep.append(i)
    return a[keep]


def _replace_nodes(g, replacement: np.ndarray, remove: np.ndarray) -> None:
    """Point every reference to a `remove` node at the matching `replacement`
    node (MRST `replaceNodes`), welding the coincident nodes of a pair."""
    node_map = np.arange(g.nodes.num, dtype=np.int64)
    node_map[remove] = replacement
    g.faces.nodes = node_map[np.asarray(g.faces.nodes, dtype=np.int64)]


def remove_internal_boundary(g, N) -> Tuple[object, np.ndarray]:
    """Weld the boundary-face pairs in `N` back into internal faces.

    `N` is the `(m, 2)` pairing returned by `make_internal_boundary`. Returns the
    grid and the indices of the merged faces.
    """
    N = np.atleast_2d(np.asarray(N, dtype=float))
    ok = ~np.any(np.isnan(N), axis=1) & (N[:, 0] != N[:, 1])
    N = _unique_stable_rows(N[ok].astype(np.int64))
    if N.size == 0:
        return g, np.array([], dtype=np.int64)

    f1, f2 = N[:, 0], N[:, 1]
    fnodes1, nnodes1, neigh1 = _copy_faces(g, f1)
    fnodes2, _, neigh2 = _copy_faces(g, f2)

    _replace_nodes(g, fnodes1, fnodes2)

    g, _ = remove_faces(g, N.ravel())

    # restore the internal neighbour pair: fill each boundary face's outside
    # slot (-1) with the real cell from its partner
    merged = neigh1.copy()
    real2 = neigh2[neigh2 != -1]                 # one real cell per pair, in order
    mask = merged == -1
    merged[mask] = real2

    g, f = add_faces(g, fnodes1, nnodes1, merged)
    return g, f


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.gridprocessing.make_internal_boundary import make_internal_boundary

    def _sorted_pairs(g):
        nb = np.sort(np.asarray(g.faces.neighbors, dtype=np.int64), axis=1)
        return nb[np.lexsort((nb[:, 1], nb[:, 0]))]

    def test_round_trip_recovers_topology():
        """make_internal_boundary then remove_internal_boundary returns the grid
        to its original face count and neighbour-pair multiset."""
        g0 = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
        neigh = np.asarray(g0.faces.neighbors)
        cut = np.flatnonzero(np.all(neigh >= 0, axis=1))[::2]
        pairs0 = _sorted_pairs(g0)
        nf0 = g0.faces.num

        g1, N = make_internal_boundary(g0, cut)
        g2, _ = remove_internal_boundary(g1, N)

        assert g2.faces.num == nf0, (g2.faces.num, nf0)
        assert np.array_equal(_sorted_pairs(g2), pairs0), \
            "round trip must restore the neighbour-pair multiset"
        print(f"ok test_round_trip_recovers_topology ({nf0} faces recovered)")

    def test_single_cut_round_trip():
        """A single internal face: cut it, weld it, get one internal face back
        connecting the same two cells."""
        g0 = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        cut = np.flatnonzero(np.all(np.asarray(g0.faces.neighbors) >= 0, axis=1))
        c1, c2 = np.asarray(g0.faces.neighbors)[cut[0]]
        g1, N = make_internal_boundary(g0, cut)
        g2, f = remove_internal_boundary(g1, N)
        nn = np.sort(np.asarray(g2.faces.neighbors), axis=1)
        assert np.any(np.all(nn == np.sort([c1, c2]), axis=1)), \
            "the welded face must reconnect the two cells"
        print("ok test_single_cut_round_trip")

    def test_nan_rows_are_ignored():
        """A pairing with NaN rows (boundary faces that were not cut) merges only
        the real pairs."""
        g0 = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        cut = np.flatnonzero(np.all(np.asarray(g0.faces.neighbors) >= 0, axis=1))[:1]
        nf0 = g0.faces.num
        g1, N = make_internal_boundary(g0, cut)
        N_padded = np.vstack([N, [np.nan, np.nan]])
        g2, _ = remove_internal_boundary(g1, N_padded)
        assert g2.faces.num == nf0
        print("ok test_nan_rows_are_ignored")

    test_round_trip_recovers_topology()
    test_single_cut_round_trip()
    test_nan_rows_are_ignored()
    print("\nAll tests passed!")
