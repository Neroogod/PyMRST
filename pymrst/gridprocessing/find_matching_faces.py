"""Find pairs of faces sharing the same node set (MRST `findMatchingFaces`).

Two faces built on the identical set of nodes are geometric duplicates: the two
coincident faces of an internal boundary, or the matching faces on either side of
a fault where two grids meet. This groups faces by their (sorted) node set and
returns the pairs - the first step in re-welding a fault or reconnecting a split.
At most two faces may share a node set (an assertion, as in MRST).
"""

import numpy as np

from pymrst.utils.gridtools.rlencode import rlencode


def find_matching_faces(g) -> np.ndarray:
    """Return an `(m, 2)` array of face-index pairs sharing the same nodes."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    counts = np.diff(node_pos)
    maxn = int(counts.max()) if g.faces.num else 0

    # each face's sorted node set as a fixed-width key, padded with -1 (0 is a
    # valid node here, unlike MRST's 1-based padding)
    key = np.full((g.faces.num, maxn), -1, dtype=np.int64)
    for f in range(g.faces.num):
        seg = np.sort(fnodes[node_pos[f]:node_pos[f + 1]])
        key[f, :seg.size] = seg

    order = np.lexsort(key.T[::-1])                 # sortrows
    _, num = rlencode(key[order])
    is_pair = np.repeat(num == 2, num)
    return order[is_pair].reshape(-1, 2)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.gridprocessing.make_internal_boundary import make_internal_boundary

    def _face_node_set(g, f):
        np_ = np.asarray(g.faces.nodePos)
        return frozenset(np.asarray(g.faces.nodes)[np_[f]:np_[f + 1]].tolist())

    def test_no_matches_in_a_plain_grid():
        """A Cartesian grid has no coincident faces."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        assert find_matching_faces(g).shape[0] == 0
        print("ok test_no_matches_in_a_plain_grid")

    def test_finds_the_pairs_made_by_an_internal_boundary():
        """After cutting k internal faces, findMatchingFaces recovers exactly k
        coincident pairs, each sharing the same node set."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        cut = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))[:3]
        g2, _ = make_internal_boundary(g, cut)
        pairs = find_matching_faces(g2)
        assert pairs.shape[0] == len(cut), (pairs.shape, len(cut))
        for a, b in pairs:
            assert _face_node_set(g2, a) == _face_node_set(g2, b)
        print(f"ok test_finds_the_pairs_made_by_an_internal_boundary "
              f"({pairs.shape[0]} pairs)")

    def test_matched_faces_are_boundary_faces():
        """The coincident faces from a cut are both boundary faces."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        cut = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))[:1]
        g2, _ = make_internal_boundary(g, cut)
        pairs = find_matching_faces(g2)
        neigh = np.asarray(g2.faces.neighbors)
        for a, b in pairs:
            assert (neigh[a] == -1).any() and (neigh[b] == -1).any()
        print("ok test_matched_faces_are_boundary_faces")

    test_no_matches_in_a_plain_grid()
    test_finds_the_pairs_made_by_an_internal_boundary()
    test_matched_faces_are_boundary_faces()
    print("\nAll tests passed!")
