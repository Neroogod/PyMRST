"""Split faces into internal boundaries (MRST `makeInternalBoundary`).

An *internal boundary* is a pair of coincident faces where cells used to share
one: the grid is cut along those faces so the two sides no longer communicate
through them. This is how a sealing fault or an explicit fracture is put into a
grid - the flow across the cut is then re-introduced (if at all) as a
transmissibility multiplier or a non-neighbour connection, not as an ordinary
face. Each supplied *internal* face is replaced by two boundary faces with the
same nodes, one carrying each of the original neighbour cells; boundary faces in
the input are left alone.

Composed from the Phase-23 face-editing primitives (`copy` the faces, remove the
originals, add the doubled replacements), exactly as MRST composes it from
`copyFaces` / `removeFaces` / `addFaces`.
"""

from typing import Tuple

import numpy as np

from pymrst.gridprocessing.private.add_faces import add_faces
from pymrst.gridprocessing.private.remove_faces import remove_faces


def _copy_faces(g, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node lists, node counts and neighbours of `faces` (MRST `copyFaces`)."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)
    nnodes = node_pos[faces + 1] - node_pos[faces]
    fnodes = np.concatenate(
        [nodes[node_pos[f]:node_pos[f + 1]] for f in faces]
    ) if faces.size else np.array([], dtype=np.int64)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)[faces]
    return fnodes, nnodes, neigh


def make_internal_boundary(g, faces) -> Tuple[object, np.ndarray]:
    """Cut `g` along `faces`, returning `(grid, N)`.

    `N` has one row per input face: the two indices of the boundary faces that
    replaced it, or `NaN` for an input face that was already on the boundary and
    left unchanged.
    """
    faces_u, iu = np.unique(np.asarray(faces, dtype=np.int64), return_inverse=True)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)[faces_u]
    internal = ~np.any(neigh == -1, axis=1)
    n = int(internal.sum())
    if n == 0:
        return g, np.full((iu.size, 2), np.nan)

    faces_int = faces_u[internal]
    fnodes, nnodes, neigh_int = _copy_faces(g, faces_int)

    # each internal face becomes two boundary faces: [c1, outside] and
    # [outside, c2], interleaved so a face's pair is adjacent
    neigh_new = np.empty((2 * n, 2), dtype=np.int64)
    neigh_new[0::2, 0] = neigh_int[:, 0]
    neigh_new[0::2, 1] = -1
    neigh_new[1::2, 0] = -1
    neigh_new[1::2, 1] = neigh_int[:, 1]

    g, _ = remove_faces(g, faces_int)

    # duplicate each face's node list (the two replacements are coincident)
    offs = np.concatenate([[0], np.cumsum(nnodes)])
    pieces = []
    for i in range(n):
        seg = fnodes[offs[i]:offs[i + 1]]
        pieces.append(seg)
        pieces.append(seg)
    fnodes_new = np.concatenate(pieces) if pieces else np.array([], dtype=np.int64)
    nnodes_new = np.repeat(nnodes, 2)

    base = g.faces.num                                  # new faces append here
    new_idx = base + np.arange(2 * n).reshape(n, 2)
    N = np.full((faces_u.size, 2), np.nan)
    N[internal] = new_idx
    N = N[iu]

    g, _ = add_faces(g, fnodes_new, nnodes_new, neigh_new)
    return g, N


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def _internal_faces(g):
        neigh = np.asarray(g.faces.neighbors)
        return np.flatnonzero(np.all(neigh >= 0, axis=1))

    def test_split_raises_face_count_by_number_of_internal_faces():
        """Each internal face (1) becomes two boundary faces (+1 net)."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        nf0 = g.faces.num
        cut = _internal_faces(g)[:2]
        g2, N = make_internal_boundary(g, cut)
        assert g2.faces.num == nf0 + len(cut), (g2.faces.num, nf0)
        assert N.shape == (len(cut), 2) and np.all(np.isfinite(N))
        print(f"ok test_split_raises_face_count_by_number_of_internal_faces "
              f"({nf0} -> {g2.faces.num})")

    def test_replacements_are_boundary_faces():
        """The two faces that replace a split face are each boundary faces, one
        per original neighbour cell."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        cut = _internal_faces(g)[:1]
        c1, c2 = np.asarray(g.faces.neighbors)[cut[0]]
        g2, N = make_internal_boundary(g, cut)
        f0, f1 = N[0].astype(int)
        nn = np.asarray(g2.faces.neighbors)
        # each replacement has exactly one outside (-1) neighbour
        assert (nn[f0] == -1).sum() == 1 and (nn[f1] == -1).sum() == 1
        cells = set(nn[f0]) | set(nn[f1])
        assert {c1, c2} <= cells, (cells, c1, c2)
        print("ok test_replacements_are_boundary_faces")

    def test_split_cells_are_disconnected():
        """After cutting the shared face, the two cells no longer share any
        face - the defining property of an internal boundary."""
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        cut = _internal_faces(g)                     # the single interior face
        c1, c2 = np.asarray(g.faces.neighbors)[cut[0]]
        g2, _ = make_internal_boundary(g, cut)
        nn = np.sort(np.asarray(g2.faces.neighbors), axis=1)
        shared = np.any(np.all(nn == np.sort([c1, c2]), axis=1))
        assert not shared, "the split cells must no longer share a face"
        print("ok test_split_cells_are_disconnected")

    def test_boundary_faces_in_input_are_left_alone():
        """Passing only boundary faces changes nothing and returns all-NaN N."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        neigh = np.asarray(g.faces.neighbors)
        bnd = np.flatnonzero(np.any(neigh == -1, axis=1))[:2]
        nf0 = g.faces.num
        g2, N = make_internal_boundary(g, bnd)
        assert g2.faces.num == nf0
        assert np.all(np.isnan(N))
        print("ok test_boundary_faces_in_input_are_left_alone")

    test_split_raises_face_count_by_number_of_internal_faces()
    test_replacements_are_boundary_faces()
    test_split_cells_are_disconnected()
    test_boundary_faces_in_input_are_left_alone()
    print("\nAll tests passed!")
