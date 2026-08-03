"""Order a 2D coarse grid's sub-edges (MRST ``sortSubEdges.m``).

`generate_coarse_grid` groups fine faces into coarse faces, and coarse faces
into blocks, but leaves both in fine-grid numbering order. This puts them in
geometric order:

* each coarse face's fine faces are ordered along the face, running from one end
  to the other, and turned round if the walk came out against the coarse face's
  own `neighbors` orientation;
* each block's coarse faces are ordered around the block's boundary.

Anything that traces a coarse face or walks a block's outline needs that.
Plotting is the obvious consumer: drawing the fine faces of a coarse face in
numbering order produces a shape that jumps back and forth across the face.

**Two dimensions only**, as MRST states and enforces. A face is a segment with
two endpoints there, so ordering is a walk along a chain. In three dimensions a
coarse face is a polygon patch and "order" has no single meaning. Nested coarse
grids are refused for the same reason MRST refuses them: the fine faces would
themselves be coarse faces, and their endpoints are not stored.

The ordering is applied in place, on `fconn` and `faces`. `conn_pos` and
`face_pos` are untouched, since only the contents of each run move.
"""

from __future__ import annotations

import warnings

import numpy as np

from pymrst.modules.coarsegrid.private.sort_edges import sort_edges


def sort_sub_edges(cg, warn: bool = True):
    """Order `cg.fconn` along each coarse face and `cg.faces` around each block.

    Returns `cg`. Raises for anything but a flat 2D coarse grid.
    """
    if cg.parent is None:
        raise ValueError("sort_sub_edges needs a coarse grid with a parent")
    if getattr(cg.parent, "partition", None) is not None:
        raise ValueError("sort_sub_edges does not support nested coarse grids")
    if int(cg.griddim) != 2:
        raise ValueError(
            f"sort_sub_edges is two-dimensional only (MRST states the same "
            f"restriction); this grid has griddim {cg.griddim}")
    if cg.fconn is None or cg.conn_pos is None:
        raise ValueError("this coarse grid carries no face connectivity")

    _sort_fconn(cg, warn)
    _sort_block_faces(cg)
    return cg


def _fine_edges(cg, faces) -> np.ndarray:
    """The two endpoint nodes of each given fine face."""
    node_pos = np.asarray(cg.parent.faces.nodePos, dtype=np.int64)
    nodes = np.asarray(cg.parent.faces.nodes, dtype=np.int64).ravel()
    counts = node_pos[faces + 1] - node_pos[faces]
    if np.any(counts != 2):
        raise ValueError(
            "a fine face has more than two nodes; sort_sub_edges is 2D only")
    return np.column_stack([nodes[node_pos[faces]], nodes[node_pos[faces] + 1]])


def _sort_fconn(cg, warn: bool) -> None:
    """Order each coarse face's fine faces along the face."""
    fconn = np.asarray(cg.fconn, dtype=np.int64)
    conn_pos = np.asarray(cg.conn_pos, dtype=np.int64)
    order, sign, success = sort_edges(_fine_edges(cg, fconn), conn_pos)

    if warn and not np.all(success):
        warnings.warn(
            f"{int(np.count_nonzero(~success))} coarse faces are disconnected "
            "sets of fine faces and could not be ordered",
            RuntimeWarning, stacklevel=3)

    ordered = fconn[order]
    # Which block each ordered fine face runs from, once its stored orientation
    # is corrected by the walk's sign.
    neighbors = np.asarray(cg.parent.faces.neighbors, dtype=np.int64)[ordered]
    flipped = sign < 0
    neighbors[flipped] = neighbors[flipped][:, ::-1]
    partition = np.asarray(cg.partition, dtype=np.int64)
    blocks = np.where(neighbors >= 0, partition[np.maximum(neighbors, 0)], -1)

    # A coarse face whose walk came out the other way round is reversed whole,
    # so `fconn` reads in the direction `neighbors` declares.
    agrees = np.all(blocks[conn_pos[:-1]] == np.asarray(cg.neighbors,
                                                        dtype=np.int64), axis=1)
    for f in np.flatnonzero(~agrees):
        lo, hi = int(conn_pos[f]), int(conn_pos[f + 1])
        ordered[lo:hi] = ordered[lo:hi][::-1]

    cg.fconn = ordered


def _sort_block_faces(cg) -> None:
    """Order each block's coarse faces around its boundary."""
    if cg.faces is None or cg.face_pos is None:
        return
    edges = _coarse_edges(cg)
    faces = np.asarray(cg.faces, dtype=np.int64)
    order, _, success = sort_edges(edges[faces], np.asarray(cg.face_pos,
                                                            dtype=np.int64))
    if not np.all(success):
        raise ValueError(
            f"{int(np.count_nonzero(~success))} blocks have a boundary that is "
            "not a single closed chain of coarse faces")
    cg.faces = faces[order]


def _coarse_edges(cg) -> np.ndarray:
    """The two endpoint nodes of each coarse face.

    A coarse face's interior nodes are shared by two of its fine faces; its
    endpoints belong to one. So the endpoints are the nodes appearing exactly
    once across the face's fine faces.
    """
    conn_pos = np.asarray(cg.conn_pos, dtype=np.int64)
    counts = np.diff(conn_pos)
    nodes = _fine_edges(cg, np.asarray(cg.fconn, dtype=np.int64)).ravel()
    face_no = np.repeat(np.arange(counts.size, dtype=np.int64), 2 * counts)

    pairs, occurrences = np.unique(np.column_stack([face_no, nodes]), axis=0,
                                   return_counts=True)
    ends = pairs[occurrences == 1]
    if ends.shape[0] != 2 * counts.size:
        raise ValueError(
            "a coarse face does not have exactly two endpoints; it is probably "
            "a disconnected set of fine faces")
    return ends[:, 1].reshape(-1, 2)


__all__ = ["sort_sub_edges"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(6, 6), part=(2, 2)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        p = partition_cart_grid(list(dims) + [1], list(part) + [1])
        return g, p, generate_coarse_grid(g, p)

    def _edges(cg, faces):
        node_pos = np.asarray(cg.parent.faces.nodePos, dtype=np.int64)
        nodes = np.asarray(cg.parent.faces.nodes, dtype=np.int64)
        return np.column_stack([nodes[node_pos[faces]], nodes[node_pos[faces] + 1]])

    def test_fine_faces_come_out_end_to_end():
        """The defining property, and the one the unsorted grid fails.

        Consecutive fine faces of a coarse face must share a node. In fine
        numbering order they generally do not, and a plot of the face then
        jumps back and forth along it.
        """
        _, _, cg = _case()
        before = cg.fconn.copy()
        sort_sub_edges(cg)
        assert not np.array_equal(before, cg.fconn), "nothing was reordered"
        for f in range(cg.num_faces):
            sub = cg.sub_faces(f)
            if sub.size < 2:
                continue
            e = _edges(cg, sub)
            for a, b in zip(e, e[1:]):
                assert set(a.tolist()) & set(b.tolist()), (f, a, b)
        print(f"ok test_fine_faces_come_out_end_to_end ({cg.num_faces} faces)")

    def test_it_is_a_permutation_within_each_coarse_face():
        """Ordering must not move a fine face between coarse faces."""
        _, _, cg = _case()
        before = [set(cg.sub_faces(f).tolist()) for f in range(cg.num_faces)]
        sort_sub_edges(cg)
        after = [set(cg.sub_faces(f).tolist()) for f in range(cg.num_faces)]
        assert before == after, "membership changed"
        print("ok test_it_is_a_permutation_within_each_coarse_face")

    def test_the_run_agrees_with_the_declared_neighbours():
        """A coarse face must read in the direction `neighbors` states.

        The walk starts at whichever fine face came first, so half the faces
        come out backwards and have to be turned round. Without that, tracing
        a coarse face from block A to block B sometimes runs B to A.
        """
        _, p, cg = _case()
        sort_sub_edges(cg)
        nb = np.asarray(cg.parent.faces.neighbors, dtype=np.int64)
        e = _edges(cg, cg.fconn)
        for f in range(cg.num_faces):
            sub = cg.sub_faces(f)
            lo = int(cg.conn_pos[f])
            # Re-derive the sign from the chain, then check the block order.
            first = e[lo]
            forwards = True
            if sub.size > 1 and int(first[1]) not in e[lo + 1].tolist():
                forwards = False
            pair = nb[sub[0]] if forwards else nb[sub[0]][::-1]
            blocks = [int(p[c]) if c >= 0 else -1 for c in pair]
            assert blocks == cg.neighbors[f].tolist(), (f, blocks,
                                                        cg.neighbors[f])
        print("ok test_the_run_agrees_with_the_declared_neighbours")

    def test_block_faces_come_out_around_the_block():
        """Consecutive coarse faces of a block must share an endpoint."""
        _, _, cg = _case()
        sort_sub_edges(cg)
        ends = _coarse_edges(cg)
        for b in range(cg.n_blocks):
            own = cg.faces[cg.face_pos[b]:cg.face_pos[b + 1]]
            assert own.size >= 3, b
            for a, c in zip(own, own[1:]):
                assert set(ends[a].tolist()) & set(ends[c].tolist()), (b, a, c)
            # and the boundary closes
            assert set(ends[own[-1]].tolist()) & set(ends[own[0]].tolist()), b
        print(f"ok test_block_faces_come_out_around_the_block "
              f"({cg.n_blocks} blocks)")

    def test_block_membership_is_unchanged():
        _, _, cg = _case()
        before = [set(cg.faces[cg.face_pos[b]:cg.face_pos[b + 1]].tolist())
                  for b in range(cg.n_blocks)]
        sort_sub_edges(cg)
        after = [set(cg.faces[cg.face_pos[b]:cg.face_pos[b + 1]].tolist())
                 for b in range(cg.n_blocks)]
        assert before == after, "a coarse face changed block"
        print("ok test_block_membership_is_unchanged")

    def test_the_packed_offsets_are_untouched():
        """Only the contents of each run move, never the run boundaries."""
        _, _, cg = _case()
        conn_pos, face_pos = cg.conn_pos.copy(), cg.face_pos.copy()
        sort_sub_edges(cg)
        assert np.array_equal(cg.conn_pos, conn_pos)
        assert np.array_equal(cg.face_pos, face_pos)
        assert cg.fconn.size == conn_pos[-1]
        assert cg.faces.size == face_pos[-1]
        print("ok test_the_packed_offsets_are_untouched")

    def test_fconn_is_stable_but_the_block_cycle_is_not():
        """Half of this is a limitation, and it is MRST's too.

        `fconn` settles, because its direction is pinned by `neighbors`: a run
        that comes out backwards is turned round, so a second pass finds
        nothing to do.

        `faces` does not settle. A block's boundary is a cycle with no natural
        first face, and the walk seeds from whichever face is currently first,
        so sorting again rotates it. The cycle is the same either way, and any
        caller storing an index into `cells.faces` across a second call is
        reading a different face.
        """
        _, _, cg = _case()
        sort_sub_edges(cg)
        once_fconn, once_faces = cg.fconn.copy(), cg.faces.copy()
        sort_sub_edges(cg)
        assert np.array_equal(cg.fconn, once_fconn), "fconn is not stable"
        assert not np.array_equal(cg.faces, once_faces), \
            "the block cycle settled; the caveat above may no longer hold"
        # It is still the same cycle, just started elsewhere.
        for b in range(cg.n_blocks):
            lo, hi = cg.face_pos[b], cg.face_pos[b + 1]
            assert set(cg.faces[lo:hi].tolist()) == set(once_faces[lo:hi].tolist())
        print("ok test_fconn_is_stable_but_the_block_cycle_is_not")

    def test_a_single_block_is_handled():
        _, _, cg = _case(part=(1, 1))
        sort_sub_edges(cg)
        assert cg.n_blocks == 1
        own = cg.faces[cg.face_pos[0]:cg.face_pos[1]]
        assert own.size == cg.num_faces
        print(f"ok test_a_single_block_is_handled ({cg.num_faces} faces)")

    def test_three_dimensions_is_refused():
        """A coarse face is a polygon patch there, with no single ordering."""
        g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
        cg = generate_coarse_grid(g, partition_cart_grid([3, 3, 2], [1, 1, 2]))
        try:
            sort_sub_edges(cg)
        except ValueError as exc:
            assert "two-dimensional" in str(exc), exc
        else:
            raise AssertionError("expected a 3D coarse grid to be refused")
        print("ok test_three_dimensions_is_refused")

    def test_a_nested_coarse_grid_is_refused():
        _, p, cg = _case()
        outer = generate_coarse_grid(cg.g, np.zeros(cg.g.cells.num, dtype=np.int64))
        outer.parent = cg
        try:
            sort_sub_edges(outer)
        except ValueError as exc:
            assert "nested" in str(exc), exc
        else:
            raise AssertionError("expected a nested coarse grid to be refused")
        print("ok test_a_nested_coarse_grid_is_refused")

    test_fine_faces_come_out_end_to_end()
    test_it_is_a_permutation_within_each_coarse_face()
    test_the_run_agrees_with_the_declared_neighbours()
    test_block_faces_come_out_around_the_block()
    test_block_membership_is_unchanged()
    test_the_packed_offsets_are_untouched()
    test_fconn_is_stable_but_the_block_cycle_is_not()
    test_a_single_block_is_handled()
    test_three_dimensions_is_refused()
    test_a_nested_coarse_grid_is_refused()
    print("\nAll tests passed!")
