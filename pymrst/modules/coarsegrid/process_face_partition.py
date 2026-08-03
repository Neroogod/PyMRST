"""Split disconnected coarse faces (MRST ``processFacePartition.m``).

The face-side counterpart of `process_partition`. A face partition can give the
same id to fine faces that never touch, exactly as a cell partition can give
the same block to cells that never touch. `generate_coarse_grid` would then
build one coarse face out of two separate strips, and the resulting coarse grid
still looks entirely reasonable: the block pairs are right, the fine faces are
all accounted for, and only the geometry gives it away.

This walks along the faces, cutting the walk wherever two faces disagree, and
gives each connected run its own id.

**Two dimensions only.** MRST states this and enforces it, and the port does the
same rather than inventing a 3D extension. In 2D a face is a segment with two
endpoints, so "connected" means sharing an endpoint and the relation is a
simple graph. In 3D a face is a polygon and adjacency runs along shared edges,
which is a different construction MRST has not written.

Only faces on a block boundary are considered: a face interior to a block is
not part of any coarse face, and `generate_coarse_grid` discards it. Their ids
are left untouched, which means the result may reuse an id that some interior
face already carries. That is harmless for the same reason, and it is what
MATLAB does.

**Which nodes conduct.** MRST keeps a node only when *every* face meeting there
agrees on both its partition id and its block pair. A node where three faces
meet, or where two faces of different coarse faces meet, is a junction, and a
walk must not pass through it. Otherwise two coarse faces that merely touch at
a corner would be merged back together.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def process_face_partition(g, partition, face_partition) -> np.ndarray:
    """Renumber `face_partition` so each coarse face is a connected run.

    Returns a new face partition. Faces interior to a block keep their id.
    """
    griddim = int(getattr(g, "griddim", 0))
    if griddim != 2:
        raise ValueError(
            f"process_face_partition is two-dimensional only (MRST states the "
            f"same restriction); this grid has griddim {griddim}")

    partition = np.asarray(partition, dtype=np.int64).ravel()
    if partition.size != int(g.cells.num):
        raise ValueError(
            f"partition has {partition.size} entries for {g.cells.num} cells")
    face_partition = np.asarray(face_partition, dtype=np.int64).ravel().copy()
    if face_partition.size != int(g.faces.num):
        raise ValueError(
            f"face_partition has {face_partition.size} entries for "
            f"{g.faces.num} faces")

    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    blocks = np.sort(
        np.where(neighbors >= 0, partition[np.maximum(neighbors, 0)], -1), axis=1)

    # Faces on the boundary of a block: everything a coarse face can be made of.
    on_boundary = np.any(neighbors < 0, axis=1) | (blocks[:, 0] != blocks[:, 1])
    f = np.flatnonzero(on_boundary)
    if f.size == 0:
        return face_partition

    node, incident_face = _node_to_face(g, f)
    keep = _conducting_nodes(node, incident_face, face_partition, blocks)

    node, incident_face = node[keep], incident_face[keep]
    counts = np.bincount(np.unique(node, return_inverse=True)[1])
    if counts.size and np.any(counts != 2):
        raise ValueError(
            "a conducting node joins "
            f"{counts.min()} to {counts.max()} faces, expected exactly 2")

    local = np.full(int(g.faces.num), -1, dtype=np.int64)
    local[f] = np.arange(f.size)
    pairs = local[incident_face].reshape(-1, 2)
    graph = sp.coo_matrix(
        (np.ones(pairs.shape[0], dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(f.size, f.size))
    _, component = connected_components(graph, directed=False)

    face_partition[f] = component.astype(np.int64)
    return face_partition


def _node_to_face(g, f: np.ndarray):
    """Every (node, face) incidence for the faces in `f`, sorted by node."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    nodes = np.asarray(g.faces.nodes, dtype=np.int64).ravel()
    counts = node_pos[f + 1] - node_pos[f]
    idx = np.concatenate([np.arange(node_pos[a], node_pos[a + 1]) for a in f]) \
        if f.size else np.empty(0, dtype=np.int64)
    incident_face = np.repeat(f, counts)
    node = nodes[idx]
    order = np.lexsort((incident_face, node))
    return node[order], incident_face[order]


def _conducting_nodes(node, incident_face, face_partition, blocks) -> np.ndarray:
    """Mask over incidences: keep those at nodes where every face agrees.

    Agreement is on the face-partition id and on the block pair together. Both
    matter: two faces can share a partition id while separating different
    blocks, and joining them would produce a coarse face with no single pair of
    neighbours.
    """
    _, group, group_counts = np.unique(node, return_inverse=True,
                                       return_counts=True)
    attributes = np.column_stack([face_partition[incident_face],
                                  blocks[incident_face]])
    first = np.concatenate([[0], np.cumsum(group_counts)[:-1]])
    same = np.all(attributes == attributes[np.repeat(first, group_counts)], axis=1)
    unanimous = np.bincount(group, weights=same.astype(np.int64)) == group_counts
    return unanimous[group]


__all__ = ["process_face_partition"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.cell_partition_to_face_partition import (
        cell_partition_to_face_partition,
    )
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _checkerboard(dims=(8, 8)):
        """MRST's own fixture, from ``showCoarseGrid.m``.

        Diagonal stripes across a 2x2 block partition. The stripes cross each
        block interface several times, so one pf id lands on several disjoint
        runs of the *same* block pair. That is the case the function exists
        for, and a simple checkerboard does not produce it: there the ids
        already differ wherever the runs do.
        """
        g = compute_geometry(cart_grid(list(dims), [1.0, 1.0]))
        pv = partition_cart_grid(list(dims) + [1], [2, 2, 1])
        c = np.asarray(g.cells.centroids, dtype=float)
        cell_tag = (np.sin(3.0 * np.pi * (c[:, 0] - c[:, 1])) > 0).astype(np.int64)
        return g, pv, cell_partition_to_face_partition(g, cell_tag)

    def _runs(g, pf, faces):
        """Group the given faces by id, then by connectivity through nodes."""
        node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
        nodes = np.asarray(g.faces.nodes, dtype=np.int64)
        out = 0
        for tag in np.unique(pf[faces]):
            sel = faces[pf[faces] == tag]
            index = {int(a): i for i, a in enumerate(sel)}
            graph = sp.lil_matrix((sel.size, sel.size))
            by_node: dict[int, list[int]] = {}
            for a in sel:
                for n in nodes[node_pos[a]:node_pos[a + 1]]:
                    by_node.setdefault(int(n), []).append(index[int(a)])
            for members in by_node.values():
                for x in members[1:]:
                    graph[members[0], x] = 1
            out += connected_components(graph.tocsr(), directed=False)[0]
        return out

    def test_a_disconnected_coarse_face_is_split():
        """The reason the function exists.

        Without it the two arms of a checkerboard indicator share one id, and
        `generate_coarse_grid` builds a single coarse face out of strips that
        never touch. Nothing raises and no count looks wrong.
        """
        g, pv, pf = _checkerboard()
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        b = np.sort(np.where(nb >= 0, pv[np.maximum(nb, 0)], -1), axis=1)
        f = np.flatnonzero(np.any(nb < 0, axis=1) | (b[:, 0] != b[:, 1]))
        before = len(set(pf[f].tolist()))
        after_pf = process_face_partition(g, pv, pf)
        after = len(set(after_pf[f].tolist()))
        assert after > before, (before, after)
        print(f"ok test_a_disconnected_coarse_face_is_split "
              f"({before} ids became {after})")

    def test_every_resulting_group_is_connected():
        """The postcondition, checked directly rather than by re-running."""
        g, pv, pf = _checkerboard()
        q = process_face_partition(g, pv, pf)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        b = np.sort(np.where(nb >= 0, pv[np.maximum(nb, 0)], -1), axis=1)
        f = np.flatnonzero(np.any(nb < 0, axis=1) | (b[:, 0] != b[:, 1]))
        assert _runs(g, q, f) == len(set(q[f].tolist())), "a group is still split"
        print("ok test_every_resulting_group_is_connected")

    def test_it_only_ever_splits():
        """Faces given different ids must never be merged.

        Merging would coarsen the face partition the user asked for, and the
        result would still be a valid coarse grid.
        """
        g, pv, pf = _checkerboard()
        q = process_face_partition(g, pv, pf)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        b = np.sort(np.where(nb >= 0, pv[np.maximum(nb, 0)], -1), axis=1)
        f = np.flatnonzero(np.any(nb < 0, axis=1) | (b[:, 0] != b[:, 1]))
        for i, a in enumerate(f):
            for c in f[i + 1:]:
                if q[a] == q[c]:
                    assert pf[a] == pf[c], (a, c)
        print("ok test_it_only_ever_splits")

    def test_the_block_pair_is_never_crossed():
        """A run must not walk from one block pair into another.

        Two coarse faces can meet at a corner while separating different
        blocks. Joining them there gives a coarse face with no single pair of
        neighbours, which `generate_coarse_grid` then resolves by taking
        whichever pair happens to come first.
        """
        g, pv, pf = _checkerboard()
        q = process_face_partition(g, pv, pf)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        b = np.sort(np.where(nb >= 0, pv[np.maximum(nb, 0)], -1), axis=1)
        f = np.flatnonzero(np.any(nb < 0, axis=1) | (b[:, 0] != b[:, 1]))
        for tag in np.unique(q[f]):
            sel = f[q[f] == tag]
            assert np.all(b[sel] == b[sel[0]]), (tag, b[sel])
        print("ok test_the_block_pair_is_never_crossed")

    def test_an_already_connected_partition_keeps_its_grouping():
        """Ids may be renumbered, but nothing may move between groups."""
        g = compute_geometry(cart_grid([4, 4], [4.0, 4.0]))
        pv = partition_cart_grid([4, 4, 1], [2, 2, 1])
        pf = cell_partition_to_face_partition(g, pv)
        q = process_face_partition(g, pv, pf)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        b = np.sort(np.where(nb >= 0, pv[np.maximum(nb, 0)], -1), axis=1)
        f = np.flatnonzero(np.any(nb < 0, axis=1) | (b[:, 0] != b[:, 1]))
        assert (len(set(zip(pf[f].tolist(), map(tuple, b[f]))))
                == len(set(q[f].tolist()))), "grouping changed"
        print("ok test_an_already_connected_partition_keeps_its_grouping")

    def test_interior_faces_are_left_alone():
        g, pv, pf = _checkerboard()
        q = process_face_partition(g, pv, pf)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        b = np.where(nb >= 0, pv[np.maximum(nb, 0)], -1)
        inside = np.flatnonzero(np.all(nb >= 0, axis=1) & (b[:, 0] == b[:, 1]))
        assert inside.size > 0
        assert np.array_equal(q[inside], pf[inside]), "an interior face was renumbered"
        print(f"ok test_interior_faces_are_left_alone ({inside.size} faces)")

    def test_the_input_is_not_mutated():
        g, pv, pf = _checkerboard()
        before = pf.copy()
        process_face_partition(g, pv, pf)
        assert np.array_equal(pf, before), "the face partition was modified in place"
        print("ok test_the_input_is_not_mutated")

    def test_it_feeds_generate_coarse_grid():
        """The whole point: more coarse faces, each of them connected."""
        g, pv, pf = _checkerboard()
        plain = generate_coarse_grid(g, pv, pf)
        processed = generate_coarse_grid(g, pv, process_face_partition(g, pv, pf))
        assert processed.num_faces > plain.num_faces, (plain.num_faces,
                                                       processed.num_faces)
        seen = np.concatenate([processed.sub_faces(i)
                               for i in range(processed.num_faces)])
        assert np.unique(seen).size == seen.size, "a fine face was reused"
        print(f"ok test_it_feeds_generate_coarse_grid "
              f"({plain.num_faces} coarse faces became {processed.num_faces})")

    def test_three_dimensions_is_refused():
        """MRST refuses too. Guessing a 3D rule would be worse than refusing."""
        g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
        p = partition_cart_grid([3, 3, 2], [1, 1, 2])
        try:
            process_face_partition(g, p, cell_partition_to_face_partition(g, p))
        except ValueError as exc:
            assert "two-dimensional" in str(exc), exc
        else:
            raise AssertionError("expected a 3D grid to be refused")
        print("ok test_three_dimensions_is_refused")

    def test_bad_input_is_rejected():
        g, pv, pf = _checkerboard()
        for args, why in (((g, pv[:-1], pf), "wrong-length partition"),
                          ((g, pv, pf[:-1]), "wrong-length face partition")):
            try:
                process_face_partition(*args)
            except ValueError as exc:
                assert "entries for" in str(exc), exc
            else:
                raise AssertionError(f"expected {why} to be rejected")
        print("ok test_bad_input_is_rejected")

    test_a_disconnected_coarse_face_is_split()
    test_every_resulting_group_is_connected()
    test_it_only_ever_splits()
    test_the_block_pair_is_never_crossed()
    test_an_already_connected_partition_keeps_its_grouping()
    test_interior_faces_are_left_alone()
    test_the_input_is_not_mutated()
    test_it_feeds_generate_coarse_grid()
    test_three_dimensions_is_refused()
    test_bad_input_is_rejected()
    print("\nAll tests passed!")
