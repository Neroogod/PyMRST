"""Canonical ordering of a grid (MRST ``utils/gridtools/sortGrid.m``).

Two grids can describe the same geometry with their nodes, faces and cells
numbered differently - built by different routes, or round-tripped through a
file that does not preserve order. This puts a grid into a canonical form so
that equivalent grids become *identical*, which is what makes them comparable
at all.

Three passes, each depending on the one before:

1. **nodes** are sorted by coordinate, last axis first, and `faces.nodes` is
   remapped;
2. **faces** have their node lists rotated to start at the lowest node, then
   the faces are sorted by those lists, and `cells.faces` is remapped;
3. **cells** have their face lists sorted, then the cells are sorted, and
   `faces.neighbors` and `cells.indexMap` are remapped.

The rotation in step 2 is what makes the comparison order-independent: a face's
nodes are cyclic, so `(3, 7, 9)` and `(7, 9, 3)` describe the same face, and
only rotating both to start at 3 makes them compare equal.

**In 2-D that rotation is a reversal, and the neighbours are swapped to
match.** A 2-D face has exactly two nodes, so rotating `(7, 3)` to start at its
lowest gives `(3, 7)` - which is the edge traversed backwards, not the same
edge relabelled. The face normal computed from it then points the other way,
breaking the invariant that a normal runs from `neighbors[0]` to
`neighbors[1]`. This swaps those two columns for exactly the faces that were
reversed, so the invariant survives. MRST does not, because it leaves the
geometry in place and stale and so never recomputes a normal to notice.

**The face tags stay with their faces.** Where `cells.faces` has a second
column naming each face's direction, MRST rotates that column but then writes
the *sorted* face numbers beside it, leaving tag and face mismatched - its own
source closes `sortCells` with the comment `% cellFaces(:,2) buggy`. Here the
tags are permuted by the same index as the faces, so they still describe the
face they sit next to.

**Geometry is dropped, not reordered.** `cells.volumes` and the centroids are
derived quantities, and permuting them consistently is more error-prone than
recomputing: call `compute_geometry` afterwards if they are needed. Dropping
them is also what makes the point above safe - a stale normal that silently
contradicts the topology is worse than no normal at all.
"""

import copy

import numpy as np


def _packed(values, pos):
    """The packed array as a list of tuples, one per item."""
    return [tuple(values[pos[i]:pos[i + 1]]) for i in range(len(pos) - 1)]


def _rotated(block):
    """Rotate a cyclic sequence to start at its lowest entry.

    Returns the rotation and whether it reversed the sequence's orientation.
    Rotating a cycle of three or more never does; rotating a two-element one
    always does, because its only rotation swaps the pair. See the module
    docstring for why that distinction has to be tracked.
    """
    start = int(np.argmin(block))
    rotated = tuple(block[start:]) + tuple(block[:start])
    return rotated, len(block) == 2 and start == 1


def _order(blocks):
    """Sort blocks of unequal length the way MRST does.

    MRST lays the blocks out as columns of a rectangular array padded with
    `NaN`, and MATLAB's `sortrows` puts `NaN` last - so where one block is a
    prefix of another, the *longer* one sorts first. Python tuples order the
    shorter one first, which would give a different canonical form on any grid
    whose faces have differing node counts. Padding with infinity reproduces
    MATLAB's answer.
    """
    width = max((len(b) for b in blocks), default=0)
    return sorted(range(len(blocks)),
                  key=lambda i: blocks[i] + (np.inf,) * (width - len(blocks[i])))


def _repack(blocks):
    values = np.concatenate([np.asarray(b, dtype=np.int64) for b in blocks]) \
        if blocks else np.zeros(0, dtype=np.int64)
    pos = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    return values, pos.astype(np.int64)


def sort_grid(g):
    """Return a copy of ``g`` in canonical node, face and cell order."""
    g = copy.deepcopy(g)

    # ---------------------------------------------------------------- nodes
    coords = np.asarray(g.nodes.coords, dtype=float)
    order = np.lexsort(tuple(coords[:, d] for d in range(coords.shape[1])))
    g.nodes.coords = coords[order]
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    face_nodes = rank[np.asarray(g.faces.nodes, dtype=np.int64)]

    # ---------------------------------------------------------------- faces
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    rotations = [_rotated(b) for b in _packed(face_nodes, node_pos)]
    blocks = [block for block, _ in rotations]
    reversed_face = np.array([flip for _, flip in rotations], dtype=bool)
    face_order = _order(blocks)
    face_rank = np.empty(len(blocks), dtype=np.int64)
    face_rank[face_order] = np.arange(len(blocks))

    values, pos = _repack([blocks[i] for i in face_order])
    g.faces.nodes, g.faces.nodePos = values, pos

    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)[face_order]
    # A reversed face points the other way now, so its two sides trade places.
    flipped = reversed_face[face_order]
    neighbours[flipped] = neighbours[flipped][:, ::-1]

    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    has_tags = cell_faces.ndim > 1 and cell_faces.shape[1] > 1
    tags = cell_faces[:, 1:] if has_tags else None
    flat = face_rank[cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces]

    # ---------------------------------------------------------------- cells
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    # Sort each cell's faces, carrying the tag columns along with them.
    within = np.concatenate([
        face_pos[c] + np.argsort(flat[face_pos[c]:face_pos[c + 1]],
                                 kind="stable")
        for c in range(len(face_pos) - 1)]).astype(np.int64)
    flat, tags = flat[within], (tags[within] if has_tags else None)

    blocks = _packed(flat, face_pos)
    cell_order = _order(blocks)
    cell_rank = np.empty(len(blocks), dtype=np.int64)
    cell_rank[cell_order] = np.arange(len(blocks))

    rows = np.concatenate([np.arange(face_pos[i], face_pos[i + 1])
                           for i in cell_order]).astype(np.int64)
    g.cells.faces = (np.column_stack([flat[rows], tags[rows]]) if has_tags
                     else flat[rows])
    g.cells.facePos = _repack([blocks[i] for i in cell_order])[1]

    inside = neighbours >= 0
    neighbours[inside] = cell_rank[neighbours[inside]]
    g.faces.neighbors = neighbours

    index_map = getattr(g.cells, "indexMap", None)
    if index_map is not None:
        g.cells.indexMap = np.asarray(index_map)[cell_order]

    # Derived geometry is now stale; see the module docstring.
    for holder, names in ((g.cells, ("volumes", "centroids")),
                          (g.faces, ("areas", "normals", "centroids"))):
        for name in names:
            if getattr(holder, name, None) is not None:
                setattr(holder, name, None)
    return g


sortGrid = sort_grid


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.compare_grids import compare_grids

    def build(dims=(3, 3), sizes=(3.0, 3.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_sorting_is_idempotent() -> None:
        once = sort_grid(build())
        twice = sort_grid(once)
        same, differences = compare_grids(once, twice)
        assert same, differences
        print("ok test_sorting_is_idempotent")

    def test_the_cell_and_face_counts_are_preserved() -> None:
        g = build()
        s = sort_grid(g)
        assert (s.cells.num, s.faces.num, s.nodes.num) == \
               (g.cells.num, g.faces.num, g.nodes.num)
        print("ok test_the_cell_and_face_counts_are_preserved")

    def test_the_nodes_end_up_sorted() -> None:
        coords = np.asarray(sort_grid(build()).nodes.coords)
        keys = np.lexsort(tuple(coords[:, d] for d in range(coords.shape[1])))
        assert np.array_equal(keys, np.arange(coords.shape[0]))
        print("ok test_the_nodes_end_up_sorted")

    def test_a_permuted_grid_sorts_to_the_same_thing() -> None:
        """The property the function exists for."""
        g = build()
        shuffled = copy.deepcopy(g)
        rng = np.random.default_rng(0)
        order = rng.permutation(g.nodes.num)
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        shuffled.nodes.coords = np.asarray(g.nodes.coords)[order]
        shuffled.faces.nodes = rank[np.asarray(g.faces.nodes, dtype=np.int64)]

        same, differences = compare_grids(sort_grid(g), sort_grid(shuffled))
        assert same, differences
        print("ok test_a_permuted_grid_sorts_to_the_same_thing")

    def test_an_unsorted_permutation_does_not_compare_equal() -> None:
        """Guard: the test above must be able to fail."""
        g = build()
        shuffled = copy.deepcopy(g)
        order = np.roll(np.arange(g.nodes.num), 3)
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        shuffled.nodes.coords = np.asarray(g.nodes.coords)[order]
        shuffled.faces.nodes = rank[np.asarray(g.faces.nodes, dtype=np.int64)]
        assert not compare_grids(g, shuffled)[0]
        print("ok test_an_unsorted_permutation_does_not_compare_equal")

    def test_the_topology_still_describes_the_same_grid() -> None:
        """Recomputing geometry after sorting gives the same volumes."""
        g = build()
        s = compute_geometry(sort_grid(g))
        assert np.isclose(np.sum(s.cells.volumes), np.sum(g.cells.volumes))
        assert np.allclose(np.sort(np.asarray(s.cells.volumes)),
                           np.sort(np.asarray(g.cells.volumes)))
        print("ok test_the_topology_still_describes_the_same_grid")

    def test_the_sorted_grid_passes_the_grid_checks() -> None:
        from pymrst.utils.gridtools.check_grid import check_grid
        ok, problems = check_grid(compute_geometry(sort_grid(build([4, 3]))))
        assert ok, problems
        print("ok test_the_sorted_grid_passes_the_grid_checks")

    def test_stale_geometry_is_dropped_not_left_wrong() -> None:
        s = sort_grid(build())
        assert getattr(s.cells, "volumes", None) is None
        print("ok test_stale_geometry_is_dropped_not_left_wrong")

    def test_the_input_is_not_modified() -> None:
        g = build()
        before = np.asarray(g.nodes.coords).copy()
        sort_grid(g)
        assert np.array_equal(np.asarray(g.nodes.coords), before)
        print("ok test_the_input_is_not_modified")

    def test_it_works_in_three_dimensions() -> None:
        g = build((2, 2, 2), (2.0, 2.0, 2.0))
        s = compute_geometry(sort_grid(g))
        assert np.isclose(np.sum(s.cells.volumes), np.sum(g.cells.volumes))
        from pymrst.utils.gridtools.check_grid import check_grid
        ok, problems = check_grid(s)
        assert ok, problems
        print("ok test_it_works_in_three_dimensions")

    def _normals_point_from_first_to_second_neighbour(g) -> bool:
        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
        interior = np.all(neighbours >= 0, axis=1)
        centroids = np.asarray(g.cells.centroids, dtype=float)
        offset = (centroids[neighbours[interior, 1]]
                  - centroids[neighbours[interior, 0]])
        normals = np.asarray(g.faces.normals, dtype=float)[interior]
        return bool(np.all(np.sum(normals * offset, axis=1) > 0))

    def test_the_normal_orientation_survives_in_two_dimensions() -> None:
        """The 2-D reversal fix; see the module docstring.

        Without swapping the neighbour columns of the reversed faces, roughly
        half the 2-D normals come back pointing the wrong way.
        """
        for dims in ([4, 3], [2, 2], [5, 1]):
            s = compute_geometry(sort_grid(build(dims, (4.0, 3.0))))
            assert _normals_point_from_first_to_second_neighbour(s), dims
        print("ok test_the_normal_orientation_survives_in_two_dimensions")

    def test_some_two_dimensional_faces_really_are_reversed() -> None:
        """Guard on the test above: if nothing were reversed it would pass
        trivially and the fix it covers would be untested."""
        from pymrst.utils.gridtools.sort_grid import _rotated
        g = build([4, 3], (4.0, 3.0))
        pos = np.asarray(g.faces.nodePos, dtype=np.int64)
        nodes = np.asarray(g.faces.nodes, dtype=np.int64)
        coords = np.asarray(g.nodes.coords, dtype=float)
        order = np.lexsort(tuple(coords[:, d] for d in range(coords.shape[1])))
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        flips = sum(_rotated(rank[nodes[pos[i]:pos[i + 1]]])[1]
                    for i in range(len(pos) - 1))
        assert flips > 0, "no face was reversed, so the fix is untested here"
        print(f"ok test_some_two_dimensional_faces_really_are_reversed ({flips})")

    def test_three_dimensional_faces_are_never_reversed() -> None:
        """The other half of the claim: only 2-D faces rotate into a reversal."""
        from pymrst.utils.gridtools.sort_grid import _rotated
        assert _rotated([7, 3])[1]
        assert not _rotated([9, 3, 7])[1]
        assert not _rotated([9, 3, 7, 5])[1]
        print("ok test_three_dimensional_faces_are_never_reversed")

    def test_tags_travel_with_their_faces() -> None:
        """MRST's `cartGrid` tags each cell-face with a direction, so a grid
        read back from MATLAB has a second column. PyMRST's builders do not, so
        the tag is attached here rather than leaving the path unexercised."""
        g = build([3, 2], (3.0, 2.0))
        faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
        # Tag each cell-face with the face it names, so any tag that parts
        # company with its face becomes visible as a mismatch.
        g.cells.faces = np.column_stack([faces, faces * 10])
        s = sort_grid(g)
        got = np.asarray(s.cells.faces, dtype=np.int64)
        assert got.ndim == 2 and got.shape[1] == 2, got.shape

        # The renumbering is a bijection, so recover it and check it is one.
        mapping = {}
        for new, tag in got:
            mapping.setdefault(tag // 10, set()).add(new)
        assert all(len(v) == 1 for v in mapping.values()), "a tag split faces"
        assert len({next(iter(v)) for v in mapping.values()}) == len(mapping)
        print("ok test_tags_travel_with_their_faces")

    def test_an_untagged_grid_stays_one_dimensional() -> None:
        assert np.asarray(sort_grid(build()).cells.faces).ndim == 1
        print("ok test_an_untagged_grid_stays_one_dimensional")

    def test_every_cell_keeps_its_own_faces() -> None:
        """Cells are permuted and their face lists renumbered; the two have to
        move together, or a cell ends up owning another's faces."""
        g = build([3, 2], (3.0, 2.0))
        s = sort_grid(g)
        neighbours = np.asarray(s.faces.neighbors, dtype=np.int64)
        pos = np.asarray(s.cells.facePos, dtype=np.int64)
        faces = np.asarray(s.cells.faces, dtype=np.int64).ravel()
        for cell in range(s.cells.num):
            for face in faces[pos[cell]:pos[cell + 1]]:
                assert cell in neighbours[face], (cell, face, neighbours[face])
        print("ok test_every_cell_keeps_its_own_faces")

    test_sorting_is_idempotent()
    test_the_cell_and_face_counts_are_preserved()
    test_the_nodes_end_up_sorted()
    test_a_permuted_grid_sorts_to_the_same_thing()
    test_an_unsorted_permutation_does_not_compare_equal()
    test_the_topology_still_describes_the_same_grid()
    test_the_sorted_grid_passes_the_grid_checks()
    test_stale_geometry_is_dropped_not_left_wrong()
    test_the_input_is_not_modified()
    test_it_works_in_three_dimensions()
    test_the_normal_orientation_survives_in_two_dimensions()
    test_some_two_dimensional_faces_really_are_reversed()
    test_three_dimensional_faces_are_never_reversed()
    test_tags_travel_with_their_faces()
    test_an_untagged_grid_stays_one_dimensional()
    test_every_cell_keeps_its_own_faces()
    print("\nAll tests passed!")
