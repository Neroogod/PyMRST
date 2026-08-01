"""Cell-node-halfface-face table (MRST ``utils/gridtools/createGridMappings.m``).

One row for every (cell, node, half-face, face) incidence in the grid: a face
contributes one row per node **per adjacent cell**, so an interior face's nodes
appear twice and a boundary face's once. Sorted lexicographically.

Everything `createAugmentedGrid` builds is a projection of this table - node to
face from columns 2 and 4, cell to node from 1 and 2, node to cell from 2 and
1. Computing the incidences once and projecting is why it exists as its own
function.

**Half-faces are the third column and are easy to misread.** A half-face is a
`cells.faces` *row index*, not a face number: face 7 seen from cell 3 and face
7 seen from cell 5 are different half-faces. That is exactly what makes the
table useful for mimetic and VEM assembly, where a face carries a different
contribution from each side.

**PyMRST's `create_augmented_grid` does not use this**, and is a partial
implementation: `pymrst/modules/vemmech/vem.py` computes `cells.nodes` and
`cells.nodePos` by its own route and stops there, leaving MRST's
`nodes.faces`/`nodes.facePos` and `nodes.cells`/`nodes.cellPos` unbuilt. The
two agree on the part they share, and a test below checks that rather than
assuming it. Completing `create_augmented_grid` is a separate decision - the
vemmech version is used and validated throughout that module, so this does not
quietly rewire it.
"""

import numpy as np

from pymrst.utils.rldecode import rldecode


def create_grid_mappings(g) -> np.ndarray:
    """Return the sorted ``(n, 4)`` table of ``(cell, node, half-face, face)``."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    per_face = np.diff(node_pos)

    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    cell_no = rldecode(np.arange(g.cells.num), np.diff(face_pos))
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)

    # Which side of its face each half-face sits on, then the half-face number
    # reachable from (face, side).
    side = (cell_no == neighbours[cell_faces, 1]).astype(np.int64)
    half_face = np.full((g.faces.num, 2), -1, dtype=np.int64)
    half_face[cell_faces, side] = np.arange(cell_faces.size)

    # Expand per face to per (face, node), keeping both sides as columns.
    cells = np.repeat(neighbours, per_face, axis=0)
    halves = np.repeat(half_face, per_face, axis=0)
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)
    faces = rldecode(np.arange(g.faces.num), per_face)

    # Flatten side-major, so each column pairs with the same nodes and faces.
    cells, halves = cells.T.ravel(), halves.T.ravel()
    nodes, faces = np.tile(nodes, 2), np.tile(faces, 2)

    keep = cells >= 0
    if np.any(halves[keep] < 0):
        raise ValueError(
            "a face names a cell that does not name it back; the grid's "
            "cells.faces and faces.neighbors disagree")

    table = np.column_stack([cells[keep], nodes[keep], halves[keep], faces[keep]])
    return table[np.lexsort((table[:, 3], table[:, 2],
                             table[:, 1], table[:, 0]))]


createGridMappings = create_grid_mappings


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(3, 2, 2), sizes=(3.0, 2.0, 2.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_the_row_count_is_one_per_cell_face_node() -> None:
        g = build()
        cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
        per_face = np.diff(np.asarray(g.faces.nodePos, dtype=np.int64))
        assert create_grid_mappings(g).shape == (int(per_face[cell_faces].sum()), 4)
        print("ok test_the_row_count_is_one_per_cell_face_node")

    def test_every_row_is_a_real_incidence() -> None:
        """The defining property, checked against the grid itself."""
        g = build()
        node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
        face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
        nodes = np.asarray(g.faces.nodes, dtype=np.int64)
        cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
        for cell, node, half, face in create_grid_mappings(g):
            assert node in nodes[node_pos[face]:node_pos[face + 1]]
            assert face in cell_faces[face_pos[cell]:face_pos[cell + 1]]
            # The half-face must be this cell's view of this face.
            assert cell_faces[half] == face
            assert face_pos[cell] <= half < face_pos[cell + 1]
        print("ok test_every_row_is_a_real_incidence")

    def test_the_table_is_sorted() -> None:
        table = create_grid_mappings(build())
        order = np.lexsort((table[:, 3], table[:, 2], table[:, 1], table[:, 0]))
        assert np.array_equal(order, np.arange(table.shape[0]))
        print("ok test_the_table_is_sorted")

    def test_an_interior_face_appears_twice_and_a_boundary_face_once() -> None:
        g = build()
        table = create_grid_mappings(g)
        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
        per_face = np.diff(np.asarray(g.faces.nodePos, dtype=np.int64))
        counts = np.bincount(table[:, 3], minlength=g.faces.num)
        sides = np.count_nonzero(neighbours >= 0, axis=1)
        assert np.array_equal(counts, per_face * sides)
        print("ok test_an_interior_face_appears_twice_and_a_boundary_face_once")

    def test_half_faces_are_per_side_not_per_face() -> None:
        """The distinction the third column exists to make."""
        g = build()
        table = create_grid_mappings(g)
        interior = np.flatnonzero(
            np.all(np.asarray(g.faces.neighbors, dtype=np.int64) >= 0, axis=1))
        face = int(interior[0])
        halves = np.unique(table[table[:, 3] == face, 2])
        assert halves.size == 2, halves
        print("ok test_half_faces_are_per_side_not_per_face")

    def test_every_half_face_is_used() -> None:
        g = build()
        table = create_grid_mappings(g)
        assert np.array_equal(np.unique(table[:, 2]),
                              np.arange(np.asarray(g.cells.faces).shape[0]))
        print("ok test_every_half_face_is_used")

    def test_the_cell_to_node_projection_matches_vemmech() -> None:
        """The cross-check on the partial implementation; see the docstring."""
        from pymrst.modules.vemmech.vem import create_augmented_grid
        g = build()
        table = create_grid_mappings(g)
        mine = np.unique(table[:, [0, 1]], axis=0)

        augmented = create_augmented_grid(build())
        pos = np.asarray(augmented.cells.nodePos, dtype=np.int64)
        theirs = np.column_stack([
            rldecode(np.arange(augmented.cells.num), np.diff(pos)),
            np.asarray(augmented.cells.nodes, dtype=np.int64)])
        assert np.array_equal(mine, np.unique(theirs, axis=0)), \
            (mine.shape, theirs.shape)
        print("ok test_the_cell_to_node_projection_matches_vemmech")

    def test_a_cube_cell_touches_eight_nodes() -> None:
        table = create_grid_mappings(build())
        pairs = np.unique(table[:, [0, 1]], axis=0)
        assert np.all(np.bincount(pairs[:, 0]) == 8)
        print("ok test_a_cube_cell_touches_eight_nodes")

    def test_it_works_in_two_dimensions() -> None:
        table = create_grid_mappings(build((3, 3), (3.0, 3.0)))
        pairs = np.unique(table[:, [0, 1]], axis=0)
        assert np.all(np.bincount(pairs[:, 0]) == 4)
        print("ok test_it_works_in_two_dimensions")

    def test_it_works_on_an_unstructured_grid() -> None:
        from pymrst.gridprocessing import triangle_grid
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        g = compute_geometry(triangle_grid(points, triangles))
        pairs = np.unique(create_grid_mappings(g)[:, [0, 1]], axis=0)
        assert np.all(np.bincount(pairs[:, 0]) == 3)
        print("ok test_it_works_on_an_unstructured_grid")

    def test_an_inconsistent_grid_is_rejected() -> None:
        """Guard: cells.faces and faces.neighbors must agree.

        The corruption has to be a face claiming a cell that does not claim it
        back. Simply blanking a face's neighbours is *not* inconsistent - it
        makes the face unreferenced, and its rows drop out cleanly.
        """
        g = build((2, 2, 2), (2.0, 2.0, 2.0))
        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64).copy()
        face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
        cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()

        boundary = np.flatnonzero(neighbours[:, 1] < 0)
        face = int(boundary[0])
        stranger = next(c for c in range(g.cells.num)
                        if face not in cell_faces[face_pos[c]:face_pos[c + 1]])
        neighbours[face, 1] = stranger
        g.faces.neighbors = neighbours
        try:
            create_grid_mappings(g)
        except ValueError as error:
            assert "does not name it back" in str(error)
            print("ok test_an_inconsistent_grid_is_rejected")
            return
        raise AssertionError("an inconsistent grid must be rejected")

    def test_an_unreferenced_face_is_not_an_error() -> None:
        """The other half of the distinction above."""
        g = build((2, 2, 2), (2.0, 2.0, 2.0))
        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64).copy()
        interior = np.flatnonzero(np.all(neighbours >= 0, axis=1))
        neighbours[interior[0]] = [-1, -1]
        g.faces.neighbors = neighbours
        table = create_grid_mappings(g)
        assert int(interior[0]) not in set(table[:, 3].tolist())
        print("ok test_an_unreferenced_face_is_not_an_error")

    test_the_row_count_is_one_per_cell_face_node()
    test_every_row_is_a_real_incidence()
    test_the_table_is_sorted()
    test_an_interior_face_appears_twice_and_a_boundary_face_once()
    test_half_faces_are_per_side_not_per_face()
    test_every_half_face_is_used()
    test_the_cell_to_node_projection_matches_vemmech()
    test_a_cube_cell_touches_eight_nodes()
    test_it_works_in_two_dimensions()
    test_it_works_on_an_unstructured_grid()
    test_an_inconsistent_grid_is_rejected()
    test_an_unreferenced_face_is_not_an_error()
    print("\nAll tests passed!")
