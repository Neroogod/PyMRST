"""Nodes of given cells, packed (MRST ``utils/gridtools/gridCellNodes.m``).

A cell's nodes are gathered through its faces, so a node shared by several of
the cell's faces arrives several times. `unique=True` (the default)
deduplicates and sorts within each cell; `unique=False` keeps every
face-by-face occurrence.

**The two differ in more than length.** The deduplicated form is sorted by node
index, so it is *not* a boundary walk - `get_sorted_cell_nodes` is the function
for that. Using this where cyclic order is needed gives a polygon with crossed
edges.
"""

from typing import Tuple

import numpy as np

from pymrst.utils.mcolon import mcolon
from pymrst.utils.rldecode import rldecode


def grid_cell_nodes(g, cells=None, unique: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(nodes, pos)`` for the named cells, or all of them."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    cells = (np.arange(g.cells.num) if cells is None
             else np.atleast_1d(np.asarray(cells, dtype=np.int64)).ravel())

    all_faces = np.asarray(g.cells.faces, dtype=np.int64)
    all_faces = all_faces[:, 0] if all_faces.ndim > 1 else all_faces

    counts = face_pos[cells + 1] - face_pos[cells]
    cell_no = rldecode(np.arange(cells.size), counts)
    faces = all_faces[mcolon(face_pos[cells], face_pos[cells + 1])]

    per_face = node_pos[faces + 1] - node_pos[faces]
    owner = rldecode(cell_no, per_face)
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)[
        mcolon(node_pos[faces], node_pos[faces + 1])]

    if unique:
        pairs = np.unique(np.column_stack([owner, nodes]), axis=0)
        owner, nodes = pairs[:, 0], pairs[:, 1]

    per_cell = np.bincount(owner, minlength=cells.size)
    return nodes, np.concatenate([[0], np.cumsum(per_cell)])


gridCellNodes = grid_cell_nodes


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    G2 = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
    G3 = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))

    def test_a_square_cell_has_four_distinct_nodes() -> None:
        nodes, pos = grid_cell_nodes(G2)
        assert np.all(np.diff(pos) == 4), np.diff(pos)
        print("ok test_a_square_cell_has_four_distinct_nodes")

    def test_a_cube_cell_has_eight() -> None:
        _, pos = grid_cell_nodes(G3)
        assert np.all(np.diff(pos) == 8), np.diff(pos)
        print("ok test_a_cube_cell_has_eight")

    def test_without_unique_every_face_contributes() -> None:
        """Each of a square's 4 nodes belongs to 2 of its faces: 8 entries."""
        _, pos = grid_cell_nodes(G2, unique=False)
        assert np.all(np.diff(pos) == 8), np.diff(pos)
        print("ok test_without_unique_every_face_contributes")

    def test_unique_and_raw_hold_the_same_node_set() -> None:
        for grid in (G2, G3):
            fine, fpos = grid_cell_nodes(grid, unique=False)
            deduped, dpos = grid_cell_nodes(grid)
            for c in range(grid.cells.num):
                assert (set(fine[fpos[c]:fpos[c + 1]].tolist())
                        == set(deduped[dpos[c]:dpos[c + 1]].tolist())), c
        print("ok test_unique_and_raw_hold_the_same_node_set")

    def test_the_unique_form_is_sorted_not_cyclic() -> None:
        """Which is why get_sorted_cell_nodes exists separately."""
        nodes, pos = grid_cell_nodes(G2, [0])
        block = nodes[pos[0]:pos[1]]
        assert np.array_equal(block, np.sort(block)), block
        print("ok test_the_unique_form_is_sorted_not_cyclic")

    def test_it_disagrees_with_the_cyclic_order() -> None:
        from pymrst.utils.get_sorted_cell_nodes import get_sorted_cell_nodes
        nodes, pos = grid_cell_nodes(G2, [0])
        cyclic = get_sorted_cell_nodes(G2)[:4]
        assert set(nodes[pos[0]:pos[1]].tolist()) == set(cyclic.tolist())
        assert not np.array_equal(nodes[pos[0]:pos[1]], cyclic)
        print("ok test_it_disagrees_with_the_cyclic_order")

    def test_a_subset_returns_only_those_cells() -> None:
        _, pos = grid_cell_nodes(G3, [0, 3])
        assert pos.size == 3, pos
        print("ok test_a_subset_returns_only_those_cells")

    def test_the_nodes_bound_the_cell() -> None:
        nodes, pos = grid_cell_nodes(G2, [0])
        coords = np.asarray(G2.nodes.coords)[nodes[pos[0]:pos[1]], :2]
        assert np.allclose(coords.min(axis=0), [0.0, 0.0])
        assert np.allclose(coords.max(axis=0), [1.0, 1.0])
        print("ok test_the_nodes_bound_the_cell")

    test_a_square_cell_has_four_distinct_nodes()
    test_a_cube_cell_has_eight()
    test_without_unique_every_face_contributes()
    test_unique_and_raw_hold_the_same_node_set()
    test_the_unique_form_is_sorted_not_cyclic()
    test_it_disagrees_with_the_cyclic_order()
    test_a_subset_returns_only_those_cells()
    test_the_nodes_bound_the_cell()
    print("\nAll tests passed!")
