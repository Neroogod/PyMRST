"""Nodes of given faces, packed (MRST ``utils/gridtools/gridFaceNodes.m``).

`(nodes, pos)` in the same packed form as `grid_cell_faces`: face `i`'s nodes
are `nodes[pos[i]:pos[i+1]]`.

The nodes come back in the grid's stored order, which for a 3-D face is
cyclic around its boundary. That order is what makes a face's polygon
well-defined, so it must not be sorted.
"""

from typing import Tuple

import numpy as np

from pymrst.utils.mcolon import mcolon


def grid_face_nodes(g, faces=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(nodes, pos)`` for the named faces, or all of them."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    faces = (np.arange(g.faces.num) if faces is None
             else np.atleast_1d(np.asarray(faces, dtype=np.int64)).ravel())
    counts = node_pos[faces + 1] - node_pos[faces]
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)[
        mcolon(node_pos[faces], node_pos[faces + 1])]
    return nodes, np.concatenate([[0], np.cumsum(counts)])


gridFaceNodes = grid_face_nodes


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    G2 = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
    G3 = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))

    def test_a_two_dimensional_face_has_two_nodes() -> None:
        nodes, pos = grid_face_nodes(G2)
        assert np.all(np.diff(pos) == 2)
        print("ok test_a_two_dimensional_face_has_two_nodes")

    def test_a_three_dimensional_face_has_four() -> None:
        _, pos = grid_face_nodes(G3)
        assert np.all(np.diff(pos) == 4)
        print("ok test_a_three_dimensional_face_has_four")

    def test_all_faces_reproduces_the_grid_s_own_list() -> None:
        nodes, pos = grid_face_nodes(G3)
        assert np.array_equal(nodes, np.asarray(G3.faces.nodes))
        assert np.array_equal(pos, np.asarray(G3.faces.nodePos))
        print("ok test_all_faces_reproduces_the_grid_s_own_list")

    def test_a_subset_returns_only_those_faces() -> None:
        nodes, pos = grid_face_nodes(G3, [5])
        node_pos = np.asarray(G3.faces.nodePos)
        assert np.array_equal(
            nodes, np.asarray(G3.faces.nodes)[node_pos[5]:node_pos[6]])
        print("ok test_a_subset_returns_only_those_faces")

    def test_the_stored_order_is_preserved() -> None:
        """A face's nodes are cyclic; sorting them would break the polygon."""
        nodes, pos = grid_face_nodes(G3, [0])
        block = nodes[pos[0]:pos[1]]
        assert not np.array_equal(block, np.sort(block)) or block.size <= 2
        print("ok test_the_stored_order_is_preserved")

    def test_the_nodes_of_a_face_are_distinct() -> None:
        nodes, pos = grid_face_nodes(G3)
        for i in range(G3.faces.num):
            block = nodes[pos[i]:pos[i + 1]]
            assert len(set(block.tolist())) == block.size, i
        print("ok test_the_nodes_of_a_face_are_distinct")

    test_a_two_dimensional_face_has_two_nodes()
    test_a_three_dimensional_face_has_four()
    test_all_faces_reproduces_the_grid_s_own_list()
    test_a_subset_returns_only_those_faces()
    test_the_stored_order_is_preserved()
    test_the_nodes_of_a_face_are_distinct()
    print("\nAll tests passed!")
