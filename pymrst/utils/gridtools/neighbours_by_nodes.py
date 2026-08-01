"""Cells that share a node (MRST ``utils/gridtools/neighboursByNodes.m``).

A *wider* neighbourship than `get_neighbourship`: two cells count as neighbours
when they share any node, not only when they share a face. On a Cartesian grid
that adds the diagonal neighbours, so an interior cell has 8 rather than 4.

The distinction matters for anything that needs a stencil rather than a flux
path - multipoint transmissibility, node-based upscaling, smoothing - where a
face-sharing neighbourship misses the corner contributions entirely.

Pairs are returned sorted within each row and unique across rows, so a
connection appears once as `(i, j)` with `i < j`, never twice.
"""

import numpy as np

from pymrst.utils.rldecode import rldecode


def neighbours_by_nodes(g, include_boundary: bool = False) -> np.ndarray:
    """The `(n, 2)` list of cell pairs sharing at least one node."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    face_nodes = np.asarray(g.faces.nodes, dtype=np.int64)
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    per_face = np.diff(node_pos)

    # Each face contributes its nodes once per side.
    nodes = np.tile(face_nodes, 2)
    cells = np.concatenate([rldecode(neighbours[:, 0], per_face),
                            rldecode(neighbours[:, 1], per_face)])
    if not include_boundary:
        keep = cells >= 0
        nodes, cells = nodes[keep], cells[keep]

    pairs = np.unique(np.column_stack([nodes, cells]), axis=0)

    # Within each node, every pair of incident cells is a neighbourship.
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    boundaries = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    out = []
    for block in np.split(pairs[:, 1], boundaries):
        if block.size < 2:
            continue
        left, right = np.meshgrid(block, block, indexing="ij")
        out.append(np.column_stack([left.ravel(), right.ravel()]))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)

    stacked = np.vstack(out)
    stacked = np.sort(stacked, axis=1)
    stacked = stacked[stacked[:, 0] != stacked[:, 1]]
    return np.unique(stacked, axis=0)


neighboursByNodes = neighbours_by_nodes


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.get_neighbourship import get_neighbourship

    G = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))

    def test_it_is_wider_than_face_neighbourship() -> None:
        by_node = neighbours_by_nodes(G).shape[0]
        by_face = get_neighbourship(G).shape[0]
        assert by_node > by_face, (by_node, by_face)
        print(f"ok test_it_is_wider_than_face_neighbourship "
              f"({by_node} vs {by_face} pairs)")

    def test_it_contains_every_face_neighbourship() -> None:
        """Cells sharing a face necessarily share nodes."""
        by_node = {tuple(sorted(p)) for p in neighbours_by_nodes(G)}
        for pair in get_neighbourship(G):
            assert tuple(sorted(pair)) in by_node, pair
        print("ok test_it_contains_every_face_neighbourship")

    def test_a_centre_cell_has_eight_neighbours() -> None:
        """The 3x3 grid's middle cell touches all the others."""
        pairs = neighbours_by_nodes(G)
        centre = 4
        touching = set(pairs[pairs[:, 0] == centre, 1].tolist())
        touching |= set(pairs[pairs[:, 1] == centre, 0].tolist())
        assert len(touching) == 8, sorted(touching)
        print("ok test_a_centre_cell_has_eight_neighbours")

    def test_diagonal_neighbours_are_included() -> None:
        """Cells 0 and 4 share only a corner, so face neighbourship misses them."""
        pairs = {tuple(sorted(p)) for p in neighbours_by_nodes(G)}
        faces = {tuple(sorted(p)) for p in get_neighbourship(G)}
        assert (0, 4) in pairs and (0, 4) not in faces
        print("ok test_diagonal_neighbours_are_included")

    def test_each_pair_appears_once_and_is_sorted() -> None:
        pairs = neighbours_by_nodes(G)
        assert np.all(pairs[:, 0] < pairs[:, 1])
        assert len({tuple(p) for p in pairs}) == pairs.shape[0]
        print("ok test_each_pair_appears_once_and_is_sorted")

    def test_no_cell_is_its_own_neighbour() -> None:
        pairs = neighbours_by_nodes(G)
        assert not np.any(pairs[:, 0] == pairs[:, 1])
        print("ok test_no_cell_is_its_own_neighbour")

    def test_it_works_in_three_dimensions() -> None:
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        pairs = neighbours_by_nodes(g)
        # every cell of a 2x2x2 block touches every other
        assert pairs.shape[0] == 8 * 7 // 2, pairs.shape
        print("ok test_it_works_in_three_dimensions")

    test_it_is_wider_than_face_neighbourship()
    test_it_contains_every_face_neighbourship()
    test_a_centre_cell_has_eight_neighbours()
    test_diagonal_neighbours_are_included()
    test_each_pair_appears_once_and_is_sorted()
    test_no_cell_is_its_own_neighbour()
    test_it_works_in_three_dimensions()
    print("\nAll tests passed!")
