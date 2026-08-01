"""Cyclically ordered nodes of each cell (MRST ``utils/getSortedCellNodes.m``).

A 2-D cell's faces arrive in whatever order the grid builder produced. Anything
that walks a cell's boundary - computing a polygon area, building a VEM
projection, drawing it - needs them in cyclic order instead, each face starting
where the last one ended.

The ordering comes from the *faces*, not from the node coordinates. Each face
is flipped so it points out of the cell, which gives a directed edge; the edges
are then chained head to tail. Sorting by angle about the centroid would be the
obvious alternative and it is wrong for a non-convex cell, where the angular
order and the boundary order differ.

**Two dimensions only**, as in MRST: the implementation reshapes
`faces.nodes` into pairs, which is a face's node count only in 2-D.
"""

import numpy as np

from pymrst.utils.rldecode import rldecode


def sort_edges(edges: np.ndarray) -> np.ndarray:
    """Chain directed edges head to tail, flipping any that point backwards.

    `edges` is `(n, 2)` of node pairs. The result visits the same edges in an
    order where each row's second node is the next row's first.
    """
    edges = np.array(edges, dtype=np.int64, copy=True)
    n = edges.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if edges[i, 1] in edges[j]:
                if edges[i, 1] == edges[j, 1]:
                    edges[j] = edges[j, ::-1]
                edges[[i + 1, j]] = edges[[j, i + 1]]
                break
            if edges[0, 0] in edges[j]:
                if edges[0, 0] == edges[j, 0]:
                    edges[j] = edges[j, ::-1]
                edges = np.concatenate(
                    [edges[[j]], edges[:j], edges[j + 1:]])
                break
    return edges


def get_sorted_cell_nodes(g) -> np.ndarray:
    """One node per cell-face, in cyclic order around each cell."""
    if g.griddim != 2:
        raise NotImplementedError(
            "getSortedCellNodes is two-dimensional; it reshapes faces.nodes "
            "into pairs, which is a face's node count only in 2-D")

    cached = getattr(g.cells, "sortedCellNodes", None)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    if cached is not None:
        cached = np.asarray(cached).ravel()
        if cached.size != cell_faces.size:
            raise ValueError(
                f"cells.sortedCellNodes has {cached.size} entries but the grid "
                f"has {cell_faces.size} cell-faces; delete the field and call "
                f"get_sorted_cell_nodes again")
        return cached

    edges = np.asarray(g.faces.nodes, dtype=np.int64).reshape(-1, 2)
    cell_edges = edges[cell_faces].copy()

    # Flip the faces whose stored orientation points into the cell, so every
    # edge runs the same way round the boundary.
    owner = rldecode(np.arange(g.cells.num), np.diff(face_pos))
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    inward = neighbours[cell_faces, 0] != owner
    cell_edges[inward] = cell_edges[inward][:, ::-1]

    for cell in range(g.cells.num):
        block = slice(face_pos[cell], face_pos[cell + 1])
        cell_edges[block] = sort_edges(cell_edges[block])
    return cell_edges[:, 0].copy()


getSortedCellNodes = get_sorted_cell_nodes
sortEdges = sort_edges


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def _cells(g, nodes):
        pos = np.asarray(g.cells.facePos, dtype=np.int64)
        return [nodes[pos[c]:pos[c + 1]] for c in range(g.cells.num)]

    def test_each_cell_lists_each_of_its_nodes_once() -> None:
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        for block in _cells(g, get_sorted_cell_nodes(g)):
            assert len(set(block.tolist())) == len(block), block
            assert len(block) == 4, block
        print("ok test_each_cell_lists_each_of_its_nodes_once")

    def test_consecutive_nodes_share_a_face() -> None:
        """The defining property: the list walks the boundary."""
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        edges = {frozenset(e) for e in
                 np.asarray(g.faces.nodes, dtype=np.int64).reshape(-1, 2)}
        for block in _cells(g, get_sorted_cell_nodes(g)):
            for a, b in zip(block, np.roll(block, -1)):
                assert frozenset((int(a), int(b))) in edges, (a, b)
        print("ok test_consecutive_nodes_share_a_face")

    def test_the_order_traces_a_simple_polygon() -> None:
        """A correct cyclic order gives the cell's own area by the shoelace
        formula; any other order gives less."""
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        coords = np.asarray(g.nodes.coords, dtype=float)
        volumes = np.asarray(g.cells.volumes, dtype=float)
        for cell, block in enumerate(_cells(g, get_sorted_cell_nodes(g))):
            xy = coords[block]
            area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                             - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
            assert np.isclose(area, volumes[cell]), (cell, area, volumes[cell])
        print("ok test_the_order_traces_a_simple_polygon")

    def test_an_unsorted_order_would_fail_that_test() -> None:
        """Guard on the test above: the shoelace check has to be able to fail."""
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        crossed = square[[0, 2, 1, 3]]
        area = 0.5 * abs(np.dot(crossed[:, 0], np.roll(crossed[:, 1], -1))
                         - np.dot(crossed[:, 1], np.roll(crossed[:, 0], -1)))
        assert not np.isclose(area, 1.0), area
        print("ok test_an_unsorted_order_would_fail_that_test")

    def test_sort_edges_chains_head_to_tail() -> None:
        chained = sort_edges(np.array([[3, 4], [1, 2], [2, 3], [4, 1]]))
        for a, b in zip(chained, np.roll(chained, -1, axis=0)):
            assert a[1] == b[0], chained
        print("ok test_sort_edges_chains_head_to_tail")

    def test_sort_edges_flips_a_reversed_edge() -> None:
        chained = sort_edges(np.array([[1, 2], [3, 2], [3, 1]]))
        for a, b in zip(chained, np.roll(chained, -1, axis=0)):
            assert a[1] == b[0], chained
        print("ok test_sort_edges_flips_a_reversed_edge")

    def test_a_cached_field_is_returned_unchanged() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        computed = get_sorted_cell_nodes(g)
        g.cells.sortedCellNodes = computed
        assert np.array_equal(get_sorted_cell_nodes(g), computed)
        print("ok test_a_cached_field_is_returned_unchanged")

    def test_a_malformed_cache_is_rejected() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        g.cells.sortedCellNodes = np.arange(3)
        try:
            get_sorted_cell_nodes(g)
        except ValueError as error:
            assert "delete the field" in str(error)
            print("ok test_a_malformed_cache_is_rejected")
            return
        finally:
            g.cells.sortedCellNodes = None
        raise AssertionError("a wrong-length cache must be rejected")

    def test_three_dimensions_is_rejected() -> None:
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        try:
            get_sorted_cell_nodes(g)
        except NotImplementedError as error:
            assert "two-dimensional" in str(error)
            print("ok test_three_dimensions_is_rejected")
            return
        raise AssertionError("a 3-D grid must be rejected")

    test_each_cell_lists_each_of_its_nodes_once()
    test_consecutive_nodes_share_a_face()
    test_the_order_traces_a_simple_polygon()
    test_an_unsorted_order_would_fail_that_test()
    test_sort_edges_chains_head_to_tail()
    test_sort_edges_flips_a_reversed_edge()
    test_a_cached_field_is_returned_unchanged()
    test_a_malformed_cache_is_rejected()
    test_three_dimensions_is_rejected()
    print("\nAll tests passed!")
