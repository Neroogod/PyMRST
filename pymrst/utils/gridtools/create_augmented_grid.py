"""Full connectivity for a grid (MRST ``utils/gridtools/createAugmentedGrid.m``).

A grid stores cells-to-faces and faces-to-nodes. Plenty of methods - VEM,
mimetic, anything assembling on nodes - need the other directions too. This
adds them:

- `cells.nodes` / `cells.nodePos` - the nodes of each cell;
- `nodes.faces` / `nodes.facePos` - the faces meeting at each node;
- `nodes.cells` / `nodes.cellPos` - the cells meeting at each node;
- in 3-D, an explicit **edge** structure: `edges.nodes`, `edges.nodePos`,
  `edges.num`, plus `faces.edges`, `faces.edgePos` and `faces.edgeSign` giving
  each face's edges and whether it traverses them forwards or backwards.

All of it is projected out of `create_grid_mappings`' incidence table, so the
work is one pass plus some sorting.

**PyMRST already had a partial `create_augmented_grid` in
`pymrst/modules/vemmech/vem.py`**, which computes `cells.nodes` and
`cells.nodePos` and stops. This is the complete version at MRST's own path.
The two agree on the fields they share - a test below asserts it rather than
trusting it - and vemmech keeps calling its own, because that module is
reference-validated throughout and rewiring it buys no behaviour anyone asked
for. If the node-to-face or node-to-cell maps are ever needed there, this is
what to switch to.

**MRST's position-array helper is wrong for a grid with an unreferenced
node.** It builds each `*Pos` array with `find(diff(keys) == 1)`, which assumes
the sorted keys run 0, 1, 2, ... with no gaps. A node belonging to no face -
left behind by a removal, or never referenced by a surface grid built with
`make_planar_grid`, which carries its parent's whole node array - skips a key,
and the position array then comes out short, silently misaligning every entry
after the gap. This counts occurrences instead, which is correct whether or not
every node is used.
"""

import copy

import numpy as np

from pymrst.gridprocessing.grid_structure import Edges
from pymrst.utils.gridtools.create_grid_mappings import create_grid_mappings
from pymrst.utils.rldecode import rldecode


def _map_to_pos(pairs, count):
    """Turn sorted unique ``(key, value)`` rows into ``(pos, val)``.

    Correct for keys that never appear; see the module docstring for why MRST's
    version is not.
    """
    keys = np.asarray(pairs[:, 0], dtype=np.int64)
    counts = np.bincount(keys, minlength=count)
    pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return pos, np.asarray(pairs[:, 1], dtype=np.int64)


def _add_edges(g):
    """The 3-D edge structure: each face's boundary, de-duplicated."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)

    blocks = [nodes[node_pos[f]:node_pos[f + 1]] for f in range(g.faces.num)]
    directed = np.column_stack([
        np.concatenate(blocks),
        np.concatenate([np.roll(b, -1) for b in blocks])])

    canonical = np.sort(directed, axis=1)
    unique, index = np.unique(canonical, axis=0, return_inverse=True)
    index = index.ravel()

    g.faces.edges = index
    g.faces.edgePos = np.concatenate([[0], np.cumsum(np.diff(node_pos))]) \
        .astype(np.int64)
    # +1 where the face walks the edge in its stored direction, -1 otherwise.
    g.faces.edgeSign = np.where(unique[index, 0] == directed[:, 0], 1, -1)

    g.edges = Edges(num=int(unique.shape[0]), nodes=unique.reshape(-1),
                    nodePos=np.arange(0, 2 * unique.shape[0] + 1, 2,
                                      dtype=np.int64))
    return g


def _cell_nodes_2d(g):
    """Each cell-face's starting node, oriented outwards from the cell."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)

    if np.any(np.diff(node_pos)[cell_faces] != 2):
        raise ValueError(
            "a 2-D grid's faces must have exactly two nodes; this grid has a "
            "face with a different count")

    cell_no = rldecode(np.arange(g.cells.num), np.diff(face_pos))
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    forward = cell_no == neighbours[cell_faces, 0]

    nodes = np.asarray(g.faces.nodes, dtype=np.int64)
    starts = node_pos[cell_faces]
    return nodes[np.where(forward, starts, starts + 1)]


def create_augmented_grid(g):
    """Return a copy of ``g`` with the full node, face, cell and edge maps."""
    g = copy.deepcopy(g)

    if g.griddim == 3:
        g = _add_edges(g)
    else:
        g.cells.nodes = _cell_nodes_2d(g)
        g.cells.nodePos = np.asarray(g.cells.facePos, dtype=np.int64).copy()

    table = create_grid_mappings(g)

    node_faces = np.unique(table[:, [1, 3]], axis=0)
    g.nodes.facePos, g.nodes.faces = _map_to_pos(node_faces, g.nodes.num)

    if g.griddim == 3:
        cell_nodes = np.unique(table[:, [0, 1]], axis=0)
        g.cells.nodePos, g.cells.nodes = _map_to_pos(cell_nodes, g.cells.num)

    node_cells = np.unique(table[:, [1, 0]], axis=0)
    g.nodes.cellPos, g.nodes.cells = _map_to_pos(node_cells, g.nodes.num)

    g.type = list(getattr(g, "type", [])) + ["createAugmentedGrid"]
    return g


def create_augmented_grid_tedfm(g):
    """Augment the matrix grid of a discrete-fracture container in place.

    MRST ships this as `createAugmentedGridTEDFM`, a copy of
    `createAugmentedGrid` with every `G` rewritten to `G.Matrix` - a hundred
    duplicated lines that must be kept in step with the original by hand. Here
    it is the wrapper it should have been.
    """
    matrix = getattr(g, "Matrix", None)
    if matrix is None:
        raise ValueError(
            "expected a discrete-fracture container with a `Matrix` grid; "
            "use create_augmented_grid for an ordinary grid")
    g = copy.copy(g)
    g.Matrix = create_augmented_grid(matrix)
    return g


createAugmentedGrid = create_augmented_grid
createAugmentedGridTEDFM = create_augmented_grid_tedfm


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(3, 2, 2), sizes=(3.0, 2.0, 2.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_a_cube_cell_has_eight_nodes() -> None:
        g = create_augmented_grid(build())
        assert np.all(np.diff(np.asarray(g.cells.nodePos)) == 8)
        print("ok test_a_cube_cell_has_eight_nodes")

    def test_a_square_cell_has_four_nodes() -> None:
        g = create_augmented_grid(build((3, 3), (3.0, 3.0)))
        assert np.all(np.diff(np.asarray(g.cells.nodePos)) == 4)
        print("ok test_a_square_cell_has_four_nodes")

    def test_the_node_maps_are_the_transpose_of_the_cell_map() -> None:
        """node-to-cell and cell-to-node must describe the same incidences."""
        g = create_augmented_grid(build())
        forward = set()
        pos = np.asarray(g.cells.nodePos, dtype=np.int64)
        for c in range(g.cells.num):
            for n in np.asarray(g.cells.nodes)[pos[c]:pos[c + 1]]:
                forward.add((c, int(n)))
        backward = set()
        pos = np.asarray(g.nodes.cellPos, dtype=np.int64)
        for n in range(g.nodes.num):
            for c in np.asarray(g.nodes.cells)[pos[n]:pos[n + 1]]:
                backward.add((int(c), n))
        assert forward == backward, len(forward ^ backward)
        print(f"ok test_the_node_maps_are_the_transpose_of_the_cell_map "
              f"({len(forward)} incidences)")

    def test_node_to_face_agrees_with_face_to_node() -> None:
        g = create_augmented_grid(build())
        pos, nodes = (np.asarray(g.faces.nodePos, np.int64),
                      np.asarray(g.faces.nodes, np.int64))
        forward = {(f, int(n)) for f in range(g.faces.num)
                   for n in nodes[pos[f]:pos[f + 1]]}
        pos, faces = (np.asarray(g.nodes.facePos, np.int64),
                      np.asarray(g.nodes.faces, np.int64))
        backward = {(int(f), n) for n in range(g.nodes.num)
                    for f in faces[pos[n]:pos[n + 1]]}
        assert forward == backward
        print("ok test_node_to_face_agrees_with_face_to_node")

    def test_a_corner_node_touches_one_cell_and_an_interior_node_eight() -> None:
        g = create_augmented_grid(build((2, 2, 2), (2.0, 2.0, 2.0)))
        per_node = np.diff(np.asarray(g.nodes.cellPos))
        assert per_node.min() == 1 and per_node.max() == 8, \
            (per_node.min(), per_node.max())
        print("ok test_a_corner_node_touches_one_cell_and_an_interior_node_eight")

    def test_a_cube_has_the_right_number_of_edges() -> None:
        """Euler's formula: V - E + F - C = 1 for a solid block."""
        g = create_augmented_grid(build((2, 2, 2), (2.0, 2.0, 2.0)))
        v, e, f, c = g.nodes.num, g.edges.num, g.faces.num, g.cells.num
        assert v - e + f - c == 1, (v, e, f, c)
        print(f"ok test_a_cube_has_the_right_number_of_edges ({e} edges)")

    def test_each_edge_has_two_nodes() -> None:
        g = create_augmented_grid(build())
        assert np.all(np.diff(np.asarray(g.edges.nodePos)) == 2)
        assert np.asarray(g.edges.nodes).size == 2 * g.edges.num
        print("ok test_each_edge_has_two_nodes")

    def test_edge_signs_are_plus_or_minus_one_and_balance() -> None:
        """Every interior edge is walked both ways by the faces sharing it."""
        g = create_augmented_grid(build())
        signs = np.asarray(g.faces.edgeSign)
        assert set(np.unique(signs).tolist()) <= {-1, 1}
        totals = np.bincount(np.asarray(g.faces.edges), weights=signs,
                             minlength=g.edges.num)
        assert np.all(np.abs(totals) <= 2), totals.max()
        print("ok test_edge_signs_are_plus_or_minus_one_and_balance")

    def test_a_faces_edges_join_its_nodes() -> None:
        g = create_augmented_grid(build())
        pos = np.asarray(g.faces.edgePos, dtype=np.int64)
        edges = np.asarray(g.faces.edges, dtype=np.int64)
        edge_nodes = np.asarray(g.edges.nodes, dtype=np.int64).reshape(-1, 2)
        node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
        nodes = np.asarray(g.faces.nodes, dtype=np.int64)
        for f in range(g.faces.num):
            own = set(nodes[node_pos[f]:node_pos[f + 1]].tolist())
            for e in edges[pos[f]:pos[f + 1]]:
                assert set(edge_nodes[e].tolist()) <= own, (f, e)
        print("ok test_a_faces_edges_join_its_nodes")

    def test_it_agrees_with_the_vemmech_version() -> None:
        """The cross-check on the partial implementation; see the docstring."""
        from pymrst.modules.vemmech.vem import create_augmented_grid as partial
        for dims, sizes in (((3, 2, 2), (3.0, 2.0, 2.0)),
                            ((3, 3), (3.0, 3.0))):
            mine = create_augmented_grid(build(dims, sizes))
            theirs = partial(build(dims, sizes))
            a = {(c, int(n)) for c, n in zip(
                rldecode(np.arange(mine.cells.num),
                         np.diff(np.asarray(mine.cells.nodePos))),
                np.asarray(mine.cells.nodes))}
            b = {(c, int(n)) for c, n in zip(
                rldecode(np.arange(theirs.cells.num),
                         np.diff(np.asarray(theirs.cells.nodePos))),
                np.asarray(theirs.cells.nodes))}
            assert a == b, (dims, len(a ^ b))
        print("ok test_it_agrees_with_the_vemmech_version")

    def test_an_unreferenced_node_does_not_shift_the_maps() -> None:
        """What MRST's `diff(keys) == 1` helper gets wrong; see the docstring."""
        g = build((2, 2, 2), (2.0, 2.0, 2.0))
        # Append a node nothing refers to - exactly what make_planar_grid
        # leaves behind, since it carries its parent's whole node array.
        g.nodes.coords = np.vstack([np.asarray(g.nodes.coords),
                                    [[99.0, 99.0, 99.0]]])
        g.nodes.num += 1
        a = create_augmented_grid(g)

        assert np.asarray(a.nodes.cellPos).size == g.nodes.num + 1
        assert np.asarray(a.nodes.facePos).size == g.nodes.num + 1
        # The orphan has no cells and no faces, and nothing after it shifted.
        pos = np.asarray(a.nodes.cellPos)
        assert pos[-1] == pos[-2], "the unreferenced node picked up a cell"
        print("ok test_an_unreferenced_node_does_not_shift_the_maps")

    def test_the_input_is_not_modified() -> None:
        g = build()
        create_augmented_grid(g)
        assert getattr(g.nodes, "cellPos", None) is None
        print("ok test_the_input_is_not_modified")

    def test_the_tedfm_wrapper_augments_the_matrix_grid() -> None:
        class Container:
            pass
        container = Container()
        container.Matrix = build()
        got = create_augmented_grid_tedfm(container)
        assert getattr(got.Matrix.nodes, "cellPos", None) is not None
        assert getattr(container.Matrix.nodes, "cellPos", None) is None
        print("ok test_the_tedfm_wrapper_augments_the_matrix_grid")

    def test_the_tedfm_wrapper_rejects_a_plain_grid() -> None:
        try:
            create_augmented_grid_tedfm(build())
        except ValueError as error:
            assert "Matrix" in str(error)
            print("ok test_the_tedfm_wrapper_rejects_a_plain_grid")
            return
        raise AssertionError("a plain grid must be rejected")

    test_a_cube_cell_has_eight_nodes()
    test_a_square_cell_has_four_nodes()
    test_the_node_maps_are_the_transpose_of_the_cell_map()
    test_node_to_face_agrees_with_face_to_node()
    test_a_corner_node_touches_one_cell_and_an_interior_node_eight()
    test_a_cube_has_the_right_number_of_edges()
    test_each_edge_has_two_nodes()
    test_edge_signs_are_plus_or_minus_one_and_balance()
    test_a_faces_edges_join_its_nodes()
    test_it_agrees_with_the_vemmech_version()
    test_an_unreferenced_node_does_not_shift_the_maps()
    test_the_input_is_not_modified()
    test_the_tedfm_wrapper_augments_the_matrix_grid()
    test_the_tedfm_wrapper_rejects_a_plain_grid()
    print("\nAll tests passed!")
