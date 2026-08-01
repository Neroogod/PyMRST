"""Surface grid from faces of a 3-D grid (MRST ``utils/gridtools/makePlanarGrid.m``).

Each selected face of the parent becomes a *cell* of the new grid, and each
edge of those faces becomes a *face*. The result is two-dimensional in its
topology - `griddim == 2` - while its node coordinates stay three-dimensional,
so it is a surface sitting in space. `Grid.spatial_dim` reports the 3, and
anything assuming `griddim == coords.shape[1]` will be wrong about this grid.

`sgn` flips a face's node order, which reverses the traversal of its boundary
and so flips the resulting cell's orientation. Passing the sign array that
`boundary_faces` or a cell's own `cells.faces` implies gives a surface whose
cells are all wound the same way; passing all ones does not, on a surface whose
parent faces disagree about which side is out.

**The parent's nodes are carried over whole**, exactly as MRST does. A surface
built from four faces of a large grid still has that grid's entire node array,
almost all of it unreferenced. That keeps node numbers comparable with the
parent's - which is usually what you want, since the point is to relate the
surface back - but it means `nodes.num` is not the number of nodes in the
surface, and anything sizing an array by it will over-allocate enormously.

**A non-manifold edge is rejected rather than silently mis-resolved.** The
neighbour construction assigns each edge at most one cell per orientation,
which is exactly right when two faces meet at an edge and wrong when three do:
MRST overwrites, keeping whichever cell it happened to visit last and losing
the other. Three faces meeting at an edge is normal in a fracture network or at
a fault, so this is reachable, and a wrong answer there is worse than a stop.
"""

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes
from pymrst.utils.rldecode import rldecode


def make_planar_grid(h, faces, sgn=None) -> Grid:
    """Build the surface grid on ``faces`` of ``h``, oriented by ``sgn``."""
    faces = np.atleast_1d(np.asarray(faces, dtype=np.int64))
    sgn = (np.ones(faces.size, dtype=np.int64) if sgn is None
           else np.atleast_1d(np.asarray(sgn, dtype=np.int64)))
    if sgn.size != faces.size:
        raise ValueError(
            f"sgn has {sgn.size} entries but {faces.size} faces were given")

    node_pos = np.asarray(h.faces.nodePos, dtype=np.int64)
    parent_nodes = np.asarray(h.faces.nodes, dtype=np.int64)
    per_face = np.diff(node_pos)[faces]

    # Each parent face becomes a cell; reverse its nodes where sgn < 0.
    blocks = [parent_nodes[node_pos[f]:node_pos[f + 1]][::-1 if s < 0 else 1]
              for f, s in zip(faces, sgn)]
    cell_nodes = np.concatenate(blocks) if blocks else np.zeros(0, np.int64)
    pos = np.concatenate([[0], np.cumsum(per_face)]).astype(np.int64)
    cell_no = rldecode(np.arange(faces.size), per_face)

    # Boundary edges of each cell: node k to node k+1, wrapping within the cell.
    following = np.concatenate([np.roll(np.arange(pos[c], pos[c + 1]), -1)
                                for c in range(faces.size)])
    pairs = np.column_stack([cell_nodes, cell_nodes[following]])

    canonical = np.sort(pairs, axis=1)
    edges, edge_no = np.unique(canonical, axis=0, return_inverse=True)
    edge_no = edge_no.ravel()
    # Which way round the cell traverses the edge: that picks the neighbour slot.
    side = (pairs[:, 0] > pairs[:, 1]).astype(np.int64)

    # Each (edge, direction) slot takes one cell. Two cells wanting the same
    # slot means three or more faces meet at that edge; see the docstring.
    slot = edge_no * 2 + side
    crowded = np.flatnonzero(np.bincount(slot, minlength=2 * edges.shape[0]) > 1)
    if crowded.size:
        clash = edges[crowded[0] // 2]
        raise ValueError(
            f"an edge is traversed the same way by more than one face, so the "
            f"surface is non-manifold there and its neighbours are not well "
            f"defined; first such edge joins nodes {clash.tolist()}")

    neighbours = np.full((edges.shape[0], 2), -1, dtype=np.int64)
    neighbours[edge_no, side] = cell_no

    return Grid(
        cells=Cells(num=int(faces.size), facePos=pos, faces=edge_no,
                    indexMap=faces.copy()),
        faces=Faces(num=int(edges.shape[0]), nodePos=np.arange(
            0, 2 * edges.shape[0] + 1, 2, dtype=np.int64),
            nodes=edges.reshape(-1), neighbors=neighbours),
        nodes=Nodes(num=int(h.nodes.num),
                    coords=np.array(h.nodes.coords, dtype=float, copy=True)),
        griddim=2, type=list(getattr(h, "type", [])) + ["makePlanarGrid"])


makePlanarGrid = make_planar_grid


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.boundary_face_indices import boundary_face_indices

    def parent(dims=(3, 3, 2), sizes=(3.0, 3.0, 2.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def top_faces(h):
        """The faces on one side of the parent - a surface with a boundary."""
        centroids = np.asarray(h.faces.centroids, dtype=float)
        normals = np.asarray(h.faces.normals, dtype=float)
        flat = np.abs(normals[:, 2]) > np.abs(normals[:, :2]).sum(axis=1)
        return np.flatnonzero(flat & np.isclose(centroids[:, 2],
                                                centroids[:, 2].max()))

    def test_each_selected_face_becomes_a_cell() -> None:
        h = parent()
        chosen = top_faces(h)
        g = make_planar_grid(h, chosen)
        assert g.cells.num == chosen.size
        assert g.griddim == 2 and g.spatial_dim == 3
        print("ok test_each_selected_face_becomes_a_cell")

    def test_a_square_face_gives_a_four_edge_cell() -> None:
        h = parent()
        g = make_planar_grid(h, top_faces(h))
        assert np.all(np.diff(np.asarray(g.cells.facePos)) == 4)
        print("ok test_a_square_face_gives_a_four_edge_cell")

    def test_shared_edges_are_identified() -> None:
        """Nine squares in a 3x3 patch share edges: 24 distinct, not 36."""
        h = parent()
        g = make_planar_grid(h, top_faces(h))
        assert g.cells.num == 9, g.cells.num
        assert g.faces.num == 24, g.faces.num
        print("ok test_shared_edges_are_identified")

    def test_interior_edges_have_two_cells_and_boundary_edges_one() -> None:
        h = parent()
        g = make_planar_grid(h, top_faces(h))
        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
        sides = np.count_nonzero(neighbours >= 0, axis=1)
        assert np.count_nonzero(sides == 2) == 12, np.bincount(sides)
        assert np.count_nonzero(sides == 1) == 12
        print("ok test_interior_edges_have_two_cells_and_boundary_edges_one")

    def test_every_cell_edge_names_that_cell() -> None:
        h = parent()
        g = make_planar_grid(h, top_faces(h))
        pos = np.asarray(g.cells.facePos, dtype=np.int64)
        faces = np.asarray(g.cells.faces, dtype=np.int64)
        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
        for cell in range(g.cells.num):
            for face in faces[pos[cell]:pos[cell + 1]]:
                assert cell in neighbours[face], (cell, face)
        print("ok test_every_cell_edge_names_that_cell")

    def test_the_index_map_points_back_at_the_parent() -> None:
        h = parent()
        chosen = top_faces(h)
        assert np.array_equal(np.asarray(make_planar_grid(h, chosen).cells.indexMap),
                              chosen)
        print("ok test_the_index_map_points_back_at_the_parent")

    def test_the_parent_nodes_are_carried_over_whole() -> None:
        """Documented, and easy to be surprised by; see the module docstring."""
        h = parent()
        g = make_planar_grid(h, top_faces(h))
        assert g.nodes.num == h.nodes.num
        used = np.unique(np.asarray(g.faces.nodes))
        assert used.size < g.nodes.num, (used.size, g.nodes.num)
        print(f"ok test_the_parent_nodes_are_carried_over_whole "
              f"({used.size} of {g.nodes.num} used)")

    def test_a_negative_sign_reverses_the_cell() -> None:
        h = parent()
        chosen = top_faces(h)[:1]
        forward = make_planar_grid(h, chosen)
        reversed_ = make_planar_grid(h, chosen, -np.ones(chosen.size, np.int64))
        # Same edges, but the cell traverses each the other way round, so the
        # neighbour lands in the other column.
        assert np.array_equal(np.sort(np.asarray(forward.faces.nodes)),
                              np.sort(np.asarray(reversed_.faces.nodes)))
        assert not np.array_equal(np.asarray(forward.faces.neighbors),
                                  np.asarray(reversed_.faces.neighbors))
        print("ok test_a_negative_sign_reverses_the_cell")

    def test_a_single_face_gives_a_one_cell_grid() -> None:
        h = parent()
        g = make_planar_grid(h, [int(top_faces(h)[0])])
        assert g.cells.num == 1 and g.faces.num == 4
        assert np.all(np.count_nonzero(
            np.asarray(g.faces.neighbors) >= 0, axis=1) == 1)
        print("ok test_a_single_face_gives_a_one_cell_grid")

    def test_a_non_manifold_selection_is_rejected() -> None:
        """Three faces on one edge: MRST would silently drop one. Building it
        by duplicating a face is enough - the copy traverses its edges the same
        way as the original."""
        h = parent()
        face = int(top_faces(h)[0])
        try:
            make_planar_grid(h, [face, face])
        except ValueError as error:
            assert "non-manifold" in str(error)
            print("ok test_a_non_manifold_selection_is_rejected")
            return
        raise AssertionError("a repeated face must be rejected")

    def test_opposite_signs_are_not_non_manifold() -> None:
        """The same face twice with opposite winding traverses each edge once
        in each direction, which is a legitimate two-sided surface."""
        h = parent()
        face = int(top_faces(h)[0])
        g = make_planar_grid(h, [face, face], [1, -1])
        assert g.cells.num == 2 and g.faces.num == 4
        assert np.all(np.asarray(g.faces.neighbors) >= 0)
        print("ok test_opposite_signs_are_not_non_manifold")

    def test_a_mismatched_sign_array_is_rejected() -> None:
        h = parent()
        try:
            make_planar_grid(h, top_faces(h), [1, -1])
        except ValueError as error:
            assert "sgn has" in str(error)
            print("ok test_a_mismatched_sign_array_is_rejected")
            return
        raise AssertionError("a wrong-length sgn must be rejected")

    test_each_selected_face_becomes_a_cell()
    test_a_square_face_gives_a_four_edge_cell()
    test_shared_edges_are_identified()
    test_interior_edges_have_two_cells_and_boundary_edges_one()
    test_every_cell_edge_names_that_cell()
    test_the_index_map_points_back_at_the_parent()
    test_the_parent_nodes_are_carried_over_whole()
    test_a_negative_sign_reverses_the_cell()
    test_a_single_face_gives_a_one_cell_grid()
    test_a_non_manifold_selection_is_rejected()
    test_opposite_signs_are_not_non_manifold()
    test_a_mismatched_sign_array_is_rejected()
    print("\nAll tests passed!")
