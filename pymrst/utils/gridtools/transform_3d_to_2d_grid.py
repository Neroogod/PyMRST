"""Collapse a one-cell-thick 3-D grid to 2-D (MRST ``utils/gridtools/transform3Dto2Dgrid.m``).

A grid that is a single cell thick in x, y or z is really a 2-D problem carried
around in 3-D arrays. This drops the redundant direction: the two cap faces
normal to it go away, volumes become areas and areas become lengths by dividing
out the thickness, and coordinates keep the other two axes.

**MRST's version does not run on a current MRST grid.** It reads
`G.faces.numNodes` and `G.cells.numFaces`, which grids stopped carrying in
favour of `nodePos` and `facePos`; `computeGeometry(cartGrid(...))` produces
neither, and the function fails on its third line with
`Unrecognized field name "numFaces"`. Supplying the two legacy fields by hand
revives it, which is how the reference for this port was produced.

Two further defects show in what it returns even then:

- **The output has no `griddim` and no `type`.** `griddim` is never assigned at
  all, and the line meant to record the transform reads
  `G = [G.type, mfilename]` - assigning to the *input* variable, which is dead
  at that point, instead of to `G2.type`. The result is therefore not a usable
  grid structure: almost everything downstream branches on `griddim`.
- **The face tags are remapped in one branch and not the others.** Collapsing x
  renumbers the surviving tags 3..6 to 1..4; collapsing y or z leaves them at
  their original values. Whichever is right, all three cannot be.

This port emits a complete `Grid` - `griddim`, `type`, `nodePos`, `facePos`,
`cartDims`, remapped tags - and writes the three branches once, parameterised
by the collapsed axis, since they differ only in which coordinate is dropped.

**Geometry is recomputed rather than transcribed.** MRST scales the stored
areas and volumes and re-sorts each face's nodes with a pair of loops to fix
their orientation. Here the topology is built, the node pairs are oriented
against the retained 3-D normals, and `compute_geometry` does the rest - and a
test checks the recomputed areas and volumes against the scaled 3-D ones, which
is a real cross-check rather than a restatement.
"""

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes

TOL = 1e-12


def _collapsed_axis(g) -> int:
    """Which coordinate direction the grid is one cell thick in."""
    centroids = np.asarray(g.cells.centroids, dtype=float)
    flat = [d for d in range(3)
            if np.ptp(centroids[:, d]) <= TOL * max(1.0, np.ptp(centroids))]
    if not flat:
        raise ValueError(
            "the grid varies in all three directions, so there is no "
            "redundant one to collapse; transform_3d_to_2d_grid needs a grid "
            "that is a single cell thick in x, y or z")
    if len(flat) > 1:
        raise ValueError(
            f"the grid is a single cell thick in more than one direction "
            f"{flat}, so it is a line rather than a surface")
    return flat[0]


def transform_3d_to_2d_grid(g) -> Grid:
    """Return the 2-D grid equivalent to the one-cell-thick 3-D grid ``g``."""
    if g.griddim != 3:
        raise ValueError(f"expected a 3-D grid, got griddim {g.griddim}")
    if getattr(g.faces, "normals", None) is None:
        raise ValueError(
            "face normals are missing; run compute_geometry before "
            "transform_3d_to_2d_grid")

    axis = _collapsed_axis(g)
    kept_axes = [d for d in range(3) if d != axis]
    coords = np.asarray(g.nodes.coords, dtype=float)
    thickness = float(np.ptp(coords[:, axis]))
    if thickness <= 0:
        raise ValueError("the collapsed direction has zero thickness")

    # Keep the nodes on one side; each surviving face has exactly two of them.
    on_face = np.isclose(coords[:, axis], coords[:, axis].min(), atol=TOL)
    node_map = np.full(coords.shape[0], -1, dtype=np.int64)
    node_map[on_face] = np.arange(int(np.count_nonzero(on_face)))

    normals = np.asarray(g.faces.normals, dtype=float)
    # Faces perpendicular to the collapsed axis are the caps; they disappear.
    keep = np.abs(normals[:, axis]) <= TOL * np.maximum(
        1.0, np.abs(normals).max())
    kept_faces = np.flatnonzero(keep)
    face_map = np.full(g.faces.num, -1, dtype=np.int64)
    face_map[kept_faces] = np.arange(kept_faces.size)

    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    face_nodes = np.asarray(g.faces.nodes, dtype=np.int64)
    pairs = np.empty((kept_faces.size, 2), dtype=np.int64)
    for new, old in enumerate(kept_faces):
        own = face_nodes[node_pos[old]:node_pos[old + 1]]
        survivors = node_map[own[node_map[own] >= 0]]
        survivors = np.unique(survivors)
        if survivors.size != 2:
            raise ValueError(
                f"face {old} keeps {survivors.size} nodes on the collapsed "
                f"plane, not 2; the grid is not a clean single layer")
        pairs[new] = survivors

    # Orient each edge so its 2-D normal agrees with the 3-D one it came from.
    flat = coords[on_face][:, kept_axes]
    edge = flat[pairs[:, 1]] - flat[pairs[:, 0]]
    computed = np.column_stack([edge[:, 1], -edge[:, 0]])
    target = normals[kept_faces][:, kept_axes]
    pairs[np.sum(computed * target, axis=1) < 0] = \
        pairs[np.sum(computed * target, axis=1) < 0][:, ::-1]

    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    survives = keep[cell_faces]
    new_face_pos = np.concatenate([[0], np.cumsum(
        [int(survives[face_pos[c]:face_pos[c + 1]].sum())
         for c in range(g.cells.num)])]).astype(np.int64)

    tags = getattr(g.cells, "tags", None)
    if tags is not None:
        # Surviving tags are the four of the two kept axes; renumber to 0..3
        # in the order those axes now appear.
        order = {2 * a + s: 2 * i + s
                 for i, a in enumerate(kept_axes) for s in (0, 1)}
        tags = np.array([order[int(t)] for t in np.asarray(tags)[survives]],
                        dtype=np.int64)

    dims = (np.asarray(g.cartDims)[kept_axes].tolist()
            if g.cartDims is not None else None)

    return Grid(
        cells=Cells(num=int(g.cells.num), facePos=new_face_pos,
                    faces=face_map[cell_faces[survives]],
                    indexMap=np.arange(g.cells.num) if
                    getattr(g.cells, "indexMap", None) is None
                    else np.asarray(g.cells.indexMap).copy(),
                    tags=tags),
        faces=Faces(num=int(kept_faces.size),
                    nodePos=np.arange(0, 2 * kept_faces.size + 1, 2,
                                      dtype=np.int64),
                    nodes=pairs.reshape(-1),
                    neighbors=np.asarray(g.faces.neighbors,
                                         dtype=np.int64)[kept_faces].copy()),
        nodes=Nodes(num=int(np.count_nonzero(on_face)), coords=flat.copy()),
        griddim=2, cartDims=tuple(dims) if dims else None,
        type=list(getattr(g, "type", [])) + ["transform3Dto2Dgrid"])


transform3Dto2Dgrid = transform_3d_to_2d_grid


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.check_grid import check_grid

    def thin(dims, sizes):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_a_slab_in_x_collapses_to_its_cross_section() -> None:
        g = transform_3d_to_2d_grid(thin([1, 4, 3], [1.0, 4.0, 3.0]))
        assert g.griddim == 2 and g.cells.num == 12
        assert g.faces.num == 31 and g.nodes.num == 20
        assert tuple(g.cartDims) == (4, 3)
        print("ok test_a_slab_in_x_collapses_to_its_cross_section")

    def test_it_works_for_every_axis() -> None:
        for dims, sizes, expect in (([1, 4, 3], [1.0, 4.0, 3.0], (4, 3)),
                                    ([4, 1, 3], [4.0, 1.0, 3.0], (4, 3)),
                                    ([4, 3, 1], [4.0, 3.0, 1.0], (4, 3))):
            g = transform_3d_to_2d_grid(thin(dims, sizes))
            assert tuple(g.cartDims) == expect, (dims, g.cartDims)
            assert g.cells.num == 12 and g.faces.num == 31
        print("ok test_it_works_for_every_axis")

    def test_the_result_is_a_valid_grid() -> None:
        """MRST's output is not: it has neither griddim nor type."""
        for dims, sizes in (([1, 4, 3], [1.0, 4.0, 3.0]),
                            ([4, 1, 3], [4.0, 1.0, 3.0]),
                            ([4, 3, 1], [4.0, 3.0, 1.0])):
            g = compute_geometry(transform_3d_to_2d_grid(thin(dims, sizes)))
            ok, problems = check_grid(g)
            assert ok, (dims, problems)
            assert "transform3Dto2Dgrid" in g.type
        print("ok test_the_result_is_a_valid_grid")

    def test_volumes_become_areas() -> None:
        """The scaling: divide out the thickness."""
        for dims, sizes, thickness in (([1, 4, 3], [2.0, 4.0, 3.0], 2.0),
                                       ([4, 1, 3], [4.0, 5.0, 3.0], 5.0),
                                       ([4, 3, 1], [4.0, 3.0, 7.0], 7.0)):
            source = thin(dims, sizes)
            g = compute_geometry(transform_3d_to_2d_grid(source))
            assert np.allclose(np.asarray(g.cells.volumes),
                               np.asarray(source.cells.volumes) / thickness)
        print("ok test_volumes_become_areas")

    def test_recomputed_geometry_matches_the_scaled_original() -> None:
        """The cross-check the port buys by rebuilding instead of scaling."""
        source = thin([1, 5, 4], [3.0, 5.0, 4.0])
        g = compute_geometry(transform_3d_to_2d_grid(source))
        assert np.isclose(np.sum(g.cells.volumes), 5.0 * 4.0)
        # Face areas become lengths: the total must be the total 3-D side area
        # divided by the thickness.
        kept = np.abs(np.asarray(source.faces.normals)[:, 0]) < 1e-12
        assert np.isclose(np.sum(g.faces.areas),
                          np.sum(np.asarray(source.faces.areas)[kept]) / 3.0)
        print("ok test_recomputed_geometry_matches_the_scaled_original")

    def test_the_normals_keep_their_direction() -> None:
        """Node pairs are oriented against the 3-D normals, so the 2-D ones
        must point the same way in the surviving coordinates."""
        source = thin([1, 4, 3], [1.0, 4.0, 3.0])
        g = compute_geometry(transform_3d_to_2d_grid(source))
        kept = np.flatnonzero(np.abs(np.asarray(source.faces.normals)[:, 0])
                              < 1e-12)
        before = np.asarray(source.faces.normals)[kept][:, [1, 2]]
        after = np.asarray(g.faces.normals)
        assert np.all(np.sum(before * after, axis=1) > 0)
        print("ok test_the_normals_keep_their_direction")

    def test_the_neighbours_are_preserved() -> None:
        source = thin([1, 4, 3], [1.0, 4.0, 3.0])
        g = transform_3d_to_2d_grid(source)
        kept = np.flatnonzero(np.abs(np.asarray(source.faces.normals)[:, 0])
                              < 1e-12)
        assert np.array_equal(np.asarray(g.faces.neighbors),
                              np.asarray(source.faces.neighbors)[kept])
        print("ok test_the_neighbours_are_preserved")

    def test_the_tags_are_renumbered_to_the_surviving_axes() -> None:
        """MRST does this in one branch of three; here it is uniform."""
        for dims, sizes in (([1, 4, 3], [1.0, 4.0, 3.0]),
                            ([4, 1, 3], [4.0, 1.0, 3.0]),
                            ([4, 3, 1], [4.0, 3.0, 1.0])):
            g = transform_3d_to_2d_grid(thin(dims, sizes))
            if g.cells.tags is None:
                print("ok test_the_tags_are_renumbered_to_the_surviving_axes "
                      "(no tags on this build)")
                return
            assert set(np.unique(g.cells.tags).tolist()) == {0, 1, 2, 3}
        print("ok test_the_tags_are_renumbered_to_the_surviving_axes")

    def test_every_cell_still_has_four_faces() -> None:
        g = transform_3d_to_2d_grid(thin([1, 4, 3], [1.0, 4.0, 3.0]))
        assert np.all(np.diff(np.asarray(g.cells.facePos)) == 4)
        print("ok test_every_cell_still_has_four_faces")

    def test_a_fully_three_dimensional_grid_is_rejected() -> None:
        try:
            transform_3d_to_2d_grid(thin([2, 2, 2], [2.0, 2.0, 2.0]))
        except ValueError as error:
            assert "no" in str(error) and "redundant" in str(error)
            print("ok test_a_fully_three_dimensional_grid_is_rejected")
            return
        raise AssertionError("a genuinely 3-D grid must be rejected")

    def test_a_line_is_rejected() -> None:
        try:
            transform_3d_to_2d_grid(thin([1, 1, 4], [1.0, 1.0, 4.0]))
        except ValueError as error:
            assert "more than one direction" in str(error)
            print("ok test_a_line_is_rejected")
            return
        raise AssertionError("a one-cell-thick-in-two-directions grid must "
                             "be rejected")

    def test_a_two_dimensional_input_is_rejected() -> None:
        try:
            transform_3d_to_2d_grid(compute_geometry(
                cart_grid([3, 3], [3.0, 3.0])))
        except ValueError as error:
            assert "expected a 3-D grid" in str(error)
            print("ok test_a_two_dimensional_input_is_rejected")
            return
        raise AssertionError("a 2-D input must be rejected")

    test_a_slab_in_x_collapses_to_its_cross_section()
    test_it_works_for_every_axis()
    test_the_result_is_a_valid_grid()
    test_volumes_become_areas()
    test_recomputed_geometry_matches_the_scaled_original()
    test_the_normals_keep_their_direction()
    test_the_neighbours_are_preserved()
    test_the_tags_are_renumbered_to_the_surviving_axes()
    test_every_cell_still_has_four_faces()
    test_a_fully_three_dimensional_grid_is_rejected()
    test_a_line_is_rejected()
    test_a_two_dimensional_input_is_rejected()
    print("\nAll tests passed!")
