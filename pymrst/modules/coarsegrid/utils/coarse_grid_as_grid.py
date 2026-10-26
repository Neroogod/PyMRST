"""A coarse grid as an ordinary grid (PyMRST bridge, no MRST counterpart).

MRST needs none of this: `generateCoarseGrid` returns a struct with `cells`
and `faces` fields, so a coarse grid *is* a grid and `computeTrans`,
`setupOperators` and the models take it unchanged. PyMRST's `CoarseGrid` is a
dataclass with flat `cell_volumes` / `face_normals` names, deliberately - they
would otherwise shadow the optional geometric grid it carries in `.g`.

That `.g` is populated only when the partition is a **tensor product**, which
`partition_cart_grid` gives and almost nothing else does. A partition split
along faults - the one `simulatorWorkflowExample` builds, and the reason the
script splits at all - is not one, so `.g` is `None` and every consumer that
reached for it failed on a coarse model MRST builds without complaint.

This converts the topology and geometry `coarsen_geometry` computes into the
`Grid` the rest of PyMRST expects. Nothing is recomputed: the volumes,
centroids, areas and normals are the ones already aggregated from the fine
grid.

The cell-face map is the one part that has to be *derived* rather than copied.
PyMRST stores it as a flat list of face indices and takes the orientation from
`faces.neighbors` - `compute_trans` reads `neighbors[face, 0] == cell` to decide
the sign - where MRST stores an explicit `[face, sign]` pair. The flat form is
what `compute_trans` expects, so that is what is built here; a two-column array
would be read as twice as many half-faces.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def coarse_grid_as_grid(cg: Any) -> Any:
    """Return a `Grid`-shaped view of ``cg``, geometry included."""
    if cg.neighbors is None:
        raise ValueError(
            "the coarse grid carries no topology; build it with "
            "generate_coarse_grid")
    if cg.cell_volumes is None or cg.face_normals is None:
        from pymrst.modules.coarsegrid.utils.coarsen_geometry import (
            coarsen_geometry,
        )
        cg = coarsen_geometry(cg)

    neighbors = np.asarray(cg.neighbors, dtype=np.int64).reshape(-1, 2)
    n_cells = int(cg.n_blocks)
    n_faces = neighbors.shape[0]

    # Cell -> faces, in block order. PyMRST takes the sign from
    # `faces.neighbors`, so the map itself is a flat list of face indices.
    cell_faces, cell_face_pos = _cell_faces(neighbors, n_cells)

    cells = Cells(
        num=n_cells,
        facePos=cell_face_pos,
        faces=cell_faces,
        indexMap=np.arange(n_cells, dtype=np.int64),
        volumes=np.asarray(cg.cell_volumes, dtype=float).ravel(),
        centroids=np.asarray(cg.cell_centroids, dtype=float),
    )
    faces = Faces(
        num=n_faces,
        # A coarse face has no nodes of its own; its geometry is the aggregate
        # `coarsen_geometry` computed, and nothing downstream of TPFA needs the
        # node lists.
        nodePos=np.zeros(n_faces + 1, dtype=np.int64),
        nodes=np.zeros(0, dtype=np.int64),
        neighbors=neighbors,
        areas=np.asarray(cg.face_areas, dtype=float).ravel(),
        normals=np.asarray(cg.face_normals, dtype=float),
        centroids=np.asarray(cg.face_centroids, dtype=float),
    )
    grid = Grid(
        cells=cells, faces=faces,
        nodes=Nodes(num=0, coords=np.zeros((0, int(cg.griddim)))),
        griddim=int(cg.griddim), cartDims=None,
        type=["coarse_grid_as_grid"],
    )
    # Kept so a caller can go back to the fine model without threading the
    # `CoarseGrid` separately.
    grid.coarse_grid = cg
    grid.partition = np.asarray(cg.partition, dtype=np.int64).ravel()
    return grid


coarseGridAsGrid = coarse_grid_as_grid


def _cell_faces(neighbors: np.ndarray, n_cells: int):
    """`(faces, facePos)`: the coarse faces of each block, in block order.

    Both sides of an interior face are listed, once under each block, because
    a half-face belongs to a cell. Listing a face only under `neighbors[:, 0]`
    gives every block half its connections and the coarse model leaks.
    """
    faces = np.arange(neighbors.shape[0], dtype=np.int64)
    rows = []
    for column in (0, 1):
        owner = neighbors[:, column]
        inside = owner >= 0
        rows.append(np.column_stack([owner[inside], faces[inside]]))
    table = np.vstack(rows) if rows else np.zeros((0, 2), dtype=np.int64)

    order = np.lexsort((table[:, 1], table[:, 0]))
    table = table[order]
    counts = np.bincount(table[:, 0], minlength=n_cells)
    face_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return table[:, 1].copy(), face_pos


__all__ = ["coarseGridAsGrid", "coarse_grid_as_grid"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import (
        generate_coarse_grid,
    )
    from pymrst.modules.coarsegrid.partition_ui import partition_ui

    FINE = compute_geometry(cart_grid([4, 4, 2], [40.0, 40.0, 10.0]))
    PARTITION = partition_ui(FINE, [2, 2, 1])
    CG = generate_coarse_grid(FINE, PARTITION)
    COARSE = coarse_grid_as_grid(CG)

    def test_it_has_one_cell_per_block():
        assert COARSE.cells.num == CG.n_blocks == 4, COARSE.cells.num
        print("ok test_it_has_one_cell_per_block")

    def test_volume_is_conserved():
        """The coarse model must hold the same rock as the fine one; a coarse
        volume that is not the summed fine volume changes the pore volume and
        with it every material balance."""
        assert np.isclose(COARSE.cells.volumes.sum(),
                          np.asarray(FINE.cells.volumes).sum())
        print("ok test_volume_is_conserved")

    def test_the_centroids_are_volume_weighted_block_centres():
        expected = np.array([[10.0, 10.0, 5.0], [30.0, 10.0, 5.0],
                             [10.0, 30.0, 5.0], [30.0, 30.0, 5.0]])
        assert np.allclose(np.sort(COARSE.cells.centroids, axis=0),
                           np.sort(expected, axis=0)), COARSE.cells.centroids
        print("ok test_the_centroids_are_volume_weighted_block_centres")

    def test_an_interior_face_is_listed_under_both_its_blocks():
        """A half-face belongs to a cell, so an interior face appears twice.
        Listing it only under `neighbors[:, 0]` gives every block half its
        connections and the coarse model leaks."""
        assert COARSE.cells.faces.ndim == 1, COARSE.cells.faces.shape
        for face, (a, b) in enumerate(COARSE.faces.neighbors):
            if a < 0 or b < 0:
                continue
            for block in (a, b):
                owned = COARSE.cells.faces[
                    COARSE.cells.facePos[block]:COARSE.cells.facePos[block + 1]]
                assert face in owned, (face, block, owned)
        print("ok test_an_interior_face_is_listed_under_both_its_blocks")

    def test_every_block_appears_in_the_map():
        counts = np.diff(COARSE.cells.facePos)
        assert counts.size == COARSE.cells.num
        assert np.all(counts > 0), counts
        print("ok test_every_block_appears_in_the_map")

    def test_a_non_tensor_partition_still_converts():
        """The case `.g` cannot cover, and the reason this exists: a partition
        that is not a tensor product leaves `CoarseGrid.g` as `None`."""
        odd = PARTITION.copy()
        odd[0] = odd.max() + 1               # peel one cell into its own block
        cg = generate_coarse_grid(FINE, odd)
        assert cg.g is None, "this partition was supposed to be non-tensor"
        grid = coarse_grid_as_grid(cg)
        assert grid.cells.num == cg.n_blocks
        assert np.isclose(grid.cells.volumes.sum(),
                          np.asarray(FINE.cells.volumes).sum())
        print("ok test_a_non_tensor_partition_still_converts")

    def test_it_feeds_the_tpfa_operator_setup():
        """The whole point: a coarse model has to be runnable."""
        from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
        from pymrst.params import make_rock
        from pymrst.utils.units import DARCY, MILLI

        rock = make_rock(COARSE, 100*MILLI*DARCY, 0.3)
        ops = setup_operators_tpfa(COARSE, rock)
        assert ops.T.size == ops.N.shape[0] > 0
        assert np.all(ops.T > 0.0)
        print(f"ok test_it_feeds_the_tpfa_operator_setup "
              f"({ops.T.size} interior connections)")

    def test_a_topology_less_coarse_grid_is_refused():
        from pymrst.modules.coarsegrid.generate_coarse_grid import CoarseGrid
        try:
            coarse_grid_as_grid(CoarseGrid(g=None, partition=PARTITION,
                                           n_blocks=4))
        except ValueError as refused:
            assert "topology" in str(refused)
            print("ok test_a_topology_less_coarse_grid_is_refused")
            return
        raise AssertionError("a grid cannot be built without its topology")

    test_it_has_one_cell_per_block()
    test_volume_is_conserved()
    test_the_centroids_are_volume_weighted_block_centres()
    test_an_interior_face_is_listed_under_both_its_blocks()
    test_every_block_appears_in_the_map()
    test_a_non_tensor_partition_still_converts()
    test_it_feeds_the_tpfa_operator_setup()
    test_a_topology_less_coarse_grid_is_refused()
    print("\nAll tests passed!")
