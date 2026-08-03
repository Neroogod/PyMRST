"""More About Coarse-Grid Geometry and Topology (``coarsegridExample3.m``).

Three things a coarse grid carries beyond "which cells are in which block":

* **subdivided coarse faces.** Passing a second, finer partition as a face
  indicator splits each block pair into several coarse faces, so the interface
  between two blocks can be resolved rather than lumped.
* **coarse geometry.** `coarsen_geometry` gives blocks volumes and centroids and
  coarse faces areas, centroids and normals, all summed from the fine grid.
* **the lookup tables back to the fine grid.** `conn_pos`/`fconn` name the fine
  faces behind a coarse face and `partition` names the fine cells in a block,
  which is what lets a coarse result be drawn on the fine model.

MRST's last section picks coarse face 66, expands it into its fine faces, and
colours the two blocks it separates. That is a lookup exercise dressed as a
figure, so here it is `face_detail`, which returns the same information and can
be checked: the fine faces behind a coarse face must all separate the same two
blocks, and those must be the blocks the coarse face claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.modules.coarsegrid.cell_partition_to_face_partition import (
    cell_partition_to_face_partition,
)
from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_ui import partition_ui
from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry


def notched_model(dims=(20, 20, 6), blocks=(2, 2, 2), sub_blocks=(4, 4, 2)):
    """The example's grid: a Cartesian block with a corner notched out.

    Returns ``(g, p, cg)``. `cg` is built with `sub_blocks` as a face indicator,
    so each block pair is split into several coarse faces.
    """
    g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
    c = np.asarray(g.cells.centroids, dtype=float)
    notch = np.flatnonzero((c[:, 0] < 10) & (c[:, 1] < 10) & (c[:, 2] < 3))
    g = compute_geometry(remove_cells(g, notch))

    p = compress_partition(partition_ui(g, list(blocks)))
    q = partition_ui(g, list(sub_blocks))
    cg = coarsen_geometry(
        generate_coarse_grid(g, p, cell_partition_to_face_partition(g, q)))
    return g, p, cg


def without_subdivision(dims=(20, 20, 6), blocks=(2, 2, 2)):
    """The same model with no face indicator, for comparison."""
    g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
    c = np.asarray(g.cells.centroids, dtype=float)
    g = compute_geometry(remove_cells(
        g, np.flatnonzero((c[:, 0] < 10) & (c[:, 1] < 10) & (c[:, 2] < 3))))
    p = compress_partition(partition_ui(g, list(blocks)))
    return g, p, coarsen_geometry(generate_coarse_grid(g, p))


@dataclass
class FaceDetail:
    """Everything MRST's last figure draws, as data."""

    face: int
    fine_faces: np.ndarray
    blocks: tuple
    left_cells: np.ndarray
    right_cells: np.ndarray


def face_detail(cg, face: int) -> FaceDetail:
    """Expand one coarse face into its fine faces and neighbouring blocks."""
    if not 0 <= face < cg.num_faces:
        raise ValueError(f"coarse face must lie in 0 to {cg.num_faces - 1}")
    neighbours = tuple(int(b) for b in cg.neighbors[face])
    partition = np.asarray(cg.partition, dtype=np.int64)
    return FaceDetail(
        face=face,
        fine_faces=cg.sub_faces(face),
        blocks=neighbours,
        left_cells=np.flatnonzero(partition == neighbours[0]),
        right_cells=np.flatnonzero(partition == neighbours[1]),
    )


def coarse_boundary_faces(cg, blocks=None) -> np.ndarray:
    """Coarse faces on the model boundary, optionally of given blocks only.

    MRST calls `boundaryFaces(CG)` and `boundaryFaces(CG, neigh)`. A coarse
    boundary face is one with no block on one side, which is -1 here and 0 in
    MATLAB.
    """
    neighbors = np.asarray(cg.neighbors, dtype=np.int64)
    on_boundary = np.any(neighbors < 0, axis=1)
    if blocks is None:
        return np.flatnonzero(on_boundary)
    wanted = np.atleast_1d(np.asarray(blocks, dtype=np.int64))
    wanted = wanted[wanted >= 0]
    return np.flatnonzero(on_boundary & np.any(np.isin(neighbors, wanted), axis=1))


def block_faces(cg, block: int) -> np.ndarray:
    """Coarse faces bounding one block (MRST ``gridCellFaces``)."""
    return np.asarray(cg.faces, dtype=np.int64)[
        cg.face_pos[block]:cg.face_pos[block + 1]]


__all__ = ["FaceDetail", "block_faces", "coarse_boundary_faces", "face_detail",
           "notched_model", "without_subdivision"]


if __name__ == "__main__":
    def test_a_face_indicator_subdivides_the_interfaces():
        """The section's point: more coarse faces for the same blocks.

        Without it each block pair gets one coarse face, however large and
        however curved. The blocks are unchanged either way, so only the face
        count shows the difference.
        """
        _, p, fine_faced = notched_model()
        _, q, plain = without_subdivision()
        assert np.array_equal(p, q), "the two cases must share a partition"
        assert fine_faced.n_blocks == plain.n_blocks
        assert fine_faced.num_faces > plain.num_faces, (plain.num_faces,
                                                        fine_faced.num_faces)
        print(f"ok test_a_face_indicator_subdivides_the_interfaces "
              f"({plain.num_faces} coarse faces became {fine_faced.num_faces})")

    def test_subdivision_does_not_lose_or_duplicate_a_fine_face():
        """Splitting must repartition the same fine faces, not resample them."""
        _, _, subdivided = notched_model()
        _, _, plain = without_subdivision()
        assert set(subdivided.fconn.tolist()) == set(plain.fconn.tolist())
        assert np.unique(subdivided.fconn).size == subdivided.fconn.size
        print(f"ok test_subdivision_does_not_lose_or_duplicate_a_fine_face "
              f"({subdivided.fconn.size} fine faces)")

    def test_coarse_geometry_sums_the_fine_grid():
        g, p, cg = notched_model()
        fine = np.asarray(g.cells.volumes, dtype=float)
        assert np.isclose(cg.cell_volumes.sum(), fine.sum(), rtol=1e-12)
        for b in range(cg.n_blocks):
            assert np.isclose(cg.cell_volumes[b], fine[p == b].sum(), rtol=1e-12), b
        print(f"ok test_coarse_geometry_sums_the_fine_grid "
              f"({cg.n_blocks} blocks, {fine.sum():.0f} m3)")

    def test_a_block_centroid_lies_among_its_fine_centroids():
        """MRST plots both and lets the eye judge; this measures it."""
        g, p, cg = notched_model()
        c = np.asarray(g.cells.centroids, dtype=float)
        for b in range(cg.n_blocks):
            inside = c[p == b]
            assert np.all(cg.cell_centroids[b] >= inside.min(axis=0) - 1e-9), b
            assert np.all(cg.cell_centroids[b] <= inside.max(axis=0) + 1e-9), b
        print("ok test_a_block_centroid_lies_among_its_fine_centroids")

    def test_a_face_expands_to_fine_faces_between_its_own_blocks():
        """The last figure, checked instead of drawn.

        A lookup that returns the right *number* of fine faces from the wrong
        coarse face produces a picture that looks entirely convincing.
        """
        g, p, cg = notched_model()
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        for face in (0, cg.num_faces // 2, cg.num_faces - 1):
            d = face_detail(cg, face)
            assert d.fine_faces.size > 0
            pairs = np.sort(np.where(nb[d.fine_faces] >= 0,
                                     p[np.maximum(nb[d.fine_faces], 0)], -1), axis=1)
            assert np.all(pairs == pairs[0]), face
            assert tuple(np.sort(d.blocks)) == tuple(pairs[0]), face
        print(f"ok test_a_face_expands_to_fine_faces_between_its_own_blocks "
              f"({cg.num_faces} coarse faces checked at three points)")

    def test_the_neighbouring_blocks_hold_the_right_cells():
        g, p, cg = notched_model()
        interior = np.flatnonzero(np.all(np.asarray(cg.neighbors) >= 0, axis=1))
        d = face_detail(cg, int(interior[0]))
        assert np.all(p[d.left_cells] == d.blocks[0])
        assert np.all(p[d.right_cells] == d.blocks[1])
        assert not set(d.left_cells.tolist()) & set(d.right_cells.tolist())
        print(f"ok test_the_neighbouring_blocks_hold_the_right_cells "
              f"({d.left_cells.size} and {d.right_cells.size} cells)")

    def test_block_faces_agrees_with_the_neighbour_list():
        g, _, cg = notched_model()
        for b in range(cg.n_blocks):
            own = block_faces(cg, b)
            assert own.size > 0, b
            for f in own:
                assert b in cg.neighbors[f].tolist(), (b, f)
        print("ok test_block_faces_agrees_with_the_neighbour_list")

    def test_boundary_faces_have_no_block_on_one_side():
        g, _, cg = notched_model()
        boundary = coarse_boundary_faces(cg)
        assert boundary.size > 0
        assert np.all(np.any(np.asarray(cg.neighbors)[boundary] < 0, axis=1))
        interior = np.setdiff1d(np.arange(cg.num_faces), boundary)
        assert np.all(np.asarray(cg.neighbors)[interior] >= 0)
        print(f"ok test_boundary_faces_have_no_block_on_one_side "
              f"({boundary.size} of {cg.num_faces})")

    def test_boundary_faces_can_be_restricted_to_blocks():
        g, _, cg = notched_model()
        interior = np.flatnonzero(np.all(np.asarray(cg.neighbors) >= 0, axis=1))
        pair = tuple(int(b) for b in cg.neighbors[interior[0]])
        subset = coarse_boundary_faces(cg, pair)
        assert set(subset.tolist()) <= set(coarse_boundary_faces(cg).tolist())
        for f in subset:
            assert set(cg.neighbors[f].tolist()) & set(pair), f
        print(f"ok test_boundary_faces_can_be_restricted_to_blocks "
              f"({subset.size} faces on blocks {pair})")

    def test_the_notch_is_actually_missing():
        """Otherwise this is a plain box and the topology is uninteresting."""
        g, _, _ = notched_model()
        full = int(np.prod([20, 20, 6]))
        assert g.cells.num < full, "nothing was removed"
        c = np.asarray(g.cells.centroids, dtype=float)
        assert not np.any((c[:, 0] < 10) & (c[:, 1] < 10) & (c[:, 2] < 3))
        print(f"ok test_the_notch_is_actually_missing "
              f"({full - g.cells.num} of {full} cells removed)")

    def test_an_out_of_range_face_is_rejected():
        _, _, cg = notched_model()
        try:
            face_detail(cg, cg.num_faces)
        except ValueError as exc:
            assert "must lie in" in str(exc), exc
        else:
            raise AssertionError("expected an out-of-range face to fail")
        print("ok test_an_out_of_range_face_is_rejected")

    test_a_face_indicator_subdivides_the_interfaces()
    test_subdivision_does_not_lose_or_duplicate_a_fine_face()
    test_coarse_geometry_sums_the_fine_grid()
    test_a_block_centroid_lies_among_its_fine_centroids()
    test_a_face_expands_to_fine_faces_between_its_own_blocks()
    test_the_neighbouring_blocks_hold_the_right_cells()
    test_block_faces_agrees_with_the_neighbour_list()
    test_boundary_faces_have_no_block_on_one_side()
    test_boundary_faces_can_be_restricted_to_blocks()
    test_the_notch_is_actually_missing()
    test_an_out_of_range_face_is_rejected()
    print("\nAll tests passed!")
