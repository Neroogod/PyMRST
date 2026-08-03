"""How to Read, Display, and Manipulate Corner-Point Grids.

Port of MRST ``examples/grids/gridTutorialCornerPoint.m``.

A corner-point grid describes geology that a Cartesian grid cannot: layers that
pinch out, and faults that offset one side of the model against the other. This
tutorial builds such a grid, marks its faults, deactivates part of it, and then
partitions it into coarse blocks.

The most instructive step is the last one. `partition_layers` builds blocks from
logical indices, which is reasonable and produces 60 blocks here. But a fault
cuts through some of them, so the two halves of a block are not connected and
flow cannot pass between them. `process_partition` finds those and splits them,
giving 64. Four blocks in this small model were not coarse cells at all, and
nothing about the partition looked wrong.

MRST closes by building the coarse grid with `generateCoarseGrid`. That step
used to raise here, because `generate_coarse_grid` accepted only rectangular
Cartesian partitions and the split blocks are not one. It now builds the coarse
topology for any partition, so the tutorial completes: 64 blocks and 200 coarse
faces, each standing for the fine faces between one pair of blocks.

The optional geometric grid `.g` is still ``None`` for this case, since a
faulted partition is not a tensor product. The topology is what a coarse model
needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.gridprocessing.testgrids.simple_grdecl import simple_grdecl
from pymrst.modules.coarsegrid import (
    generate_coarse_grid,
    partition_layers,
    process_partition,
)
from pymrst.modules.coarsegrid.plotting.explosion_view import (
    block_displacements,
    exploded_block_grids,
)


@dataclass
class CornerPointTutorial:
    """The faulted grid and the partitions built on it."""

    G: Any
    grdecl: dict
    partition: np.ndarray          # from partition_layers
    connected: np.ndarray          # after process_partition

    @property
    def n_split(self) -> int:
        """How many blocks the fault disconnected."""
        return int(self.connected.max() - self.partition.max())


def small_faulted_grid(dims=(2, 1, 2), throw: float = 0.15):
    """MRST's first figure: a grid small enough to count its faces."""
    grdecl = simple_grdecl(list(dims), throw)
    return compute_geometry(process_grdecl(grdecl)), grdecl


def elliptical_actnum(cart_dims) -> np.ndarray:
    """MRST's ACTNUM: an ellipse in the areal plane, repeated down the layers.

    Cells outside `i^2/4 + j^2 > 0.5` on a grid normalised to [-0.95, 0.95] are
    switched off.
    """
    nx, ny, nz = (int(v) for v in np.asarray(cart_dims).ravel()[:3])
    i = np.linspace(-0.95, 0.95, nx)[:, None]
    j = np.linspace(-0.95, 0.95, ny)[None, :]
    areal = np.ones((nx, ny), dtype=np.int64)
    areal[(i ** 2 / 4.0 + j ** 2) > 0.5] = 0
    return np.repeat(areal[:, :, None], nz, axis=2).ravel(order="F")


def fault_adjacent_cells(g, *, both_sides: bool = False) -> np.ndarray:
    """Cells touching a fault face.

    With `both_sides`, only faces whose two neighbours are both present count,
    which is MRST's second selection: cells facing each other across a fault
    rather than merely beside one.
    """
    tag = getattr(g.faces, "tag", None)
    if tag is None:
        return np.zeros(0, dtype=np.int64)
    faces = np.flatnonzero(np.asarray(tag).ravel() > 0)
    nb = np.asarray(g.faces.neighbors, dtype=np.int64)[faces]
    if both_sides:
        nb = nb[np.all(nb >= 0, axis=1)]
    return np.unique(nb[nb >= 0])


def run_grid_tutorial_corner_point(dims=(30, 20, 5),
                                   throw: float = 0.12) -> CornerPointTutorial:
    """Build the faulted grid and both partitions."""
    grdecl = simple_grdecl(list(dims), throw)
    g = compute_geometry(process_grdecl(grdecl))

    nz = int(np.asarray(g.cartDims).ravel()[2])
    # MRST: ceil([1 nz/2 nz nz+1]) gives three layer groups.
    bounds = [int(np.ceil(v)) for v in (1, nz / 2, nz, nz + 1)]
    p = partition_layers(g, [5, 4], bounds)
    q = process_partition(g, p)
    return CornerPointTutorial(G=g, grdecl=grdecl, partition=p, connected=q)


def coarse_grid_step(case: CornerPointTutorial | None = None):
    """MRST's closing step, which now completes.

    Returns ``(coarse_grid, None)``. The second element is kept so callers
    written against the earlier behaviour, when this could not complete, still
    work unchanged.
    """
    case = run_grid_tutorial_corner_point() if case is None else case
    return generate_coarse_grid(case.G, case.connected), None


if __name__ == "__main__":
    def test_a_small_grid_has_faults():
        g, _ = small_faulted_grid()
        tag = np.asarray(getattr(g.faces, "tag", np.zeros(g.faces.num))).ravel()
        assert g.cells.num == 4, g.cells.num
        assert np.count_nonzero(tag > 0) > 0, "no fault faces in a faulted grid"
        print(f"ok test_a_small_grid_has_faults "
              f"({g.cells.num} cells, {int(np.count_nonzero(tag > 0))} fault faces)")

    def test_throw_creates_the_fault():
        """A zero throw gives a grid with no fault faces at all."""
        flat, _ = small_faulted_grid(dims=(4, 2, 2), throw=0.0)
        faulted, _ = small_faulted_grid(dims=(4, 2, 2), throw=0.3)
        n_flat = int(np.count_nonzero(
            np.asarray(getattr(flat.faces, "tag", np.zeros(flat.faces.num))) > 0))
        n_fault = int(np.count_nonzero(
            np.asarray(getattr(faulted.faces, "tag", np.zeros(faulted.faces.num))) > 0))
        assert n_fault > n_flat, (n_flat, n_fault)
        print(f"ok test_throw_creates_the_fault ({n_flat} vs {n_fault} fault faces)")

    def test_elliptical_actnum_deactivates_the_corners():
        actnum = elliptical_actnum((30, 20, 5))
        assert actnum.size == 30 * 20 * 5
        assert 0 < actnum.sum() < actnum.size, actnum.sum()
        # The ellipse is centred, so the middle stays active and corners do not.
        a = actnum.reshape((30, 20, 5), order="F")
        assert a[15, 10, 0] == 1, "the centre was deactivated"
        assert a[0, 0, 0] == 0, "a corner survived"
        print(f"ok test_elliptical_actnum_deactivates_the_corners "
              f"({int(actnum.sum())} of {actnum.size} active)")

    def test_fault_cells_are_found_on_both_sides():
        """The two selections MRST draws differ, and the second is smaller.

        Cells beside a fault include those whose other side is outside the
        model; cells facing each other across one do not.
        """
        case = run_grid_tutorial_corner_point()
        beside = fault_adjacent_cells(case.G)
        facing = fault_adjacent_cells(case.G, both_sides=True)
        assert beside.size > 0, "no cells beside a fault"
        assert facing.size <= beside.size, (facing.size, beside.size)
        assert set(facing.tolist()) <= set(beside.tolist())
        print(f"ok test_fault_cells_are_found_on_both_sides "
              f"({beside.size} beside, {facing.size} facing)")

    def test_layer_partition_gives_the_expected_block_count():
        case = run_grid_tutorial_corner_point()
        assert case.partition.max() + 1 == 5 * 4 * 3, case.partition.max() + 1
        print(f"ok test_layer_partition_gives_the_expected_block_count "
              f"({case.partition.max() + 1} blocks)")

    def test_the_fault_disconnects_blocks():
        """The lesson of the tutorial, as a number rather than a picture.

        `partition_layers` produces blocks from logical indices alone, so a
        fault can leave a block in two pieces with no connection between them.
        Such a block is not a coarse cell, and nothing about the partition
        looks wrong until it is checked.
        """
        case = run_grid_tutorial_corner_point()
        assert case.n_split > 0, "the fault split no blocks; the case proves nothing"
        assert case.connected.max() + 1 == case.partition.max() + 1 + case.n_split
        print(f"ok test_the_fault_disconnects_blocks "
              f"({case.partition.max() + 1} blocks -> {case.connected.max() + 1}, "
              f"{case.n_split} split)")

    def test_splitting_only_subdivides():
        """No cell may change block partner, only lose one."""
        case = run_grid_tutorial_corner_point()
        p, q = case.partition, case.connected
        idx = np.argsort(q, kind="stable")
        for b in np.unique(q):
            cells = idx[q[idx] == b]
            assert len(set(p[cells].tolist())) == 1, b
        print("ok test_splitting_only_subdivides")

    def test_processed_partition_is_stable():
        case = run_grid_tutorial_corner_point()
        again = process_partition(case.G, case.connected)
        assert np.array_equal(again, case.connected), "still disconnected"
        print("ok test_processed_partition_is_stable")

    def test_explosion_view_moves_every_block_outward():
        case = run_grid_tutorial_corner_point()
        shifts = block_displacements(case.G, case.connected, radius=0.5)
        centroids = np.asarray(case.G.cells.centroids, dtype=float)
        o = centroids.mean(axis=0)
        moved = 0
        for b in range(shifts.shape[0]):
            sel = case.connected == b
            if not np.any(sel):
                continue
            outward = centroids[sel].mean(axis=0) - o
            if np.linalg.norm(outward) > 1e-9:
                assert np.dot(shifts[b], outward) > 0.0, b
                moved += 1
        assert moved > 0
        grids = exploded_block_grids(case.G, case.connected, radius=0.5)
        assert len(grids) == case.connected.max() + 1, len(grids)
        print(f"ok test_explosion_view_moves_every_block_outward ({moved} blocks)")

    def test_coarse_grid_is_built_for_the_faulted_partition():
        """The step that used to be impossible here.

        `generate_coarse_grid` once accepted only rectangular Cartesian
        partitions and raised on the split blocks. It now builds the coarse
        topology directly, so every block is represented and every coarse face
        stands for the fine faces between one pair of blocks.
        """
        case = run_grid_tutorial_corner_point()
        cg, reason = coarse_grid_step(case)
        assert cg is not None, reason
        assert cg.n_blocks == case.connected.max() + 1, cg.n_blocks
        assert cg.num_faces > 0
        assert cg.g is None, "a faulted partition is not a tensor product"
        nb = np.asarray(case.G.faces.neighbors, dtype=np.int64)
        for f in range(cg.num_faces):
            sub = cg.sub_faces(f)
            pairs = np.sort(np.where(nb[sub] >= 0,
                                     case.connected[np.maximum(nb[sub], 0)], -1),
                            axis=1)
            assert np.all(pairs == pairs[0]), f
        print(f"ok test_coarse_grid_is_built_for_the_faulted_partition "
              f"({cg.n_blocks} blocks, {cg.num_faces} coarse faces)")

    def test_coarse_grid_works_on_an_unsplit_partition():
        """The Cartesian case it does support, to show the limit is specific."""
        case = run_grid_tutorial_corner_point()
        if np.array_equal(case.partition, case.connected):
            print("ok test_coarse_grid_works_on_an_unsplit_partition (skipped)")
            return
        from pymrst.gridprocessing import cart_grid
        from pymrst.modules.coarsegrid import partition_cart_grid
        g = compute_geometry(cart_grid([6, 4, 2], [6.0, 4.0, 2.0]))
        p = partition_cart_grid([6, 4, 2], [3, 2, 1])
        cg = generate_coarse_grid(g, p)
        assert cg.n_blocks == 6, cg.n_blocks
        print("ok test_coarse_grid_works_on_an_unsplit_partition")

    test_a_small_grid_has_faults()
    test_throw_creates_the_fault()
    test_elliptical_actnum_deactivates_the_corners()
    test_fault_cells_are_found_on_both_sides()
    test_layer_partition_gives_the_expected_block_count()
    test_the_fault_disconnects_blocks()
    test_splitting_only_subdivides()
    test_processed_partition_is_stable()
    test_explosion_view_moves_every_block_outward()
    test_coarse_grid_is_built_for_the_faulted_partition()
    test_coarse_grid_works_on_an_unsplit_partition()
    print("\nAll tests passed!")
