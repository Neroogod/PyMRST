"""Partitioning More Complex Grids (``coarsegridExample2.m``).

The first tutorial partitioned grids that fill a box. This one covers the four
cases where that assumption breaks, and each has a different answer:

* **a grid with a piece cut out** — `partition_ui` partitions the *bounding
  box*, so it works where `partition_cart_grid` cannot;
* **a grid that fills little of its box** — the partition then has gaps in its
  numbering, and `compress_partition` closes them, because most of MRST
  requires a contiguous partition;
* **an unstructured grid** — there are no logical indices at all, so
  `sample_from_box` assigns blocks from cell centroids instead;
* **a faulted grid** — a logically sensible partition can be geometrically
  disconnected, and `process_partition` splits the blocks that are.

The last is the one that matters most and is the hardest to see. A block cut in
two by a fault is still a block as far as the partition vector is concerned. The
coarse grid built from it treats the two halves as one control volume, so flow
passes between them at no cost, through rock that does not connect.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, tensor_grid
from pymrst.gridprocessing.grid_structure import extract_subgrid
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_layers import partition_layers
from pymrst.modules.coarsegrid.partition_ui import partition_ui
from pymrst.modules.coarsegrid.process_partition import process_partition


def semicircular_cutout(blocks=(5, 4)):
    """A rectilinear grid with a semi-circular bite out of one corner.

    `partition_cart_grid` cannot help here: the remaining cells do not form a
    logical rectangle. `partition_ui` partitions the bounding box and takes
    whatever falls inside.
    """
    dx = 1.0 - 0.5 * np.cos(np.arange(-1.0, 1.05, 0.1) * np.pi)
    x = -1.15 + 0.1 * np.cumsum(dx)
    y = np.arange(0.0, 1.05, 0.05)
    g = compute_geometry(tensor_grid(x, y ** 0.75))
    r = np.sqrt(np.sum(np.asarray(g.cells.centroids, dtype=float) ** 2, axis=1))
    g = compute_geometry(extract_subgrid(g, np.flatnonzero(r > 0.6)))
    p = partition_ui(g, list(blocks))
    return g, p, generate_coarse_grid(g, compress_partition(p))


def cup_grid(n: int = 41, blocks=(5, 5, 4)):
    """A hemispherical shell carved out of a Cartesian block.

    Returns ``(g, p, q)``: the grid, the raw partition and the compressed one.
    The grid occupies a small part of its bounding box, so most of the boxes
    `partition_ui` lays down are empty and the raw numbering has large gaps.
    """
    x = np.linspace(-2.0, 2.0, n)
    g = compute_geometry(tensor_grid(x, x, x))
    c = np.asarray(g.cells.centroids, dtype=float)
    r = c[:, 0] ** 2 + c[:, 1] ** 2 + c[:, 2] ** 2
    drop = np.flatnonzero((r > 1.0) | (r < 0.25) | (c[:, 2] < 0.0))
    g = compute_geometry(remove_cells(g, drop))
    p = partition_ui(g, list(blocks))
    return g, p, compress_partition(p)


def unstructured_partition(n: int = 25, boxes=(4, 4)):
    """A Voronoi grid partitioned by sampling a Cartesian box.

    `partition_ui` needs an underlying Cartesian topology. An unstructured grid
    has none, so `sample_from_box` reads a block number off a uniform box using
    each cell's centroid.
    """
    from pymrst.gridprocessing.pebi import pebi
    from pymrst.gridprocessing.triangle_grid import triangle_grid
    from pymrst.utils.gridtools.sample_from_box import sample_from_box
    from pymrst.gridprocessing.testgrids.twister import twister
    from scipy.spatial import Delaunay

    lin = np.linspace(0.0, 1.0, n)
    x, y = np.meshgrid(lin, lin)
    pts = twister(np.column_stack([x.ravel(), y.ravel()]))
    g = compute_geometry(pebi(triangle_grid(pts, Delaunay(pts).simplices)))
    r = np.sqrt(np.sum(np.asarray(g.cells.centroids, dtype=float) ** 2, axis=1))
    g = compute_geometry(extract_subgrid(g, np.flatnonzero(r > 0.6)))
    values = np.arange(int(np.prod(boxes))).reshape(boxes, order="F")
    return g, sample_from_box(g, values)


def faulted_layers(dims=(10, 10, 7), throw: float = 0.15,
                   blocks=(5, 5), layers=(1, 3, 6, 8)):
    """A faulted grid partitioned in ij and by layer in k.

    Returns ``(g, before, after)``. `before` is the partition as
    `partition_layers` produces it; `after` is what `process_partition` makes of
    it. The difference is the blocks the fault cut in two.

    `layers` is MRST's `L` unchanged. Layer boundaries stay **1-based**
    throughout this module, unlike every other index in it: they name
    stratigraphic layers, and renumbering them would invite the wrong groups.
    The block index that comes back is 0-based as usual.
    """
    from pymrst.gridprocessing.testgrids.simple_grdecl import simple_grdecl

    g = compute_geometry(process_grdecl(simple_grdecl(list(dims), throw)))
    before = partition_layers(g, list(blocks), list(layers))
    return g, before, process_partition(g, before)


__all__ = ["cup_grid", "faulted_layers", "semicircular_cutout",
           "unstructured_partition"]


if __name__ == "__main__":
    def test_partition_ui_handles_a_non_rectangular_grid():
        """`partition_cart_grid` cannot, because the cells are not a box."""
        g, p, cg = semicircular_cutout()
        assert p.size == g.cells.num
        assert np.all(p >= 0)
        assert cg.n_blocks <= 20, cg.n_blocks
        assert cg.n_blocks > 1
        print(f"ok test_partition_ui_handles_a_non_rectangular_grid "
              f"({g.cells.num} cells into {cg.n_blocks} blocks)")

    def test_the_cutout_really_removed_cells():
        """Otherwise the case is a plain tensor grid and proves nothing."""
        g, _, _ = semicircular_cutout()
        c = np.asarray(g.cells.centroids, dtype=float)
        assert np.all(np.sqrt(np.sum(c ** 2, axis=1)) > 0.6 - 1e-9)
        full = compute_geometry(tensor_grid(
            -1.15 + 0.1 * np.cumsum(1.0 - 0.5 * np.cos(np.arange(-1.0, 1.05, 0.1)
                                                       * np.pi)),
            np.arange(0.0, 1.05, 0.05) ** 0.75))
        assert g.cells.num < full.cells.num, "nothing was cut out"
        print(f"ok test_the_cutout_really_removed_cells "
              f"({full.cells.num - g.cells.num} cells removed)")

    def test_a_sparse_grid_leaves_gaps_in_the_numbering():
        """The reason `compress_partition` exists.

        Most of MRST requires a contiguous partition. A gapped one still
        indexes correctly with `p[cell]`, so nothing fails until a routine
        allocates `max(p)` blocks and finds most of them empty.
        """
        g, p, q = cup_grid()
        used = np.unique(p)
        assert used.size < int(p.max()) + 1, "the case has no gaps to close"
        assert np.array_equal(np.unique(q), np.arange(q.max() + 1))
        assert q.max() + 1 == used.size
        print(f"ok test_a_sparse_grid_leaves_gaps_in_the_numbering "
              f"({int(p.max()) + 1} slots for {used.size} non-empty blocks)")

    def test_compressing_preserves_membership():
        """Renumbering must not move a cell between blocks."""
        from pymrst.modules.coarsegrid.utils.is_same_partition import same_grouping
        _, p, q = cup_grid()
        assert same_grouping(p, q)
        print("ok test_compressing_preserves_membership")

    def test_the_cup_coarse_grid_conserves_volume():
        from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
        g, _, q = cup_grid()
        cg = coarsen_geometry(generate_coarse_grid(g, q))
        assert np.isclose(cg.cell_volumes.sum(), float(np.sum(g.cells.volumes)),
                          rtol=1e-12)
        print(f"ok test_the_cup_coarse_grid_conserves_volume "
              f"({cg.n_blocks} blocks, {cg.cell_volumes.sum():.4f} m3)")

    def test_an_unstructured_grid_is_partitioned_by_sampling():
        g, p = unstructured_partition()
        assert p.size == g.cells.num
        assert len(set(p.tolist())) > 1
        print(f"ok test_an_unstructured_grid_is_partitioned_by_sampling "
              f"({g.cells.num} cells into {len(set(p.tolist()))} blocks)")

    def test_sampling_follows_position_not_cell_number():
        """A block must be a region, not an arbitrary set of cells.

        Sampling from the wrong axis or a transposed array gives a partition
        that looks fine in a histogram and is spatially scrambled.
        """
        g, p = unstructured_partition()
        c = np.asarray(g.cells.centroids, dtype=float)
        span = c.max(axis=0) - c.min(axis=0)
        for b in np.unique(p):
            inside = c[p == b]
            extent = inside.max(axis=0) - inside.min(axis=0)
            assert np.all(extent <= 0.5 * span + 1e-9), (b, extent, span)
        print("ok test_sampling_follows_position_not_cell_number")

    def test_the_fault_disconnects_blocks():
        """The case the example is built around.

        `partition_layers` is perfectly reasonable in logical space; the fault
        is what makes some of its blocks two separate pieces of rock.
        """
        g, before, after = faulted_layers()
        assert after.max() > before.max(), (before.max(), after.max())
        print(f"ok test_the_fault_disconnects_blocks "
              f"({before.max() + 1} blocks became {after.max() + 1})")

    def test_processing_only_ever_splits():
        g, before, after = faulted_layers()
        for a in range(after.size):
            for b in range(a + 1, after.size):
                if after[a] == after[b]:
                    assert before[a] == before[b], (a, b)
        print("ok test_processing_only_ever_splits")

    def test_every_processed_block_is_connected():
        """The postcondition, checked by re-running the splitter."""
        g, _, after = faulted_layers()
        assert np.array_equal(process_partition(g, after), after)
        print("ok test_every_processed_block_is_connected")

    def test_the_layers_have_the_requested_thickness():
        """MRST prints `diff(L)`; the thicknesses are the point of the section."""
        g, before, _ = faulted_layers()
        from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
        k = np.asarray(grid_logical_indices(g)[2], dtype=np.int64) + 1
        thickness = []
        for layer_start, layer_end in zip((1, 3, 6), (3, 6, 8)):
            here = (k >= layer_start) & (k < layer_end)
            thickness.append(layer_end - layer_start)
            # no block may straddle a layer boundary
            for b in np.unique(before[here]):
                assert np.all(here[before == b]), (b, layer_start)
        assert thickness == [2, 3, 2], thickness
        print(f"ok test_the_layers_have_the_requested_thickness ({thickness})")

    def test_a_coarse_grid_builds_on_the_faulted_partition():
        from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
        g, _, after = faulted_layers()
        cg = coarsen_geometry(generate_coarse_grid(g, after))
        assert cg.n_blocks == after.max() + 1
        assert np.isclose(cg.cell_volumes.sum(), float(np.sum(g.cells.volumes)),
                          rtol=1e-12)
        assert cg.num_faces > 0
        print(f"ok test_a_coarse_grid_builds_on_the_faulted_partition "
              f"({cg.n_blocks} blocks, {cg.num_faces} coarse faces)")

    test_partition_ui_handles_a_non_rectangular_grid()
    test_the_cutout_really_removed_cells()
    test_a_sparse_grid_leaves_gaps_in_the_numbering()
    test_compressing_preserves_membership()
    test_the_cup_coarse_grid_conserves_volume()
    test_an_unstructured_grid_is_partitioned_by_sampling()
    test_sampling_follows_position_not_cell_number()
    test_the_fault_disconnects_blocks()
    test_processing_only_ever_splits()
    test_every_processed_block_is_connected()
    test_the_layers_have_the_requested_thickness()
    test_a_coarse_grid_builds_on_the_faulted_partition()
    print("\nAll tests passed!")
