"""Coarse Grids with Near-Well Refinement (``coarsegridExample5.m``).

Uniform blocks resolve a well badly. Flow near a wellbore is radial and the
pressure drop goes as `log r`, so most of it happens inside the first block and
a coarse model cannot see it. `refine_near_well` builds rings and sectors about
the well instead, and this shows three ways of using it:

* **a different refinement per well**, replacing whichever coarse block each
  well sits in;
* **more angular sectors further out**, by passing one sector count per ring;
* **a local patch on an unstructured partition**, where METIS partitions the
  field by transmissibility and the refinement is dropped in around the well.

**Grid size.** MRST uses 150x150x6, which is 135,000 cells. The defaults here
are smaller so the self-tests run in seconds; pass `dims` for the full model.
Nothing about the method depends on the size.

**The last section needs METIS**, an optional dependency. Everything else works
without it, and the test covering it prints `SKIP` rather than passing quietly.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid
from pymrst.modules.coarsegrid.partition_ui import partition_ui
from pymrst.modules.coarsegrid.refine_near_well import refine_near_well
from pymrst.utils.units import METER


def field_model(dims=(50, 50, 6), extent=(100.0, 100.0, 1.0)):
    """The example's grid: a Cartesian field with a gentle dome in the top.

    The dome is cosmetic in MRST and kept because it makes the vertical
    coordinates non-trivial, which is a better test of the geometry than a flat
    box would be.
    """
    g = cart_grid(list(dims), [e * METER for e in extent])
    coords = np.asarray(g.nodes.coords, dtype=float)
    x, y = coords[:, 0], coords[:, 1]
    coords[:, 2] = coords[:, 2] - np.exp(-(x - 60.0) ** 2 / 1000.0
                                         - (y - 40.0) ** 2 / 2000.0)
    g.nodes.coords = coords
    return compute_geometry(g)


def well_cells(g, ij) -> np.ndarray:
    """Cells of a vertical well at logical column `ij`, top to bottom."""
    from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices

    i, j, _ = (np.asarray(a, dtype=np.int64) for a in grid_logical_indices(g))
    return np.flatnonzero((i == int(ij[0])) & (j == int(ij[1])))


def refine_at_wells(g, base_blocks=(5, 5, 1), wells=((25, 25), (0, 0)),
                    angle_bins=(6, 1), radius_bins=(5, 5)):
    """Replace each well's coarse block by a radial refinement about the well.

    MRST refines only the cells of the block the well sits in, so the far field
    keeps its original blocks. Returns ``(partition, base, well_cell_ids)``.
    """
    base = partition_ui(g, list(base_blocks))
    p = base.copy()
    centroids = np.asarray(g.cells.centroids, dtype=float)
    heads = []
    for w, (na, nr) in zip(wells, zip(angle_bins, radius_bins)):
        cells = well_cells(g, w)
        if cells.size == 0:
            raise ValueError(f"no cells at logical column {w}")
        head = int(cells[0])
        heads.append(head)
        here = np.flatnonzero(p == p[head])
        local, _, _, _ = refine_near_well(
            centroids[here], centroids[head], angle_bins=na, radius_bins=nr)
        p[here] = int(p.max()) + 1 + local
        p = compress_partition(p)
    return p, base, np.asarray(heads, dtype=np.int64)


def layered_refinement(g, p, layers: int = 3) -> np.ndarray:
    """Cross a partition with a vertical split, as MRST's `p1` does."""
    dims = list(g.cartDims)
    vertical = partition_cart_grid(dims, [1, 1, layers])
    return compress_partition(p + (int(p.max()) + 1) * vertical)


def graded_sectors(g, well_head: int, angle_bins=(2, 4, 8, 16),
                   radius_bins: int = 4) -> np.ndarray:
    """More angular sectors further from the well, one count per ring."""
    centroids = np.asarray(g.cells.centroids, dtype=float)
    p, _, _, _ = refine_near_well(centroids, centroids[well_head],
                                  angle_bins=list(angle_bins),
                                  radius_bins=radius_bins)
    return p


def metis_with_local_patch(g, well_head: int, n_blocks: int = 20,
                           max_radius=(15.0, 10.0)):
    """Partition the field with METIS, then drop a radial patch at the well.

    Raises `ImportError` naming the package when `pymetis` is absent.
    """
    from pymrst.modules.coarsegrid.partition_metis import partition_metis
    from pymrst.params.rock.make_rock import make_rock
    from pymrst.solvers.compute_trans import compute_trans
    from pymrst.utils.units import DARCY, MILLI

    from pymrst.gridprocessing.testgrids.log_norm_layers import log_norm_layers

    perm = log_norm_layers(list(g.cartDims), [300 * MILLI * DARCY,
                                              200 * MILLI * DARCY])
    rock = make_rock(g, perm=np.asarray(perm, dtype=float).ravel(), poro=0.3)
    p, _ = partition_metis(g, compute_trans(g, rock), n_blocks, use_log=True)

    centroids = np.asarray(g.cells.centroids, dtype=float)
    local, _, _, _ = refine_near_well(
        centroids, centroids[well_head], angle_bins=6, radius_bins=4,
        max_radius=np.asarray(max_radius) * METER)
    inside = local >= 0
    p = p.copy()
    p[inside] = local[inside] + int(p.max()) + 1
    return compress_partition(p), inside


__all__ = ["field_model", "graded_sectors", "layered_refinement",
           "metis_with_local_patch", "refine_at_wells", "well_cells"]


if __name__ == "__main__":
    from pymrst.modules.coarsegrid.utils.call_metis_matrix import HAVE_METIS

    _G = field_model(dims=(30, 30, 4))

    def test_the_well_column_runs_top_to_bottom():
        cells = well_cells(_G, (15, 15))
        assert cells.size == _G.cartDims[2], cells.size
        z = np.asarray(_G.cells.centroids, dtype=float)[cells, 2]
        assert np.all(np.diff(z) > 0.0), z
        print(f"ok test_the_well_column_runs_top_to_bottom ({cells.size} cells)")

    def test_refinement_replaces_only_the_wells_own_block():
        """The far field must keep the blocks it had.

        A refinement that leaked outside the well's block would still give a
        valid partition, with the far field quietly resegmented.
        """
        p, base, heads = refine_at_wells(_G, wells=((15, 15), (0, 0)))
        # The complement has to exclude *every* refined block, not one at a
        # time: the second well refines a block that lies outside the first's.
        refined = {int(base[h]) for h in heads}
        untouched = ~np.isin(base, list(refined))
        assert np.any(untouched), "every block was refined; nothing is tested"
        pairs = set(zip(base[untouched].tolist(), p[untouched].tolist()))
        assert len(pairs) == len(set(base[untouched].tolist())), \
            "a block outside the wells was resegmented"
        print(f"ok test_refinement_replaces_only_the_wells_own_block "
              f"({base.max() + 1} base blocks became {p.max() + 1}; "
              f"{int(untouched.sum())} cells untouched)")

    def test_the_refinement_is_finer_near_the_well():
        """The whole purpose: small blocks where the gradient is steep."""
        p, base, heads = refine_at_wells(_G, wells=((15, 15), (0, 0)))
        assert p.max() > base.max(), (base.max(), p.max())
        centroids = np.asarray(_G.cells.centroids, dtype=float)
        head = int(heads[0])
        near = np.linalg.norm(centroids[:, :2] - centroids[head, :2], axis=1) < 10.0
        sizes_near = np.bincount(p[near], minlength=p.max() + 1)
        sizes_all = np.bincount(p, minlength=p.max() + 1)
        occupied_near = sizes_near > 0
        assert sizes_all[occupied_near].mean() < sizes_all.mean(), \
            "blocks near the well are not smaller than average"
        print(f"ok test_the_refinement_is_finer_near_the_well "
              f"({base.max() + 1} blocks became {p.max() + 1})")

    def test_every_cell_keeps_a_block():
        p, _, _ = refine_at_wells(_G, wells=((15, 15), (0, 0)))
        assert p.size == _G.cells.num
        assert np.all(p >= 0)
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        print("ok test_every_cell_keeps_a_block")

    def test_two_wells_get_different_refinements():
        """MRST gives one well 6 sectors and the other 1, and it must show."""
        p, base, heads = refine_at_wells(_G, wells=((15, 15), (0, 0)),
                                         angle_bins=(6, 1), radius_bins=(5, 5))
        counts = []
        for head in heads:
            here = base == base[head]
            counts.append(len(set(p[here].tolist())))
        assert counts[0] > counts[1], counts
        print(f"ok test_two_wells_get_different_refinements "
              f"({counts[0]} blocks against {counts[1]})")

    def test_a_vertical_split_crosses_the_partition():
        p, _, _ = refine_at_wells(_G, wells=((15, 15), (0, 0)))
        q = layered_refinement(_G, p, layers=2)
        assert q.max() > p.max()
        for b in np.unique(q):
            assert len(set(p[q == b].tolist())) == 1, b
        print(f"ok test_a_vertical_split_crosses_the_partition "
              f"({p.max() + 1} blocks became {q.max() + 1})")

    def test_graded_sectors_give_more_blocks_further_out():
        """One sector count per ring, increasing outward.

        Bucketed by the ring index the function returns, not by distance
        quantiles: logarithmic rings are nowhere near equal in population
        (84, 800 and 2716 cells here), so quantile buckets straddle them and
        the counts come out meaningless.
        """
        head = int(well_cells(_G, (15, 15))[0])
        centroids = np.asarray(_G.cells.centroids, dtype=float)
        wanted = (2, 4, 8)
        p, _, rings, _ = refine_near_well(centroids, centroids[head],
                                          angle_bins=list(wanted), radius_bins=3)
        counts = [len(set(p[rings == r].tolist())) for r in range(3)]
        assert counts == list(wanted), (counts, wanted)
        assert np.array_equal(p, graded_sectors(_G, head, angle_bins=wanted,
                                                radius_bins=3))
        print(f"ok test_graded_sectors_give_more_blocks_further_out "
              f"({counts} blocks in rings of {np.bincount(rings).tolist()} cells)")

    def test_the_dome_reaches_the_geometry():
        """Otherwise the case is a flat box and the z coordinates are trivial."""
        flat = compute_geometry(cart_grid([30, 30, 4], [100.0, 100.0, 1.0]))
        z_flat = np.asarray(flat.cells.centroids, dtype=float)[:, 2]
        z_dome = np.asarray(_G.cells.centroids, dtype=float)[:, 2]
        assert not np.allclose(z_flat, z_dome), "the dome had no effect"
        print(f"ok test_the_dome_reaches_the_geometry "
              f"(top surface varies by {z_dome.max() - z_dome.min():.3f} m)")

    def test_a_coarse_grid_builds_on_the_refined_partition():
        from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
        p, _, _ = refine_at_wells(_G, wells=((15, 15), (0, 0)))
        cg = coarsen_geometry(generate_coarse_grid(_G, p))
        assert cg.n_blocks == p.max() + 1
        assert np.isclose(cg.cell_volumes.sum(),
                          float(np.sum(_G.cells.volumes)), rtol=1e-12)
        print(f"ok test_a_coarse_grid_builds_on_the_refined_partition "
              f"({cg.n_blocks} blocks, {cg.num_faces} coarse faces)")

    def test_a_missing_well_column_is_rejected():
        try:
            refine_at_wells(_G, wells=((999, 999),))
        except ValueError as exc:
            assert "no cells at" in str(exc), exc
        else:
            raise AssertionError("expected a well outside the grid to fail")
        print("ok test_a_missing_well_column_is_rejected")

    def test_the_metis_section_when_the_package_is_present():
        """Skipped without pymetis, and loudly."""
        if not HAVE_METIS:
            print("SKIP test_the_metis_section_when_the_package_is_present "
                  "(pymetis is not installed; this section is unexercised)")
            return
        head = int(well_cells(_G, (15, 15))[0])
        p, inside = metis_with_local_patch(_G, head, n_blocks=10)
        assert p.size == _G.cells.num
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        assert np.any(inside), "the local patch covered nothing"
        assert len(set(p[inside].tolist())) > 1
        print(f"ok test_the_metis_section_when_the_package_is_present "
              f"({p.max() + 1} blocks, {int(inside.sum())} cells refined)")

    test_the_well_column_runs_top_to_bottom()
    test_refinement_replaces_only_the_wells_own_block()
    test_the_refinement_is_finer_near_the_well()
    test_every_cell_keeps_a_block()
    test_two_wells_get_different_refinements()
    test_a_vertical_split_crosses_the_partition()
    test_graded_sectors_give_more_blocks_further_out()
    test_the_dome_reaches_the_geometry()
    test_a_coarse_grid_builds_on_the_refined_partition()
    test_a_missing_well_column_is_rejected()
    test_the_metis_section_when_the_package_is_present()
    print("\nAll tests passed!")
