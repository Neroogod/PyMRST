"""Coarsening the SAIGUP Model (``coarsegridExampleSAIGUP.m``).

A logical partition of a faulted model produces blocks of wildly uneven size.
Splitting the disconnected ones makes that worse: every sliver cut off by a
fault becomes a block of its own, and a coarse block holding three cells beside
one holding three thousand is a badly conditioned model.

The example's answer is to **merge the small blocks away**, and its substance is
that loop rather than the pictures. Two variants are given, and the difference
between them is the interesting part:

* `merge_small_blocks(..., strategy=...)` follows MRST's first loop, rebuilding
  the coarse grid after every merge. Simple and slow: each pass is `O(grid)`.
* `merge_small_blocks_incremental` follows its second loop, which never rebuilds.
  It keeps a `new_block` redirection table and adds the merged volumes directly,
  so the cost is `O(blocks)` per merge.

**They are not equivalent, even given the same neighbour rule**, and it took a
failing test to see why. The fast loop reads adjacency once, from the coarse
grid it starts with, and maps it forward through the redirection table. But
merging changes adjacency: a block that absorbs another inherits its
neighbours. The rebuilding loop sees that, the fast one does not, so after a
few merges the two can pick different targets. On the fixture below they agree
for one rule and differ by a single block for the other.

That is a property of MRST's two loops, not a defect in either, and it is the
sort of thing that is invisible until the two are run side by side. What *is*
asserted here is sharper than agreement: the fast loop's running volumes must
equal the volumes of a coarse grid rebuilt from its own answer. A wrong
redirection table or a mis-added volume fails that exactly.

**The merging works on any coarse grid**, so it is tested on a synthetic model
that runs in a second. Building that fixture took a correction worth recording:
a *faulted* grid of uniform cells does not reproduce the problem at all. What
produces tiny blocks is cell volumes varying by orders of magnitude, which is
what graded stratigraphy gives and what SAIGUP has. See `_graded` below.

SAIGUP itself is a 2.5 MB download that is not installed here; the functions
that need it return ``None`` and their tests report the absence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_ui import partition_ui
from pymrst.modules.coarsegrid.process_partition import process_partition
from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry


def find_saigup_grdecl(path=None):
    """Locate ``SAIGUP.GRDECL``, or return ``None``. Never downloads."""
    if path is not None:
        path = Path(path)
        return path if path.exists() else None
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = get_dataset_path("SAIGUP", download=False)
    except Exception:
        return None
    if root is None:
        return None
    for candidate in Path(root).rglob("SAIGUP.GRDECL"):
        return candidate
    return None


def fetch_saigup():
    """Download SAIGUP. Never called by the loaders."""
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    return get_dataset_path("SAIGUP", download=True)


def load_saigup(path=None):
    """Read SAIGUP and return the grid, or ``None`` when it is not installed."""
    grdecl_file = find_saigup_grdecl(path)
    if grdecl_file is None:
        return None
    # `read_grdecl` does not follow INCLUDE, and these decks split their GRID
    # section across included files, so the operators arrive without the
    # SPECGRID they need. `read_grid` splices first and parses once.
    from pymrst.modules.deckformat.deckinput.read_grid import read_grid

    grids = process_grdecl(read_grid(grdecl_file))
    first = grids[0] if isinstance(grids, (list, tuple)) else grids
    return compute_geometry(first)


def logical_partition(g, blocks=(6, 12, 3), min_face_area=None) -> np.ndarray:
    """Partition logically, compress, then split the disconnected blocks.

    `min_face_area` is MRST's `G.faces.areas < 250` variant: faces smaller than
    this do not conduct, so a block joined only by a sliver of contact is split
    rather than kept.
    """
    p = compress_partition(partition_ui(g, list(blocks)))
    blocked = None
    if min_face_area is not None:
        blocked = np.asarray(g.faces.areas, dtype=float) < float(min_face_area)
    return process_partition(g, p, face_list=blocked)


def _neighbour_blocks(cg, block: int) -> np.ndarray:
    neighbors = np.asarray(cg.neighbors, dtype=np.int64)
    touching = neighbors[np.any(neighbors == block, axis=1)].ravel()
    return np.unique(touching[(touching >= 0) & (touching != block)])


def merge_small_blocks(g, partition, threshold: float = 0.1,
                       strategy: str = "largest", max_merges: int = 10_000):
    """Merge away every block below `threshold` times the mean block volume.

    MRST's first loop: rebuild the coarse grid after each merge and take the
    smallest block again. Correct and slow.

    `strategy` picks the neighbour to merge into. MRST offers three and leaves
    two commented out; all three are here because the choice changes the result
    and none is obviously right. Returns ``(partition, merges)``.
    """
    if strategy not in ("largest", "smallest", "median"):
        raise ValueError(f"unknown strategy {strategy!r}")
    p = compress_partition(np.asarray(partition, dtype=np.int64).ravel())
    cg = coarsen_geometry(generate_coarse_grid(g, p))
    volumes = np.asarray(cg.cell_volumes, dtype=float)
    if volumes.size < 2:
        return p, 0
    # The threshold is fixed against the *starting* mean, as MRST fixes it.
    # Recomputing it each pass would make it a moving target: every merge
    # raises the mean, so the loop would keep finding new blocks below it and
    # run far past what was asked for.
    limit = threshold * volumes.mean()

    merges = 0
    while merges < max_merges:
        if merges:
            cg = coarsen_geometry(generate_coarse_grid(g, p))
            volumes = np.asarray(cg.cell_volumes, dtype=float)
        if volumes.size < 2:
            break
        block = int(np.argmin(volumes))
        if volumes[block] >= limit:
            break
        nlist = _neighbour_blocks(cg, block)
        if nlist.size == 0:
            break
        target = int(nlist[_pick(volumes[nlist], strategy)])
        p[p == block] = target
        p = compress_partition(p)
        merges += 1
    return p, merges


def merge_small_blocks_incremental(g, partition, threshold: float = 0.1,
                                   strategy: str = "smallest",
                                   max_merges: int = 10_000):
    """The same, without rebuilding the coarse grid (MRST's second loop).

    The bookkeeping is what makes this work. `new_block` redirects a merged
    block to whatever absorbed it, so a neighbour list gathered from the
    original coarse grid still resolves to live blocks; the absorbed block's
    volume becomes infinite so it is never chosen again, and the absorber's
    grows by what it took.

    Returns ``(partition, merges, volumes)``. The volumes are the loop's own
    running figures, kept so a caller can check them against a rebuilt coarse
    grid; that is the test that the bookkeeping is right.

    **Adjacency is read once and never updated**, which is what makes this fast
    and what makes it differ from `merge_small_blocks`: a block that absorbs
    another really does gain its neighbours, and this does not see them.
    """
    if strategy not in ("largest", "smallest", "median"):
        raise ValueError(f"unknown strategy {strategy!r}")
    p = compress_partition(np.asarray(partition, dtype=np.int64).ravel())
    cg = coarsen_geometry(generate_coarse_grid(g, p))
    volumes = np.asarray(cg.cell_volumes, dtype=float).copy()
    if volumes.size < 2:
        return p, 0, volumes

    limit = threshold * volumes.mean()
    new_block = np.arange(volumes.size, dtype=np.int64)
    merges = 0
    while merges < max_merges:
        block = int(np.argmin(volumes))
        if volumes[block] >= limit:
            break
        nlist = np.unique(new_block[_neighbour_blocks(cg, block)])
        nlist = nlist[nlist != new_block[block]]
        if nlist.size == 0:
            break
        target = int(nlist[_pick(volumes[nlist], strategy)])
        p[p == block] = target
        # Redirect anything that pointed at the absorbed block, so a later
        # neighbour lookup never returns a block that no longer exists.
        new_block[new_block == block] = target
        volumes[target] += volumes[block]
        volumes[block] = np.inf
        merges += 1
    return compress_partition(p), merges, volumes


def _pick(volumes: np.ndarray, strategy: str) -> int:
    if strategy == "largest":
        return int(np.argmax(volumes))
    if strategy == "smallest":
        return int(np.argmin(volumes))
    order = np.argsort(volumes)[::-1]
    return int(order[len(order) // 2])


def block_volume_spread(g, partition, reference_mean=None, threshold=0.1):
    """Smallest, mean and largest block volume, and how many are tiny.

    `reference_mean` is the mean "tiny" is measured against. It matters after
    merging: the loops fix their threshold against the mean they *started*
    with, as MRST does, while merging raises the mean. Measured against the
    final mean, a partition the loop considers finished still shows small
    blocks, and the postcondition looks violated when it is not.
    """
    cg = coarsen_geometry(generate_coarse_grid(g, partition))
    v = np.asarray(cg.cell_volumes, dtype=float)
    mean = float(v.mean()) if reference_mean is None else float(reference_mean)
    return {"min": float(v.min()), "mean": float(v.mean()),
            "max": float(v.max()), "blocks": int(v.size),
            "tiny": int(np.count_nonzero(v < threshold * mean))}


__all__ = ["block_volume_spread", "fetch_saigup", "find_saigup_grdecl",
           "load_saigup", "logical_partition", "merge_small_blocks",
           "merge_small_blocks_incremental"]


if __name__ == "__main__":
    from pymrst.gridprocessing import tensor_grid
    from pymrst.gridprocessing.remove_cells import remove_cells

    def _graded(blocks=(3, 3, 5)):
        """A model that reproduces SAIGUP's sliver problem in a second.

        A faulted grid of uniform cells does **not** reproduce it, which was
        worth finding out: `simple_grdecl` at several throws and partitions
        gave a volume spread under 4x and no block below a tenth of the mean,
        so the merging loop had nothing to do and the tests were vacuous.

        SAIGUP's tiny blocks come from cell *volumes* varying by orders of
        magnitude, not from the fault as such. Real stratigraphy is thin at the
        top and thick at the base, so this grades the layers geometrically and
        notches one corner. That gives an 80x spread and 8 blocks of 42 below
        the threshold, which is the condition the loop exists for.
        """
        z = np.concatenate([[0.0], np.cumsum(np.geomspace(0.02, 2.0, 10))])
        lateral = np.linspace(0.0, 10.0, 13)
        g = compute_geometry(tensor_grid(lateral, lateral, z))
        c = np.asarray(g.cells.centroids, dtype=float)
        notch = np.flatnonzero((c[:, 0] > 6.0) & (c[:, 1] > 6.0) & (c[:, 2] < 1.0))
        g = compute_geometry(remove_cells(g, notch))
        return g, logical_partition(g, blocks)

    _G, _P = _graded()

    def test_splitting_creates_tiny_blocks():
        """The problem the section solves, measured rather than asserted.

        A partition with one block a thousandth the size of another is not
        obviously wrong; it just conditions badly.
        """
        spread = block_volume_spread(_G, _P)
        assert spread["tiny"] > 0, "the case has no small blocks to merge"
        assert spread["max"] / spread["min"] > 5.0, spread
        print(f"ok test_splitting_creates_tiny_blocks "
              f"({spread['tiny']} of {spread['blocks']} blocks below a tenth "
              f"of the mean; spread {spread['max'] / spread['min']:.0f}x)")

    _START_MEAN = block_volume_spread(_G, _P)["mean"]

    def test_merging_removes_every_tiny_block():
        """The postcondition, measured against the mean the loop started from."""
        p, merges = merge_small_blocks(_G, _P)
        assert merges > 0
        spread = block_volume_spread(_G, p, reference_mean=_START_MEAN)
        assert spread["tiny"] == 0, spread
        print(f"ok test_merging_removes_every_tiny_block "
              f"({merges} merges, {spread['blocks']} blocks left)")

    def test_merging_conserves_volume():
        """Merging moves cells between blocks; it must not lose any."""
        p, _ = merge_small_blocks(_G, _P)
        fine = float(np.sum(_G.cells.volumes))
        cg = coarsen_geometry(generate_coarse_grid(_G, p))
        assert np.isclose(cg.cell_volumes.sum(), fine, rtol=1e-12)
        assert p.size == _G.cells.num
        print(f"ok test_merging_conserves_volume ({fine:.4f} m3)")

    def test_merging_only_ever_joins():
        """No cell may end up in a block with cells it was not with before.

        A merge that split as well as joined would still produce a valid
        partition with a reasonable volume distribution.
        """
        p, _ = merge_small_blocks(_G, _P)
        for a in range(p.size):
            for b in range(a + 1, p.size):
                if _P[a] == _P[b]:
                    assert p[a] == p[b], (a, b)
        print("ok test_merging_only_ever_joins")

    def test_the_incremental_volumes_match_a_rebuilt_grid():
        """The sharp test of the fast loop's bookkeeping.

        Its running volumes must equal those of a coarse grid built from its
        own answer. A wrong redirection or a mis-added volume fails this
        exactly, where comparing partitions would not: the two loops legitimately
        disagree because one sees adjacency change and the other does not.
        """
        for strategy in ("largest", "smallest"):
            p, merges, running = merge_small_blocks_incremental(
                _G, _P, strategy=strategy)
            live = np.sort(running[np.isfinite(running)])
            rebuilt = np.sort(np.asarray(
                coarsen_geometry(generate_coarse_grid(_G, p)).cell_volumes,
                dtype=float))
            assert live.size == rebuilt.size, (strategy, live.size, rebuilt.size)
            assert np.allclose(live, rebuilt, rtol=1e-12, atol=0.0), strategy
        print(f"ok test_the_incremental_volumes_match_a_rebuilt_grid "
              f"({merges} merges, two strategies)")

    def test_the_two_loops_reach_the_same_postcondition():
        """Both must clear every tiny block and conserve volume.

        They need not agree on the partition. The fast loop reads adjacency
        once, so after a few merges it can pick a neighbour the rebuilding loop
        would not; on this fixture they agree under one rule and differ by a
        single block under the other. That is inherent to MRST's two loops.
        """
        from pymrst.modules.coarsegrid.utils.is_same_partition import same_grouping
        fine = float(np.sum(_G.cells.volumes))
        agreed = []
        for strategy in ("largest", "smallest"):
            slow, n_slow = merge_small_blocks(_G, _P, strategy=strategy)
            fast, n_fast, _ = merge_small_blocks_incremental(_G, _P,
                                                             strategy=strategy)
            for p in (slow, fast):
                spread = block_volume_spread(_G, p, reference_mean=_START_MEAN)
                assert spread["tiny"] == 0, (strategy, spread)
                cg = coarsen_geometry(generate_coarse_grid(_G, p))
                assert np.isclose(cg.cell_volumes.sum(), fine, rtol=1e-12)
            assert n_slow == n_fast, (strategy, n_slow, n_fast)
            agreed.append(same_grouping(slow, fast))
        assert any(agreed), "the two loops agreed on nothing at all"
        print(f"ok test_the_two_loops_reach_the_same_postcondition "
              f"(agree on {sum(agreed)} of {len(agreed)} rules)")

    def test_the_strategy_changes_the_result():
        """MRST offers three and comments two out; the choice is not neutral."""
        from pymrst.modules.coarsegrid.utils.is_same_partition import same_grouping
        largest, _ = merge_small_blocks(_G, _P, strategy="largest")
        smallest, _ = merge_small_blocks(_G, _P, strategy="smallest")
        assert not same_grouping(largest, smallest), \
            "the case cannot distinguish the strategies"
        print(f"ok test_the_strategy_changes_the_result "
              f"({largest.max() + 1} blocks against {smallest.max() + 1})")

    def test_the_incremental_loop_never_targets_a_dead_block():
        """The trap the redirection table exists for.

        Neighbour lists come from the coarse grid built once at the start, so
        without the redirection they name blocks that have since been absorbed.
        Merging into one would silently resurrect it.
        """
        p, _, _ = merge_small_blocks_incremental(_G, _P)
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        cg = coarsen_geometry(generate_coarse_grid(_G, p))
        v = np.asarray(cg.cell_volumes, dtype=float)
        assert np.all(v > 0.0)
        assert int(np.count_nonzero(v < 0.1 * _START_MEAN)) == 0
        print(f"ok test_the_incremental_loop_never_targets_a_dead_block "
              f"({cg.n_blocks} blocks, all above the threshold)")

    def test_blocking_small_faces_splits_more():
        """MRST's `G.faces.areas < 250` variant.

        A block joined to its neighbour only by a sliver of fault contact is
        not really connected. Refusing those faces splits it.
        """
        areas = np.asarray(_G.faces.areas, dtype=float)
        cutoff = float(np.quantile(areas, 0.3))
        strict = logical_partition(_G, (3, 3, 5), min_face_area=cutoff)
        assert strict.max() >= _P.max(), (strict.max(), _P.max())
        print(f"ok test_blocking_small_faces_splits_more "
              f"({_P.max() + 1} blocks became {strict.max() + 1} at a "
              f"{cutoff:.3f} m2 cutoff)")

    def test_an_unknown_strategy_is_rejected():
        for fn in (merge_small_blocks, merge_small_blocks_incremental):
            try:
                fn(_G, _P, strategy="whatever")
            except ValueError as exc:
                assert "unknown strategy" in str(exc), exc
            else:
                raise AssertionError(f"expected {fn.__name__} to reject it")
        print("ok test_an_unknown_strategy_is_rejected")

    def test_saigup_absence_is_reported_not_raised():
        g = load_saigup()
        if g is None:
            print("ok test_saigup_absence_is_reported_not_raised (absent)")
            print("   (SAIGUP is a 2.5 MB download; call fetch_saigup() to "
                  "run the merging above on the real model)")
        else:
            assert g.cells.num > 0
            print(f"ok test_saigup_absence_is_reported_not_raised "
                  f"(present, {g.cells.num} cells)")

    def test_loading_never_downloads():
        import inspect
        source = inspect.getsource(load_saigup) + inspect.getsource(find_saigup_grdecl)
        assert "download=True" not in source, "a loader requests a download"
        assert "fetch_saigup" not in source, "a loader calls the fetcher"
        print("ok test_loading_never_downloads")

    test_splitting_creates_tiny_blocks()
    test_merging_removes_every_tiny_block()
    test_merging_conserves_volume()
    test_merging_only_ever_joins()
    test_the_incremental_volumes_match_a_rebuilt_grid()
    test_the_two_loops_reach_the_same_postcondition()
    test_the_strategy_changes_the_result()
    test_the_incremental_loop_never_targets_a_dead_block()
    test_blocking_small_faces_splits_more()
    test_an_unknown_strategy_is_rejected()
    test_saigup_absence_is_reported_not_raised()
    test_loading_never_downloads()
    print("\nAll tests passed!")
