"""Coarsening the Norne Simulation Model (``coarsegridExampleNorne.m``).

Norne is a real field model, and it is where the three partition steps stop
being formalities. A logical 6x12x3 partition of it produces:

* **empty blocks**, because the field fills only part of its logical box, which
  `compress_partition` removes;
* **disconnected blocks**, because faults and pinch-outs cut logical boxes in
  two, which `process_partition` splits.

MRST asserts `all(accumarray(p, 1) > 0)` after both steps, that is, that every
remaining block holds at least one cell. That assertion is the point of the
section and it is kept here.

The rest of the script plots individual blocks with their neighbours. Those
are lookups rather than pictures, so `block_and_neighbours` returns the block,
its neighbouring blocks and the coarse faces between them.

**Needs the Norne dataset.** It is installed here. If it is not, every function
returns ``None`` and the self-tests report the absence rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_ui import partition_ui
from pymrst.modules.coarsegrid.process_partition import process_partition


def find_norne_grdecl(path=None):
    """Locate ``NORNE.GRDECL``, or return ``None``. Never downloads."""
    if path is not None:
        path = Path(path)
        return path if path.exists() else None

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = get_dataset_path("norne", download=False)
    except Exception:
        return None
    if root is None:
        return None
    for candidate in Path(root).rglob("NORNE.GRDECL"):
        return candidate
    return None


def fetch_norne():
    """Download Norne through the dataset manager.

    Kept out of the loaders and never called by them, so importing or testing
    this module cannot start a network transfer.
    """
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    return get_dataset_path("norne", download=True)


def load_norne(path=None):
    """Read Norne and return the first (largest) connected grid, or ``None``.

    `process_grdecl` can return several disconnected grids for a faulted model.
    MRST takes `G(1)`, and so does this.
    """
    grdecl_file = find_norne_grdecl(path)
    if grdecl_file is None:
        return None
    # `read_grdecl` does not follow INCLUDE, and these decks split their GRID
    # section across included files, so the operators arrive without the
    # SPECGRID they need. `read_grid` splices first and parses once.
    from pymrst.modules.deckformat.deckinput.read_grid import read_grid

    grids = process_grdecl(read_grid(grdecl_file))
    first = grids[0] if isinstance(grids, (list, tuple)) else grids
    return compute_geometry(first)


@dataclass
class PartitionStages:
    """The partition after each of MRST's three steps."""

    raw: np.ndarray
    compressed: np.ndarray
    processed: np.ndarray

    def counts(self, which: str) -> np.ndarray:
        p = getattr(self, which)
        return np.bincount(p, minlength=int(p.max()) + 1)


def partition_stages(g, blocks=(6, 12, 3)) -> PartitionStages:
    """Partition logically, compress, then split the disconnected blocks."""
    raw = partition_ui(g, list(blocks))
    compressed = compress_partition(raw)
    return PartitionStages(raw=raw, compressed=compressed,
                           processed=process_partition(g, compressed))


@dataclass
class BlockDetail:
    """A block, the blocks around it, and the coarse faces between."""

    block: int
    cells: np.ndarray
    neighbours: np.ndarray
    faces: np.ndarray


def block_and_neighbours(cg, block: int) -> BlockDetail:
    """What `plotBlockAndNeighbors` draws, as data."""
    if not 0 <= block < cg.n_blocks:
        raise ValueError(f"block must lie in 0 to {cg.n_blocks - 1}")
    neighbors = np.asarray(cg.neighbors, dtype=np.int64)
    faces = np.flatnonzero(np.any(neighbors == block, axis=1))
    others = neighbors[faces].ravel()
    others = np.unique(others[(others >= 0) & (others != block)])
    return BlockDetail(
        block=block,
        cells=np.flatnonzero(np.asarray(cg.partition, dtype=np.int64) == block),
        neighbours=others,
        faces=faces,
    )


__all__ = ["BlockDetail", "PartitionStages", "block_and_neighbours",
           "fetch_norne", "find_norne_grdecl", "load_norne", "partition_stages"]


if __name__ == "__main__":
    def _skip(name, extra=""):
        print(f"ok {name} (skipped){extra}")

    _MODEL: dict[str, Any] = {}

    def _norne():
        """Load once; Norne is large and every test would otherwise reread it."""
        if "g" not in _MODEL:
            _MODEL["g"] = load_norne()
        return _MODEL["g"]

    def test_absence_is_reported_not_raised():
        g = _norne()
        if g is None:
            print("ok test_absence_is_reported_not_raised (absent)")
            print("   (call fetch_norne() to enable the checks below)")
        else:
            print(f"ok test_absence_is_reported_not_raised "
                  f"(present, {g.cells.num} cells)")

    def test_loading_never_downloads():
        """A test sweep must not pull data or block on a prompt."""
        import inspect
        source = inspect.getsource(load_norne) + inspect.getsource(find_norne_grdecl)
        assert "download=True" not in source, "a loader requests a download"
        assert "fetch_norne" not in source, "a loader calls the fetcher"
        print("ok test_loading_never_downloads")

    def test_the_logical_partition_leaves_empty_blocks():
        """The field fills part of its logical box, so most boxes are empty.

        This is what `compress_partition` is for, and on a real model the gap
        is large rather than incidental.
        """
        g = _norne()
        if g is None:
            return _skip("test_the_logical_partition_leaves_empty_blocks")
        s = partition_stages(g)
        slots = int(s.raw.max()) + 1
        used = np.unique(s.raw).size
        assert used < slots, "no empty blocks; the case proves nothing"
        assert int(s.compressed.max()) + 1 == used
        print(f"ok test_the_logical_partition_leaves_empty_blocks "
              f"({slots} slots for {used} non-empty blocks)")

    def test_faults_disconnect_blocks():
        g = _norne()
        if g is None:
            return _skip("test_faults_disconnect_blocks")
        s = partition_stages(g)
        assert s.processed.max() > s.compressed.max(), (
            s.compressed.max(), s.processed.max())
        print(f"ok test_faults_disconnect_blocks "
              f"({s.compressed.max() + 1} blocks became {s.processed.max() + 1})")

    def test_every_block_holds_at_least_one_cell():
        """MRST's own assertion, kept verbatim in intent."""
        g = _norne()
        if g is None:
            return _skip("test_every_block_holds_at_least_one_cell")
        s = partition_stages(g)
        assert np.all(s.counts("processed") > 0)
        assert np.all(s.counts("compressed") > 0)
        print(f"ok test_every_block_holds_at_least_one_cell "
              f"(smallest block has {s.counts('processed').min()} cells)")

    def test_the_partition_is_contiguous_after_processing():
        g = _norne()
        if g is None:
            return _skip("test_the_partition_is_contiguous_after_processing")
        p = partition_stages(g).processed
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        print("ok test_the_partition_is_contiguous_after_processing")

    def test_every_processed_block_is_connected():
        """The postcondition, checked by re-running the splitter."""
        g = _norne()
        if g is None:
            return _skip("test_every_processed_block_is_connected")
        p = partition_stages(g).processed
        assert np.array_equal(process_partition(g, p), p)
        print("ok test_every_processed_block_is_connected")

    def test_the_coarse_grid_conserves_volume():
        from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
        g = _norne()
        if g is None:
            return _skip("test_the_coarse_grid_conserves_volume")
        cg = coarsen_geometry(generate_coarse_grid(g, partition_stages(g).processed))
        fine = float(np.sum(g.cells.volumes))
        assert np.isclose(cg.cell_volumes.sum(), fine, rtol=1e-10), (
            cg.cell_volumes.sum(), fine)
        print(f"ok test_the_coarse_grid_conserves_volume "
              f"({cg.n_blocks} blocks, {fine:.4e} m3)")

    def test_a_block_and_its_neighbours_are_consistent():
        """Each returned coarse face must actually touch the block.

        A neighbour list built from the wrong column of `neighbors` gives a
        plausible set of blocks that are not adjacent at all.
        """
        g = _norne()
        if g is None:
            return _skip("test_a_block_and_its_neighbours_are_consistent")
        cg = generate_coarse_grid(g, partition_stages(g).processed)
        for block in (0, cg.n_blocks // 2, cg.n_blocks - 1):
            d = block_and_neighbours(cg, block)
            assert d.cells.size > 0, block
            for f in d.faces:
                assert block in cg.neighbors[f].tolist(), (block, f)
            for other in d.neighbours:
                shared = [f for f in d.faces
                          if other in cg.neighbors[f].tolist()]
                assert shared, (block, other)
        print(f"ok test_a_block_and_its_neighbours_are_consistent "
              f"({cg.n_blocks} blocks)")

    def test_an_out_of_range_block_is_rejected():
        g = _norne()
        if g is None:
            return _skip("test_an_out_of_range_block_is_rejected")
        cg = generate_coarse_grid(g, partition_stages(g).processed)
        try:
            block_and_neighbours(cg, cg.n_blocks)
        except ValueError as exc:
            assert "must lie in" in str(exc), exc
        else:
            raise AssertionError("expected an out-of-range block to fail")
        print("ok test_an_out_of_range_block_is_rejected")

    test_absence_is_reported_not_raised()
    test_loading_never_downloads()
    test_the_logical_partition_leaves_empty_blocks()
    test_faults_disconnect_blocks()
    test_every_block_holds_at_least_one_cell()
    test_the_partition_is_contiguous_after_processing()
    test_every_processed_block_is_connected()
    test_the_coarse_grid_conserves_volume()
    test_a_block_and_its_neighbours_are_consistent()
    test_an_out_of_range_block_is_rejected()
    print("\nAll tests passed!")
