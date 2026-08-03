"""Split disconnected coarse blocks (MRST ``processPartition.m``).

A partition can assign cells to a block without those cells touching. That
happens whenever a block is cut by inactive cells, by a fault, or simply by a
partition built from logical indices on a grid whose geometry does not follow
them. Such a block is not a coarse cell in any useful sense: flow cannot pass
between its halves, yet a coarse model would treat it as one control volume.

`process_partition` finds the connected components of each block and gives each
extra component a new block number. Blocks that are already connected keep their
number, so `q == p` whenever nothing needed splitting, which MRST states as a
guarantee and this port preserves.

Connectivity runs across the fine grid's internal faces. `face_list` marks faces
that should not conduct, which is how a fault is made to split the blocks it cuts.

MRST finds the components with `dmperm`, a Dulmage-Mendelsohn decomposition, and
loops over blocks building a sparse adjacency per block. The port builds one
graph over the whole grid and calls `scipy.sparse.csgraph.connected_components`,
which is the same computation done once instead of per block.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def process_partition(g, partition, face_list=None) -> np.ndarray:
    """Split any block whose cells are not connected.

    Returns a new 0-based partition. Blocks that were already connected keep
    their number; each additional component of a split block is appended after
    the highest block in use.
    """
    partition = np.asarray(partition, dtype=np.int64).ravel()
    n_cells = int(g.cells.num)
    if partition.size != n_cells:
        raise ValueError(
            f"partition has {partition.size} entries for {n_cells} cells")

    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    # An exterior face has -1 on one side in PyMRST, 0 in MRST. Only faces with
    # a cell on both sides can connect anything.
    interior = np.all(neighbors >= 0, axis=1)

    # Connections count only inside a block: a face between two blocks says
    # nothing about whether either block is internally connected.
    same_block = np.zeros(neighbors.shape[0], dtype=bool)
    same_block[interior] = (partition[neighbors[interior, 0]]
                            == partition[neighbors[interior, 1]])

    if face_list is not None:
        blocked = _face_mask(face_list, neighbors.shape[0])
        same_block &= ~blocked

    rows = neighbors[same_block, 0]
    cols = neighbors[same_block, 1]
    adjacency = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(n_cells, n_cells))
    _, component = connected_components(adjacency, directed=False)

    return _relabel(partition, component)


def _relabel(partition: np.ndarray, component: np.ndarray) -> np.ndarray:
    """Give each (block, component) pair a block number.

    The component holding a block's lowest-numbered cell keeps the block's
    original number, which is what makes `q == p` hold for an already-connected
    partition. Every other component is appended after the highest block in use.
    """
    out = np.empty_like(partition)
    next_block = int(partition.max()) + 1 if partition.size else 0

    order = np.lexsort((component, partition))
    boundaries = np.flatnonzero(
        np.diff(partition[order]) | np.diff(component[order])) + 1
    groups = np.split(order, boundaries)

    seen: set[int] = set()
    for cells in groups:
        block = int(partition[cells[0]])
        if block in seen:
            out[cells] = next_block
            next_block += 1
        else:
            seen.add(block)
            out[cells] = block
    return out


def _face_mask(face_list, n_faces: int) -> np.ndarray:
    """Accept either a boolean mask over faces or a list of face indices."""
    arr = np.asarray(face_list)
    if arr.dtype == bool:
        if arr.size != n_faces:
            raise ValueError(
                f"a boolean face_list needs one entry per face ({n_faces}), "
                f"got {arr.size}")
        return arr
    arr = arr.astype(np.int64).ravel()
    if arr.size and (arr.min() < 0 or arr.max() >= n_faces):
        raise ValueError(
            f"face numbers must lie in 0 to {n_faces - 1}; got "
            f"{arr.min()} to {arr.max()}")
    mask = np.zeros(n_faces, dtype=bool)
    mask[arr] = True
    return mask


__all__ = ["process_partition"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.gridprocessing.remove_cells import remove_cells
    from pymrst.modules.coarsegrid.compress_partition import compress_partition
    from pymrst.modules.coarsegrid.partition_ui import partition_ui

    def _grid(dims=(6, 4, 1)):
        return compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))

    def test_connected_partition_is_returned_unchanged():
        """MRST states this as a guarantee: `all(q == p)` when nothing splits.

        A version that renumbers regardless would still return a valid
        partition, and every downstream result would shift for no reason.
        """
        g = _grid((6, 4, 2))
        p = partition_ui(g, [3, 2, 1])
        q = process_partition(g, p)
        assert np.array_equal(q, p), "a connected partition was renumbered"
        print("ok test_connected_partition_is_returned_unchanged")

    def test_a_split_block_gains_one_number():
        """Two opposite corners share a block but do not touch.

        The block must become two, and the rest of the partition must be
        untouched.
        """
        g = _grid((4, 1, 1))
        p = np.array([0, 1, 1, 0])           # cells 0 and 3 are not adjacent
        q = process_partition(g, p)
        assert q[1] == q[2], "the connected half was split"
        assert q[0] != q[3], "the disconnected halves kept one block"
        assert len(set(q.tolist())) == 3, q.tolist()
        print(f"ok test_a_split_block_gains_one_number ({q.tolist()})")

    def test_the_first_component_keeps_the_block_number():
        g = _grid((4, 1, 1))
        p = np.array([0, 1, 1, 0])
        q = process_partition(g, p)
        assert q[0] == 0, "the lowest-numbered cell's component changed number"
        assert q[3] == 2, q.tolist()
        print("ok test_the_first_component_keeps_the_block_number")

    def test_a_hole_splits_the_block_around_it():
        """Removing the middle of a strip disconnects its two ends.

        This is the case the function exists for: a partition built on logical
        indices is perfectly reasonable until inactive cells cut a block in two.
        """
        g = compute_geometry(remove_cells(cart_grid([5, 1, 1], [5.0, 1.0, 1.0]),
                                          np.array([2])))
        p = np.zeros(g.cells.num, dtype=np.int64)
        q = process_partition(g, p)
        assert len(set(q.tolist())) == 2, q.tolist()
        assert q[0] == q[1] and q[2] == q[3] and q[0] != q[2], q.tolist()
        print("ok test_a_hole_splits_the_block_around_it")

    def test_face_list_blocks_connections():
        """A fault is expressed by refusing to conduct across its faces."""
        g = _grid((4, 1, 1))
        p = np.zeros(4, dtype=np.int64)
        assert len(set(process_partition(g, p).tolist())) == 1
        # The internal face between cells 1 and 2.
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        cut = np.flatnonzero((nb[:, 0] == 1) & (nb[:, 1] == 2))
        assert cut.size == 1, cut
        q = process_partition(g, p, face_list=cut)
        assert q[0] == q[1] and q[2] == q[3] and q[0] != q[2], q.tolist()
        print("ok test_face_list_blocks_connections")

    def test_face_list_accepts_a_boolean_mask():
        g = _grid((4, 1, 1))
        p = np.zeros(4, dtype=np.int64)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        mask = (nb[:, 0] == 1) & (nb[:, 1] == 2)
        q = process_partition(g, p, face_list=mask)
        assert len(set(q.tolist())) == 2, q.tolist()
        print("ok test_face_list_accepts_a_boolean_mask")

    def test_every_output_block_is_connected():
        """The defining postcondition, checked by re-running the function.

        A second pass must find nothing left to split.
        """
        g = compute_geometry(remove_cells(cart_grid([6, 4, 1], [6.0, 4.0, 1.0]),
                                          np.array([7, 8, 13, 14])))
        p = partition_ui(g, [2, 2, 1])
        q = process_partition(g, p)
        assert np.array_equal(process_partition(g, q), q), "still disconnected"
        print("ok test_every_output_block_is_connected")

    def test_membership_only_ever_splits():
        """Cells in different blocks never end up together.

        Splitting is the only permitted change; merging would silently coarsen
        the model.
        """
        g = compute_geometry(remove_cells(cart_grid([6, 4, 1], [6.0, 4.0, 1.0]),
                                          np.array([7, 8, 13, 14])))
        p = partition_ui(g, [2, 2, 1])
        q = process_partition(g, p)
        for a in range(q.size):
            for b in range(a + 1, q.size):
                if q[a] == q[b]:
                    assert p[a] == p[b], (a, b)
        print("ok test_membership_only_ever_splits")

    def test_result_is_contiguous():
        g = compute_geometry(remove_cells(cart_grid([5, 1, 1], [5.0, 1.0, 1.0]),
                                          np.array([2])))
        q = process_partition(g, np.zeros(4, dtype=np.int64))
        assert np.array_equal(compress_partition(q), q), q.tolist()
        print("ok test_result_is_contiguous")

    def test_bad_input_is_rejected():
        g = _grid((4, 1, 1))
        try:
            process_partition(g, np.zeros(3, dtype=np.int64))
        except ValueError as exc:
            assert "entries for" in str(exc), exc
        else:
            raise AssertionError("expected a wrong-length partition to fail")
        try:
            process_partition(g, np.zeros(4, dtype=np.int64), face_list=[999])
        except ValueError as exc:
            assert "face numbers" in str(exc), exc
        else:
            raise AssertionError("expected an out-of-range face to fail")
        print("ok test_bad_input_is_rejected")

    test_connected_partition_is_returned_unchanged()
    test_a_split_block_gains_one_number()
    test_the_first_component_keeps_the_block_number()
    test_a_hole_splits_the_block_around_it()
    test_face_list_blocks_connections()
    test_face_list_accepts_a_boolean_mask()
    test_every_output_block_is_connected()
    test_membership_only_ever_splits()
    test_result_is_contiguous()
    test_bad_input_is_rejected()
    print("\nAll tests passed!")
