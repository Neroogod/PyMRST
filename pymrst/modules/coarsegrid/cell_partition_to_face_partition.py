"""Face partition induced by a cell partition (MRST ``cellPartitionToFacePartition.m``).

Every fine face lies between two coarse blocks, or between a block and the
outside. Grouping faces by that pair is the first step in building a coarse
grid: all fine faces separating the same two blocks become one coarse face.

The pair is unordered, so a face with blocks `(2, 5)` groups with one having
`(5, 2)`. MRST sorts each row before grouping, and the port does the same.

**Exterior faces.** MRST marks a missing neighbour with 0 and prepends a zero
to the partition so that "outside" behaves as block 0, which puts every
boundary face of a given block into one group. PyMRST marks it -1, which groups
the same way without the prepend. With ``all_boundary_faces`` each boundary face
instead gets an id of its own, matching MRST's option of the same name.

MRST groups with `sortrows` followed by `rlencode`. `np.unique` over rows with
`return_inverse` is the same operation in one call.
"""

from __future__ import annotations

import numpy as np


def cell_partition_to_face_partition(g, partition,
                                     all_boundary_faces: bool = False) -> np.ndarray:
    """Give each distinct pair of adjacent blocks a face-partition id.

    Returns a 0-based id per fine face. Faces separating the same two blocks
    share an id.
    """
    partition = np.asarray(partition, dtype=np.int64).ravel()
    if partition.size != int(g.cells.num):
        raise ValueError(
            f"partition has {partition.size} entries for {g.cells.num} cells")

    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    # Block on each side, with -1 kept for "outside" so it sorts below every
    # real block and groups consistently.
    blocks = np.where(neighbors >= 0, partition[np.maximum(neighbors, 0)], -1)
    pairs = np.sort(blocks, axis=1)

    _, face_partition = np.unique(pairs, axis=0, return_inverse=True)
    face_partition = face_partition.astype(np.int64).ravel()

    if all_boundary_faces:
        boundary = np.any(neighbors < 0, axis=1)
        n_boundary = int(np.count_nonzero(boundary))
        if n_boundary:
            face_partition[boundary] = (face_partition.max() + 1
                                        + np.arange(n_boundary, dtype=np.int64))
    return face_partition


__all__ = ["cell_partition_to_face_partition"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(4, 4, 1), part=(2, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        return g, partition_cart_grid(list(dims), list(part))

    def test_faces_between_the_same_blocks_share_an_id():
        """The defining property: the id identifies the block pair, not the face."""
        g, p = _case()
        pf = cell_partition_to_face_partition(g, p)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        blocks = np.where(nb >= 0, p[np.maximum(nb, 0)], -1)
        pairs = np.sort(blocks, axis=1)
        for a in range(pf.size):
            for b in range(a + 1, pf.size):
                same_pair = np.array_equal(pairs[a], pairs[b])
                assert (pf[a] == pf[b]) == same_pair, (a, b, pairs[a], pairs[b])
        print(f"ok test_faces_between_the_same_blocks_share_an_id "
              f"({pf.max() + 1} groups from {pf.size} faces)")

    def test_the_pair_is_unordered():
        """A face is between blocks 2 and 5 whichever way its normal points.

        Grouping on the ordered pair would split every coarse face in two, and
        the resulting coarse grid would still look reasonable.
        """
        g, p = _case()
        pf = cell_partition_to_face_partition(g, p)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        interior = np.all(nb >= 0, axis=1)
        # Flip every neighbour pair and require the same grouping.
        flipped = g.faces.neighbors.copy()
        flipped[interior] = flipped[interior][:, ::-1]
        original = g.faces.neighbors
        try:
            g.faces.neighbors = flipped
            pf2 = cell_partition_to_face_partition(g, p)
        finally:
            g.faces.neighbors = original
        # Ids may be renumbered, but the grouping must be identical.
        assert np.array_equal(
            np.unique(pf, return_inverse=True)[1],
            np.unique(pf2, return_inverse=True)[1]), "orientation changed the grouping"
        print("ok test_the_pair_is_unordered")

    def test_internal_faces_of_one_block_form_one_group():
        """Faces inside a block have the same block on both sides."""
        g, p = _case()
        pf = cell_partition_to_face_partition(g, p)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        interior = np.all(nb >= 0, axis=1)
        internal = interior & (p[nb[:, 0]] == p[nb[:, 1]])
        for b in np.unique(p):
            sel = internal & (p[np.maximum(nb[:, 0], 0)] == b)
            if np.any(sel):
                assert len(set(pf[sel].tolist())) == 1, b
        print("ok test_internal_faces_of_one_block_form_one_group")

    def test_boundary_faces_of_a_block_group_together_by_default():
        g, p = _case()
        pf = cell_partition_to_face_partition(g, p)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        boundary = np.any(nb < 0, axis=1)
        assert np.any(boundary)
        # Fewer distinct ids than boundary faces: they are grouped by block.
        assert len(set(pf[boundary].tolist())) < int(np.count_nonzero(boundary))
        print("ok test_boundary_faces_of_a_block_group_together_by_default")

    def test_all_boundary_faces_gives_each_its_own_id():
        g, p = _case()
        pf = cell_partition_to_face_partition(g, p, all_boundary_faces=True)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        boundary = np.any(nb < 0, axis=1)
        ids = pf[boundary]
        assert len(set(ids.tolist())) == ids.size, "boundary faces still share ids"
        # Interior grouping is untouched.
        interior_ids = pf[~boundary]
        assert len(set(interior_ids.tolist())) < interior_ids.size
        print(f"ok test_all_boundary_faces_gives_each_its_own_id ({ids.size} faces)")

    def test_a_single_block_gives_two_groups():
        """One block: interior faces in one group, boundary faces in another."""
        g, p = _case(part=(1, 1, 1))
        pf = cell_partition_to_face_partition(g, p)
        assert pf.max() + 1 == 2, pf.max() + 1
        print("ok test_a_single_block_gives_two_groups")

    def test_full_resolution_separates_every_face():
        """One cell per block: no two interior faces share a block pair."""
        g, p = _case(dims=(3, 3, 1), part=(3, 3, 1))
        pf = cell_partition_to_face_partition(g, p)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        interior = np.all(nb >= 0, axis=1)
        ids = pf[interior]
        assert len(set(ids.tolist())) == ids.size, "distinct cell pairs merged"
        print("ok test_full_resolution_separates_every_face")

    def test_ids_are_contiguous():
        g, p = _case(dims=(6, 4, 2), part=(3, 2, 2))
        pf = cell_partition_to_face_partition(g, p)
        assert np.array_equal(np.unique(pf), np.arange(pf.max() + 1))
        print("ok test_ids_are_contiguous")

    def test_wrong_length_partition_is_rejected():
        g, p = _case()
        try:
            cell_partition_to_face_partition(g, p[:-1])
        except ValueError as exc:
            assert "entries for" in str(exc), exc
        else:
            raise AssertionError("expected a wrong-length partition to fail")
        print("ok test_wrong_length_partition_is_rejected")

    test_faces_between_the_same_blocks_share_an_id()
    test_the_pair_is_unordered()
    test_internal_faces_of_one_block_form_one_group()
    test_boundary_faces_of_a_block_group_together_by_default()
    test_all_boundary_faces_gives_each_its_own_id()
    test_a_single_block_gives_two_groups()
    test_full_resolution_separates_every_face()
    test_ids_are_contiguous()
    test_wrong_length_partition_is_rejected()
    print("\nAll tests passed!")
