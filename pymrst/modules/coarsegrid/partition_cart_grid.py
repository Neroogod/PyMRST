"""Uniform Cartesian partitioning (MRST ``partitionCartGrid.m`` layout)."""

from typing import Sequence

import numpy as np


def partition_cart_grid(cart_dims: Sequence[int], part_dims: Sequence[int]) -> np.ndarray:
    """Uniform Cartesian partition, returning zero-based coarse block ids."""
    cart = list(cart_dims) + [1] * (3 - len(cart_dims))
    part = list(part_dims) + [1] * (3 - len(part_dims))
    nx, ny, nz = cart
    cnx, cny, cnz = part
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    bi = np.minimum((i * cnx) // nx, cnx - 1)
    bj = np.minimum((j * cny) // ny, cny - 1)
    bk = np.minimum((k * cnz) // nz, cnz - 1)
    p = bi + cnx * (bj + cny * bk)
    return np.ravel(p, order="F").astype(np.int64)


__all__ = ["partition_cart_grid"]


if __name__ == "__main__":
    from pymrst.modules.coarsegrid.compress_partition import compress_partition

    def test_block_count_is_the_product_of_the_request():
        p = partition_cart_grid([6, 4, 2], [3, 2, 1])
        assert p.size == 6 * 4 * 2, p.size
        assert p.max() + 1 == 3 * 2 * 1, p.max() + 1
        print("ok test_block_count_is_the_product_of_the_request")

    def test_every_cell_lands_in_exactly_one_block():
        p = partition_cart_grid([5, 3, 2], [2, 3, 1])
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        assert np.bincount(p).sum() == p.size
        print("ok test_every_cell_lands_in_exactly_one_block")

    def test_blocks_are_boxes_in_index_space():
        """A block must be a contiguous box, not a scattering of cells.

        A partition with the right block sizes but interleaved membership is a
        classic Fortran/C order mistake, and the coarse grid still builds.
        """
        nx, ny, nz = 6, 4, 2
        p = partition_cart_grid([nx, ny, nz], [3, 2, 2])
        i, j, k = np.unravel_index(np.arange(p.size), (nx, ny, nz), order="F")
        for b in range(p.max() + 1):
            sel = p == b
            for axis in (i, j, k):
                v = axis[sel]
                assert v.max() - v.min() + 1 == np.unique(v).size, (b, v)
        print("ok test_blocks_are_boxes_in_index_space")

    def test_uneven_division_stays_balanced():
        """7 cells into 3 blocks gives 3, 2, 2 rather than 5, 1, 1."""
        p = partition_cart_grid([7, 1, 1], [3, 1, 1])
        sizes = np.bincount(p)
        assert sizes.sum() == 7 and sizes.max() - sizes.min() <= 1, sizes.tolist()
        print(f"ok test_uneven_division_stays_balanced ({sizes.tolist()})")

    def test_full_resolution_gives_one_cell_per_block():
        p = partition_cart_grid([3, 2, 2], [3, 2, 2])
        assert p.max() + 1 == p.size
        assert len(set(p.tolist())) == p.size
        print("ok test_full_resolution_gives_one_cell_per_block")

    def test_single_block_holds_everything():
        p = partition_cart_grid([4, 3, 2], [1, 1, 1])
        assert p.max() == 0 and p.size == 24
        print("ok test_single_block_holds_everything")

    def test_result_needs_no_compression():
        p = partition_cart_grid([7, 5, 3], [3, 2, 2])
        assert np.array_equal(compress_partition(p), p)
        print("ok test_result_needs_no_compression")

    def test_two_dimensional_input_is_padded():
        p = partition_cart_grid([4, 4], [2, 2])
        assert p.size == 16 and p.max() + 1 == 4
        print("ok test_two_dimensional_input_is_padded")

    test_block_count_is_the_product_of_the_request()
    test_every_cell_lands_in_exactly_one_block()
    test_blocks_are_boxes_in_index_space()
    test_uneven_division_stays_balanced()
    test_full_resolution_gives_one_cell_per_block()
    test_single_block_holds_everything()
    test_result_needs_no_compression()
    test_two_dimensional_input_is_padded()
    print("\nAll tests passed!")
