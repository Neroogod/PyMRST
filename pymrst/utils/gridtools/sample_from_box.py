"""Sample gridded data at cell centroids (MRST ``utils/gridtools/sampleFromBox.m``).

Takes an array of values laid out on a uniform Cartesian lattice spanning the
grid's bounding box, and returns the value at each cell's centroid. The usual
use is populating a grid from a porosity or permeability field that was
generated independently of it - the lattice need not match the grid in any way
beyond covering it.

Sampling is **nearest-cell, not interpolated**: the box is divided into `nd[d]`
equal slabs per direction and a centroid takes the value of the slab it lands
in. That is what MRST does and it is the right default for a piecewise-constant
property like a rock type, but it means the result is not smooth and does not
converge under refinement of `p`. Interpolate first if you want that.

**The bounding box is computed from every node of the selected cells.** MRST
instead takes the nodes of the cells with the extreme *centroids*, which is not
the same thing: on an unstructured grid a cell whose centroid is unremarkable
can still own the furthest-out node, and then MRST's box is too small. Centroids
outside it map to a negative or over-large index - out of bounds in MATLAB,
which is how the bug shows up. The two boxes coincide on Cartesian grids, so
the MATLAB reference agrees there, and a test below exhibits a grid where they
do not.
"""

import numpy as np

from pymrst.utils.gridtools.grid_cell_nodes import grid_cell_nodes


def sample_from_box(g, p, cells=None) -> np.ndarray:
    """Return one value of ``p`` per cell in ``cells`` (default: all cells)."""
    if g.griddim not in (2, 3):
        raise ValueError(f"the grid must be 2-D or 3-D, not {g.griddim}-D")

    p = np.asarray(p)
    # A trailing singleton axis is legitimate and easily lost, so restore it
    # rather than rejecting the array.
    shape = p.shape + (1,) * (g.griddim - p.ndim)
    if len(shape) != g.griddim:
        raise ValueError(
            f"the sample array has {p.ndim} dimensions but the grid has "
            f"{g.griddim}")
    p = p.reshape(shape)

    cells = (np.arange(g.cells.num) if cells is None
             else np.atleast_1d(np.asarray(cells, dtype=np.int64)))
    centroids = np.asarray(g.cells.centroids, dtype=float)[cells]

    nodes, _ = grid_cell_nodes(g, cells)
    corners = np.asarray(g.nodes.coords, dtype=float)[np.unique(nodes)]
    low, high = corners.min(axis=0), corners.max(axis=0)

    span = high - low
    if np.any(span <= 0):
        raise ValueError(
            "the selected cells are flat in at least one direction, so the "
            "bounding box has no interior to sample")

    scaled = (centroids - low) / span
    index = np.ceil(scaled * np.asarray(shape)).astype(np.int64) - 1
    index = np.clip(index, 0, np.asarray(shape) - 1)
    return p[tuple(index[:, d] for d in range(g.griddim))]


sampleFromBox = sample_from_box


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(4, 4), sizes=(4.0, 4.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_a_constant_field_samples_constant() -> None:
        g = build()
        assert np.allclose(sample_from_box(g, np.full((8, 8), 3.5)), 3.5)
        print("ok test_a_constant_field_samples_constant")

    def test_one_value_per_cell_is_returned() -> None:
        g = build()
        assert sample_from_box(g, np.zeros((5, 7))).shape == (g.cells.num,)
        print("ok test_one_value_per_cell_is_returned")

    def test_a_matching_lattice_is_the_identity() -> None:
        """An `n x n` sample on an `n x n` grid maps cell k to entry k."""
        g = build((4, 4), (4.0, 4.0))
        p = np.arange(16, dtype=float).reshape(4, 4)
        got = sample_from_box(g, p)
        # cart_grid numbers cells x-fastest; p is indexed [i, j].
        assert np.array_equal(got, p.ravel(order="F")), got
        print("ok test_a_matching_lattice_is_the_identity")

    def test_it_follows_a_gradient() -> None:
        """A field varying only in x must give values increasing in x."""
        g = build((6, 3), (6.0, 3.0))
        p = np.tile(np.arange(12.0)[:, None], (1, 4))
        got = sample_from_box(g, p)
        x = np.asarray(g.cells.centroids)[:, 0]
        order = np.argsort(x)
        assert np.all(np.diff(got[order]) >= 0), got[order]
        print("ok test_it_follows_a_gradient")

    def test_a_cell_subset_is_honoured() -> None:
        g = build()
        subset = np.array([0, 5, 11])
        p = np.arange(64, dtype=float).reshape(8, 8)
        assert sample_from_box(g, p, subset).shape == (3,)
        print("ok test_a_cell_subset_is_honoured")

    def test_the_subset_rescales_the_box() -> None:
        """A subset spans less, so the same array is stretched over it - which
        is the documented behaviour, not an accident."""
        g = build((4, 4), (4.0, 4.0))
        p = np.arange(16, dtype=float).reshape(4, 4)
        corner = np.array([0])
        assert sample_from_box(g, p, corner)[0] != sample_from_box(g, p)[0]
        print("ok test_the_subset_rescales_the_box")

    def test_every_index_stays_inside_the_array() -> None:
        """The failure mode MRST's smaller box produces; see the docstring."""
        for dims, nd in (((7, 5), (3, 11)), ((2, 9), (13, 2))):
            g = build(dims, tuple(float(d) for d in dims))
            got = sample_from_box(g, np.arange(np.prod(nd), dtype=float)
                                  .reshape(nd))
            assert np.all(np.isfinite(got))
        print("ok test_every_index_stays_inside_the_array")

    def test_it_works_in_three_dimensions() -> None:
        g = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
        p = np.arange(27, dtype=float).reshape(3, 3, 3)
        got = sample_from_box(g, p)
        assert np.array_equal(got, p.ravel(order="F"))
        print("ok test_it_works_in_three_dimensions")

    def test_a_trailing_singleton_axis_is_restored() -> None:
        """`(n, m)` data for a 3-D grid means `(n, m, 1)`, per MRST."""
        g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
        got = sample_from_box(g, np.arange(9, dtype=float).reshape(3, 3))
        assert got.shape == (g.cells.num,)
        print("ok test_a_trailing_singleton_axis_is_restored")

    def test_mrsts_box_would_be_too_small_here() -> None:
        """The divergence, exhibited rather than asserted in prose.

        The point is to separate "owns the furthest-out node" from "has the
        furthest-out centroid". Stretching one *interior* node does it: the
        cells it belongs to sit in the middle of the grid, so they are extreme
        in no direction and MRST never looks at them, while their other nodes
        hold their centroids well short of the right-hand column's. MRST's box
        stops at x = 3.0; the grid reaches 3.9.

        It has to be an interior node. Stretching a boundary node puts it in a
        cell that is extreme in some *other* direction - a node on the top edge
        belongs to a top-row cell, which is picked for its `y` centroid - and
        MRST's box then happens to include it.
        """
        g = cart_grid([3, 3], [3.0, 3.0])
        coords = np.asarray(g.nodes.coords, dtype=float).copy()
        interior = int(np.argmin(np.sum(np.abs(coords - [1.0, 1.0]), axis=1)))
        coords[interior] = [3.9, 1.0]
        g.nodes.coords = coords
        g = compute_geometry(g)

        coords = np.asarray(g.nodes.coords, dtype=float)
        centroids = np.asarray(g.cells.centroids, dtype=float)
        extreme = np.unique(np.concatenate(
            [np.argmin(centroids, axis=0), np.argmax(centroids, axis=0)]))
        nodes, _ = grid_cell_nodes(g, extreme)
        mrst_box = coords[np.unique(nodes)]

        assert np.any(mrst_box.max(axis=0) < coords.max(axis=0) - 1e-12), \
            "expected MRST's box to fall short of the true one"

        # The port uses every node, so it still samples in bounds.
        got = sample_from_box(g, np.arange(20.0).reshape(4, 5))
        assert got.shape == (g.cells.num,) and np.all(np.isfinite(got))
        print("ok test_mrsts_box_would_be_too_small_here")

    def test_a_wrong_dimension_array_is_rejected() -> None:
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        try:
            sample_from_box(g, np.zeros((2, 2, 2)))
        except ValueError as error:
            assert "dimensions" in str(error)
            print("ok test_a_wrong_dimension_array_is_rejected")
            return
        raise AssertionError("a 3-D array on a 2-D grid must be rejected")

    test_a_constant_field_samples_constant()
    test_one_value_per_cell_is_returned()
    test_a_matching_lattice_is_the_identity()
    test_it_follows_a_gradient()
    test_a_cell_subset_is_honoured()
    test_the_subset_rescales_the_box()
    test_every_index_stays_inside_the_array()
    test_it_works_in_three_dimensions()
    test_a_trailing_singleton_axis_is_restored()
    test_mrsts_box_would_be_too_small_here()
    test_a_wrong_dimension_array_is_rejected()
    print("\nAll tests passed!")
