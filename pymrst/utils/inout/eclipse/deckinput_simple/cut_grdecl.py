"""Cut a logical box out of a corner-point description (MRST cutGrdecl.m).

A full field deck is usually far larger than the piece anyone wants to
simulate. `cut_grdecl` takes the (i, j, k) box out of the raw GRDECL arrays
*before* the grid is built, so `process_grdecl` only ever sees the subset.

The awkward part, and the reason this is a function rather than three lines
of slicing: the three arrays are indexed differently. Cell properties are one
value per cell; COORD is one pillar per (i, j) *node* - so it needs one extra
column and row - and ZCORN is eight values per cell, laid out as a
(2nx, 2ny, 2nz) array of corners. Each has to be sliced in its own index
space, and all three are stored column-major.
"""

from typing import Dict, Sequence

import numpy as np

CELL_KEYWORDS = ("PERMX", "PERMY", "PERMZ", "PORO", "ACTNUM", "SATNUM", "NTG")


def cut_grdecl(grdecl: Dict, ranges: Sequence[Sequence[int]]) -> Dict:
    """The sub-box `ranges` of a GRDECL dict, as a new GRDECL dict.

    `ranges` is ((i0, i1), (j0, j1), (k0, k1)), **0-based and inclusive** at
    both ends - so PyMRST's `((9, 24), (34, 54), (0, 21))` is the same box as
    MATLAB's `[10 25; 35 55; 1 22]`.
    """
    ranges = np.asarray(ranges, dtype=np.int64).reshape(3, 2)
    dims = np.asarray(grdecl["cartDims"], dtype=np.int64)
    assert np.all(ranges[:, 0] >= 0) and np.all(ranges[:, 1] < dims), \
        f"box {ranges.tolist()} is outside cartDims {dims.tolist()}"
    assert np.all(ranges[:, 1] >= ranges[:, 0]), "empty box"

    new_dims = tuple(int(n) for n in (ranges[:, 1] - ranges[:, 0] + 1))
    out: Dict = {"cartDims": new_dims}

    # Cell properties: one value per cell, column-major over cartDims.
    cell_index = _box_index(dims, ranges)
    for kw in CELL_KEYWORDS:
        if kw in grdecl and grdecl[kw] is not None:
            out[kw] = np.asarray(grdecl[kw]).ravel(order="F")[cell_index]

    # COORD: 6 values per pillar, on an (nx+1, ny+1) node grid.
    xyz = np.asarray(grdecl["COORD"]).reshape(
        (6, dims[0] + 1, dims[1] + 1), order="F"
    )
    xyz = xyz[:,
              ranges[0, 0]:ranges[0, 1] + 2,
              ranges[1, 0]:ranges[1, 1] + 2]
    out["COORD"] = xyz.ravel(order="F")

    # ZCORN: 8 corners per cell, as a (2nx, 2ny, 2nz) corner array - so the
    # box in corner space is [2i, 2i+1] for each cell index i.
    zc_ranges = np.column_stack([2 * ranges[:, 0], 2 * ranges[:, 1] + 1])
    zc_index = _box_index(2 * dims, zc_ranges)
    out["ZCORN"] = np.asarray(grdecl["ZCORN"]).ravel(order="F")[zc_index]
    return out


def _box_index(dims: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Flat column-major indices of the box `ranges` in an array of `dims`."""
    i, j, k = (np.arange(lo, hi + 1) for lo, hi in ranges)
    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
    flat = ii + dims[0] * (jj + dims[1] * kk)          # column-major
    return flat.ravel(order="F")


if __name__ == "__main__":
    from pymrst.gridprocessing.geometry import compute_geometry
    from pymrst.gridprocessing.process_grdecl import process_grdecl
    from pymrst.gridprocessing.tensor_grid import cart_grid

    def _box_grdecl(nx=4, ny=3, nz=2):
        """A GRDECL description of a unit-cell Cartesian box, built by hand."""
        coord = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                coord.extend([i, j, 0.0, i, j, float(nz)])
        coord = np.asarray(coord, float).reshape(-1, 6)
        # COORD is stored pillar-major, 6 values each, column-major in (i, j)
        coord = coord.reshape((ny + 1, nx + 1, 6)).transpose(2, 1, 0)

        z = np.zeros((2 * nx, 2 * ny, 2 * nz))
        for k in range(nz):
            z[:, :, 2 * k] = k
            z[:, :, 2 * k + 1] = k + 1
        poro = np.arange(nx * ny * nz, dtype=float)
        return {
            "cartDims": (nx, ny, nz),
            "COORD": coord.ravel(order="F"),
            "ZCORN": z.ravel(order="F"),
            "PORO": poro,
            "ACTNUM": np.ones(nx * ny * nz),
        }

    def test_cut_matches_the_cells_it_names():
        """The cut keeps exactly the cells of the box, in the right order."""
        g = _box_grdecl()
        sub = cut_grdecl(g, ((1, 2), (0, 1), (0, 0)))
        assert sub["cartDims"] == (2, 2, 1)
        # cells (i, j, 0) for i in 1..2, j in 0..1, column-major
        want = [1 + 4 * j for j in (0, 1)]
        want = [1, 2, 5, 6]
        assert np.array_equal(sub["PORO"], want)
        assert sub["COORD"].size == 6 * 3 * 3
        assert sub["ZCORN"].size == 8 * 4
        print("ok test_cut_matches_the_cells_it_names")

    def test_cut_grid_is_the_subgrid():
        """Building the cut gives the same geometry as cutting the build."""
        g = _box_grdecl()
        sub = cut_grdecl(g, ((1, 2), (0, 1), (0, 1)))
        cut = compute_geometry(process_grdecl(sub))
        assert cut.cells.num == 2 * 2 * 2
        assert np.allclose(cut.cells.volumes, 1.0)
        # the box spans x in [1, 3], y in [0, 2], z in [0, 2]
        assert np.isclose(cut.nodes.coords[:, 0].min(), 1.0)
        assert np.isclose(cut.nodes.coords[:, 0].max(), 3.0)
        assert np.isclose(cut.nodes.coords[:, 1].max(), 2.0)
        print("ok test_cut_grid_is_the_subgrid")

    def test_whole_box_is_a_no_op():
        """Cutting everything returns the same grid."""
        g = _box_grdecl()
        sub = cut_grdecl(g, ((0, 3), (0, 2), (0, 1)))
        assert sub["cartDims"] == g["cartDims"]
        for kw in ("COORD", "ZCORN", "PORO"):
            assert np.allclose(sub[kw], g[kw]), kw
        print("ok test_whole_box_is_a_no_op")

    def test_out_of_range_box_raises():
        """A box outside the deck is an error, not a silent clamp."""
        g = _box_grdecl()
        try:
            cut_grdecl(g, ((0, 9), (0, 2), (0, 1)))
            raise SystemExit("should have raised")
        except AssertionError:
            pass
        print("ok test_out_of_range_box_raises")

    test_cut_matches_the_cells_it_names()
    test_cut_grid_is_the_subgrid()
    test_whole_box_is_a_no_op()
    test_out_of_range_box_raises()
    print("\nAll tests passed!")
