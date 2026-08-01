"""Refine a corner-point grid vertically (MRST ``utils/gridtools/refineGrdeclLayers.m``).

Splits a chosen band of layers into `ref` sub-layers each, leaving everything
above and below untouched. The usual reason is that one interval - a thief
zone, the sand you actually care about - needs more vertical resolution than
the overburden, and refining the whole model to get it would be wasteful.

The work is done by cutting the band out, refining the cut, and stitching the
three pieces back into one `ZCORN` array. `COORD` is unchanged, which is what
makes this cheap and also what bounds it: **the pillars are not touched, so
only vertical refinement is possible here** - the areal grid is whatever it
was.

`layers` is a `(first, last)` pair, **0-based and inclusive**, naming the band
to refine. MRST's is 1-based; `layers=(2, 4)` here is `[3 5]` there.

**The refined band's corner depths come from `refine_grdecl`, not from
interpolating the original two surfaces.** For a band whose layers vary in
thickness the two are not the same thing, and the difference is invisible on
the uniform test grids that make this easy to check. Stated because a caller
reading "refine" may reasonably expect the simpler behaviour.
"""

from typing import Dict, Sequence

import numpy as np

from pymrst.gridprocessing.cut_grdecl import cut_grdecl
from pymrst.gridprocessing.refine_grdecl import grdecl_xyz, refine_grdecl


def refine_grdecl_layers(grdecl: Dict, layers: Sequence[int],
                         ref: int) -> Dict:
    """Split layers ``first..last`` of ``grdecl`` into ``ref`` each."""
    first, last = (int(v) for v in layers)
    dims = np.asarray(grdecl["cartDims"], dtype=np.int64)
    if not 0 <= first <= last < dims[2]:
        raise ValueError(
            f"layers {(first, last)} lie outside 0..{dims[2] - 1}; note "
            f"PyMRST numbers layers from 0 where MRST numbers from 1")
    if ref < 1:
        raise ValueError(f"the refinement factor must be at least 1, got {ref}")

    _, zcorn = grdecl_xyz(grdecl)
    above = zcorn[:, :, :2 * first]
    below = zcorn[:, :, 2 * (last + 1):]

    # Cut the band out full-width, refine it vertically, and take its corners.
    band = cut_grdecl(grdecl, [[0, dims[0] - 1], [0, dims[1] - 1],
                               [first, last]])
    _, middle = grdecl_xyz(refine_grdecl(band, [1, 1, ref]))

    thickness = (last - first + 1)
    new_dims = np.array([dims[0], dims[1],
                         dims[2] - thickness + thickness * ref], dtype=np.int64)

    stitched = np.full(tuple(2 * new_dims), np.nan)
    stitched[:, :, :above.shape[2]] = above
    stitched[:, :, above.shape[2]:above.shape[2] + middle.shape[2]] = middle
    if below.shape[2]:
        stitched[:, :, -below.shape[2]:] = below
    if np.any(np.isnan(stitched)):
        raise AssertionError(
            "the reassembled ZCORN has gaps; the three pieces did not tile "
            "the refined grid")

    # ZCORN travels flat in Fortran order; `grdecl_xyz` is what unpacks it.
    return {"cartDims": new_dims, "COORD": np.array(grdecl["COORD"], copy=True),
            "ZCORN": stitched.ravel(order="F")}


refineGrdeclLayers = refine_grdecl_layers


if __name__ == "__main__":
    from pymrst.gridprocessing import compute_geometry, process_grdecl

    def build(dims=(3, 3, 4)):
        """A unit-spaced Cartesian box as a corner-point deck."""
        nx, ny, nz = dims
        xyz = np.zeros((6, nx + 1, ny + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                xyz[:, i, j] = [i, j, 0.0, i, j, float(nz)]
        zc = np.zeros((2 * nx, 2 * ny, 2 * nz))
        for k in range(nz):
            zc[:, :, 2 * k] = k
            zc[:, :, 2 * k + 1] = k + 1
        return {"cartDims": np.array([nx, ny, nz]),
                "COORD": xyz.ravel(order="F"), "ZCORN": zc.ravel(order="F")}

    def test_the_layer_count_grows_by_the_refinement() -> None:
        g = build((3, 3, 4))
        got = refine_grdecl_layers(g, (1, 2), 3)
        # Two layers become six, so 4 -> 8.
        assert tuple(got["cartDims"]) == (3, 3, 8), got["cartDims"]
        print("ok test_the_layer_count_grows_by_the_refinement")

    def test_refining_by_one_changes_nothing() -> None:
        g = build()
        got = refine_grdecl_layers(g, (1, 2), 1)
        assert tuple(got["cartDims"]) == tuple(np.asarray(g["cartDims"]))
        _, before = grdecl_xyz(g)
        _, after = grdecl_xyz(got)
        assert np.allclose(before, after)
        print("ok test_refining_by_one_changes_nothing")

    def test_the_untouched_layers_keep_their_corners() -> None:
        """The point of refining a band rather than the whole grid."""
        g = build((3, 3, 4))
        _, before = grdecl_xyz(g)
        _, after = grdecl_xyz(refine_grdecl_layers(g, (1, 2), 3))
        assert np.allclose(after[:, :, :2], before[:, :, :2]), "layer 0 moved"
        assert np.allclose(after[:, :, -2:], before[:, :, -2:]), "last moved"
        print("ok test_the_untouched_layers_keep_their_corners")

    def test_the_result_is_a_usable_grid() -> None:
        # process_grdecl returns a bare Grid when the result is connected and
        # a list of them when it is not, so a list here means the refinement
        # tore the box apart.
        parts = process_grdecl(refine_grdecl_layers(build((3, 3, 4)), (1, 2), 3))
        assert not isinstance(parts, list), \
            f"the box should stay connected, got {len(parts)} pieces"
        g = compute_geometry(parts)
        assert g.cells.num == 3 * 3 * 8, g.cells.num
        assert np.all(np.asarray(g.cells.volumes) > 0)
        print("ok test_the_result_is_a_usable_grid")

    def test_the_total_volume_is_unchanged() -> None:
        """Refinement subdivides; it must not create or destroy rock."""
        coarse = compute_geometry(process_grdecl(build((3, 3, 4))))
        fine = compute_geometry(process_grdecl(
            refine_grdecl_layers(build((3, 3, 4)), (1, 2), 3)))
        assert np.isclose(np.sum(fine.cells.volumes),
                          np.sum(coarse.cells.volumes), rtol=1e-10), \
            (np.sum(fine.cells.volumes), np.sum(coarse.cells.volumes))
        print("ok test_the_total_volume_is_unchanged")

    def test_a_single_layer_can_be_refined() -> None:
        got = refine_grdecl_layers(build((3, 3, 4)), (2, 2), 5)
        assert tuple(got["cartDims"]) == (3, 3, 8)
        print("ok test_a_single_layer_can_be_refined")

    def test_the_first_and_last_layers_can_be_refined() -> None:
        """The stitching has an empty piece at one end in each case."""
        for band in ((0, 0), (3, 3), (0, 3)):
            got = refine_grdecl_layers(build((3, 3, 4)), band, 2)
            span = band[1] - band[0] + 1
            assert tuple(got["cartDims"]) == (3, 3, 4 - span + 2 * span), band
        print("ok test_the_first_and_last_layers_can_be_refined")

    def test_coord_is_untouched() -> None:
        """Only vertical refinement is possible; see the module docstring."""
        g = build()
        got = refine_grdecl_layers(g, (1, 2), 3)
        assert np.allclose(np.asarray(got["COORD"]), np.asarray(g["COORD"]))
        print("ok test_coord_is_untouched")

    def test_the_input_is_not_modified() -> None:
        g = build()
        before = np.array(g["ZCORN"], copy=True)
        refine_grdecl_layers(g, (1, 2), 3)
        assert np.allclose(np.asarray(g["ZCORN"]), before)
        print("ok test_the_input_is_not_modified")

    def test_an_out_of_range_band_is_rejected() -> None:
        try:
            refine_grdecl_layers(build((3, 3, 4)), (2, 9), 2)
        except ValueError as error:
            assert "MRST numbers from 1" in str(error)
            print("ok test_an_out_of_range_band_is_rejected")
            return
        raise AssertionError("a band outside the grid must be rejected")

    def test_a_zero_refinement_is_rejected() -> None:
        try:
            refine_grdecl_layers(build(), (1, 2), 0)
        except ValueError as error:
            assert "at least 1" in str(error)
            print("ok test_a_zero_refinement_is_rejected")
            return
        raise AssertionError("a zero refinement factor must be rejected")

    test_the_layer_count_grows_by_the_refinement()
    test_refining_by_one_changes_nothing()
    test_the_untouched_layers_keep_their_corners()
    test_the_result_is_a_usable_grid()
    test_the_total_volume_is_unchanged()
    test_a_single_layer_can_be_refined()
    test_the_first_and_last_layers_can_be_refined()
    test_coord_is_untouched()
    test_the_input_is_not_modified()
    test_an_out_of_range_band_is_rejected()
    test_a_zero_refinement_is_rejected()
    print("\nAll tests passed!")
