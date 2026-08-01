"""Complex 3-D grids for vemmech (MRST `complex3DGrid`).

Four cases in MRST. Two are self-contained and ported here:

- ``box``: a Cartesian block whose top and bottom faces are triangulated, so
  cells become general polyhedra rather than hexahedra;
- ``grdecl``: a corner-point grid from `simple_grdecl`, with the pillar
  distortion that makes faces non-planar.

The other two, ``sbed`` and ``norne``, read ECLIPSE decks from MRST's dataset
downloads. They are recognised and refused with a clear message rather than
silently omitted, because the dataset is the missing piece, not the code.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.compute_geometry import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.gridprocessing.testgrids.simple_grdecl import simple_grdecl
from pymrst.gridprocessing.triangulate_faces import triangulate_faces
from pymrst.modules.vemmech.utils.square_grid import square_grid
from pymrst.modules.vemmech.vem import create_augmented_grid

GRID_CASES = ("box", "grdecl", "sbed", "norne")
DATASET_CASES = ("sbed", "norne")

# MRST tags top and bottom faces 5 and 6; zero-based those are 4 and 5
_TOP_BOTTOM_TAGS = (4, 5)


@dataclass
class ComplexGridOptions:
    """Options MRST passes through its `opt` struct."""

    cart_dims: Sequence[int] = (4, 4, 4)
    L: Sequence[float] = (1.0, 1.0, 1.0)
    disturb: float = 0.0
    triangulate: bool = True
    gtol: float = 0.0
    vertical: bool = False


def complex_3d_grid(opt: ComplexGridOptions, grid_case: str) -> Tuple[object, object]:
    """Return ``(grid, original_grid)`` for the named case.

    ``original_grid`` is ``None`` except where MRST keeps a pre-processing
    copy, matching its second output.
    """
    if grid_case in DATASET_CASES:
        raise NotImplementedError(
            f"the {grid_case!r} case reads an ECLIPSE deck from MRST's dataset "
            "downloads (BedModel2 / norne), which PyMRST does not ship; supply "
            "the deck and use process_grdecl directly"
        )
    if grid_case == "box":
        grid = square_grid(list(opt.cart_dims), list(opt.L),
                           grid_type="cartgrid", disturb=opt.disturb)
        grid = compute_geometry(grid)
        if opt.triangulate:
            grid = triangulate_faces(grid, _top_and_bottom_faces(grid))
        return create_augmented_grid(compute_geometry(grid)), None
    if grid_case == "grdecl":
        size = int(np.ceil(1e3 ** (1.0 / 3.0)))
        deck = simple_grdecl([2 * size, 1 * size, 2 * size], 0.15)
        grid = process_grdecl(deck)
        if isinstance(grid, list):
            grid = grid[0]
        return create_augmented_grid(compute_geometry(grid)), None
    raise ValueError(f"no such grid case: {grid_case!r}; expected {GRID_CASES}")


def _top_and_bottom_faces(grid) -> np.ndarray:
    """The faces MRST selects by cell-face tag 5 and 6."""
    if grid.cells.tags is None:
        raise ValueError("the box case needs cell-face tags to find top/bottom")
    tags = np.asarray(grid.cells.tags, dtype=np.int64)
    faces = np.asarray(grid.cells.faces, dtype=np.int64)
    return np.unique(faces[np.isin(tags, _TOP_BOTTOM_TAGS)])


complex3DGrid = complex_3d_grid


if __name__ == "__main__":

    def test_box_case_triangulates_top_and_bottom_only() -> None:
        plain, _ = complex_3d_grid(
            ComplexGridOptions(cart_dims=[3, 3, 3], triangulate=False), "box")
        cut, _ = complex_3d_grid(
            ComplexGridOptions(cart_dims=[3, 3, 3], triangulate=True), "box")
        assert cut.faces.num > plain.faces.num
        assert cut.cells.num == plain.cells.num, "cells must not be split"
        assert np.isclose(cut.cells.volumes.sum(), plain.cells.volumes.sum())
        print(f"ok test_box_case_triangulates_top_and_bottom_only "
              f"({plain.faces.num} -> {cut.faces.num} faces)")

    def test_box_case_is_a_valid_grid() -> None:
        grid, _ = complex_3d_grid(ComplexGridOptions(cart_dims=[3, 3, 3]), "box")
        assert np.all(grid.cells.volumes > 0)
        assert not np.any(np.all(grid.faces.neighbors == -1, axis=1))
        print("ok test_box_case_is_a_valid_grid")

    def test_grdecl_case_builds_a_corner_point_grid() -> None:
        grid, _ = complex_3d_grid(ComplexGridOptions(), "grdecl")
        assert grid.griddim == 3
        assert grid.cells.num > 0 and np.all(grid.cells.volumes > 0)
        print(f"ok test_grdecl_case_builds_a_corner_point_grid "
              f"({grid.cells.num} cells)")

    def test_box_case_solves_linear_elasticity() -> None:
        from pymrst.modules.vemmech.vem import ElasticBC, enu2c
        from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast

        grid, _ = complex_3d_grid(ComplexGridOptions(cart_dims=[3, 3, 3]), "box")
        C = enu2c(np.full(grid.cells.num, 1e9), np.full(grid.cells.num, 0.25))
        base = np.flatnonzero(np.isclose(grid.nodes.coords[:, 2], 0.0))
        bc = ElasticBC(disp_nodes=base,
                       disp_mask=np.ones((base.size, 3), dtype=bool),
                       disp_vals=np.zeros((base.size, 3)),
                       force_faces=np.zeros(0, dtype=np.int64),
                       force=np.zeros((0, 3)))
        load = lambda x: np.tile([0.0, 0.0, -1e4], (x.shape[0], 1))  # noqa: E731
        u, _ = vem_lin_elast(grid, C, bc, load)
        assert np.all(np.isfinite(u))
        assert abs(u[base]).max() < 1e-12
        print("ok test_box_case_solves_linear_elasticity")

    def test_dataset_cases_are_refused_clearly() -> None:
        for case in DATASET_CASES:
            try:
                complex_3d_grid(ComplexGridOptions(), case)
            except NotImplementedError as error:
                assert "dataset" in str(error)
                continue
            raise AssertionError(f"{case} should be refused")
        print("ok test_dataset_cases_are_refused_clearly")

    def test_unknown_case_is_rejected() -> None:
        try:
            complex_3d_grid(ComplexGridOptions(), "nope")
        except ValueError:
            print("ok test_unknown_case_is_rejected")
            return
        raise AssertionError("unknown case should be rejected")

    test_box_case_triangulates_top_and_bottom_only()
    test_box_case_is_a_valid_grid()
    test_grdecl_case_builds_a_corner_point_grid()
    test_box_case_solves_linear_elasticity()
    test_dataset_cases_are_refused_clearly()
    test_unknown_case_is_rejected()
    print("\nAll tests passed!")
