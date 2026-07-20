"""Vertical well constructor, ported from MRST `verticalWell`."""

from typing import List, Optional, Sequence

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc.add_well import add_well
from pymrst.params.wells_and_bc.types import Well


def vertical_well(
    wells: Optional[List[Well]],
    g: Grid,
    rock: Rock,
    i: int,
    j: int,
    k: Optional[Sequence[int]] = None,
    **kwargs,
) -> List[Well]:
    """Add a well down the 0-based logical `(i, j)` column."""
    assert g.cartDims is not None, "vertical_well needs a Cartesian grid"
    nx, ny, nz = _cart_dims_3d(g)
    assert 0 <= i < nx, f"i = {i} is outside 0..{nx - 1}"
    assert 0 <= j < ny, f"j = {j} is outside 0..{ny - 1}"

    layers = np.arange(nz) if k is None else np.atleast_1d(np.asarray(k, dtype=np.int64))
    assert np.all((layers >= 0) & (layers < nz)), f"layer index outside 0..{nz - 1}"
    cells = _active_column_cells(g, nx, ny, nz, i, j, layers)
    assert cells.size, (
        f"the column (i={i}, j={j}) has no active cells in layers "
        f"{layers.min()}..{layers.max()}"
    )
    kwargs.setdefault("well_dir", "z")
    return add_well(wells, g, rock, cells, **kwargs)


def _cart_dims_3d(g: Grid) -> tuple[int, int, int]:
    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    return int(dims[0]), int(dims[1]), int(dims[2])


def _active_column_cells(
    g: Grid,
    nx: int,
    ny: int,
    nz: int,
    i: int,
    j: int,
    layers: np.ndarray,
) -> np.ndarray:
    global_ix = i + j * nx + layers * nx * ny
    if g.cells.indexMap is None:
        return global_ix
    lookup = -np.ones(nx * ny * nz, dtype=np.int64)
    lookup[np.asarray(g.cells.indexMap, dtype=np.int64)] = np.arange(g.cells.num)
    cells = lookup[global_ix]
    return cells[cells >= 0]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.gridprocessing.grid import extract_subgrid
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    def test_vertical_well_perforates_a_column() -> None:
        g = compute_geometry(cart_grid([5, 4, 3], [50.0, 40.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        wells = vertical_well(
            None, g, rock, 2, 1, well_type="rate", val=1.0, compi=np.array([1.0])
        )
        assert np.array_equal(wells[0].cells, [7, 27, 47])
        print("ok test_vertical_well_perforates_a_column")

    def test_vertical_well_skips_inactive_cells() -> None:
        g = compute_geometry(cart_grid([4, 3, 4], [40.0, 30.0, 40.0]))
        drop = {1 + 1 * 4 + layer * 12 for layer in (1, 2)}
        keep = [cell for cell in range(g.cells.num) if cell not in drop]
        sub = compute_geometry(extract_subgrid(g, keep))
        rock = make_rock(sub, 100 * MILLI * DARCY, 0.2)
        wells = vertical_well(None, sub, rock, 1, 1, well_type="bhp", val=1e5)
        global_ids = np.asarray(sub.cells.indexMap)[wells[0].cells]
        assert set(global_ids.tolist()) == {1 + 4, 1 + 4 + 36}
        print("ok test_vertical_well_skips_inactive_cells")

    test_vertical_well_perforates_a_column()
    test_vertical_well_skips_inactive_cells()
    print("\nAll tests passed!")

