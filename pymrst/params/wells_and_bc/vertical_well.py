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
    j: Optional[int] = None,
    k: Optional[Sequence[int]] = None,
    *,
    top_cell: bool = False,
    **kwargs,
) -> List[Well]:
    """Add a vertical well using zero-based logical or top-cell mode.

    Logical mode uses ``i, j, k``.  Set ``top_cell=True`` to interpret ``i``
    as the top-layer logical cell and omit ``j``.  Empty/``None`` layers mean
    all layers, matching MRST.
    """
    layer_size, layer_count = _layer_shape(g)
    if top_cell:
        if j is not None:
            raise ValueError("j must be omitted in top-cell mode")
        column = int(i)
        if not 0 <= column < layer_size:
            raise IndexError("top cell is outside the first logical layer")
    else:
        if g.cartDims is None:
            raise ValueError("logical (i, j) mode needs cartDims")
        nx, ny, _ = _cart_dims_3d(g)
        if j is None:
            raise ValueError("j is required in logical (i, j) mode")
        if not 0 <= i < nx or not 0 <= j < ny:
            raise IndexError("logical well position is outside cartDims")
        column = int(i + j*nx)

    layers = np.asarray([] if k is None else k, dtype=np.int64).ravel()
    if layers.size == 0:
        layers = np.arange(layer_count, dtype=np.int64)
    if np.any((layers < 0) | (layers >= layer_count)):
        raise IndexError(f"layer index outside 0..{layer_count - 1}")
    cells = _active_layer_cells(g, layer_size, layer_count, column, layers)
    kwargs.pop("well_dir", None)
    kwargs["well_dir"] = "z"
    return add_well(wells, g, rock, cells, **kwargs)


def _cart_dims_3d(g: Grid) -> tuple[int, int, int]:
    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    return int(dims[0]), int(dims[1]), int(dims[2])


def _layer_shape(g: Grid) -> tuple[int, int]:
    if hasattr(g, "layerSize") and hasattr(g, "numLayers"):
        return int(g.layerSize), int(g.numLayers)
    if g.cartDims is not None:
        nx, ny, nz = _cart_dims_3d(g)
        return nx*ny, nz if g.griddim == 3 else 1
    raise ValueError("grid does not expose a layered structure")


def _active_layer_cells(g, layer_size, layer_count, column, layers):
    global_ix = column + layers*layer_size
    if g.cells.indexMap is None:
        return global_ix
    lookup = -np.ones(layer_size*layer_count, dtype=np.int64)
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
