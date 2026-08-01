"""Rock construction, ported from MRST `makeRock`."""

from typing import Optional, Union
import warnings

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock.rock import Rock

Scalar = Union[float, np.ndarray]


def make_rock(
    g: Grid,
    perm: Scalar,
    poro: Scalar,
    ntg: Optional[Scalar] = None,
) -> Rock:
    """Create rock properties with per-cell permeability and porosity."""
    nc = g.cells.num
    perm_arr = _expand_to_cells(perm, nc)
    valid_cols = {1: (1,), 2: (1, 2, 3), 3: (1, 3, 6)}[g.griddim]
    if perm_arr.shape[1] not in valid_cols:
        raise ValueError(
            f"permeability must have {valid_cols} columns in {g.griddim}D, "
            f"got {perm_arr.shape[1]}"
        )
    poro_arr = _expand_to_cells(poro, nc)
    if poro_arr.shape[1] != 1:
        raise ValueError("porosity must be a single column")
    if np.any(poro_arr <= 0):
        warnings.warn("Zero or negative porosity found")

    rock = Rock(perm=perm_arr, poro=poro_arr.ravel())
    if ntg is not None:
        ntg_arr = _expand_to_cells(ntg, nc)
        if ntg_arr.shape[1] != 1:
            raise ValueError("net-to-gross must be a single column")
        rock.ntg = ntg_arr.ravel()
    return rock


def _expand_to_cells(vals: Scalar, nc: int) -> np.ndarray:
    """Broadcast a scalar or single row to one row per cell."""
    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        # NumPy has no row/column distinction. Treat nc entries as a
        # per-cell scalar field; use [[kx, ky, kz]] for a uniform tensor.
        arr = arr.reshape(-1, 1) if arr.size == nc else arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError("rock values must be scalar, vector, or matrix")
    if arr.shape[0] == 1:
        arr = np.repeat(arr, nc, axis=0)
    if arr.shape[0] != nc:
        raise ValueError(
            "rock values must be one row per cell or one uniform row"
        )
    return arr


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid
    from pymrst.utils.units import DARCY, MILLI

    def test_make_rock_scalar_expansion() -> None:
        g = cart_grid([2, 2, 2])
        rock = make_rock(g, 100 * MILLI * DARCY, 0.3)
        assert rock.perm.shape == (8, 1)
        assert rock.poro.shape == (8,)
        assert np.allclose(rock.perm, 100 * MILLI * DARCY)
        print("ok test_make_rock_scalar_expansion")

    def test_make_rock_ntg() -> None:
        g = cart_grid([2, 2, 1])
        rock = make_rock(g, 1.0, 0.25, ntg=0.5)
        assert rock.ntg is not None
        assert np.allclose(rock.ntg, 0.5)
        print("ok test_make_rock_ntg")

    test_make_rock_scalar_expansion()
    test_make_rock_ntg()
    print("\nAll tests passed!")
