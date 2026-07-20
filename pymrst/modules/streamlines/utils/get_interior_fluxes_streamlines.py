"""MRST ``getInteriorFluxesStreamlines.m`` cell-face flux gathering."""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid import Grid


def interior_fluxes(
    g: Grid,
    flux: np.ndarray,
    pvol: Optional[np.ndarray] = None,
    reverse: bool = False,
) -> np.ndarray:
    """Gather tagged face fluxes per cell/direction, divided by pore volume."""
    flux = np.asarray(flux, dtype=np.float64)
    if flux.ndim > 1:
        flux = flux.sum(axis=1)
    if reverse:
        flux = -flux
    if g.cells.tags is None:
        raise ValueError("pollock needs a logically Cartesian grid "
                         "(cells.tags is None)")
    pv = (np.ones(g.cells.num) if pvol is None
          else np.asarray(pvol, dtype=np.float64).ravel())
    if not np.all(pv > 0):
        raise ValueError("pore volume must be strictly positive in all cells")

    d = g.griddim
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    out = np.zeros((g.cells.num, 2 * d))
    np.add.at(out, (cell_no, g.cells.tags), flux[g.cells.faces])
    return out / pv[:, None]

__all__ = ["interior_fluxes"]
