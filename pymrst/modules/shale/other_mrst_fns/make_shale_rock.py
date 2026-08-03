"""Shale rock construction, ported from MRST ``makeShaleRock``."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import make_rock
from pymrst.modules.shale.data import ShaleRock


def make_shale_rock(
    grid: Grid,
    perm,
    poro,
    *,
    ntg=None,
    is_matrix=True,
    shale_mechanisms: Mapping[str, Any] | None = None,
) -> ShaleRock:
    """Create rock properties and mark matrix/fracture cells explicitly."""
    base = make_rock(grid, perm, poro, ntg=ntg)
    flags = np.asarray(is_matrix)
    if flags.ndim == 0:
        flags = np.full(grid.cells.num, bool(flags), dtype=bool)
    else:
        flags = flags.reshape(-1)
        if flags.size != grid.cells.num:
            raise ValueError("is_matrix must be scalar or one value per cell")
        flags = flags.astype(bool, copy=False)
    return ShaleRock(
        perm=base.perm,
        poro=base.poro,
        ntg=base.ntg,
        is_matrix=flags,
        shale_mechanisms=dict(shale_mechanisms or {}),
    )


__all__ = ["make_shale_rock"]
