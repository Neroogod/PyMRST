"""MRST-style pore-volume-conserving porosity upscaling."""

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.upscaling.upscale import CoarseGrid
from pymrst.params.rock import Rock, pore_volume


def upscale_poro(g: Grid, rock: Rock, cg: CoarseGrid) -> np.ndarray:
    """Volume-weighted porosity: total pore volume is conserved per block."""
    pv = pore_volume(g, rock)
    num = np.bincount(cg.partition, weights=pv, minlength=cg.n_blocks)
    den = np.bincount(cg.partition, weights=g.cells.volumes,
                      minlength=cg.n_blocks)
    return num / den

__all__ = ["upscale_poro"]
