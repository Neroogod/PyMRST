"""Rock upscaling (MRST ``upscaleGridRockAndWells.m`` rock subset)."""

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.upscaling.coarsegrid.generate_coarse_grid import CoarseGrid
from pymrst.modules.upscaling.flow.upscale_permeability_fixed import (
    upscale_perm_fixed,
)
from pymrst.modules.upscaling.flow.upscale_poro import upscale_poro
from pymrst.params.rock import Rock


def upscale_rock(g: Grid, rock: Rock, cg: CoarseGrid) -> Rock:
    """Upscaled rock: flow-based permeability and volume-weighted porosity."""
    return Rock(perm=upscale_perm_fixed(g, rock, cg), poro=upscale_poro(g, rock, cg))


__all__ = ["upscale_rock"]
