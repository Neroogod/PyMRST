"""MRST coarse-grid compatibility landing modules."""

from pymrst.modules.upscaling.coarsegrid.generate_coarse_grid import (
    CoarseGrid,
    generate_coarse_grid,
)
from pymrst.modules.upscaling.coarsegrid.partition_cart_grid import (
    partition_cart_grid,
)

__all__ = ["CoarseGrid", "generate_coarse_grid", "partition_cart_grid"]
