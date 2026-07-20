"""Grid coarsening and upscaling (MRST modules/coarsegrid + upscaling)."""

from pymrst.modules.upscaling.coarsegrid.generate_coarse_grid import (
    CoarseGrid,
    generate_coarse_grid,
)
from pymrst.modules.upscaling.coarsegrid.partition_cart_grid import (
    partition_cart_grid,
)
from pymrst.modules.upscaling.flow.upscale_permeability_fixed import (
    upscale_perm_fixed,
)
from pymrst.modules.upscaling.flow.upscale_poro import (
    upscale_poro,
)
from pymrst.modules.upscaling.flow.upscale_rock import (
    upscale_rock,
)

__all__ = [
    "partition_cart_grid", "CoarseGrid", "generate_coarse_grid",
    "upscale_perm_fixed", "upscale_poro", "upscale_rock",
]
