"""Utilities matching MRST ``modules/mpfa/utils``."""

from pymrst.modules.mpfa.utils.block_inverter import block_inverter
from pymrst.modules.mpfa.utils.compute_local_flux_mimetic import (
    compute_local_flux_mimetic,
)
from pymrst.modules.mpfa.utils.matrix_blocks_from_sparse import (
    matrix_blocks_from_sparse,
)

__all__ = [
    "block_inverter",
    "compute_local_flux_mimetic",
    "matrix_blocks_from_sparse",
]
