"""Constants for the Norne field deck workflow."""

from typing import Dict, Tuple

import numpy as np

REPO = "OPM/opm-tests"
REVISION = "5969ec63b"
FILES = (
    "GRID/IRAP_1005.GRDECL",
    "GRID/ACTNUM_0704.prop",
    "PETRO/PERM_0704.prop",
    "PETRO/PORO_0704.prop",
    "PETRO/NTG_0704.prop",
)

CART_DIMS: Tuple[int, int, int] = (46, 112, 22)
PERMZ_MULTIPLIERS = np.array([
    0.2, 0.04, 0.25, 0.0, 0.13, 0.13, 0.13, 0.13, 0.09, 0.07, 0.19,
    0.13, 0.64, 0.64, 0.64, 0.64, 0.64, 0.016, 0.004, 0.004, 1.0, 1.0,
])

SUBSETS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "full": ((9, 24), (34, 54), (0, 21)),
    "mini": ((9, 19), (34, 44), (0, 4)),
}

