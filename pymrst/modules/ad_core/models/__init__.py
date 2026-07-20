"""MRST-style model entry points for `ad_core`."""

from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.facilities.separators import (
    BasicSeparator,
    EOSSeparator,
    SeparatorFlashResult,
    SeparatorGroup,
    SeparatorSurfaceResult,
)
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel

__all__ = [
    "BasicSeparator",
    "EOSSeparator",
    "FacilityModel",
    "PhysicalModel",
    "ReservoirModel",
    "SeparatorFlashResult",
    "SeparatorGroup",
    "SeparatorSurfaceResult",
]
