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
from pymrst.modules.ad_core.models.generic_reservoir_model import (
    GenericReservoirModel,
)
from pymrst.modules.ad_core.models.wrapper_model import WrapperModel

__all__ = [
    "BasicSeparator",
    "EOSSeparator",
    "FacilityModel",
    "GenericReservoirModel",
    "PhysicalModel",
    "ReservoirModel",
    "WrapperModel",
    "SeparatorFlashResult",
    "SeparatorGroup",
    "SeparatorSurfaceResult",
]
