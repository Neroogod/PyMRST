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
from pymrst.modules.ad_core.models.composite import CompositeModel
from pymrst.modules.ad_core.models.wells import (
    setup_well_control_equation_single_well,
    update_switched_controls,
    violated_limit,
)

__all__ = [
    "BasicSeparator",
    "CompositeModel",
    "EOSSeparator",
    "FacilityModel",
    "GenericReservoirModel",
    "PhysicalModel",
    "ReservoirModel",
    "WrapperModel",
    "SeparatorFlashResult",
    "SeparatorGroup",
    "SeparatorSurfaceResult",
    "setup_well_control_equation_single_well",
    "update_switched_controls",
    "violated_limit",
]
