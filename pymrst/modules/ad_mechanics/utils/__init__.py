"""MRST-shaped ad-mechanics utility landing modules."""

from pymrst.modules.ad_mechanics.utils.add_derived_quantities import (
    add_derived_quantities,
)
from pymrst.modules.ad_mechanics.utils.compute_init_disp import (
    compute_init_disp,
)
from pymrst.modules.ad_mechanics.utils.equations_black_oil_mech import (
    MechBlackOilModel,
)
from pymrst.modules.ad_mechanics.utils.equations_poro_mechanics import (
    MechanicModel,
)
from pymrst.modules.ad_mechanics.utils.equations_water_mech import (
    MechWaterModel,
)
from pymrst.modules.ad_mechanics.utils.poro_params import (
    consolidation_coefficient,
    poro_params,
)

__all__ = [
    "add_derived_quantities",
    "compute_init_disp",
    "MechBlackOilModel",
    "MechanicModel",
    "MechWaterModel",
    "consolidation_coefficient",
    "poro_params",
]
