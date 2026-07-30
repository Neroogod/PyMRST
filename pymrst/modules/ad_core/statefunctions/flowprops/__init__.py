"""MRST ad-core flow properties (`.../statefunctions/flowprops/`).

The PVT half of the flow-property graph, ported on MRST's own class hierarchy:
a pressure-only base and a black-oil subclass for each property, so that
`ad-eor`, `solvent` and compositional-native can subclass the same bases MRST
does instead of re-deriving PVT.

- `ShrinkageFactors` / `BlackOilShrinkageFactors`
- `Viscosity` / `BlackOilViscosity`
- `PoreVolume` / `BlackOilPoreVolume`
- `SurfaceDensity`
- `BlackOilDensity` / `DensityDerivedShrinkageFactors`
- `RsMax` / `RvMax`

These reproduce the pre-split monolithic PVT statefunctions in
`ad_core.statefunctions.pvt_property_functions` bit-for-bit
(`validate_phase340.py`). The relative-permeability hierarchy
(`BaseRelativePermeability`, `BlackOilCapillaryPressure`) and the
pressure-reduction factors (AC5) are separate follow-ons.
"""

from pymrst.modules.ad_core.statefunctions.flowprops.shrinkage_factors import (
    BlackOilShrinkageFactors,
    ShrinkageFactors,
)
from pymrst.modules.ad_core.statefunctions.flowprops.viscosity import (
    BlackOilViscosity,
    Viscosity,
)
from pymrst.modules.ad_core.statefunctions.flowprops.pore_volume import (
    BlackOilPoreVolume,
    PoreVolume,
)
from pymrst.modules.ad_core.statefunctions.flowprops.surface_density import (
    SurfaceDensity,
)
from pymrst.modules.ad_core.statefunctions.flowprops.density import (
    BlackOilDensity,
    DensityDerivedShrinkageFactors,
)
from pymrst.modules.ad_core.statefunctions.flowprops.saturation_limits import (
    RsMax,
    RvMax,
)
from pymrst.modules.ad_core.statefunctions.flowprops.pressure_reduction_factors import (  # noqa: E501
    BlackOilPressureReductionFactors,
    NumericalPressureReductionFactors,
    PressureReductionFactors,
)
from pymrst.modules.ad_core.statefunctions.flowprops.component_properties import (
    ComponentMobility,
    ComponentPhaseDensity,
    ComponentProperty,
)
from pymrst.modules.ad_core.statefunctions.flowprops.virtual import (
    SaturationProperty,
)
from pymrst.modules.ad_core.statefunctions.flowprops.base_relative_permeability import (  # noqa: E501
    BaseRelativePermeability,
)
from pymrst.modules.ad_core.statefunctions.flowprops.hysteresis import (
    HystereticRelativePermeability,
)

__all__ = [
    "BaseRelativePermeability",
    "BlackOilDensity",
    "BlackOilPoreVolume",
    "BlackOilPressureReductionFactors",
    "NumericalPressureReductionFactors",
    "ComponentMobility",
    "ComponentPhaseDensity",
    "ComponentProperty",
    "HystereticRelativePermeability",
    "SaturationProperty",
    "BlackOilShrinkageFactors",
    "BlackOilViscosity",
    "DensityDerivedShrinkageFactors",
    "PoreVolume",
    "PressureReductionFactors",
    "RsMax",
    "RvMax",
    "ShrinkageFactors",
    "SurfaceDensity",
    "Viscosity",
]
