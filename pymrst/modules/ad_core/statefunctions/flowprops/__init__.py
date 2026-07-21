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

__all__ = [
    "BlackOilDensity",
    "BlackOilPoreVolume",
    "BlackOilShrinkageFactors",
    "BlackOilViscosity",
    "DensityDerivedShrinkageFactors",
    "PoreVolume",
    "RsMax",
    "RvMax",
    "ShrinkageFactors",
    "SurfaceDensity",
    "Viscosity",
]
