"""MRST ``ad-core/statefunctions/PVTPropertyFunctions.m``.

The grouping of pressure-driven fluid properties: density, viscosity,
shrinkage factors, surface density, phase pressures, pore volume, the
pressure-reduction weights, and for live oil / wet gas the saturated limits
`RsMax` / `RvMax`. All are evaluated per PVT region.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.black_oil_density import (
    BlackOilDensity,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.black_oil_pore_volume import (
    BlackOilPoreVolume,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.black_oil_pressure_reduction_factors import (
    BlackOilPressureReductionFactors,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.black_oil_shrinkage_factors import (
    BlackOilShrinkageFactors,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.black_oil_viscosity import (
    BlackOilViscosity,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.phase_pressures import (
    PhasePressures,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.pore_volume import PoreVolume
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.pressure_reduction_factors import (
    PressureReductionFactors,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.rs_max import RsMax
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.rv_max import RvMax
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.shrinkage_factors import (
    ShrinkageFactors,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.surface_density import (
    SurfaceDensity,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.viscosity import Viscosity
from pymrst.modules.ad_core.statefunctions.state_function import (
    StateFunctionGrouping,
)


class PVTPropertyFunctions(StateFunctionGrouping):
    """MRST's `PVTProps` container."""

    def __init__(self, model: Any):
        super().__init__("PVTProps")
        from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
            ThreePhaseBlackOilModel,
        )
        pvt = self.get_region_pvt(model)
        is_bo = isinstance(model, ThreePhaseBlackOilModel) and (
            model.disgas or model.vapoil)
        self.set_state_function("Density", BlackOilDensity(model, pvt))
        if is_bo:
            mu = BlackOilViscosity(model, pvt)
            bfactors = BlackOilShrinkageFactors(model, pvt)
            if model.disgas:
                self.set_state_function("RsMax", RsMax(model, pvt))
            if model.vapoil:
                self.set_state_function("RvMax", RvMax(model, pvt))
            w_p = BlackOilPressureReductionFactors(model)
        else:
            mu = Viscosity(model, pvt)
            bfactors = ShrinkageFactors(model, pvt)
            w_p = PressureReductionFactors(model)
        self.set_state_function("SurfaceDensity", SurfaceDensity(model, pvt))
        self.set_state_function("Viscosity", mu)
        self.set_state_function("ShrinkageFactors", bfactors)
        self.set_state_function("PressureReductionFactors", w_p)
        self.set_state_function("PhasePressures", PhasePressures(model))
        if getattr(model.fluid, "pvMultR", None) is not None:
            pv = BlackOilPoreVolume(model, pvt)
        else:
            pv = PoreVolume(model, pvt)
        self.set_state_function("PoreVolume", pv)

    def get_region_pvt(self, model: Any) -> np.ndarray:
        regions = getattr(model.rock, "regions", None)
        pvt = (regions.get("pvt") if isinstance(regions, dict)
               else getattr(regions, "pvt", None))
        if pvt is None:
            return np.ones(model.G.cells.num, dtype=np.int64)
        return np.asarray(pvt, dtype=np.int64).ravel()

    getRegionPVT = get_region_pvt


__all__ = ["PVTPropertyFunctions"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model

    def test_a_dead_oil_model_gets_the_plain_pvt_functions():
        model = toy_model("WO", nx=2)
        group = model.PVTPropertyFunctions
        assert type(group.get_state_function("Viscosity")) is Viscosity
        assert type(group.get_state_function("ShrinkageFactors")) is ShrinkageFactors
        assert not group.has_state_function("RsMax")
        assert type(group.get_state_function("PoreVolume")) is PoreVolume
        print("ok test_a_dead_oil_model_gets_the_plain_pvt_functions")

    def test_rock_compressibility_selects_the_pressure_dependent_pore_volume():
        model = toy_model("W", nx=2, c_r=1e-9)
        pv = model.PVTPropertyFunctions.get_state_function("PoreVolume")
        assert type(pv) is BlackOilPoreVolume
        print("ok test_rock_compressibility_selects_the_pressure_dependent_pore_volume")

    test_a_dead_oil_model_gets_the_plain_pvt_functions()
    test_rock_compressibility_selects_the_pressure_dependent_pore_volume()
    print("\nAll tests passed!")
