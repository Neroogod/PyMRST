"""MRST ``ad-core/statefunctions/FlowPropertyFunctions.m``.

The grouping of saturation-driven properties: relative permeability,
capillary pressure, mobility, and the per-component masses, mobilities and
densities. Relative permeability and capillary pressure are evaluated per
saturation region, and an input deck switches on end-point scaling
(`ENDSCALE`, three-point with `SCALECRS`) and J-function capillarity (`JFUNC`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.base_relative_permeability import (
    BaseRelativePermeability,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.black_oil_capillary_pressure import (
    BlackOilCapillaryPressure,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_mobility import (
    ComponentMobility,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_phase_density import (
    ComponentPhaseDensity,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_phase_mass import (
    ComponentPhaseMass,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_total_mass import (
    ComponentTotalMass,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.mobility import Mobility
from pymrst.modules.ad_core.statefunctions.state_function import (
    StateFunctionGrouping,
)


class FlowPropertyFunctions(StateFunctionGrouping):
    """MRST's `FlowProps` container."""

    def __init__(self, model: Any):
        super().__init__("FlowProps")
        sat = self.get_region_saturation(model)
        deck = model.inputdata
        has_deck = deck is not None
        scale_options = {}
        kr_options = {}
        if has_deck:
            runspec, props = _field(deck, "RUNSPEC"), _field(deck, "PROPS")
            do_scaling = _field(runspec, "ENDSCALE") is not None
            scalecrs = _field(props, "SCALECRS")
            three_point = scalecrs is not None and \
                str(np.ravel(scalecrs)[0])[:1].lower() == "y"
            kr_options = {"relpermPoints": 2 + int(three_point)}
            scale_options = {"scalingActive": do_scaling}
        if getattr(model.fluid, "krHyst", None) is not None:
            from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.hysteretic_relative_permeability import (
                HystereticRelativePermeability,
            )
            kr = HystereticRelativePermeability(model, sat, **scale_options, **kr_options)
        else:
            kr = BaseRelativePermeability(model, sat, **scale_options, **kr_options)
        self.set_state_function("RelativePermeability", kr)
        pc = BlackOilCapillaryPressure(model, sat, **scale_options)
        if has_deck:
            jfunc = _field(_field(deck, "GRID"), "JFUNC")
            if jfunc is not None:
                jtype = str(jfunc[0]).lower()
                if model.water and model.oil and jtype in ("both", "water"):
                    pc.set_surface_tension(jfunc[1], "ow")
                if model.gas and model.oil and jtype in ("both", "gas"):
                    pc.set_surface_tension(jfunc[2], "og")
                unit = _field(deck, "PCUNIT")
                pc.set_j_function_constants(jfunc[3], jfunc[4], jfunc[5],
                                            1.0 if unit is None else unit)
        self.set_state_function("CapillaryPressure", pc)
        self.set_state_function("Mobility", Mobility(model, sat))
        self.set_state_function("ComponentPhaseMass", ComponentPhaseMass(model))
        self.set_state_function("ComponentTotalMass", ComponentTotalMass(model))
        self.set_state_function("ComponentMobility", ComponentMobility(model))
        self.set_state_function("ComponentPhaseDensity", ComponentPhaseDensity(model))

    def get_region_saturation(self, model: Any) -> np.ndarray:
        return _region(model, "saturation")

    def get_region_surfactant(self, model: Any) -> np.ndarray:
        return _region(model, "surfactant")

    getRegionSaturation = get_region_saturation
    getRegionSurfactant = get_region_surfactant


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _region(model: Any, kind: str) -> np.ndarray:
    """`rock.regions.<kind>`, or region 1 in every cell."""
    regions = _field(_field(model.rock, "regions"), kind)
    if regions is None:
        return np.ones(model.G.cells.num, dtype=np.int64)
    return np.asarray(regions, dtype=np.int64).ravel()


__all__ = ["FlowPropertyFunctions"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_the_grouping_carries_mrsts_functions():
        model = toy_model("WO", nx=2)
        names = set(model.FlowPropertyFunctions.get_names())
        assert {"RelativePermeability", "CapillaryPressure", "Mobility",
                "ComponentPhaseMass", "ComponentTotalMass", "ComponentMobility",
                "ComponentPhaseDensity"} <= names
        assert model.FlowPropertyFunctions.struct_name == "FlowProps"
        print("ok test_the_grouping_carries_mrsts_functions")

    def test_saturation_regions_come_from_the_rock():
        model = toy_model("WO", nx=3, rock_regions={"saturation": [1, 2, 1]})
        kr = model.FlowPropertyFunctions.get_state_function("RelativePermeability")
        np.testing.assert_array_equal(kr.regions, [1, 2, 1])
        print("ok test_saturation_regions_come_from_the_rock")

    def test_hysteresis_selects_the_hysteretic_relative_permeability():
        """A fluid with `krHyst` gets MRST's HystereticRelativePermeability;
        without `krPts` or `sWcon` its gas end points default (MRST warns)."""
        import warnings
        from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.hysteretic_relative_permeability import (
            HystereticRelativePermeability,
        )
        model = toy_model("WO", nx=2)
        model.fluid.krHyst = 1
        model.fluid.krPts = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            props = FlowPropertyFunctions(model)
        kr = props.get_state_function("RelativePermeability")
        assert isinstance(kr, HystereticRelativePermeability), type(kr)
        print("ok test_hysteresis_selects_the_hysteretic_relative_permeability")

    test_the_grouping_carries_mrsts_functions()
    test_saturation_regions_come_from_the_rock()
    test_hysteresis_selects_the_hysteretic_relative_permeability()
    print("\nAll tests passed!")
