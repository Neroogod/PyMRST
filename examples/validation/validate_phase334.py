"""Phase 334: MRST GenericFacilityModel/UniformFacilityModel surfaces."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import (  # noqa: E402
    FacilityModel,
    GenericFacilityModel,
    UniformFacilityModel,
)
from pymrst.utils.units import ATM  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_generic_defaults_and_delegates() -> int:
    reservoir = SimpleNamespace(
        AutoDiffBackend=SimpleNamespace(name="backend"),
        getNumberOfComponents=lambda: 3,
        getNumberOfPhases=lambda: 2,
        getPhaseNames=lambda: ("W", "O"),
    )
    facility = GenericFacilityModel(reservoir)

    failures = 0
    failures += _check("GenericFacilityModel subclasses FacilityModel",
                       isinstance(facility, FacilityModel))
    failures += _check("GenericFacilityModel follows MRST setWellTargets default",
                       facility.setWellTargets is False)
    failures += _check("GenericFacilityModel stores MRST standard surface conditions",
                       np.isclose(facility.T, 288.15)
                       and np.isclose(facility.pressure, ATM)
                       and facility.getSurfaceConditions() == (float(ATM), 288.15))
    failures += _check("GenericFacilityModel delegates component and phase counts",
                       facility.getNumberOfComponents() == 3
                       and facility.getNumberOfPhases() == 2)
    failures += _check("GenericFacilityModel exposes MRST output/separator slots",
                       facility.outputFluxes is True
                       and facility.output_fluxes is True
                       and facility.SeparatorGroup is None
                       and facility.separator_group is None)
    return failures


def validate_component_sources_and_uniform() -> int:
    class FlowDiscretization:
        def __init__(self, empty: bool):
            self.empty = empty

        def get(self, facility, state, name):
            del facility, state
            if name == "FacilityWellMapping":
                return SimpleNamespace(W=[] if self.empty else [object()], cells=np.array([1, 3]))
            if name == "ComponentTotalFlux":
                return [np.array([1.0]), np.array([2.0])]
            raise KeyError(name)

    generic = GenericFacilityModel([])
    generic.FacilityFlowDiscretization = FlowDiscretization(empty=True)
    empty_sources = generic.getComponentSources({})
    generic.FacilityFlowDiscretization = FlowDiscretization(empty=False)
    sources = generic.get_component_sources({})
    uniform = UniformFacilityModel([])

    failures = 0
    failures += _check("GenericFacilityModel getComponentSources handles empty well mapping",
                       empty_sources["value"] == []
                       and np.array_equal(empty_sources["cells"], np.array([1, 3])))
    failures += _check("GenericFacilityModel getComponentSources returns component fluxes",
                       len(sources["value"]) == 2
                       and np.allclose(sources["value"][0], [1.0])
                       and np.array_equal(sources["cells"], np.array([1, 3])))
    failures += _check("UniformFacilityModel subclasses FacilityModel with standard target behavior",
                       isinstance(uniform, FacilityModel)
                       and uniform.setWellTargets is True)
    return failures


def main() -> int:
    print("\nPhase 334: MRST GenericFacilityModel/UniformFacilityModel surfaces")
    failures = 0
    failures += validate_generic_defaults_and_delegates()
    failures += validate_component_sources_and_uniform()
    if failures:
        print(f"\nPhase 334 failed ({failures} checks).")
        return 1
    print("\nGeneric/Uniform facility model surfaces are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
