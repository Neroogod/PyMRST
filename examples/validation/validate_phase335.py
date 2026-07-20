"""Phase 335: MRST SimpleWell surface and facility integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel, SimpleWell  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str = "W1", well_type: str = "rate", val: float = 3.0) -> Well:
    return Well(
        cells=np.array([0, 2]),
        type=well_type,
        val=val,
        r=np.array([0.1, 0.1]),
        dir="z",
        WI=np.array([2.0, 1.0]),
        dZ=np.array([0.0, 3.0]),
        name=name,
        compi=np.array([0.25, 0.75]),
        refDepth=0.0,
        sign=1,
        status=True,
        lims={"bhp": 10.0, "rate": 5.0},
    )


def _reservoir_model():
    return SimpleNamespace(
        getPhaseNames=lambda: ("W", "O"),
        getNumberOfComponents=lambda: 2,
        getExtraWellPrimaryVariableNames=lambda: ["well_temperature"],
        getExtraWellEquationNames=lambda: (["well_energy"], ["well"]),
    )


def validate_defaults_and_metadata() -> int:
    well = _well()
    model = SimpleWell(well)

    failures = 0
    failures += _check("SimpleWell stores MRST default flags",
                       model.W is well
                       and model.allowCrossflow is True
                       and model.allowSignChange is False
                       and model.allowControlSwitching is True
                       and model.doUpdatePressureDrop is True
                       and np.isinf(model.dpMaxRel)
                       and np.isinf(model.dpMaxAbs)
                       and np.isinf(model.dsMaxAbs))
    failures += _check("SimpleWell advertises reservoir-added variables/equations",
                       model.getExtraPrimaryVariableNames(_reservoir_model()) == (["well_temperature"], [True])
                       and model.getExtraEquationNames(_reservoir_model()) == (["well_energy"], ["well"]))
    replacement = _well("W2", "bhp", 12.0)
    failures += _check("updateWell replaces schedule well and returns self",
                       model.updateWell(replacement) is model and model.W is replacement)
    return failures


def validate_state_and_facility_setup() -> int:
    well = _well()
    reservoir = _reservoir_model()
    simple = SimpleWell(well)
    facility = FacilityModel([well])
    facility.setReservoirModel(reservoir)
    facility.setupWells([well], wellmodels=[simple])
    state = facility.validateState({"wellSol": [SimpleNamespace(status=True)]})
    wsol = state["wellSol"][0]

    failures = 0
    failures += _check("setupWells keeps SimpleWell in WellModels and collects metadata",
                       facility.WellModels == [simple]
                       and facility.addedPrimaryVarNames == ["well_temperature"]
                       and facility.addedEquationNames == ["well_energy"])
    failures += _check("validateWellSol fills MRST-like fields without replacing row",
                       wsol.name == "W1"
                       and np.allclose(wsol.cdp, [0.0, 0.0])
                       and np.allclose(wsol.massfractions, [0.5, 0.5])
                       and wsol.well_temperature == 0.0)
    return failures


def validate_equation_hooks_and_limits() -> int:
    well = _well(val=4.0)
    model = SimpleWell(well)
    reservoir = SimpleNamespace(getPhaseNames=lambda: ("W", "O"), getNumberOfComponents=lambda: 2)
    wsol = model.validateWellSol(reservoir, SimpleNamespace(), {})
    q_s = [np.array([1.0]), np.array([3.0])]
    names, types = model.getWellEquationNames(reservoir)
    result = model.computeWellEquations(wsol, wsol, reservoir, q_s, np.array([9.0]), SimpleNamespace(), 1.0, 1)
    weqs, ctrl_eq, extra, extra_names, q_mass, q_vol, wsol = result

    failures = 0
    failures += _check("getWellEquationNames follows MRST phase-name convention",
                       names == ["WWells", "OWells"] and types == ["perf", "perf"])
    failures += _check("computeWellEquations returns per-phase equations and sources",
                       len(weqs) == 2
                       and np.allclose([eq[0] for eq in weqs], [0.0, 0.0])
                       and np.allclose(ctrl_eq, [0.0])
                       and extra == []
                       and extra_names == []
                       and len(q_mass) == 2
                       and np.allclose(q_vol[0], [2.0 / 3.0, 1.0 / 3.0])
                       and np.allclose(wsol.cqs[:, 1], [2.0, 1.0]))
    comp_result = model.computeComponentContributions(wsol, wsol, reservoir, q_s, np.array([9.0]), SimpleNamespace(),
                                                      q_mass, q_vol, 1.0, 1)
    failures += _check("computeComponentContributions splits total mass by components",
                       len(comp_result[1]) == 2
                       and np.allclose(comp_result[1][0], [2.0 / 3.0, 1.0 / 3.0])
                       and np.allclose(comp_result[1][1], [2.0, 1.0]))
    q_s2, bhp2, wsol2, ok = model.updateLimits(wsol, wsol, reservoir, q_s, np.array([11.0]), SimpleNamespace(), 1.0, 1)
    failures += _check("updateLimits switches active control when a limit is violated",
                       ok is False
                       and model.W.type == "bhp"
                       and np.allclose(bhp2, [10.0])
                       and wsol2.type == "bhp"
                       and q_s2 is q_s)
    return failures


def main() -> int:
    print("\nPhase 335: MRST SimpleWell surface and facility integration")
    failures = 0
    failures += validate_defaults_and_metadata()
    failures += validate_state_and_facility_setup()
    failures += validate_equation_hooks_and_limits()
    if failures:
        print(f"\nPhase 335 failed ({failures} checks).")
        return 1
    print("\nSimpleWell surface and facility integration are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
