"""Phase 318: setupWells harvests extra well-model metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, *, val: float = 1.0) -> Well:
    return Well(
        cells=np.array([0], dtype=np.int64),
        type="rate",
        val=val,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.array([1.0, 0.0, 0.0]),
        refDepth=0.0,
        sign=1,
        status=True,
    )


class FirstWellModel:
    def getExtraPrimaryVariableNames(self, reservoir_model):
        assert reservoir_model.tag == "reservoir"
        return ["alpha", "beta"], [False, True]

    def getExtraEquationNames(self, reservoir_model):
        assert reservoir_model.tag == "reservoir"
        return ["seg", "node"], ["connection", "node"]


class SecondWellModel:
    def get_extra_primary_variable_names(self, reservoir_model):
        assert reservoir_model.tag == "reservoir"
        return ("beta", "gamma"), (False, False)

    def get_extra_equation_names(self, reservoir_model):
        assert reservoir_model.tag == "reservoir"
        return ("node", "alphaEq"), ("node-duplicate", "constraint")


class NoArgumentWellModel:
    def getExtraPrimaryVariableNames(self):
        return "delta", True

    def getExtraEquationNames(self):
        return "deltaEq"


def validate_setup_wells_metadata() -> int:
    facility = FacilityModel([_well("I1"), _well("I2")])
    facility.setReservoirModel(SimpleNamespace(tag="reservoir"))
    facility.setupWells([_well("I1"), _well("I2")], wellmodels=[FirstWellModel(), SecondWellModel()])

    failures = 0
    failures += _check("setupWells harvests unique extra primary variables in first-seen order",
                       facility.addedPrimaryVarNames == ["alpha", "beta", "gamma"])
    failures += _check("setupWells preserves first-seen primary variable origin flags",
                       facility.addedPrimaryVarNamesIsFromResModel == [False, True, False])
    failures += _check("setupWells harvests unique extra equation names and types",
                       facility.addedEquationNames == ["seg", "node", "alphaEq"]
                       and facility.addedEquationTypes == ["connection", "node", "constraint"])
    failures += _check("getPrimaryVariableNames appends advertised extras",
                       facility.getPrimaryVariableNames(["W", "O", "G"])
                       == ["qWs", "qOs", "qGs", "bhp", "alpha", "beta", "gamma"])

    fallback = FacilityModel([_well("I1")])
    fallback.setupWells([_well("I1")], wellmodels=[NoArgumentWellModel()])
    failures += _check("metadata hooks may be no-argument methods",
                       fallback.addedPrimaryVarNames == ["delta"]
                       and fallback.addedPrimaryVarNamesIsFromResModel == [True]
                       and fallback.addedEquationNames == ["deltaEq"]
                       and fallback.addedEquationTypes == ["well"])

    fallback.setupWells([_well("I1")])
    failures += _check("direct-well setup clears advertised well-model metadata",
                       fallback.WellModels == []
                       and fallback.addedPrimaryVarNames == []
                       and fallback.addedEquationNames == [])
    return failures


def main() -> int:
    print("\nPhase 318: setupWells harvests extra well-model metadata")
    failures = validate_setup_wells_metadata()
    if failures:
        print(f"\nPhase 318 failed ({failures} checks).")
        return 1
    print("\nsetupWells well-model metadata harvesting is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
