"""Phase 267: compositional fluids/models/utils Contents.m catalog parity."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.fluids.contents import (  # noqa: E402
    FLUIDS,
    MRST_CONTENTS_FLUIDS,
    list_fluids,
    list_mrst_contents_fluids,
)
from pymrst.modules.compositional.models.contents import (  # noqa: E402
    MODELS,
    MRST_CONTENTS_MODELS,
    list_models,
    list_mrst_contents_models,
)
from pymrst.modules.compositional.utils.contents import (  # noqa: E402
    MRST_CONTENTS_UTILS,
    UTILS,
    list_mrst_contents_utils,
    list_utils,
)

EXPECTED_FLUIDS = (
    ("BlackOilPropertyModel", "Black-oil property model for compositional models"),
    ("CompositionalMixture", ""),
    ("CompositionalPropertyModel", "Property model for compositional models"),
    ("CoolPropsCompositionalMixture", "Create MRST compositional fluid by using the open CoolProp library"),
    ("getBenchmarkMixture", "Grab bag of different fluids for use in examples etc"),
    ("PropertyModel", "Base class for properties in the compositional module"),
    ("TableCompositionalMixture", "Create MRST compositional from stored tables (taken from CoolProp)"),
)
EXPECTED_MODELS = (
    ("GenericNaturalVariablesModel", ""),
    ("GenericOverallCompositionModel", ""),
    ("ThreePhaseCompositionalModel", "Base class for compositional models"),
)
EXPECTED_UTIL_NAMES = (
    "checkComponentMassBalance",
    "coolPropFluidsStructs",
    "cubicPositive",
    "ensureMinimumFraction",
    "equationsCompositional",
    "estimateEquilibriumWilson",
    "expandMatrixToCell",
    "FastAD",
    "formatMassString",
    "getComponentsTwoPhaseSimpleWater",
    "getDefaultFlashNonLinearSolver",
    "getEOSComponent",
    "getImpesWeightsOverallComposition",
    "getNonUnitMassFraction",
    "getNonUnitMoleFraction",
    "getPartialVolumes",
    "initCompositionalState",
    "initDeckEOSModel",
    "initVariablesFastAD",
    "mrstCubic",
    "newtonFugacityEquilibrium",
    "phaseStabilityTest",
    "solveRachfordRiceVLE",
    "standaloneFlash",
    "validateCompositionalForces",
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _imports(base: str, catalog: dict[str, dict[str, str | None]]) -> bool:
    ok = True
    for meta in catalog.values():
        module = meta["python_module"]
        if module is None or module.startswith("pymrst."):
            continue
        try:
            importlib.import_module(f"{base}.{module}")
        except Exception:
            ok = False
    return ok


def main() -> int:
    fluids = list_fluids()
    models = list_models()
    utils = list_utils()

    print("\nPhase 267: compositional Contents.m catalog parity")
    failures = 0
    failures += _check("fluids Contents.m entries match MRST order/text",
                       list_mrst_contents_fluids() == EXPECTED_FLUIDS)
    failures += _check("models Contents.m entries match MRST order/text",
                       list_mrst_contents_models() == EXPECTED_MODELS)
    failures += _check("utils Contents.m names match MRST order",
                       tuple(name for name, _ in list_mrst_contents_utils()) == EXPECTED_UTIL_NAMES)
    failures += _check("catalog constants are immutable tuple metadata",
                       isinstance(MRST_CONTENTS_FLUIDS, tuple)
                       and isinstance(MRST_CONTENTS_MODELS, tuple)
                       and isinstance(MRST_CONTENTS_UTILS, tuple))
    failures += _check("helpers return defensive copies",
                       fluids is not FLUIDS and fluids["CompositionalMixture"] is not FLUIDS["CompositionalMixture"]
                       and models is not MODELS and models["GenericOverallCompositionModel"] is not MODELS["GenericOverallCompositionModel"]
                       and utils is not UTILS and utils["phaseStabilityTest"] is not UTILS["phaseStabilityTest"])
    failures += _check("all Contents entries are represented in catalogs",
                       set(fluids) == {name for name, _ in EXPECTED_FLUIDS}
                       and set(models) == {name for name, _ in EXPECTED_MODELS}
                       and set(utils) == set(EXPECTED_UTIL_NAMES))
    failures += _check("implemented fluid utility modules import",
                       _imports("pymrst.modules.compositional.fluids", fluids))
    failures += _check("implemented compositional utility modules import",
                       _imports("pymrst.modules.compositional.utils", utils))
    failures += _check("formerly marked utility/fluid/model gaps are now represented by implemented ports",
                       fluids["CoolPropsCompositionalMixture"]["status"] == "implemented_optional_dependency"
                       and fluids["TableCompositionalMixture"]["status"] == "implemented"
                       and models["GenericNaturalVariablesModel"]["status"] == "implemented_core_surface"
                       and utils["newtonFugacityEquilibrium"]["status"] == "implemented_newton_flash_step"
                       and utils["getEOSComponent"]["status"] == "implemented")

    if failures:
        print(f"\nPhase 267 failed ({failures} checks).")
        return 1
    print("\nCompositional fluids/models/utils catalogs now mirror MRST Contents.m and expose remaining gaps explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
