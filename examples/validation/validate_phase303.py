"""Phase 303: MRST subfolder Contents catalogs for compositional ports."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


CATALOGS = {
    "pymrst.modules.compositional.fluids.blackoil.contents": (
        "list_mrst_contents_blackoil",
        (
            "blackOilToMassFraction",
            "convertBlackOilModelToCompositionalModel",
            "convertBlackOilStateToCompositional",
            "convertRSRVtoKValue",
        ),
    ),
    "pymrst.modules.compositional.models.eos.contents": (
        "list_mrst_contents_eos",
        ("EquationOfStateModel", "EquilibriumConstantModel"),
    ),
    "pymrst.modules.compositional.models.natvars.contents": (
        "list_mrst_contents_natvars",
        ("equationsNaturalVariables", "NaturalVariablesCompositionalModel", "ReducedLinearizedSystem"),
    ),
    "pymrst.modules.compositional.models.natvars.sequential.contents": (
        "list_mrst_contents_natvars_sequential",
        (
            "PressureNaturalVariablesModel",
            "PressureReducedLinearSystem",
            "transportEquationsNaturalVars",
            "TransportNaturalVariablesModel",
        ),
    ),
    "pymrst.modules.compositional.models.natvars.sequential.utils.contents": (
        "list_mrst_contents_natvars_sequential_utils",
        ("correctSequentialBHP", "getUpdatesFromPressure"),
    ),
    "pymrst.modules.compositional.models.overall.contents": (
        "list_mrst_contents_overall",
        ("OverallCompositionCompositionalModel",),
    ),
    "pymrst.modules.compositional.models.overall.sequential.contents": (
        "list_mrst_contents_overall_sequential",
        (
            "PressureCompositionalModel",
            "pressureEquationCompositional",
            "PressureOverallCompositionModel",
            "TransportCompositionalModel",
            "transportEquationCompositional",
            "transportEquationOverallComposition",
            "TransportOverallCompositionModel",
        ),
    ),
}


def main() -> int:
    print("\nPhase 303: MRST subfolder Contents catalogs")
    failures = 0
    for module_name, (function_name, expected_names) in CATALOGS.items():
        module = import_module(module_name)
        rows = getattr(module, function_name)()
        names = tuple(row[0] for row in rows)
        failures += _check(f"{module_name} imports", module is not None)
        failures += _check(f"{module_name} preserves MRST entries", names == expected_names)
    if failures:
        print(f"\nPhase 303 failed ({failures} checks).")
        return 1
    print("\nCompositional subfolder Contents catalogs now mirror the MRST layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
