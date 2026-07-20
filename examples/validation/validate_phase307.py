"""Phase 307: MRST FacilityModel primary-variable extraction parity."""

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


def _well(name: str, *, status: bool = True) -> Well:
    return Well(
        cells=np.array([0], dtype=np.int64),
        type="rate",
        val=1.0,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.array([0.2, 0.3, 0.5]),
        refDepth=0.0,
        sign=1,
        status=status,
    )


def validate_primary_variable_extraction() -> int:
    well_sol = [
        SimpleNamespace(status=True, qWs=1.0, qOs=2.0, qGs=3.0, bhp=100.0,
                        massfractions=np.array([0.2, 0.3, 0.5])),
        SimpleNamespace(status=False, qWs=9.0, qOs=9.0, qGs=9.0, bhp=900.0,
                        massfractions=np.array([0.8, 0.1, 0.1])),
        SimpleNamespace(status=True, qWs=4.0, qOs=5.0, qGs=6.0, bhp=200.0,
                        massfractions=np.array([0.1, 0.4, 0.5])),
    ]

    standard = FacilityModel([_well("I1"), _well("P1", status=False), _well("P2")])
    variables, names, mapping = standard.get_basic_primary_variables(well_sol, ["W", "O", "G"])
    state_variables, state_names, origin = standard.getPrimaryVariables({"wellSol": well_sol})

    bhp_only = FacilityModel([_well("I1"), _well("P1", status=False), _well("P2")],
                             primary_variable_set="bhp")
    bhp_vars, bhp_names, bhp_map = bhp_only.getBasicPrimaryVariables(well_sol)

    mass = FacilityModel([_well("I1"), _well("P1", status=False), _well("P2")],
                         primary_variable_set="bhp_massfractions")
    mass_vars, mass_names, _ = mass.get_basic_primary_variables(well_sol)

    none = FacilityModel([_well("I1")], primary_variable_set="none")
    none_vars, none_names, none_map = none.get_basic_primary_variables(well_sol[:1])

    failures = 0
    failures += _check("standard names match MRST q<phase>s plus bhp",
                       names == ["qWs", "qOs", "qGs", "bhp"])
    failures += _check("standard variables include active wells only in order",
                       np.allclose(variables[0], [1.0, 4.0])
                       and np.allclose(variables[1], [2.0, 5.0])
                       and np.allclose(variables[2], [3.0, 6.0])
                       and np.allclose(variables[3], [100.0, 200.0]))
    failures += _check("basic variable map marks only bhp as pressure control variable",
                       np.array_equal(mapping["isBHP"], [False, False, False, True])
                       and np.array_equal(mapping["isRate"], [True, True, True, False]))
    failures += _check("getPrimaryVariables returns variables, names and facility origin",
                       state_names == names
                       and all(np.allclose(a, b) for a, b in zip(state_variables, variables))
                       and origin == ["FacilityModel"] * 4)
    failures += _check("bhp-only primary variable set is honored",
                       bhp_names == ["bhp"]
                       and np.allclose(bhp_vars[0], [100.0, 200.0])
                       and np.array_equal(bhp_map["isBHP"], [True]))
    failures += _check("bhp_massfractions flattens all but final component per active well",
                       mass_names == ["bhp", "well_massfractions"]
                       and np.allclose(mass_vars[0], [100.0, 200.0])
                       and np.allclose(mass_vars[1], [0.2, 0.3, 0.1, 0.4]))
    failures += _check("none primary variable set returns empty MRST-shaped outputs",
                       none_vars == [] and none_names == []
                       and none_map["isBHP"].size == 0
                       and none_map["isRate"].size == 0)
    return failures


def main() -> int:
    print("\nPhase 307: MRST FacilityModel primary-variable extraction parity")
    failures = validate_primary_variable_extraction()
    if failures:
        print(f"\nPhase 307 failed ({failures} checks).")
        return 1
    print("\nFacilityModel primary-variable extraction is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
