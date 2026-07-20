"""Phase 306: MRST FacilityModel well-variable map parity."""

from __future__ import annotations

from pathlib import Path
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


def _well(name: str, compi, *, status: bool = True) -> Well:
    return Well(
        cells=np.array([0], dtype=np.int64),
        type="rate",
        val=1.0,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.asarray(compi, dtype=float),
        refDepth=0.0,
        sign=1,
        status=status,
    )


def validate_well_variable_map() -> int:
    facility = FacilityModel([
        _well("I1", [1.0, 0.0, 0.0], status=True),
        _well("P1", [0.0, 1.0, 0.0], status=False),
        _well("P2", [0.2, 0.3, 0.5], status=True),
    ])
    all_active_from_well_sol = [{"status": True}, {"status": True}, {"status": True}]
    only_last_from_well_sol = [{"status": False}, {"status": True}, {"status": True}]

    failures = 0
    failures += _check("qWs ownership maps one row per active well",
                       np.array_equal(facility.get_well_variable_map("qWs", all_active_from_well_sol), [0, 1])
                       and np.array_equal(facility.getWellVariableMap("bhp", all_active_from_well_sol), [0, 1]))
    failures += _check("wellSol.status removes inactive rows before ownership numbering",
                       np.array_equal(facility.get_well_variable_map("qOs", only_last_from_well_sol), [0]))
    failures += _check("well_massfractions expands to ncomp-1 rows per active well",
                       np.array_equal(facility.get_well_variable_map("well_massfractions", all_active_from_well_sol),
                                      [0, 0, 1, 1]))
    failures += _check("unknown variable has an empty ownership map",
                       facility.get_well_variable_map("not_a_well_variable").size == 0)
    return failures


def main() -> int:
    print("\nPhase 306: MRST FacilityModel well-variable map parity")
    failures = validate_well_variable_map()
    if failures:
        print(f"\nPhase 306 failed ({failures} checks).")
        return 1
    print("\nFacilityModel well-variable ownership maps are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
