"""Phase 305: MRST FacilityModel query helper parity."""

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


def _well(name: str, cells, *, status: bool = True, sign: int = 1) -> Well:
    cells = np.asarray(cells, dtype=np.int64)
    return Well(
        cells=cells,
        type="rate",
        val=1.0,
        r=np.full(cells.size, 0.1),
        dir="z",
        WI=np.ones(cells.size),
        dZ=np.zeros(cells.size),
        name=name,
        compi=np.array([1.0, 0.0, 0.0]),
        refDepth=0.0,
        sign=sign,
        status=status,
    )


def validate_facility_model_queries() -> int:
    wells = [
        _well("I1", [0, 2], status=True, sign=1),
        _well("P1", [5], status=False, sign=-1),
        _well("P2", [7, 8, 9], status=True, sign=-1),
    ]
    facility = FacilityModel(wells)
    well_sol = [
        {"status": True},
        {"status": True},
        {"status": False},
    ]

    failures = 0
    failures += _check("getNumberOfWells mirrors facility well count",
                       facility.get_number_of_wells() == 3
                       and facility.getNumberOfWells() == 3
                       and facility.n_wells == 3)
    failures += _check("getWellStruct returns selected wells in facility order",
                       [w.name for w in facility.get_well_struct([2, 0])] == ["P2", "I1"]
                       and [w.name for w in facility.getWellStruct(slice(1, None))] == ["P1", "P2"])
    failures += _check("status mask combines W.status with wellSol.status",
                       np.array_equal(facility.get_well_status_mask(well_sol),
                                      np.array([True, False, False]))
                       and np.array_equal(facility.getWellStatusMask(),
                                          np.array([True, False, True])))
    failures += _check("active well count and indices are MRST-shaped queries",
                       facility.get_number_of_active_wells(well_sol) == 1
                       and facility.getNumberOfActiveWells(well_sol) == 1
                       and np.array_equal(facility.get_indices_of_active_wells(well_sol), [0]))
    failures += _check("getWellCells concatenates perforations regardless of status",
                       np.array_equal(facility.get_well_cells(), [0, 2, 5, 7, 8, 9])
                       and np.array_equal(facility.getWellCells([2]), [7, 8, 9]))
    cells, p2w = facility.get_active_well_cells(well_sol)
    failures += _check("getActiveWellCells returns cells and perforation-to-well map",
                       np.array_equal(cells, [0, 2])
                       and np.array_equal(p2w, [0, 0]))
    failures += _check("primary variable names follow MRST q<phase>s plus bhp order",
                       facility.get_primary_variable_names(["W", "O", "G"]) == ["qWs", "qOs", "qGs", "bhp"]
                       and facility.getPrimaryVariableNames(["O", "G"]) == ["qOs", "qGs", "bhp"])
    return failures


def main() -> int:
    print("\nPhase 305: MRST FacilityModel query helper parity")
    failures = validate_facility_model_queries()
    if failures:
        print(f"\nPhase 305 failed ({failures} checks).")
        return 1
    print("\nFacilityModel MRST-style query helpers are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
