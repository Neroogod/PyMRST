"""Phase 333: MRST FacilityModel constructor overload parity."""

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


def _well() -> Well:
    return Well(
        cells=np.array([0]),
        type="rate",
        val=1.0e-5,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0e-12]),
        dZ=np.array([0.0]),
        name="W1",
        compi=np.array([1.0, 0.0]),
        refDepth=0.0,
        sign=1,
        status=True,
    )


def validate_reservoir_model_constructor() -> int:
    backend = SimpleNamespace(name="backend")
    reservoir = SimpleNamespace(
        AutoDiffBackend=backend,
        getPhaseNames=lambda: ("W", "O"),
        getNumberOfComponents=lambda: 2,
    )
    facility = FacilityModel(reservoir)

    failures = 0
    failures += _check("FacilityModel(reservoirModel) stores an empty well list",
                       facility.getNumberOfWells() == 0)
    failures += _check("FacilityModel(reservoirModel) attaches ReservoirModel and AutoDiffBackend",
                       facility.ReservoirModel is reservoir
                       and facility.reservoir_model is reservoir
                       and facility.AutoDiffBackend is backend
                       and facility.auto_diff_backend is backend)
    failures += _check("reservoir-model constructor feeds phase-name primary variables",
                       facility.getBasicPrimaryVariableNames() == ["qWs", "qOs", "bhp"])
    return failures


def validate_existing_well_constructors() -> int:
    well = _well()
    single = FacilityModel(well)
    listed = FacilityModel([well])
    empty = FacilityModel(None)

    failures = 0
    failures += _check("single Well constructor remains supported",
                       single.getNumberOfWells() == 1 and single.wells[0] is well)
    failures += _check("sequence-of-wells constructor remains supported",
                       listed.getNumberOfWells() == 1 and listed.wells[0] is well)
    failures += _check("None constructor creates an empty facility model",
                       empty.getNumberOfWells() == 0)
    return failures


def main() -> int:
    print("\nPhase 333: MRST FacilityModel constructor overload parity")
    failures = 0
    failures += validate_reservoir_model_constructor()
    failures += validate_existing_well_constructors()
    if failures:
        print(f"\nPhase 333 failed ({failures} checks).")
        return 1
    print("\nFacilityModel constructor overload parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
