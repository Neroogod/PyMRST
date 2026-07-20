"""Phase 315: MRST FacilityModel setup/validation hooks parity."""

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


def _well(name: str, *, val: float = 1.0, status: bool = True, sign: int = 1, vfp_index=None) -> Well:
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
        sign=sign,
        status=status,
        vfp_index=vfp_index,
    )


def validate_setup_and_model_hooks() -> int:
    backend = object()
    reservoir = SimpleNamespace(AutoDiffBackend=backend)
    facility = FacilityModel([_well("I1")])

    returned = facility.setReservoirModel(reservoir)
    grouped = facility.setupStateFunctionGroupings()
    removed = facility.removeStateFunctionGroupings()

    refreshed = [_well("I1", val=2.0)]
    setup_returned = facility.setupWells(refreshed)
    setup_ok = (
        setup_returned is facility
        and len(facility.wells) == 1
        and np.isclose(facility.wells[0].val, 2.0)
    )

    forces = {"W": [_well("I1", val=3.0)]}
    validate_returned = facility.validateModel(forces)
    validate_ok = (
        validate_returned is facility
        and np.isclose(facility.wells[0].val, 3.0)
    )

    wellmodels = [SimpleNamespace(tag="wm")]
    facility.setup_wells([_well("I1", val=4.0)], wellmodels=wellmodels)

    failures = 0
    failures += _check("setReservoirModel stores reservoir and autodiff backend",
                       returned is facility
                       and facility.ReservoirModel is reservoir
                       and facility.reservoir_model is reservoir
                       and facility.AutoDiffBackend is backend
                       and facility.auto_diff_backend is backend)
    failures += _check("state-function grouping hooks are chainable",
                       grouped is facility
                       and removed is facility
                       and facility.FacilityFlowDiscretization is None)
    facility.setupStateFunctionGroupings()
    failures += _check("setupStateFunctionGroupings creates explicit placeholder storage",
                       isinstance(facility.FacilityFlowDiscretization, dict)
                       and facility.FacilityFlowDiscretization["model"] is facility)
    facility.removeStateFunctionGroupings()
    failures += _check("setupWells refreshes direct well list without changing count", setup_ok)
    failures += _check("validateModel consumes driving-force W block", validate_ok)
    failures += _check("explicit wellmodels are retained in MRST-style slots",
                       facility.WellModels == wellmodels
                       and facility.well_models == wellmodels)

    try:
        facility.setupWells([_well("I1"), _well("P1")])
    except AssertionError:
        count_change_fails = True
    else:
        count_change_fails = False
    failures += _check("setupWells rejects count changes during simulation", count_change_fails)

    try:
        FacilityModel([_well("I1")]).setupWells([
            _well("I1", val=np.array([1.0, 2.0])),
        ])
    except AssertionError:
        bad_control_fails = True
    else:
        bad_control_fails = False
    failures += _check("setupWells rejects active wells with vector controls", bad_control_fails)
    return failures


def main() -> int:
    print("\nPhase 315: MRST FacilityModel setup/validation hooks parity")
    failures = validate_setup_and_model_hooks()
    if failures:
        print(f"\nPhase 315 failed ({failures} checks).")
        return 1
    print("\nFacilityModel setup and validation hooks are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
