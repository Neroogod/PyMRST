"""Phase 331: MRST FacilityModel property alias parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.facility_model import (  # noqa: E402
    FacilityLinearizedProblem,
    FacilityModel,
    __all__ as facility_model_all,
)
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, sign: int, vfp_index: int | None = None) -> Well:
    return Well(
        cells=np.array([0]),
        type="rate",
        val=1.0e-5,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0e-12]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.array([1.0, 0.0]),
        refDepth=0.0,
        sign=sign,
        status=True,
        vfp_index=vfp_index,
    )


def validate_primary_variable_set_alias() -> int:
    facility = FacilityModel([_well("P1", -1)])
    facility.primaryVariableSet = "bhp"
    names = facility.getBasicPrimaryVariableNames()

    failures = 0
    failures += _check("primaryVariableSet writes through to primary_variable_set",
                       facility.primary_variable_set == "bhp"
                       and facility.primaryVariableSet == "bhp")
    failures += _check("MRST primaryVariableSet spelling controls primary-variable names",
                       names == ["bhp"])
    facility.primaryVariableSet = "bhp_massfractions"
    failures += _check("primaryVariableSet supports MRST bhp_massfractions mode",
                       facility.getBasicPrimaryVariableNames() == ["bhp", "well_massfractions"])
    failures += _check("setWellTargets defaults to MRST true",
                       facility.setWellTargets is True)
    return failures


def validate_vfp_table_aliases() -> int:
    injector = _well("I1", 1)
    inj_table = SimpleNamespace(name="injector-table")
    injector_facility = FacilityModel([injector])
    injector.vfp_index = 1
    injector_facility.VFPTablesInjector = [inj_table]

    producer = _well("P1", -1)
    prod_table = SimpleNamespace(name="producer-table")
    producer_facility = FacilityModel([producer])
    producer.vfp_index = 1
    producer_facility.VFPTablesProducer = [prod_table]

    failures = 0
    failures += _check("VFPTablesInjector synchronizes with vfp_tables_injector",
                       injector_facility.vfp_tables_injector == [inj_table]
                       and injector_facility.VFPTablesInjector == [inj_table]
                       and injector.vfp_table is inj_table)
    failures += _check("VFPTablesProducer synchronizes with vfp_tables_producer",
                       producer_facility.vfp_tables_producer == [prod_table]
                       and producer_facility.VFPTablesProducer == [prod_table]
                       and producer.vfp_table is prod_table)
    return failures


def validate_module_exports() -> int:
    problem = FacilityLinearizedProblem([], [], [], [], {})

    failures = 0
    failures += _check("FacilityLinearizedProblem is exported from facility_model.__all__",
                       "FacilityLinearizedProblem" in facility_model_all)
    failures += _check("FacilityLinearizedProblem remains constructible from module import",
                       problem.equations == []
                       and problem.equationNames == []
                       and problem.primaryVariables == [])
    return failures


def main() -> int:
    print("\nPhase 331: MRST FacilityModel property alias parity")
    failures = 0
    failures += validate_primary_variable_set_alias()
    failures += validate_vfp_table_aliases()
    failures += validate_module_exports()
    if failures:
        print(f"\nPhase 331 failed ({failures} checks).")
        return 1
    print("\nFacilityModel property alias parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
