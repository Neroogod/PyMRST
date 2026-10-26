"""Phase 185: MRST-style selectFacilityFromDeck VFP table parity."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import (  # noqa: E402
    FacilityModel,
    VFPTable,
    selectFacilityFromDeck,
    select_facility_from_deck,
    select_facility_from_deck_ad,
    vfp_tables_from_deck,
)
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _producer_table(offset: float = 0.0) -> dict:
    flo = np.array([0.0, 10.0], dtype=float)
    thp = np.array([100.0, 200.0], dtype=float)
    wfr = np.array([0.0, 1.0], dtype=float)
    gfr = np.array([0.0, 2.0], dtype=float)
    q = np.zeros((2, 2, 2, 2), dtype=float)
    for i, f in enumerate(flo):
        for j, p in enumerate(thp):
            for k, wr in enumerate(wfr):
                for m, gr in enumerate(gfr):
                    q[i, j, k, m] = offset + p + 2.0 * f + 5.0 * wr + 7.0 * gr
    return {
        "FLO": flo,
        "THP": thp,
        "WFR": wfr,
        "GFR": gfr,
        "Q": q,
        "FLOID": "LIQ",
        "WFRID": "WFR",
        "GFRID": "GOR",
    }


def _injector_table(offset: float = 0.0) -> dict:
    flo = np.array([0.0, 10.0], dtype=float)
    thp = np.array([100.0, 200.0], dtype=float)
    bhp = np.zeros((2, 2), dtype=float)
    for i, f in enumerate(flo):
        for j, p in enumerate(thp):
            bhp[i, j] = offset + p + 3.0 * f
    return {"FLO": flo, "THP": thp, "BHP": bhp, "FLOID": "RATE"}


def _deck(*controls) -> dict:
    return {"SCHEDULE": {"control": list(controls)}, "RUNSPEC": {"VFPIDIMS": [1]}}


def _well(name: str, *, sign: int, vfp_index: int) -> Well:
    return Well(
        cells=np.array([0]),
        type="thp",
        val=150.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=sign,
        compi=np.array([0.0, 1.0, 0.0]),
        refDepth=0.0,
        lims={},
        vfp_index=vfp_index,
    )


class ModelShell:
    def __init__(self, wells, facility=None):
        self.wells = list(wells)
        self.FacilityModel = facility


def validate_table_collection() -> int:
    inj = _injector_table(1000.0)
    prod = _producer_table(2000.0)
    injector, producer = vfp_tables_from_deck(_deck({"VFPINJ": {2: inj}}, {"VFPPROD": [prod]}))

    failures = 0
    failures += _check("selectFacilityFromDeck collector preserves sparse injector table numbers",
                       len(injector) == 2 and injector[0] is None and isinstance(injector[1], VFPTable))
    failures += _check("selectFacilityFromDeck collector imports producer table lists",
                       len(producer) == 1 and isinstance(producer[0], VFPTable))
    failures += _check("collected injector table interpolates",
                       np.isclose(injector[1].evaluate_bhp(5.0, 150.0), 1000.0 + 150.0 + 15.0))
    failures += _check("collected producer table interpolates",
                       np.isclose(producer[0].evaluate_bhp(5.0, 150.0, 0.5, 1.0),
                                  2000.0 + 150.0 + 10.0 + 2.5 + 7.0))
    return failures


def validate_new_facility_selection() -> int:
    prod = _producer_table(300.0)
    well = _well("P1", sign=-1, vfp_index=1)
    model = ModelShell([well])
    facility = select_facility_from_deck_ad(_deck({"VFPPROD": [prod]}), model)

    failures = 0
    failures += _check("selector creates a native FacilityModel",
                       isinstance(facility, FacilityModel))
    failures += _check("selector attaches the facility to the model",
                       model.FacilityModel is facility)
    failures += _check("selector binds well vfp_index during construction",
                       facility.wells[0].vfp_table is facility.vfp_tables_producer[0])
    failures += _check("MATLAB-style alias selectFacilityFromDeck is exported",
                       selectFacilityFromDeck is select_facility_from_deck)
    return failures


def validate_existing_facility_update_and_warning() -> int:
    prod_a = _producer_table(10.0)
    prod_b = _producer_table(20.0)
    well = _well("P1", sign=-1, vfp_index=1)
    facility = FacilityModel([well], vfp_tables_producer=[VFPTable.from_deck(prod_a)])
    model = ModelShell([], facility=facility)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = select_facility_from_deck_ad(_deck({"VFPPROD": [prod_a]}, {"VFPPROD": [prod_b]}), model)

    failures = 0
    failures += _check("selector reuses an existing facility model",
                       returned is facility)
    failures += _check("existing facility receives deck VFP tables",
                       np.isclose(facility.vfp_tables_producer[0].evaluate_bhp(5.0, 150.0, 0.5, 1.0),
                                  20.0 + 150.0 + 10.0 + 2.5 + 7.0))
    failures += _check("overwriting VFP tables warns once",
                       len(caught) == 1 and "overwriting VFP tables" in str(caught[0].message))
    failures += _check("existing facility rebinds selected VFP table",
                       facility.wells[0].vfp_table is facility.vfp_tables_producer[0])
    return failures


def main() -> int:
    print("\nPhase 185: MRST-style selectFacilityFromDeck VFP table parity")
    failures = validate_table_collection()
    failures += validate_new_facility_selection()
    failures += validate_existing_facility_update_and_warning()
    if failures:
        print(f"\nPhase 185 failed ({failures} checks).")
        return 1
    print("\nselectFacilityFromDeck now imports deck VFP tables into native facilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
