"""Phase 183: VFP-index binding and compositional THP/VFP control parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel  # noqa: E402
from pymrst.modules.ad_core.models.facilities.vfp import VFPTable  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402
from pymrst.utils.adi import value  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, *, sign: int, control: str = "thp", vfp_index: int = 1) -> Well:
    return Well(
        cells=np.array([0]),
        type=control,
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


def _table(offset: float = 0.0) -> VFPTable:
    flo = np.array([0.0, 10.0], dtype=float)
    thp = np.array([100.0, 200.0], dtype=float)
    wfr = np.array([0.0, 1.0], dtype=float)
    gfr = np.array([0.0, 2.0], dtype=float)
    t = np.zeros((2, 2, 2, 2), dtype=float)
    for i, f in enumerate(flo):
        for j, p in enumerate(thp):
            for k, wr in enumerate(wfr):
                for m, gr in enumerate(gfr):
                    t[i, j, k, m] = offset + p + 2.0 * f + 5.0 * wr + 7.0 * gr
    return VFPTable(flo, thp, [wfr, gfr], ["WFR", "GFR"], t, flow_type="liquid")


def validate_generic_facility_binding() -> int:
    inj_table = _table(1000.0)
    prod_table = _table(2000.0)
    wells = [_well("INJ", sign=1), _well("PROD", sign=-1)]
    facility = FacilityModel(
        wells,
        vfp_tables_injector=[inj_table],
        vfp_tables_producer=[prod_table],
    )

    failures = 0
    failures += _check("generic FacilityModel binds injector vfp_index to injector table",
                       facility.wells[0].vfp_table is inj_table)
    failures += _check("generic FacilityModel binds producer vfp_index to producer table",
                       facility.wells[1].vfp_table is prod_table)
    try:
        FacilityModel([_well("BAD", sign=-1, vfp_index=2)], vfp_tables_producer=[prod_table])
    except IndexError:
        missing_failed = True
    else:
        missing_failed = False
    failures += _check("missing producer VFP index fails loudly",
                       missing_failed)
    return failures


def _compositional_model(table: VFPTable) -> OverallCompositionModel:
    g = compute_geometry(cart_grid([1, 1, 1], [10.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    well = _well("PROD", sign=-1)
    return OverallCompositionModel(
        g,
        rock,
        simple_mixture(),
        temperature=350.0,
        wells=[well],
        rho_s=(700.0, 100.0),
        vfp_tables_producer=[table],
    )


def validate_compositional_vfp_control() -> int:
    table = _table()
    model = _compositional_model(table)
    well = model.wells[0]
    qo = np.array([3.0])
    qg = np.array([2.0])
    qw = np.array([1.0])
    bhp = np.array([200.0])
    # liquid = water + oil = 4, WFR = 1/4, GFR = gas/oil = 2/3,
    # target BHP = THP 150 + 2*4 + 5*(1/4) + 7*(2/3).
    expected_bhp = 150.0 + 8.0 + 1.25 + 14.0 / 3.0
    eq = model._well_control_equation(well, qo, qg, bhp, qw_k=qw)
    nowater = model._well_control_equation(well, qo, qg, bhp)
    expected_nowater_bhp = 150.0 + 2.0 * 3.0 + 0.0 + 7.0 * (2.0 / 3.0)

    failures = 0
    failures += _check("compositional model binds producer VFP table",
                       well.vfp_table is table)
    failures += _check("water compositional THP equation uses table BHP",
                       np.allclose(value(eq), bhp - expected_bhp))
    failures += _check("no-water compositional THP equation uses oil/gas ratios",
                       np.allclose(value(nowater), bhp - expected_nowater_bhp))
    missing = _well("MISSING", sign=-1, vfp_index=1)
    missing.vfp_table = None
    try:
        model._well_control_equation(missing, qo, qg, bhp)
    except ValueError:
        missing_failed = True
    else:
        missing_failed = False
    failures += _check("THP control without bound VFP table fails loudly",
                       missing_failed)
    return failures


def main() -> int:
    print("\nPhase 183: VFP-index binding and compositional THP/VFP control parity")
    failures = validate_generic_facility_binding()
    failures += validate_compositional_vfp_control()
    if failures:
        print(f"\nPhase 183 failed ({failures} checks).")
        return 1
    print("\nVFP indices now bind to native tables and compositional THP controls use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
