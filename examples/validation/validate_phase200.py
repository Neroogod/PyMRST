"""Phase 200: WGRUPCON allocation provenance reaches diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.facility import (  # noqa: E402
    facility_group_controls_from_deck_control,
)
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _model():
    g = compute_geometry(cart_grid([6, 1, 1], [120.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    for cell, name in [(3, "P1"), (4, "P2"), (5, "P3")]:
        wells = add_well(wells, g, rock, np.array([cell]), well_type="bhp",
                         val=55.0 * BARSA, compi=np.array([0.0, 1.0]),
                         sign=-1, name=name)
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def validate_imported_wgrupcon_allocation_metadata() -> int:
    group = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"], ["P3", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 8.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "WGRUPCON": [
                ["P1", "YES", 1.0, "PROD", 2.0],
                ["P2", "NO", 99.0, "PROD", 1.0],
                ["P3", "YES", 3.0, "PROD", 0.5],
            ],
        }
    )[0]
    alloc = group.extra.get("WGRUPCON_ALLOCATION", {})

    failures = 0
    failures += _check("WGRUPCON allocation metadata is attached",
                       alloc.get("source_keyword") == "WGRUPCON"
                       and alloc.get("allocation") == "fraction")
    failures += _check("WGRUPCON allocation records active/inactive guides",
                       [row["well"] for row in alloc["members"]] == ["P1", "P2", "P3"]
                       and [row["used"] for row in alloc["members"]] == [True, False, True])
    failures += _check("WGRUPCON allocation stores effective fractions",
                       np.allclose(alloc["fractions"], [2.0, 0.0, 1.5]))
    return failures


def validate_wgrupcon_runtime_diagnostic() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"], ["P3", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 8.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "WGRUPCON": [
                ["P1", "YES", 1.0, "PROD", 2.0],
                ["P2", "NO", 99.0, "PROD", 1.0],
                ["P3", "YES", 3.0, "PROD", 0.5],
            ],
        }
    )
    state.qOs[1:] = [-5.0e-5, -5.0e-5, -5.0e-5]
    state.qGs[1:] = 0.0
    ok = model._update_group_controls_from_state(state)
    group = model.group_controls[0]
    report = model.last_group_switches[-1] if model.last_group_switches else {}
    alloc = group.extra.get("WGRUPCON_ALLOCATION", {})

    failures = 0
    failures += _check("WGRUPCON-driven group limit still redistributes", not ok)
    failures += _check("runtime report identifies WGRUPCON allocation source",
                       report.get("source_keyword") == "GCONPROD"
                       and report.get("allocation_source_keyword") == "WGRUPCON"
                       and report.get("allocation") == "fraction")
    failures += _check("runtime report carries WGRUPCON guide provenance",
                       report.get("allocation_guided_wells") == ["P1", "P3"]
                       and np.allclose(report.get("allocation_effective_guide_rates"), [2.0, 1.5])
                       and np.allclose(report.get("allocation_fractions"), [2.0, 0.0, 1.5]))
    failures += _check("allocation metadata is stamped with runtime action",
                       alloc.get("last_violation") == "orat"
                       and alloc.get("last_action") == "redistribute"
                       and alloc.get("last_affected_wells") == ["P1", "P2", "P3"])
    failures += _check("WGRUPCON fractions still drive targets",
                       np.allclose(
                           [model.wells[1].val, model.wells[2].val, model.wells[3].val],
                           [-4.571428571428571e-5, -0.0, -3.4285714285714284e-5],
                       ))
    return failures


def main() -> int:
    print("\nPhase 200: WGRUPCON allocation diagnostics")
    failures = validate_imported_wgrupcon_allocation_metadata()
    failures += validate_wgrupcon_runtime_diagnostic()
    if failures:
        print(f"\nPhase 200 failed ({failures} checks).")
        return 1
    print("\nWGRUPCON-derived allocation now reaches runtime diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
