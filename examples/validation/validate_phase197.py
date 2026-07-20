"""Phase 197: GCONPROD RESV imports as a native group VRAT limit."""

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
    g = compute_geometry(cart_grid([5, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = None
    for cell, name in [(3, "P1"), (4, "P2")]:
        wells = add_well(
            wells,
            g,
            rock,
            np.array([cell]),
            well_type="bhp",
            val=55.0 * BARSA,
            compi=np.array([0.5, 0.5]),
            sign=-1,
            name=name,
        )
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(), temperature=350.0, wells=wells)
    state = model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])
    state.qOs[:] = [-5.0e-5, -5.0e-5]
    state.qGs[:] = [-2.0e-5, -2.0e-5]
    return model, state


def _gconprod_resv_row(limit: float, exceed: str = "NONE") -> list[object]:
    row = ["FIELD", "NONE", np.inf, np.inf, np.inf, np.inf, exceed, "YES"]
    row += [np.inf, "", "NONE", "NONE", "NONE", limit]
    row += [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, "NONE"]
    return row


def validate_positional_gconprod_resv_limit() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "FIELD"], ["P2", "FIELD"]],
            "GCONPROD": [_gconprod_resv_row(8.0e-5)],
        }
    )
    group = model.group_controls[0]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[-1] if model.last_group_switches else {}
    meta = group.extra["GCONPROD"][0]

    failures = 0
    failures += _check("positional GCONPROD RESV becomes native vrat limit",
                       group.lims.get("vrat") == 8.0e-5)
    failures += _check("RESV/VRAT violation triggers redistribution",
                       not ok and report.get("mode") == "vrat")
    failures += _check("diagnostic keeps GCONPROD source and RESV context",
                       report.get("source_keyword") == "GCONPROD"
                       and report.get("control") == "none"
                       and meta.get("last_violation") == "vrat")
    failures += _check("member wells switch to vrat with allocated targets",
                       [w.type for w in model.wells] == ["vrat", "vrat"]
                       and np.allclose([w.val for w in model.wells], [-4.0e-5, -4.0e-5]))
    failures += _check("vrat state guess resets total oil+gas target",
                       np.allclose(state.qOs + state.qGs, [-4.0e-5, -4.0e-5]))
    return failures


def validate_mapping_gconprod_resv_limit() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "FIELD"], ["P2", "FIELD"]],
            "GCONPROD": [
                {
                    "group": "FIELD",
                    "control": "RESV",
                    "resv": 7.0e-5,
                    "exceed_action": "NONE",
                    "status": "YES",
                }
            ],
        }
    )
    group = model.group_controls[0]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[-1] if model.last_group_switches else {}

    failures = 0
    failures += _check("mapping GCONPROD RESV becomes native vrat limit",
                       group.lims.get("vrat") == 7.0e-5)
    failures += _check("mapping RESV row is selected for diagnostics",
                       not ok
                       and report.get("mode") == "vrat"
                       and group.extra["GCONPROD"][0].get("last_action") == "redistribute")
    failures += _check("mapping RESV row preserves raw MRST slot",
                       np.isclose(group.extra["GCONPROD"][0]["raw"][13], 7.0e-5))
    return failures


def main() -> int:
    print("\nPhase 197: GCONPROD RESV/VRAT group limit")
    failures = validate_positional_gconprod_resv_limit()
    failures += validate_mapping_gconprod_resv_limit()
    if failures:
        print(f"\nPhase 197 failed ({failures} checks).")
        return 1
    print("\nGCONPROD RESV now imports as an enforceable native group VRAT limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
