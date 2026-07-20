"""Phase 191: GCONPROD redistribution diagnostics."""

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
    g = compute_geometry(cart_grid([4, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    for cell, name in [(2, "P1"), (3, "P2")]:
        wells = add_well(wells, g, rock, np.array([cell]), well_type="bhp",
                         val=55.0 * BARSA, compi=np.array([0.0, 1.0]),
                         sign=-1, name=name)
        wells[-1].components = np.array([0.5, 0.3, 0.2])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def validate_gconprod_redistribution_marks_metadata() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 2.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
        }
    )
    state.qOs[1:] = np.array([-2.0e-5, -2.0e-5])
    ok = model._update_group_controls_from_state(state)
    switch = model.last_group_switches[0] if model.last_group_switches else {}
    meta = model.group_controls[0].extra["GCONPROD"][0]

    failures = 0
    failures += _check("GCONPROD NONE still redistributes violated producer limit",
                       not ok
                       and switch.get("action") == "redistribute"
                       and switch.get("mode") == "orat")
    failures += _check("GCONPROD redistribution diagnostic records MRST context",
                       switch.get("source_keyword") == "GCONPROD"
                       and switch.get("control") == "orat"
                       and switch.get("exceed_action") == "NONE")
    failures += _check("GCONPROD metadata records ordinary redistribution",
                       meta.get("last_violation") == "orat"
                       and meta.get("last_action") == "redistribute"
                       and meta.get("last_exceed_action") == "NONE")
    failures += _check("GCONPROD redistribution targets remain unchanged",
                       np.allclose([w.val for w in model.wells[1:]], [-1.0e-5, -1.0e-5])
                       and [w.type for w in model.wells[1:]] == ["orat", "orat"])
    return failures


def validate_gconprod_shut_path_still_marks_shut_action() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 2.0e-5, np.inf, np.inf, np.inf, "SHUT", "YES"]],
        }
    )
    state.qOs[1:] = np.array([-2.0e-5, -2.0e-5])
    ok = model._update_group_controls_from_state(state)
    meta = model.group_controls[0].extra["GCONPROD"][0]

    failures = 0
    failures += _check("GCONPROD SHUT path remains active",
                       not ok
                       and not model.group_controls[0].status
                       and not model.wells[1].status
                       and not model.wells[2].status)
    failures += _check("GCONPROD SHUT metadata remains explicit",
                       meta.get("last_violation") == "orat"
                       and meta.get("last_exceed_action") == "SHUT"
                       and meta.get("last_action") == "shut")
    return failures


def main() -> int:
    print("\nPhase 191: GCONPROD redistribution diagnostics")
    failures = validate_gconprod_redistribution_marks_metadata()
    failures += validate_gconprod_shut_path_still_marks_shut_action()
    if failures:
        print(f"\nPhase 191 failed ({failures} checks).")
        return 1
    print("\nGCONPROD redistribution now leaves MRST-style diagnostic context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
