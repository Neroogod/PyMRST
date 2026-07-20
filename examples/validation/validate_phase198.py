"""Phase 198: GCONPROD late-column context is named and reported."""

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
    wells = None
    for cell, name in [(2, "P1"), (3, "P2")]:
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
    state.qGs[:] = [-1.0e-5, -1.0e-5]
    return model, state


def validate_named_metadata_and_report_context() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "FIELD"], ["P2", "FIELD"]],
            "GCONPROD": [
                {
                    "group": "FIELD",
                    "control": "ORAT",
                    "orat": 7.0e-5,
                    "exceed_action": "NONE",
                    "status": "YES",
                    "guide_rate": 1.1e-4,
                    "guide_phase": "OIL",
                    "damping": "YES",
                    "free_group_control": "NO",
                    "region": "FIELDREG",
                    "voidage": 2.2e-4,
                    "wet_gas": 3.3e-4,
                    "calorific": 4.4e-4,
                    "crat": 5.5e-4,
                    "trgp": 123.0 * BARSA,
                    "procedure": "RATE",
                }
            ],
        }
    )
    group = model.group_controls[0]
    ok = model._update_group_controls_from_state(state)
    report = model.last_group_switches[-1] if model.last_group_switches else {}
    meta = group.extra["GCONPROD"][0]

    failures = 0
    failures += _check("late GCONPROD columns are named metadata",
                       meta["damping"] == "YES"
                       and meta["free_group_control"] == "NO"
                       and meta["region"] == "FIELDREG"
                       and meta["procedure"] == "RATE")
    failures += _check("numeric late GCONPROD columns are preserved",
                       np.isclose(meta["voidage"], 2.2e-4)
                       and np.isclose(meta["wet_gas"], 3.3e-4)
                       and np.isclose(meta["calorific"], 4.4e-4)
                       and np.isclose(meta["crat"], 5.5e-4)
                       and np.isclose(meta["trgp"], 123.0 * BARSA))
    failures += _check("violation report carries non-native GCONPROD context",
                       not ok
                       and report.get("source_keyword") == "GCONPROD"
                       and report.get("mode") == "orat"
                       and report.get("damping") == "YES"
                       and report.get("free_group_control") == "NO"
                       and report.get("region") == "FIELDREG"
                       and report.get("procedure") == "RATE"
                       and np.isclose(report.get("trgp"), 123.0 * BARSA))
    failures += _check("metadata is stamped with the last reported action",
                       meta.get("last_violation") == "orat"
                       and meta.get("last_action") == "redistribute"
                       and meta.get("last_exceed_action") == "NONE")
    return failures


def validate_shut_report_stamps_action() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "FIELD"], ["P2", "FIELD"]],
            "GCONPROD": [["FIELD", "ORAT", 7.0e-5, np.inf, np.inf, np.inf, "SHUT", "YES"]],
        }
    )
    group = model.group_controls[0]
    ok = model._update_group_controls_from_state(state)
    meta = group.extra["GCONPROD"][0]

    failures = 0
    failures += _check("GCONPROD SHUT still shuts the violating group",
                       not ok and not group.status and not any(w.status for w in model.wells))
    failures += _check("GCONPROD SHUT stamps last action as shut",
                       meta.get("last_violation") == "orat"
                       and meta.get("last_action") == "shut"
                       and meta.get("last_exceed_action") == "SHUT")
    return failures


def main() -> int:
    print("\nPhase 198: GCONPROD late-column diagnostics")
    failures = validate_named_metadata_and_report_context()
    failures += validate_shut_report_stamps_action()
    if failures:
        print(f"\nPhase 198 failed ({failures} checks).")
        return 1
    print("\nGCONPROD late MRST columns now survive as named metadata and runtime diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
