"""Phase 189: GCONPROD shut-style exceed actions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    run_schedule,
    simple_schedule,
)
from pymrst.modules.compositional.facility import (  # noqa: E402
    facility_group_controls_from_deck_control,
)
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI  # noqa: E402


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


def _set_production(state, *, qo, qg=0.0) -> None:
    state.qOs[1:] = np.asarray(qo, dtype=float)
    state.qGs[1:] = np.asarray(qg if np.ndim(qg) else [qg, qg], dtype=float)


def validate_gconprod_shut_action_shuts_group() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 2.0e-5, np.inf, np.inf, np.inf, "SHUT", "YES"]],
        }
    )
    _set_production(state, qo=[-2.0e-5, -2.0e-5])
    ok = model._update_group_controls_from_state(state)
    switch = model.last_group_switches[0] if model.last_group_switches else {}
    meta = model.group_controls[0].extra["GCONPROD"][0]

    failures = 0
    failures += _check("GCONPROD SHUT exceed action triggers a group switch",
                       not ok
                       and switch.get("type") == "rate_limit"
                       and switch.get("mode") == "orat")
    failures += _check("GCONPROD SHUT shuts active producer wells",
                       not model.group_controls[0].status
                       and not model.wells[1].status
                       and not model.wells[2].status)
    failures += _check("GCONPROD SHUT zeros producer state rate guesses",
                       np.allclose(state.qOs[1:], 0.0)
                       and np.allclose(state.qGs[1:], 0.0))
    failures += _check("GCONPROD SHUT diagnostic records action and old controls",
                       switch.get("action") == "shut"
                       and switch.get("exceed_action") == "SHUT"
                       and [c["type"] for c in switch.get("old_controls", [])] == ["bhp", "bhp"]
                       and switch.get("new_controls") == [])
    failures += _check("GCONPROD metadata records the last shut violation",
                       meta.get("last_violation") == "orat"
                       and meta.get("last_exceed_action") == "SHUT")
    return failures


def validate_default_action_keeps_redistribution_behavior() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GCONPROD": [["PROD", "ORAT", 2.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
        }
    )
    _set_production(state, qo=[-2.0e-5, -2.0e-5])
    ok = model._update_group_controls_from_state(state)
    switch = model.last_group_switches[0] if model.last_group_switches else {}

    failures = 0
    failures += _check("GCONPROD NONE still triggers ordinary rate redistribution",
                       not ok
                       and all(w.status for w in model.wells[1:])
                       and model.group_controls[0].status)
    failures += _check("GCONPROD NONE keeps old per-well target behavior",
                       [w.type for w in model.wells[1:]] == ["orat", "orat"]
                       and np.allclose([w.val for w in model.wells[1:]], [-1.0e-5, -1.0e-5]))
    failures += _check("GCONPROD NONE diagnostic records redistribution",
                       switch.get("action") == "redistribute"
                       and switch.get("exceed_action") == "NONE"
                       and [c["type"] for c in switch.get("new_controls", [])] == ["orat", "orat"])
    return failures


class ReportingModel:
    def __init__(self):
        self.wells = []
        self.bc = None
        self.src = None
        self.separator_group = None
        self.group_controls = []
        self.last_well_switches = []
        self.last_group_switches = []

    def step(self, state, dt, bc=None, src=None):
        self.last_group_switches = [
            {
                "group": "PROD",
                "type": "rate_limit",
                "mode": "orat",
                "action": "shut",
                "exceed_action": "SHUT",
                "wells": ["P1"],
            }
        ]
        return state + dt, True, 2


def validate_schedule_report_carries_exceed_action() -> int:
    result = run_schedule(
        ReportingModel(),
        0.0,
        simple_schedule([1.0 * DAY], control=[ScheduleControl(group_controls=[])]),
    )
    report = result.reports[0]

    failures = 0
    failures += _check("schedule report carries GCONPROD exceed action diagnostic",
                       report["GroupSwitches"][0]["action"] == "shut"
                       and report["GroupSwitches"][0]["exceed_action"] == "SHUT")
    return failures


def main() -> int:
    print("\nPhase 189: GCONPROD shut-style exceed actions")
    failures = validate_gconprod_shut_action_shuts_group()
    failures += validate_default_action_keeps_redistribution_behavior()
    failures += validate_schedule_report_carries_exceed_action()
    if failures:
        print(f"\nPhase 189 failed ({failures} checks).")
        return 1
    print("\nGCONPROD shut-style exceed actions now shut producing groups conservatively.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
