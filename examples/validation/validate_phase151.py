"""Phase 151: compositional facility switch diagnostics in reports."""

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


def validate_model_group_switch_diagnostics() -> int:
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
    failures += _check("group rate-limit switch records a diagnostic",
                       not ok
                       and switch.get("group") == "PROD"
                       and switch.get("type") == "rate_limit"
                       and switch.get("mode") == "orat")
    failures += _check("group switch diagnostic records old and new controls",
                       [c["type"] for c in switch.get("old_controls", [])] == ["bhp", "bhp"]
                       and [c["type"] for c in switch.get("new_controls", [])] == ["orat", "orat"]
                       and np.allclose([c["value"] for c in switch["new_controls"]], [-1.0e-5, -1.0e-5]))
    return failures


def validate_model_gecon_diagnostics() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GECON": [["PROD", 3.0e-5, np.inf, np.inf, np.inf, np.inf, "NONE", "NO", 0]],
        }
    )
    _set_production(state, qo=[-1.0e-5, -1.0e-5])
    ok = model._update_group_controls_from_state(state)
    switch = model.last_group_switches[0] if model.last_group_switches else {}

    failures = 0
    failures += _check("GECON shut-in records a group diagnostic",
                       not ok
                       and switch.get("group") == "PROD"
                       and switch.get("type") == "economics"
                       and switch.get("mode") == "shut"
                       and switch.get("reason") == "min_oil_rate")
    failures += _check("GECON diagnostic records affected producers",
                       switch.get("wells") == ["P1", "P2"]
                       and not model.wells[1].status
                       and not model.wells[2].status)
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
        self.last_well_switches = [
            {"well": "P1", "index": 0, "from": "bhp", "to": "orat", "reason": "orat"}
        ]
        self.last_group_switches = [
            {"group": "PROD", "type": "rate_limit", "mode": "orat", "wells": ["P1"]}
        ]
        return state + dt, True, 3


def validate_schedule_reports_facility_switches() -> int:
    model = ReportingModel()
    schedule = simple_schedule([1.0 * DAY], control=[ScheduleControl(group_controls=[])])
    result = run_schedule(model, 0.0, schedule)
    report = result.reports[0]
    model.last_group_switches[0]["group"] = "MUTATED"

    failures = 0
    failures += _check("schedule report exposes facility-change flag",
                       report["FacilityChanged"] is True)
    failures += _check("schedule report carries well and group switches",
                       report["WellSwitches"][0]["to"] == "orat"
                       and report["GroupSwitches"][0]["group"] == "PROD")
    failures += _check("schedule report deep-copies facility diagnostics",
                       report["GroupSwitches"][0]["group"] == "PROD")
    return failures


def main() -> int:
    print("\nPhase 151: compositional facility switch diagnostics in reports")
    failures = validate_model_group_switch_diagnostics()
    failures += validate_model_gecon_diagnostics()
    failures += validate_schedule_reports_facility_switches()
    if failures:
        print(f"\nPhase 151 failed ({failures} checks).")
        return 1
    print("\nFacility control switches are now visible in compositional reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
