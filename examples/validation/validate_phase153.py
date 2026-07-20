"""Phase 153: GECON end-run request reporting."""

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


def validate_model_records_gecon_end_run_request() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            "GECON": [["PROD", 3.0e-5, np.inf, np.inf, np.inf, np.inf, "NONE", "YES", 0]],
        }
    )
    state.qOs[1:] = [-1.0e-5, -1.0e-5]
    state.qGs[1:] = 0.0
    ok = model._update_group_controls_from_state(state)
    switch = model.last_group_switches[0] if model.last_group_switches else {}

    failures = 0
    failures += _check("GECON end-run economics violation is not ok",
                       not ok and model.last_end_run_requested is True)
    failures += _check("GECON end-run group name is recorded",
                       model.last_end_run_groups == ["PROD"])
    failures += _check("group switch diagnostic carries end-run flag",
                       switch.get("group") == "PROD"
                       and switch.get("type") == "economics"
                       and switch.get("end_run") is True)
    failures += _check("GECON metadata records end-run request",
                       model.group_controls[0].extra["GECON"].get("end_run_requested") is True)
    return failures


class EndRunModel:
    def __init__(self):
        self.wells = []
        self.bc = None
        self.src = None
        self.separator_group = None
        self.group_controls = []
        self.calls = 0
        self.last_well_switches = []
        self.last_group_switches = []
        self.last_end_run_requested = False
        self.last_end_run_groups = []

    def step(self, state, dt, *, bc=None, src=None):
        self.calls += 1
        self.last_well_switches = []
        if self.calls == 1:
            self.last_end_run_requested = True
            self.last_end_run_groups = ["PROD"]
            self.last_group_switches = [
                {"group": "PROD", "type": "economics", "reason": "min_oil_rate", "end_run": True}
            ]
        else:
            self.last_end_run_requested = False
            self.last_end_run_groups = []
            self.last_group_switches = []
        return state + 1, True, 1


def validate_schedule_reports_and_stops_on_end_run() -> int:
    model = EndRunModel()
    schedule = simple_schedule(
        [1.0 * DAY, 1.0 * DAY, 1.0 * DAY],
        control=[ScheduleControl(), ScheduleControl(), ScheduleControl()],
        step_control=[1, 2, 3],
    )
    result = run_schedule(model, 0, schedule)
    report = result.reports[0] if result.reports else {}

    keep_going = EndRunModel()
    full_result = run_schedule(keep_going, 0, schedule, stop_on_end_run=False)

    failures = 0
    failures += _check("run_schedule emits accepted end-run report",
                       len(result.reports or []) == 1
                       and result.final_state == 1
                       and report.get("EndRunRequested") is True)
    failures += _check("end-run report carries affected group names",
                       report.get("EndRunGroups") == ["PROD"]
                       and report.get("GroupSwitches", [{}])[0].get("end_run") is True)
    failures += _check("stop_on_end_run can be disabled for diagnostics",
                       len(full_result.reports or []) == 3
                       and full_result.final_state == 3
                       and [r["EndRunRequested"] for r in full_result.reports] == [True, False, False])
    return failures


def main() -> int:
    print("\nPhase 153: GECON end-run request reporting")
    failures = validate_model_records_gecon_end_run_request()
    failures += validate_schedule_reports_and_stops_on_end_run()
    if failures:
        print(f"\nPhase 153 failed ({failures} checks).")
        return 1
    print("\nGECON end-run requests are now visible in compositional reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
