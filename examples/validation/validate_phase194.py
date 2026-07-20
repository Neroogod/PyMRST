"""Phase 194: MRST GECON runtime diagnostic context."""

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


def _gecon_groups():
    return facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
            # name, min oil, min gas, max water cut, max GOR, max WGR,
            # workover, end-run, follow-on.
            "GECON": [["PROD", 3.0e-5, np.inf, np.inf, np.inf, np.inf,
                       "CON+", "YES", 7.0]],
        }
    )


def validate_model_gecon_runtime_context() -> int:
    model, state = _model()
    model.group_controls = _gecon_groups()
    state.qOs[1:] = [-1.0e-5, -1.0e-5]
    state.qGs[1:] = 0.0
    ok = model._update_group_controls_from_state(state)
    group = model.group_controls[0]
    switch = model.last_group_switches[0] if model.last_group_switches else {}
    meta = group.extra["GECON"]

    failures = 0
    failures += _check("GECON economics switch carries source keyword context",
                       not ok
                       and switch.get("source_keyword") == "GECON"
                       and switch.get("reason") == "min_oil_rate")
    failures += _check("GECON economics switch carries workover/follow-on context",
                       switch.get("workover") == "CON+"
                       and np.isclose(switch.get("follow_on"), 7.0)
                       and switch.get("end_run") is True)
    failures += _check("GECON metadata stamps runtime action and affected wells",
                       meta.get("last_violation") == "min_oil_rate"
                       and meta.get("last_action") == "shut"
                       and meta.get("last_affected_wells") == ["P1", "P2"]
                       and meta.get("end_run_requested") is True)
    return failures


def validate_schedule_report_preserves_gecon_context() -> int:
    class ReportingModel:
        def __init__(self):
            self.wells = []
            self.bc = None
            self.src = None
            self.separator_group = None
            self.group_controls = []
            self.last_well_switches = []
            self.last_group_switches = []
            self.last_end_run_requested = False
            self.last_end_run_groups = []

        def step(self, state, dt, *, bc=None, src=None):
            self.last_well_switches = []
            self.last_group_switches = [
                {
                    "group": "PROD",
                    "type": "economics",
                    "mode": "shut",
                    "reason": "min_oil_rate",
                    "source_keyword": "GECON",
                    "workover": "CON+",
                    "follow_on": 7.0,
                    "end_run": True,
                    "wells": ["P1", "P2"],
                    "indices": [1, 2],
                }
            ]
            self.last_end_run_requested = True
            self.last_end_run_groups = ["PROD"]
            return state + 1, True, 1

    model = ReportingModel()
    schedule = simple_schedule([1.0 * DAY], control=[ScheduleControl()])
    result = run_schedule(model, 0, schedule, stop_on_end_run=False)
    report = result.reports[0] if result.reports else {}
    switch = report.get("GroupSwitches", [{}])[0]
    # Mutate the live model after the run; reports should retain a deep copy.
    model.last_group_switches[0]["workover"] = "MUTATED"

    failures = 0
    failures += _check("schedule report carries GECON source context",
                       report.get("FacilityChanged") is True
                       and switch.get("source_keyword") == "GECON"
                       and switch.get("workover") == "CON+")
    failures += _check("schedule report carries end-run and follow-on context",
                       report.get("EndRunRequested") is True
                       and report.get("EndRunGroups") == ["PROD"]
                       and switch.get("end_run") is True
                       and np.isclose(switch.get("follow_on"), 7.0))
    failures += _check("schedule report deep-copies GECON diagnostics",
                       switch.get("workover") == "CON+")
    return failures


def main() -> int:
    print("\nPhase 194: MRST GECON runtime diagnostic context")
    failures = validate_model_gecon_runtime_context()
    failures += validate_schedule_report_preserves_gecon_context()
    if failures:
        print(f"\nPhase 194 failed ({failures} checks).")
        return 1
    print("\nGECON economics switches now preserve MRST runtime context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
