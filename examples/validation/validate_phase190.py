"""Phase 190: GCONINJE injection-limit diagnostics."""

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
    g = compute_geometry(cart_grid([3, 1, 1], [60.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5.0e-5, compi=np.array([0.0, 1.0]), sign=1,
                     name="INJ")
    wells[0].components = np.array([0.9, 0.05, 0.05])
    model = OverallCompositionModel(g, rock, simple_mixture(),
                                    temperature=350.0, wells=wells)
    return model, model.initial_state(70.0 * BARSA, [0.5, 0.3, 0.2])


def validate_gconinje_surface_limit_diagnostic() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["INJ", "GINJ"]],
            "GCONINJE": [["GINJ", "GAS", "RATE", 2.0e-5, np.inf, np.inf, np.inf, "YES"]],
        }
    )
    state.qGs[0] = 5.0e-5
    ok = model._update_group_controls_from_state(state)
    switch = model.last_group_switches[0] if model.last_group_switches else {}
    meta = model.group_controls[0].extra["GCONINJE"][0]

    failures = 0
    failures += _check("GCONINJE gas surface limit triggers redistribution",
                       not ok
                       and switch.get("type") == "rate_limit"
                       and switch.get("mode") == "grat")
    failures += _check("GCONINJE diagnostic records source keyword and row fields",
                       switch.get("source_keyword") == "GCONINJE"
                       and switch.get("phase") == "GAS"
                       and switch.get("control") == "rate"
                       and switch.get("action") == "redistribute")
    failures += _check("GCONINJE limit resets injector target and state guess",
                       model.wells[0].type == "grat"
                       and np.isclose(model.wells[0].val, 2.0e-5)
                       and np.isclose(state.qGs[0], 2.0e-5))
    failures += _check("GCONINJE metadata records the last enforced violation",
                       meta.get("last_violation") == "grat"
                       and meta.get("last_action") == "redistribute")
    return failures


def validate_gconinje_reservoir_limit_diagnostic() -> int:
    model, state = _model()
    model.group_controls = facility_group_controls_from_deck_control(
        {
            "WELSPECS": [["INJ", "GINJ"]],
            "GCONINJE": [["GINJ", "GAS", "RESV", np.inf, 3.0e-5, np.inf, np.inf, "YES"]],
        }
    )
    state.qGs[0] = 5.0e-5
    ok = model._update_group_controls_from_state(state)
    switch = model.last_group_switches[0] if model.last_group_switches else {}
    meta = model.group_controls[0].extra["GCONINJE"][0]

    failures = 0
    failures += _check("GCONINJE reservoir limit triggers vrat redistribution",
                       not ok
                       and switch.get("mode") == "vrat"
                       and model.wells[0].type == "vrat")
    failures += _check("GCONINJE reservoir diagnostic keeps phase/control context",
                       switch.get("source_keyword") == "GCONINJE"
                       and switch.get("phase") == "GAS"
                       and switch.get("control") == "vrat")
    failures += _check("GCONINJE reservoir metadata records violation",
                       meta.get("last_violation") == "vrat"
                       and meta.get("last_action") == "redistribute")
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
                "group": "GINJ",
                "type": "rate_limit",
                "mode": "grat",
                "action": "redistribute",
                "source_keyword": "GCONINJE",
                "phase": "GAS",
                "control": "rate",
                "wells": ["INJ"],
            }
        ]
        return state + dt, True, 1


def validate_schedule_report_carries_gconinje_context() -> int:
    result = run_schedule(
        ReportingModel(),
        0.0,
        simple_schedule([1.0 * DAY], control=[ScheduleControl(group_controls=[])]),
    )
    switch = result.reports[0]["GroupSwitches"][0]

    failures = 0
    failures += _check("schedule report carries GCONINJE diagnostic context",
                       switch["source_keyword"] == "GCONINJE"
                       and switch["phase"] == "GAS"
                       and switch["control"] == "rate")
    return failures


def main() -> int:
    print("\nPhase 190: GCONINJE injection-limit diagnostics")
    failures = validate_gconinje_surface_limit_diagnostic()
    failures += validate_gconinje_reservoir_limit_diagnostic()
    failures += validate_schedule_report_carries_gconinje_context()
    if failures:
        print(f"\nPhase 190 failed ({failures} checks).")
        return 1
    print("\nGCONINJE limit enforcement now leaves MRST-style diagnostic context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
