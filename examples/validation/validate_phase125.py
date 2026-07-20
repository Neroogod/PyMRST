"""Phase 125: sequential compositional schedules with changing controls.

This is a native PyMRST regression for the MRST-style schedule contract around
``setTimeDiscretization(model, 'explicit')``. The physics pieces below are
already MATLAB-validated in phases 121--124; this check proves that the
schedule wrapper drives the sequential pressure/transport dispatch across
multiple report-step controls and does not leak switched well controls back
into the stored schedule blocks.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    make_line_drive_case,
    run_schedule,
    simple_schedule,
)
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_explicit_schedule_run() -> int:
    _, _, _, info, wells, model, state = make_line_drive_case(nx=4)
    model = model.set_time_discretization("explicit")
    model.max_iterations = 60

    wells_a = copy.deepcopy(wells)
    wells_b = copy.deepcopy(wells)
    wells_a[0].components = np.array([0.9, 0.05, 0.05])
    wells_b[0].components = np.array([0.05, 0.9, 0.05])
    wells_a[0].name = "Injector-A"
    wells_b[0].name = "Injector-B"

    schedule = simple_schedule(
        [0.25 * DAY, 0.25 * DAY],
        control=[
            ScheduleControl(wells=wells_a),
            ScheduleControl(wells=wells_b),
        ],
        step_control=[1, 2],
    )
    out = run_schedule(model, state, schedule)
    final = out.final_state

    failures = 0
    failures += _check("two report steps completed", out.iterations.size == 2)
    failures += _check("explicit dispatch remained on model",
                       model.time_discretization == "explicit")
    failures += _check("finite final pressure",
                       bool(np.all(np.isfinite(final.pressure))))
    failures += _check("finite final composition",
                       bool(np.all(np.isfinite(final.z))))
    failures += _check("composition rows stay normalized",
                       bool(np.allclose(final.z.sum(axis=1), 1.0, atol=1.0e-10)))
    failures += _check("model wells restored after schedule",
                       [w.name for w in model.wells] == [w.name for w in wells])
    failures += _check("schedule control A preserved",
                       schedule.control[0].wells[0].name == "Injector-A")
    failures += _check("schedule control B preserved",
                       schedule.control[1].wells[0].name == "Injector-B")
    return failures


def validate_schedule_well_copy_isolation() -> int:
    class MutatingModel:
        def __init__(self):
            self.wells = []
            self.bc = None
            self.src = None
            self.separator_group = None
            self.group_controls = []

        def step(self, state, dt, bc=None, src=None):
            self.wells[0].type = "bhp"
            self.wells[0].val = 123.0
            return state + 1, True, 1

    _, _, _, _, wells, _, _ = make_line_drive_case(nx=3)
    wells_a = copy.deepcopy(wells)
    wells_b = copy.deepcopy(wells)
    wells_a[0].name = "Control-A"
    wells_b[0].name = "Control-B"
    model = MutatingModel()
    model.wells = copy.deepcopy(wells)
    schedule = simple_schedule(
        [1.0, 1.0],
        control=[
            ScheduleControl(wells=wells_a),
            ScheduleControl(wells=wells_b),
        ],
        step_control=[1, 2],
    )
    out = run_schedule(model, 0, schedule)

    failures = 0
    failures += _check("dummy schedule advanced both steps", out.final_state == 2)
    failures += _check("model wells restored after mutating step",
                       model.wells[0].type == wells[0].type)
    failures += _check("control A not mutated by active step",
                       schedule.control[0].wells[0].type == "rate")
    failures += _check("control B not mutated by active step",
                       schedule.control[1].wells[0].type == "rate")
    return failures


def main() -> int:
    print("\nPhase 125: sequential schedule controls")
    failures = validate_explicit_schedule_run()
    failures += validate_schedule_well_copy_isolation()
    if failures:
        print(f"\nPhase 125 failed ({failures} checks).")
        return 1
    print("\nSequential compositional schedules preserve MRST-style control blocks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
