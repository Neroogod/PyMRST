"""Phase 210: packed fallback reports infer schedule block-change flags."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    SimulationResult,
    get_multiple_packed_simulator_outputs_ad,
    pack_simulation_problem_ad,
    simple_schedule,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _legacy_reports_for(schedule):
    problem = pack_simulation_problem_ad(
        0.0,
        object(),
        schedule,
        "ValidationSuite",
        name="BlockChangeLegacyProblem",
    )
    problem.result = SimulationResult(
        states=[1.0, 2.0, 3.0],
        iterations=np.asarray([2, 2, 2], dtype=int),
        reports=None,
    )
    _, _, reports, _ = get_multiple_packed_simulator_outputs_ad([problem])
    return reports[0]


def validate_forcing_blocks_do_not_become_facility_changes() -> int:
    schedule = simple_schedule(
        [1.0, 1.0, 1.0],
        control=[
            ScheduleControl(bc={"face": 1}, src={"cell": 1}),
            ScheduleControl(bc={"face": 2}, src={"cell": 2}),
        ],
        step_control=[1, 1, 2],
    )
    reports = _legacy_reports_for(schedule)

    failures = 0
    failures += _check("fallback boundary/source flags follow control changes",
                       [r["BoundaryChanged"] for r in reports] == [True, False, True]
                       and [r["SourceChanged"] for r in reports] == [True, False, True]
                       and [r["ForcingChanged"] for r in reports] == [True, False, True])
    failures += _check("fallback forcing changes stay separate from facility changes",
                       [r["FacilityChanged"] for r in reports] == [False, False, False])
    failures += _check("fallback well/VFP flags stay false for forcing-only schedules",
                       all(not r["WellControlsChanged"] for r in reports)
                       and all(not r["VFPTablesChanged"] for r in reports))
    return failures


def validate_facility_blocks_drive_facility_change_flags() -> int:
    schedule = simple_schedule(
        [1.0, 1.0, 1.0],
        control=[
            ScheduleControl(
                wells=["W1"],
                separator_group={"name": "sep-a"},
                group_controls=[{"name": "FIELD"}],
                vfp_tables_injector=["inj-a"],
                vfp_tables_producer=["prod-a"],
            ),
            ScheduleControl(
                wells=["W2"],
                separator_group={"name": "sep-b"},
                group_controls=[{"name": "FIELD"}],
                vfp_tables_injector=["inj-b"],
                vfp_tables_producer=["prod-b"],
            ),
        ],
        step_control=[1, 2, 2],
    )
    reports = _legacy_reports_for(schedule)

    failures = 0
    failures += _check("fallback well-control flags follow facility control changes",
                       [r["WellControlsChanged"] for r in reports] == [True, True, False])
    failures += _check("fallback separator/group flags follow facility control changes",
                       [r["SeparatorChanged"] for r in reports] == [True, True, False]
                       and [r["GroupControlsChanged"] for r in reports] == [True, True, False])
    failures += _check("fallback VFP flags follow facility control changes",
                       [r["VFPInjectorTablesChanged"] for r in reports] == [True, True, False]
                       and [r["VFPProducerTablesChanged"] for r in reports] == [True, True, False]
                       and [r["VFPTablesChanged"] for r in reports] == [True, True, False])
    failures += _check("fallback facility-change flag includes managed facility blocks",
                       [r["FacilityChanged"] for r in reports] == [True, True, False])
    failures += _check("fallback facility schedules do not imply forcing changes",
                       [r["ForcingChanged"] for r in reports] == [False, False, False])
    return failures


def main() -> int:
    print("\nPhase 210: packed fallback schedule block-change flags")
    failures = validate_forcing_blocks_do_not_become_facility_changes()
    failures += validate_facility_blocks_drive_facility_change_flags()
    if failures:
        print(f"\nPhase 210 failed ({failures} checks).")
        return 1
    print("\nPacked legacy reports now infer schedule forcing/facility block changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
