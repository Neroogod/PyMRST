"""Phase 148: MRST-like compositional source-control schedules."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    make_line_drive_case,
    run_schedule,
    schedule_controls_from_source_controls,
    simple_schedule_from_source_controls,
    source_from_control,
)
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _source_case():
    grid, _, _, info, _, model, _ = make_line_drive_case(nx=4)
    model.wells = []
    state = model.initial_state(info.pressure, info.initial)
    rate = pore_volume(grid, model.rock).sum() / (1000.0 * DAY)
    return grid, info, model, state, float(rate)


def validate_direct_source_record() -> int:
    grid, _, _, _, rate = _source_case()
    src = source_from_control(
        {
            "cells": [0, grid.cells.num - 1],
            "rate": [rate, -rate],
            "sat": [[0.0, 1.0], [0.5, 0.5]],
            "components": [[0.9, 0.05, 0.05], [0.0, 0.0, 0.0]],
        }
    )

    failures = 0
    failures += _check("direct record creates one source object",
                       src is not None and src.cell.tolist() == [0, grid.cells.num - 1])
    failures += _check("direct record preserves source/sink rates",
                       np.allclose(src.rate, [rate, -rate]))
    failures += _check("direct record preserves component rows",
                       src.components.shape == (2, 3)
                       and np.allclose(src.components[0], [0.9, 0.05, 0.05]))
    return failures


def validate_nested_source_records() -> int:
    grid, _, _, _, rate = _source_case()
    src = source_from_control(
        {
            "SRC": [
                {"cell": 0, "value": rate, "sat": [0.0, 1.0], "z": [0.1, 0.8, 0.1]},
                {"cell": grid.cells.num - 1, "value": -rate, "sat": [0.5, 0.5]},
            ]
        }
    )

    failures = 0
    failures += _check("nested SRC records append source and sink",
                       src is not None and src.cell.tolist() == [0, grid.cells.num - 1])
    failures += _check("nested SRC records keep zero sink components",
                       src.components.shape == (2, 3)
                       and np.allclose(src.components[1], [0.0, 0.0, 0.0]))
    return failures


def validate_schedule_switching_from_source_controls() -> int:
    grid, _, model, state, rate = _source_case()
    controls = [
        {
            "sources": [
                {"cell": 0, "rate": rate, "sat": [0.0, 1.0], "components": [0.9, 0.05, 0.05]},
                {"cell": grid.cells.num - 1, "rate": -rate, "sat": [0.5, 0.5]},
            ],
            "label": "methane-rich",
        },
        {
            "sources": [
                {"cell": 0, "rate": rate, "sat": [0.0, 1.0], "components": [0.05, 0.9, 0.05]},
                {"cell": grid.cells.num - 1, "rate": -rate, "sat": [0.5, 0.5]},
            ],
            "label": "CO2-rich",
        },
    ]
    converted = schedule_controls_from_source_controls(controls)
    schedule = simple_schedule_from_source_controls(
        [2.0 * DAY, 2.0 * DAY],
        controls,
        step_control=[1, 2],
    )
    result = run_schedule(model, state, schedule)

    failures = 0
    failures += _check("source-control conversion preserves extras",
                       converted[0].extra == {"label": "methane-rich"}
                       and converted[1].extra == {"label": "CO2-rich"})
    failures += _check("schedule keeps one-based source controls",
                       schedule.step.control.tolist() == [1, 2])
    failures += _check("second control carries changed component stream",
                       np.allclose(schedule.control[1].src.components[0], [0.05, 0.9, 0.05]))
    failures += _check("switched source stream affects injected cell composition",
                       result.states[0].z[0, 0] > state.z[0, 0]
                       and result.states[1].z[0, 1] > result.states[0].z[0, 1])
    failures += _check("model source is restored after run", model.src is None)
    return failures


def main() -> int:
    print("\nPhase 148: MRST-like compositional source-control schedules")
    failures = validate_direct_source_record()
    failures += validate_nested_source_records()
    failures += validate_schedule_switching_from_source_controls()
    if failures:
        print(f"\nPhase 148 failed ({failures} checks).")
        return 1
    print("\nSource-control dictionaries now build switched compositional source schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
