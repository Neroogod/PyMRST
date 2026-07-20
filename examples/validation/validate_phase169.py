"""Phase 169: deck schedule conversion can import COMPDAT wells."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    schedule_controls_from_deck_controls,
    simple_schedule_from_deck_controls,
)
from pymrst.params.rock import make_rock  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _grid_and_rock():
    grid = compute_geometry(cart_grid([2, 2, 3], [20.0, 20.0, 30.0]))
    rock = make_rock(grid, 100.0, 0.25)
    return grid, rock


def _controls():
    return [
        {
            "WELSPECS": [["P1", "FIELD", 1, 1, 1000.0]],
            "COMPDAT": [["P1", 1, 1, 1, 3, "OPEN", 1, 10.0, 0.20, 0.0, 0.0, 0.0, "Z"]],
            "WCONPROD": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 200.0]],
            "WELOPEN": [["P1", "SHUT", 1, 1, 2, 1, 3]],
        },
        {
            "WCONPROD": [["P1", "OPEN", "ORAT", 5.0, np.inf, np.inf, np.inf, np.inf, 150.0]],
            "WELOPEN": [["P1", "SHUT", 1, 1, 3, 1, 3]],
        },
    ]


def validate_schedule_controls_import_compdat_geometry() -> int:
    grid, rock = _grid_and_rock()
    converted = schedule_controls_from_deck_controls(
        None,
        _controls(),
        grid=grid,
        rock=rock,
        compi=[0.0, 1.0, 0.0],
    )

    failures = 0
    failures += _check("controls become native ScheduleControl objects",
                       len(converted) == 2 and all(isinstance(c, ScheduleControl) for c in converted))
    first = converted[0].wells[0]
    second = converted[1].wells[0]
    failures += _check("first control imports COMPDAT well geometry",
                       first.name == "P1" and np.array_equal(first.cells, np.array([0, 4, 8])))
    failures += _check("imported base well receives first WCONPROD control",
                       first.type == "bhp" and first.sign == -1 and np.isclose(first.val, 200.0))
    failures += _check("first control applies grid-aware WELOPEN to imported well",
                       np.array_equal(first.cstatus, np.array([True, False, True])))
    failures += _check("second control reuses imported geometry",
                       np.array_equal(second.cells, np.array([0, 4, 8])) and np.allclose(second.WI, first.WI))
    failures += _check("second control applies its own WCONPROD row",
                       second.type == "orat" and np.isclose(second.val, -5.0))
    failures += _check("second control applies its own WELOPEN row",
                       np.array_equal(second.cstatus, np.array([True, True, False])))
    failures += _check("control wells are isolated copies",
                       np.array_equal(first.cstatus, np.array([True, False, True])))
    return failures


def validate_simple_schedule_import_compdat_geometry() -> int:
    grid, rock = _grid_and_rock()
    sched = simple_schedule_from_deck_controls(
        [1.0, 2.0],
        None,
        _controls(),
        step_control=[1, 2],
        grid=grid,
        rock=rock,
        compi=[0.0, 1.0, 0.0],
    )

    failures = 0
    failures += _check("simple schedule keeps one-based controls",
                       np.array_equal(sched.step.control, np.array([1, 2])))
    failures += _check("simple schedule imports COMPDAT geometry",
                       np.array_equal(sched.control[0].wells[0].cells, np.array([0, 4, 8])))
    failures += _check("simple schedule carries independent completion states",
                       np.array_equal(sched.control[0].wells[0].cstatus, np.array([True, False, True]))
                       and np.array_equal(sched.control[1].wells[0].cstatus, np.array([True, True, False])))
    return failures


def validate_missing_geometry_requirements_are_explicit() -> int:
    failures = 0
    try:
        schedule_controls_from_deck_controls(None, _controls(), grid=_grid_and_rock()[0])
    except ValueError as exc:
        failures += _check("missing rock for COMPDAT import fails loudly", "grid and rock" in str(exc))
    else:
        failures += _check("missing rock for COMPDAT import fails loudly", False)
    return failures


def main() -> int:
    print("\nPhase 169: deck schedule conversion can import COMPDAT wells")
    failures = validate_schedule_controls_import_compdat_geometry()
    failures += validate_simple_schedule_import_compdat_geometry()
    failures += validate_missing_geometry_requirements_are_explicit()
    if failures:
        print(f"\nPhase 169 failed ({failures} checks).")
        return 1
    print("\nDeck-style schedule conversion now imports WELSPECS/COMPDAT well geometry when needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
