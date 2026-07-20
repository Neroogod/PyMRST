"""Phase 167: schedule conversion forwards grid-aware WELOPEN matching."""

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
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _grid():
    return compute_geometry(cart_grid([2, 2, 3], [20.0, 20.0, 30.0]))


def _well(name: str) -> Well:
    return Well(
        cells=np.array([0, 4, 8]),
        type="bhp",
        val=100.0,
        r=np.ones(3),
        dir="z",
        WI=np.array([10.0, 20.0, 30.0]),
        dZ=np.zeros(3),
        name=name,
        sign=-1,
        compi=np.asarray([0.5, 0.5], dtype=float),
        refDepth=0.0,
        lims={},
    )


def validate_schedule_control_conversion_forwards_grid() -> int:
    grid = _grid()
    base_wells = [_well("P1")]
    converted = schedule_controls_from_deck_controls(
        base_wells,
        [
            {
                "WCONPROD": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
                "WELOPEN": [["P1", "SHUT", 1, 1, 2, 1, 3]],
            },
            {
                "WCONPROD": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
                "WELOPEN": [["P1", "SHUT", 1, 1, 3, 1, 3]],
            },
        ],
        grid=grid,
    )

    failures = 0
    failures += _check("deck controls still become ScheduleControl objects",
                       len(converted) == 2 and all(isinstance(c, ScheduleControl) for c in converted))
    failures += _check("first control uses grid-aware K=2 WELOPEN matching",
                       np.array_equal(converted[0].wells[0].cstatus, np.array([True, False, True])))
    failures += _check("second control uses grid-aware K=3 WELOPEN matching",
                       np.array_equal(converted[1].wells[0].cstatus, np.array([True, True, False])))
    failures += _check("base well remains unmutated by schedule conversion",
                       base_wells[0].cstatus is None)
    return failures


def validate_simple_schedule_builder_forwards_grid() -> int:
    sched = simple_schedule_from_deck_controls(
        [1.0, 1.0],
        [_well("P2")],
        [
            {
                "WCONPROD": [["P2", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
                "WELOPEN": [["P2", "SHUT", 1, 1, 1, 1, 3]],
            },
            {
                "WCONPROD": [["P2", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
                "WELOPEN": [["P2", "SHUT", 1, 1, 3, 1, 3]],
            },
        ],
        step_control=[1, 2],
        grid=_grid(),
    )

    failures = 0
    failures += _check("simple schedule keeps MRST one-based control indices",
                       np.array_equal(sched.step.control, np.array([1, 2])))
    failures += _check("simple schedule first control carries K=1 completion shut-in",
                       np.array_equal(sched.control[0].wells[0].cstatus, np.array([False, True, True])))
    failures += _check("simple schedule second control carries K=3 completion shut-in",
                       np.array_equal(sched.control[1].wells[0].cstatus, np.array([True, True, False])))
    return failures


def validate_schedule_conversion_without_grid_is_conservative() -> int:
    converted = schedule_controls_from_deck_controls(
        [_well("P3")],
        [
            {
                "WCONPROD": [["P3", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
                "WELOPEN": [["P3", "SHUT", 1, 1, 2, 1, 3]],
            }
        ],
    )

    return _check("schedule conversion remains conservative without grid", converted[0].wells[0].cstatus is None)


def main() -> int:
    print("\nPhase 167: schedule conversion forwards grid-aware WELOPEN matching")
    failures = validate_schedule_control_conversion_forwards_grid()
    failures += validate_simple_schedule_builder_forwards_grid()
    failures += validate_schedule_conversion_without_grid_is_conservative()
    if failures:
        print(f"\nPhase 167 failed ({failures} checks).")
        return 1
    print("\nDeck-style schedule conversion now forwards Cartesian grids to WELOPEN matching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
