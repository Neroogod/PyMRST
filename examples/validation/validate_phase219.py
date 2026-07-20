"""Phase 219: singular forcing container aliases."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    boundary_condition_from_control,
    schedule_controls_from_boundary_controls,
    schedule_controls_from_deck_controls,
    schedule_controls_from_forcing_controls,
    schedule_controls_from_source_controls,
    simple_schedule,
    source_from_control,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.modules.compositional.examples._shared import make_line_drive_case  # noqa: E402
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.params.wells_and_bc import add_source  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _native_source():
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    rate = float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))
    src = add_source(
        None,
        np.asarray([0, grid.cells.num - 1], dtype=np.int64),
        np.asarray([rate, -rate], dtype=float),
        sat=np.asarray([[0.0, 1.0], [0.5, 0.5]], dtype=float),
        components=np.asarray([[0.2, 0.7, 0.1], [0.0, 0.0, 0.0]], dtype=float),
    )
    return src, rate


def validate_direct_singular_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    src, rate = _native_source()

    bc = boundary_condition_from_control(
        case["grid"],
        {
            "Boundary": [
                {"SIDE": "xmin", "TYPE": "flux", "Q": 0.0, "COMPI": [0.1, 0.9, 0.0]},
            ]
        },
    )
    direct_native = boundary_condition_from_control(case["grid"], {"BoundaryCondition": case["bc"]})
    src_from_term = source_from_control(
        {
            "SourceTerm": [
                {"CELL": [0, 1], "Q": [rate, -rate], "COMPI": [[0.2, 0.8, 0.0], [0.0, 0.0, 0.0]]},
            ]
        }
    )
    direct_native_src = source_from_control({"SrcTerms": src})

    failures = 0
    failures += _check("Boundary container records are parsed",
                       bc is not None
                       and bc.face.size > 0
                       and bc.components.shape[1] == 3)
    failures += _check("BoundaryCondition native object passes through",
                       direct_native is not None
                       and direct_native is not case["bc"]
                       and np.array_equal(direct_native.face, case["bc"].face))
    failures += _check("SourceTerm container records are parsed",
                       src_from_term is not None
                       and src_from_term.cell.tolist() == [0, 1]
                       and np.allclose(src_from_term.rate, [rate, -rate]))
    failures += _check("SrcTerms native source passes through",
                       direct_native_src is not None
                       and direct_native_src is not src
                       and np.allclose(direct_native_src.rate, [rate, -rate]))
    return failures


def validate_schedule_singular_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    src, rate = _native_source()
    boundary = schedule_controls_from_boundary_controls(
        case["grid"],
        [
            {
                "Boundary": [{"SIDE": "xmin", "Q": 0.0}],
                "SrcTerms": src,
                "Label": "boundary-alias",
            }
        ],
    )[0]
    source = schedule_controls_from_source_controls(
        [
            {
                "BoundaryCondition": case["bc"],
                "SourceTerm": [{"CELL": [0, 1], "Q": [rate, -rate]}],
                "Label": "source-alias",
            }
        ],
    )[0]
    mixed = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "Boundary": [{"SIDE": "xmin", "Q": 0.0}],
                "SrcTerms": src,
                "Label": "mixed-alias",
            }
        ],
    )[0]

    failures = 0
    for name, control, label in (
        ("boundary schedule", boundary, "boundary-alias"),
        ("source schedule", source, "source-alias"),
        ("mixed schedule", mixed, "mixed-alias"),
    ):
        failures += _check(f"{name} consumes singular aliases",
                           control.bc is not None and control.src is not None)
        failures += _check(f"{name} filters singular aliases",
                           control.extra == {"Label": label})
    return failures


def validate_generic_deck_singular_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    src, rate = _native_source()
    simple = simple_schedule(
        [1.0],
        control=[
            {
                "BoundaryCondition": case["bc"],
                "SourceTerm": src,
                "Label": "simple-singular",
            }
        ],
    ).control[0]
    deck = schedule_controls_from_deck_controls(
        [],
        [
            {
                "Boundary": case["bc"],
                "SrcTerms": src,
                "Tag": "deck-singular",
            }
        ],
    )[0]

    failures = 0
    failures += _check("simple_schedule consumes singular native aliases",
                       simple.bc is not None
                       and simple.src is not None
                       and np.allclose(simple.src.rate, [rate, -rate]))
    failures += _check("simple_schedule filters singular native aliases",
                       simple.extra == {"Label": "simple-singular"})
    failures += _check("deck conversion consumes singular native aliases",
                       deck.bc is not None
                       and deck.src is not None
                       and np.allclose(deck.src.rate, [rate, -rate]))
    failures += _check("deck conversion filters singular native aliases",
                       deck.extra == {"Tag": "deck-singular"})
    return failures


def main() -> int:
    print("\nPhase 219: singular forcing container aliases")
    failures = validate_direct_singular_aliases()
    failures += validate_schedule_singular_aliases()
    failures += validate_generic_deck_singular_aliases()
    if failures:
        print(f"\nPhase 219 failed ({failures} checks).")
        return 1
    print("\nSingular boundary/source forcing aliases now parse consistently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
