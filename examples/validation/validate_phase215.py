"""Phase 215: case-insensitive native forcing object passthrough."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    boundary_condition_from_control,
    make_line_drive_case,
    schedule_controls_from_boundary_controls,
    schedule_controls_from_forcing_controls,
    schedule_controls_from_source_controls,
    source_from_control,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
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
    return grid, src, rate


def validate_native_boundary_passthrough() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    native = case["bc"]
    direct = boundary_condition_from_control(case["grid"], {"BoundaryConditions": native})
    converted = schedule_controls_from_source_controls(
        [{"Bc": native, "CELL": 0, "Q": 0.0, "Label": "native-bc"}]
    )[0]

    failures = 0
    failures += _check("BoundaryConditions native BC is passed through",
                       direct is not None
                       and direct is not native
                       and np.array_equal(direct.face, native.face)
                       and np.allclose(direct.value, native.value))
    failures += _check("source schedule keeps mixed-case native BC side block",
                       converted.bc is not None
                       and converted.bc is not native
                       and np.array_equal(converted.bc.face, native.face)
                       and converted.extra == {"Label": "native-bc"})
    return failures


def validate_native_source_passthrough() -> int:
    _, native, rate = _native_source()
    direct = source_from_control({"SourceTerms": native})
    converted = schedule_controls_from_boundary_controls(
        make_boundary_conditions_case(nx=4, ny=2)["grid"],
        [{"Src": native, "TYPE": "RATE", "SIDE": "xmin", "Q": 0.0, "Label": "native-src"}],
    )[0]

    failures = 0
    failures += _check("SourceTerms native source is passed through",
                       direct is not None
                       and direct is not native
                       and direct.cell.tolist() == native.cell.tolist()
                       and np.allclose(direct.rate, [rate, -rate]))
    failures += _check("boundary schedule keeps mixed-case native source side block",
                       converted.src is not None
                       and converted.src is not native
                       and converted.src.cell.tolist() == native.cell.tolist()
                       and converted.extra == {"Label": "native-src"})
    return failures


def validate_mixed_native_forcing_passthrough() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    _, native_src, rate = _native_source()
    converted = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "BcRecords": case["bc"],
                "SrcRecords": native_src,
                "Label": "native-forcing",
            }
        ],
    )[0]

    failures = 0
    failures += _check("mixed forcing accepts native BC under mixed-case container",
                       converted.bc is not None
                       and converted.bc is not case["bc"]
                       and np.array_equal(converted.bc.face, case["bc"].face))
    failures += _check("mixed forcing accepts native source under mixed-case container",
                       converted.src is not None
                       and converted.src is not native_src
                       and np.allclose(converted.src.rate, [rate, -rate]))
    failures += _check("native forcing containers are consumed from extras",
                       converted.extra == {"Label": "native-forcing"})
    return failures


def main() -> int:
    print("\nPhase 215: case-insensitive native forcing object passthrough")
    failures = validate_native_boundary_passthrough()
    failures += validate_native_source_passthrough()
    failures += validate_mixed_native_forcing_passthrough()
    if failures:
        print(f"\nPhase 215 failed ({failures} checks).")
        return 1
    print("\nNative BC/SRC forcing objects now pass through mixed-case container keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
