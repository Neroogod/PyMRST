"""Phase 220: MATLAB cell-array forcing record flattening."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    boundary_condition_from_control,
    schedule_controls_from_forcing_controls,
    schedule_controls_from_source_controls,
    source_from_control,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.modules.compositional.examples._shared import make_line_drive_case  # noqa: E402
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _object_row(records):
    arr = np.empty((1, len(records)), dtype=object)
    for ix, record in enumerate(records):
        arr[0, ix] = record
    return arr


def _object_column(records):
    arr = np.empty((len(records), 1), dtype=object)
    for ix, record in enumerate(records):
        arr[ix, 0] = record
    return arr


def _rate():
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    return float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))


def validate_boundary_cell_arrays() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    records = _object_row([
        {"SIDE": "xmin", "TYPE": "flux", "Q": 0.0, "COMPI": [1.0, 0.0, 0.0]},
        {"SIDE": "xmax", "TYPE": "pressure", "PRES": 100.0, "COMPI": [0.0, 1.0, 0.0]},
    ])
    bc = boundary_condition_from_control(case["grid"], {"BCRecords": records})
    bc_types = np.asarray([] if bc is None else bc.type, dtype=object)
    table = np.asarray([[0, 0.0], [1, 1.0]], dtype=object)
    table_records = source_from_control({"SourceTerm": table})

    failures = 0
    failures += _check("1-by-N object cell array creates two BC record groups",
                       bc is not None
                       and bc.face.size > 0
                       and bc.components.shape[1] == 3
                       and np.any(bc_types == "pressure")
                       and np.any(bc_types == "flux"))
    failures += _check("numeric object tables still parse as one source table",
                       table_records is not None
                       and table_records.cell.tolist() == [0, 1])
    return failures


def validate_source_cell_arrays() -> int:
    rate = _rate()
    records = _object_column([
        {"CELL": [0], "Q": [rate], "COMPI": [0.2, 0.8, 0.0]},
        {"CELL": [1], "Q": [-rate], "COMPI": [0.0, 0.0, 0.0]},
    ])
    src = source_from_control({"SrcRecords": records})

    failures = 0
    failures += _check("N-by-1 object cell array creates two source records",
                       src is not None
                       and src.cell.tolist() == [0, 1]
                       and np.allclose(src.rate, [rate, -rate]))
    failures += _check("source cell-array components stay row-aligned",
                       src is not None
                       and src.components.shape == (2, 3)
                       and np.allclose(src.components[0], [0.2, 0.8, 0.0]))
    return failures


def validate_schedule_cell_arrays() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    rate = _rate()
    bc_records = _object_row([
        {"SIDE": "xmin", "Q": 0.0, "COMPI": [1.0, 0.0, 0.0]},
        {"SIDE": "xmax", "PRES": 100.0, "COMPI": [0.0, 1.0, 0.0]},
    ])
    src_records = _object_row([
        {"CELL": [0], "Q": [rate], "COMPI": [0.3, 0.7, 0.0]},
        {"CELL": [1], "Q": [-rate], "COMPI": [0.0, 0.0, 0.0]},
    ])
    mixed = schedule_controls_from_forcing_controls(
        case["grid"],
        [{"BCRecords": bc_records, "SrcRecords": src_records, "Name": "cell-array"}],
    )[0]
    source_only = schedule_controls_from_source_controls(
        [{"SourceTerms": src_records, "Name": "source-cell-array"}],
    )[0]

    failures = 0
    failures += _check("mixed schedule flattens BC object cell arrays",
                       mixed.bc is not None and mixed.bc.components.shape[1] == 3)
    failures += _check("mixed schedule flattens source object cell arrays",
                       mixed.src is not None and mixed.src.cell.tolist() == [0, 1])
    failures += _check("mixed schedule filters consumed cell-array aliases",
                       mixed.extra == {"Name": "cell-array"})
    failures += _check("source schedule filters consumed cell-array aliases",
                       source_only.extra == {"Name": "source-cell-array"})
    return failures


def main() -> int:
    print("\nPhase 220: MATLAB cell-array forcing record flattening")
    failures = validate_boundary_cell_arrays()
    failures += validate_source_cell_arrays()
    failures += validate_schedule_cell_arrays()
    if failures:
        print(f"\nPhase 220 failed ({failures} checks).")
        return 1
    print("\nMATLAB-style forcing record cell arrays now flatten consistently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
