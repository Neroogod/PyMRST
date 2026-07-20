"""Phase 221: forcing record field-name aliases."""

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


def _rate():
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    return float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))


def validate_boundary_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    faces = case["bc"].face[:2]
    bc = boundary_condition_from_control(
        case["grid"],
        {
            "BCRecords": {
                "faceIndex": faces,
                "type": ["flux", "pressure"],
                "values": [0.0, 100.0],
                "saturations": [[1.0, 0.0], [0.0, 1.0]],
                "composition_i": [[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]],
            },
        },
    )

    failures = 0
    failures += _check("boundary faceIndex alias selects explicit faces",
                       bc is not None and np.array_equal(bc.face, faces))
    failures += _check("boundary values alias preserves per-face values",
                       bc is not None and np.allclose(bc.value, [0.0, 100.0]))
    failures += _check("boundary saturations alias preserves rows",
                       bc is not None
                       and bc.sat.shape == (2, 2)
                       and np.allclose(bc.sat[1], [0.0, 1.0]))
    failures += _check("boundary composition_i alias remains aligned",
                       bc is not None
                       and bc.components.shape == (2, 3)
                       and np.allclose(bc.components[1], [0.7, 0.2, 0.1]))
    return failures


def validate_source_aliases() -> int:
    rate = _rate()
    src = source_from_control(
        {
            "SourceTerms": {
                "cellIndex": [0, 1],
                "values": [rate, -rate],
                "S": [[0.0, 1.0], [0.5, 0.5]],
                "component_i": [[0.2, 0.7, 0.1], [0.0, 0.0, 0.0]],
            },
        },
    )

    failures = 0
    failures += _check("source cellIndex alias selects cells",
                       src is not None and src.cell.tolist() == [0, 1])
    failures += _check("source values alias preserves rates",
                       src is not None and np.allclose(src.rate, [rate, -rate]))
    failures += _check("source S alias preserves saturation rows",
                       src is not None
                       and src.sat.shape == (2, 2)
                       and np.allclose(src.sat[0], [0.0, 1.0]))
    failures += _check("source component_i alias remains aligned",
                       src is not None
                       and src.components.shape == (2, 3)
                       and np.allclose(src.components[0], [0.2, 0.7, 0.1]))
    return failures


def validate_schedule_alias_cleanup() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    rate = _rate()
    faces = case["bc"].face[:2]
    mixed = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "BCRecords": {
                    "faceIndices": faces,
                    "type": ["flux", "pressure"],
                    "vals": [0.0, 100.0],
                    "saturations": [[1.0, 0.0], [0.0, 1.0]],
                    "COMPI": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                },
                "SourceTerms": {
                    "cellIndices": [0, 1],
                    "vals": [rate, -rate],
                    "saturations": [[0.0, 1.0], [0.5, 0.5]],
                },
                "Label": "alias-cleanup",
            }
        ],
    )[0]
    source_only = schedule_controls_from_source_controls(
        [
            {
                "cellIndex": [0],
                "values": [rate],
                "saturations": [[1.0, 0.0]],
                "Label": "source-alias-cleanup",
            }
        ],
    )[0]

    failures = 0
    failures += _check("mixed schedule parses plural index aliases",
                       mixed.bc is not None
                       and mixed.src is not None
                       and np.array_equal(mixed.bc.face, faces)
                       and mixed.src.cell.tolist() == [0, 1])
    failures += _check("mixed schedule filters consumed alias fields",
                       mixed.extra == {"Label": "alias-cleanup"})
    failures += _check("source schedule filters consumed alias fields",
                       source_only.extra == {"Label": "source-alias-cleanup"})
    return failures


def main() -> int:
    print("\nPhase 221: forcing record field-name aliases")
    failures = validate_boundary_aliases()
    failures += validate_source_aliases()
    failures += validate_schedule_alias_cleanup()
    if failures:
        print(f"\nPhase 221 failed ({failures} checks).")
        return 1
    print("\nForcing records now accept plural/index-style MRST field aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
