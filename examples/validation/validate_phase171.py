"""Phase 171: COMPDAT/WELOPEN completion status text parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional import (  # noqa: E402
    apply_deck_well_controls,
    wells_from_deck_completion_records,
)
from pymrst.params.rock import make_rock  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _grid_and_rock():
    grid = compute_geometry(cart_grid([1, 1, 3], [10.0, 10.0, 30.0]))
    rock = make_rock(grid, 100.0, 0.25)
    return grid, rock


def validate_compdat_status_text_is_preserved() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [["P1", "FIELD", 1, 1, 100.0]],
            "COMPDAT": [
                ["P1", 1, 1, 1, 1, "OPEN", 1, 10.0, 0.2, -1.0, 0.0, 0.0, "Z"],
                ["P1", 1, 1, 2, 2, "SHUT", 1, 20.0, 0.2, -1.0, 0.0, 0.0, "Z"],
                ["P1", 1, 1, 3, 3, "STOP", 1, 30.0, 0.2, -1.0, 0.0, 0.0, "Z"],
            ],
        },
    )

    w = wells[0]
    failures = 0
    failures += _check("COMPDAT status text preserves OPEN/SHUT/STOP",
                       np.array_equal(getattr(w, "cstatus_text", None),
                                      np.array(["OPEN", "SHUT", "STOP"], dtype=object)))
    failures += _check("SHUT and STOP remain inactive in boolean cstatus",
                       np.array_equal(w.cstatus, np.array([True, False, False])))
    failures += _check("raw COMPDAT rows are still retained",
                       len(getattr(w, "compdat", [])) == 3)
    return failures


def validate_mapping_status_normalization_and_welopen_update() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [{"well": "P2", "group": "FIELD", "i": 1, "j": 1, "ref_depth": 100.0}],
            "COMPDAT": [
                {"well": "P2", "k1": 1, "k2": 1, "status": "open", "wi": 10.0, "radius": 0.1},
                {"well": "P2", "k1": 2, "k2": 2, "status": "stop", "wi": 20.0, "radius": 0.1},
                {"well": "P2", "k1": 3, "k2": 3, "status": "shut", "wi": 30.0, "radius": 0.1},
            ],
        },
    )

    base = wells[0]
    updated = apply_deck_well_controls(
        wells,
        {"WELOPEN": [["P2", "OPEN", 0, 0, 0, 2, 3]]},
        grid=grid,
    )[0]

    failures = 0
    failures += _check("mapping COMPDAT status text is normalized to uppercase",
                       np.array_equal(getattr(base, "cstatus_text", None),
                                      np.array(["OPEN", "STOP", "SHUT"], dtype=object)))
    failures += _check("WELOPEN updates completion status text on copied wells",
                       np.array_equal(getattr(updated, "cstatus_text", None),
                                      np.array(["OPEN", "OPEN", "OPEN"], dtype=object)))
    failures += _check("WELOPEN status text update matches boolean cstatus",
                       np.array_equal(updated.cstatus, np.array([True, True, True])))
    failures += _check("WELOPEN does not mutate imported base status text",
                       np.array_equal(getattr(base, "cstatus_text", None),
                                      np.array(["OPEN", "STOP", "SHUT"], dtype=object)))
    return failures


def main() -> int:
    print("\nPhase 171: COMPDAT/WELOPEN completion status text parity")
    failures = validate_compdat_status_text_is_preserved()
    failures += validate_mapping_status_normalization_and_welopen_update()
    if failures:
        print(f"\nPhase 171 failed ({failures} checks).")
        return 1
    print("\nCOMPDAT and WELOPEN now preserve MRST completion status text metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
