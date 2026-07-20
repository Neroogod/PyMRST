"""Phase 172: MRST default COMPDAT insertion for WELSPECS-only wells."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional import wells_from_deck_completion_records  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _grid_and_rock():
    grid = compute_geometry(cart_grid([2, 2, 3], [20.0, 20.0, 30.0]))
    rock = make_rock(grid, 100.0, 0.25)
    return grid, rock


def validate_welspecs_only_well_gets_default_vertical_compdat() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {"WELSPECS": [["P1", "FIELD", 2, 2, 123.0]]},
    )

    failures = 0
    failures += _check("WELSPECS-only control creates one well", len(wells) == 1)
    w = wells[0]
    failures += _check("default COMPDAT perforates all K layers at WELSPECS I/J",
                       np.array_equal(w.cells, np.array([3, 7, 11])))
    failures += _check("default COMPDAT status opens all completions",
                       np.array_equal(w.cstatus, np.array([True, True, True])))
    failures += _check("default COMPDAT status text is OPEN",
                       np.array_equal(getattr(w, "cstatus_text", None),
                                      np.array(["OPEN", "OPEN", "OPEN"], dtype=object)))
    failures += _check("MRST default diameter 0.1 becomes radius 0.05",
                       np.allclose(w.r, np.array([0.05, 0.05, 0.05])))
    failures += _check("default COMPDAT raw row is retained",
                       getattr(w, "compdat", [])[0][0] == "P1"
                       and getattr(w, "compdat", [])[0][3:5] == [1, 3])
    failures += _check("WELSPECS reference depth/group still attach",
                       np.isclose(w.refDepth, 123.0) and getattr(w, "group", None) == "FIELD")
    return failures


def validate_mixed_explicit_and_default_compdat_wells() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [
                {"well": "P1", "group": "A", "i": 1, "j": 1, "ref_depth": 100.0},
                {"well": "P2", "group": "B", "i": 2, "j": 1, "ref_depth": 200.0},
            ],
            "COMPDAT": [
                {"well": "P1", "k1": 2, "k2": 2, "status": "SHUT", "wi": 44.0, "radius": 0.2},
            ],
        },
    )

    by_name = {w.name: w for w in wells}
    failures = 0
    failures += _check("explicit and defaulted WELSPECS wells are both created",
                       set(by_name) == {"P1", "P2"})
    failures += _check("explicit COMPDAT well is not expanded by default rows",
                       np.array_equal(by_name["P1"].cells, np.array([4]))
                       and np.array_equal(by_name["P1"].cstatus, np.array([False])))
    failures += _check("missing COMPDAT well receives all-layer default completions",
                       np.array_equal(by_name["P2"].cells, np.array([1, 5, 9])))
    failures += _check("defaulted well keeps its own WELSPECS metadata",
                       getattr(by_name["P2"], "group", None) == "B"
                       and np.isclose(by_name["P2"].refDepth, 200.0))
    return failures


def main() -> int:
    print("\nPhase 172: MRST default COMPDAT insertion for WELSPECS-only wells")
    failures = validate_welspecs_only_well_gets_default_vertical_compdat()
    failures += validate_mixed_explicit_and_default_compdat_wells()
    if failures:
        print(f"\nPhase 172 failed ({failures} checks).")
        return 1
    print("\nWELSPECS-only wells now receive MRST-style default vertical COMPDAT completions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
