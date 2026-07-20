"""Phase 168: MRST COMPDAT/WELSPECS well geometry import."""

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
    grid = compute_geometry(cart_grid([2, 2, 3], [20.0, 20.0, 30.0]))
    rock = make_rock(grid, 100.0, 0.25)
    return grid, rock


def validate_positional_compdat_import() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [["P1", "FIELD", 1, 1, 1234.0]],
            "COMPDAT": [
                ["P1", 1, 1, 1, 2, "OPEN", 1, 100.0, 0.50, 0.0, 0.0, 0.0, "Z"],
                ["P1", 1, 1, 3, 3, "SHUT", 1, 300.0, 0.25, 0.0, 0.0, 0.0, "Z"],
            ],
        },
        compi=[0.0, 1.0, 0.0],
    )

    failures = 0
    failures += _check("one well is created from multi-row COMPDAT", len(wells) == 1)
    w = wells[0]
    failures += _check("COMPDAT K range expands to Cartesian cells",
                       np.array_equal(w.cells, np.array([0, 4, 8])))
    failures += _check("explicit COMPDAT WI is preserved per completion",
                       np.allclose(w.WI, np.array([100.0, 100.0, 300.0])))
    failures += _check("COMPDAT diameter becomes radius",
                       np.allclose(w.r, np.array([0.25, 0.25, 0.125])))
    failures += _check("COMPDAT status becomes cstatus",
                       np.array_equal(w.cstatus, np.array([True, True, False])))
    failures += _check("WELSPECS reference depth is preserved",
                       np.isclose(w.refDepth, 1234.0))
    failures += _check("WELSPECS group metadata is retained",
                       getattr(w, "group", None) == "FIELD")
    failures += _check("raw COMPDAT rows remain available for exact ports",
                       len(getattr(w, "compdat", [])) == 2)

    updated = apply_deck_well_controls(
        wells,
        {
            "WCONPROD": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 250.0]],
            "WELOPEN": [["P1", "SHUT", 1, 1, 2, 1, 3]],
        },
        grid=grid,
    )[0]
    failures += _check("imported COMPDAT well accepts WCONPROD updates",
                       updated.type == "bhp" and updated.sign == -1 and np.isclose(updated.val, 250.0))
    failures += _check("imported COMPDAT well accepts grid-aware WELOPEN updates",
                       np.array_equal(updated.cstatus, np.array([True, False, False])))
    failures += _check("deck update does not mutate imported base well",
                       np.array_equal(w.cstatus, np.array([True, True, False])))
    return failures


def validate_mapping_compdat_import() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [{"well": "P2", "group": "SAT", "i": 2, "j": 1, "ref_depth": 222.0}],
            "COMPDAT": [
                {"well": "P2", "k1": 1, "k2": 2, "status": "OPEN", "wi": 50.0, "radius": 0.2},
                {"well": "P2", "i": 2, "j": 1, "k1": 3, "k2": 3, "status": "SHUT", "wi": 75.0, "radius": 0.3},
            ],
        },
    )

    failures = 0
    failures += _check("mapping COMPDAT rows create one well", len(wells) == 1)
    w = wells[0]
    failures += _check("missing mapping I/J values fall back to WELSPECS",
                       np.array_equal(w.cells, np.array([1, 5, 9])))
    failures += _check("mapping WI/radius values are preserved",
                       np.allclose(w.WI, np.array([50.0, 50.0, 75.0]))
                       and np.allclose(w.r, np.array([0.2, 0.2, 0.3])))
    failures += _check("mapping COMPDAT status is preserved",
                       np.array_equal(w.cstatus, np.array([True, True, False])))
    failures += _check("mapping WELSPECS metadata is retained",
                       getattr(w, "group", None) == "SAT" and np.isclose(w.refDepth, 222.0))
    return failures


def main() -> int:
    print("\nPhase 168: MRST COMPDAT/WELSPECS well geometry import")
    failures = validate_positional_compdat_import()
    failures += validate_mapping_compdat_import()
    if failures:
        print(f"\nPhase 168 failed ({failures} checks).")
        return 1
    print("\nMRST-like WELSPECS/COMPDAT records now build native compositional wells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
