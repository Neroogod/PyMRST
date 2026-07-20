"""Phase 173: MRST last-record-wins overlap handling for COMPDAT."""

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
    grid = compute_geometry(cart_grid([1, 1, 4], [10.0, 10.0, 40.0]))
    rock = make_rock(grid, 100.0, 0.25)
    return grid, rock


def validate_same_cell_completion_keeps_last_record() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [["P1", "FIELD", 1, 1, 100.0]],
            "COMPDAT": [
                ["P1", 1, 1, 1, 1, "OPEN", 1, 10.0, 0.20, -1.0, 0.0, 0.0, "Z"],
                ["P1", 1, 1, 2, 2, "OPEN", 2, 20.0, 0.20, -1.0, 0.0, 0.0, "Z"],
                ["P1", 1, 1, 1, 1, "SHUT", 9, 99.0, 0.40, -1.0, 3.0, 0.5, "Z"],
            ],
        },
    )

    w = wells[0]
    failures = 0
    failures += _check("duplicate cell keeps one completion with last-row position",
                       np.array_equal(w.cells, np.array([1, 0])))
    failures += _check("last duplicate row overrides WI and radius",
                       np.allclose(w.WI, np.array([20.0, 99.0]))
                       and np.allclose(w.r, np.array([0.10, 0.20])))
    failures += _check("last duplicate row overrides status and status text",
                       np.array_equal(w.cstatus, np.array([True, False]))
                       and np.array_equal(getattr(w, "cstatus_text", None),
                                          np.array(["OPEN", "SHUT"], dtype=object)))
    failures += _check("last duplicate row overrides completion metadata",
                       np.array_equal(getattr(w, "satnum", None), np.array([2, 9]))
                       and np.allclose(getattr(w, "d_factor", None), np.array([0.0, 0.5])))
    failures += _check("raw COMPDAT rows remain available even after collapse",
                       len(getattr(w, "compdat", [])) == 3)
    return failures


def validate_overlapping_k_ranges_keep_last_record_per_cell() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [{"well": "P2", "group": "FIELD", "i": 1, "j": 1, "ref_depth": 100.0}],
            "COMPDAT": [
                {"well": "P2", "k1": 1, "k2": 3, "status": "OPEN", "wi": 10.0, "radius": 0.1, "satnum": 1},
                {"well": "P2", "k1": 2, "k2": 4, "status": "STOP", "wi": 40.0, "radius": 0.4, "satnum": 4},
            ],
        },
    )

    w = wells[0]
    failures = 0
    failures += _check("overlapping K ranges retain MRST last-occurrence ordering",
                       np.array_equal(w.cells, np.array([0, 1, 2, 3])))
    failures += _check("non-overlapped first cell keeps first row metadata",
                       np.isclose(w.WI[0], 10.0) and np.isclose(w.r[0], 0.1)
                       and w.cstatus[0] and getattr(w, "satnum", None)[0] == 1)
    failures += _check("overlapped cells keep later row metadata",
                       np.allclose(w.WI[1:], np.array([40.0, 40.0, 40.0]))
                       and np.allclose(w.r[1:], np.array([0.4, 0.4, 0.4]))
                       and np.array_equal(w.cstatus[1:], np.array([False, False, False]))
                       and np.array_equal(getattr(w, "cstatus_text", None)[1:],
                                          np.array(["STOP", "STOP", "STOP"], dtype=object)))
    return failures


def main() -> int:
    print("\nPhase 173: MRST last-record-wins overlap handling for COMPDAT")
    failures = validate_same_cell_completion_keeps_last_record()
    failures += validate_overlapping_k_ranges_keep_last_record_per_cell()
    if failures:
        print(f"\nPhase 173 failed ({failures} checks).")
        return 1
    print("\nOverlapping COMPDAT completions now keep the last record per cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
