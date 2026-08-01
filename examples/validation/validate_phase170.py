"""Phase 170: COMPDAT KH and completion metadata parity."""

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
from pymrst.utils.compute_well_index import compute_well_index  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _grid_and_rock():
    grid = compute_geometry(cart_grid([1, 1, 3], [20.0, 20.0, 30.0]))
    rock = make_rock(grid, 100.0, 0.25)
    return grid, rock


def validate_kh_drives_wi_when_connection_wi_is_defaulted() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [["P1", "FIELD", 1, 1, 100.0]],
            "COMPDAT": [
                ["P1", 1, 1, 1, 1, "OPEN", 7, -1.0, 0.20, 10.0, 0.0, 0.25, "Z"],
                ["P1", 1, 1, 2, 2, "OPEN", 8, -1.0, 0.20, 30.0, 0.0, 0.50, "Z"],
            ],
        },
    )

    w = wells[0]
    expected = compute_well_index(
        grid,
        rock,
        np.array([0.10, 0.10]),
        np.array([0, 1]),
        well_dir="z",
        skin=np.array([0.0, 0.0]),
        kh=np.array([10.0, 30.0]),
    )

    failures = 0
    failures += _check("KH column drives computed WI when COMPDAT WI is defaulted",
                       np.allclose(w.WI, expected))
    failures += _check("different KH values produce different completion WI",
                       w.WI[1] > w.WI[0])
    failures += _check("COMPDAT SATNUM is preserved per completion",
                       np.array_equal(getattr(w, "satnum", None), np.array([7, 8])))
    failures += _check("COMPDAT KH metadata is preserved per completion",
                       np.allclose(getattr(w, "kh", None), np.array([10.0, 30.0])))
    failures += _check("COMPDAT D-factor metadata is preserved per completion",
                       np.allclose(getattr(w, "d_factor", None), np.array([0.25, 0.50])))
    return failures


def validate_mixed_explicit_wi_overrides_kh_for_that_completion() -> int:
    grid, rock = _grid_and_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        {
            "WELSPECS": [{"well": "P2", "group": "FIELD", "i": 1, "j": 1, "ref_depth": 100.0}],
            "COMPDAT": [
                {"well": "P2", "k1": 1, "k2": 1, "status": "OPEN", "wi": 123.0, "kh": 10.0, "radius": 0.1},
                {"well": "P2", "k1": 2, "k2": 2, "status": "OPEN", "kh": 40.0, "radius": 0.1},
            ],
        },
    )

    w = wells[0]
    kh_wi = compute_well_index(
        grid,
        rock,
        np.array([0.1]),
        np.array([1]),
        well_dir="z",
        kh=np.array([40.0]),
    )[0]

    failures = 0
    failures += _check("explicit COMPDAT WI overrides KH on that completion",
                       np.isclose(w.WI[0], 123.0))
    failures += _check("defaulted COMPDAT WI still uses KH on other completions",
                       np.isclose(w.WI[1], kh_wi))
    failures += _check("mixed explicit/default WI preserves KH metadata",
                       np.allclose(getattr(w, "kh", None), np.array([10.0, 40.0])))
    return failures


def main() -> int:
    print("\nPhase 170: COMPDAT KH and completion metadata parity")
    failures = validate_kh_drives_wi_when_connection_wi_is_defaulted()
    failures += validate_mixed_explicit_wi_overrides_kh_for_that_completion()
    if failures:
        print(f"\nPhase 170 failed ({failures} checks).")
        return 1
    print("\nCOMPDAT KH now feeds WI calculation and completion metadata is preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
