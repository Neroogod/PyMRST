"""Phase 308: MRST repeated perforated-cell facility utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_repeated_perforation_collapse() -> int:
    wc = np.array([4, 2, 4, 7, 2], dtype=np.int64)
    phase_a = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    phase_b = [
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]),
        np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    ]

    unique, collapsed_a, collapsed_b = FacilityModel.handle_repeated_perforated_cells(wc, phase_a, phase_b)
    unchanged_wc, unchanged = FacilityModel.handleRepeatedPerforatedcells([1, 3], [np.array([8.0, 9.0])])
    combined = FacilityModel.combineCellData(
        [
            [np.array([[1.0], [2.0]]), np.array([[3.0]])],
            [np.array([[4.0]]), np.array([[5.0], [6.0]])],
        ],
        1,
    )

    failures = 0
    failures += _check("repeated cells collapse in first-occurrence order",
                       np.array_equal(unique, [4, 2, 7]))
    failures += _check("single phase source rows sum onto unique cells",
                       np.allclose(collapsed_a[0], [4.0, 7.0, 4.0]))
    failures += _check("multiple source blocks and vector rows collapse together",
                       np.allclose(collapsed_b[0], [[4.0, 40.0], [7.0, 70.0], [4.0, 40.0]])
                       and np.allclose(collapsed_b[1], [40.0, 70.0, 40.0]))
    failures += _check("no-repeat inputs are returned unchanged by alias",
                       np.array_equal(unchanged_wc, [1, 3])
                       and np.allclose(unchanged[0], [8.0, 9.0]))
    failures += _check("combineCellData vertically concatenates selected entries",
                       np.allclose(combined, [[3.0], [5.0], [6.0]]))
    return failures


def main() -> int:
    print("\nPhase 308: repeated perforated-cell facility utility parity")
    failures = validate_repeated_perforation_collapse()
    if failures:
        print(f"\nPhase 308 failed ({failures} checks).")
        return 1
    print("\nRepeated perforated-cell handling is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
