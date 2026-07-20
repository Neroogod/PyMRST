"""Phase 231: MRST-shaped performance timing plot table."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.compositional_performance_example import (  # noqa: E402
    mrst_performance_plot_table,
    run_compositional_performance_example,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p117_compositional_performance.mat"


def _labels(raw) -> list[str]:
    arr = np.asarray(raw, dtype=object).ravel()
    return [str(x[0] if isinstance(x, np.ndarray) else x) for x in arr]


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    if not REF.exists():
        print("ref_p117_compositional_performance.mat missing (run dump_phase117_reference.m)")
        return 1

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    rows = run_compositional_performance_example((3, 3, 2))
    table = mrst_performance_plot_table(rows)

    print("\nPhase 231: MRST-shaped performance timing plot table")
    failures = 0
    failures += _check("plot labels match MRST backend rows",
                       table["labels"] == _labels(ref["backend_labels"]))
    failures += _check("plot table has MRST timing columns",
                       table["legend"] == ["Assembly / each", "Linear solve / it", "Total / it"]
                       and table["ylabel"] == "Time [s]"
                       and table["time"].shape == (4, 3))
    failures += _check("plain PyMRST RowMajor row is kept out of MRST plot table",
                       "Diagonal (RowMajor)" not in table["labels"]
                       and any(row.backend == "Diagonal (RowMajor)" for row in rows))
    failures += _check("available rows have finite timing values",
                       np.all(np.isfinite(table["time"][table["available"]])))
    failures += _check("unavailable rows are represented by NaN timings and reasons",
                       np.all(np.isnan(table["time"][~table["available"]]))
                       and all(reason for reason, ok in zip(table["reasons"], table["available"])
                               if not ok))

    if failures:
        print(f"\nPhase 231 failed ({failures} checks).")
        return 1
    print("\nPerformance example now exposes MRST's exact four-row timing table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
