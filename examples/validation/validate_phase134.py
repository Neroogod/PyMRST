"""Phase 134: larger compositional tutorial timing-table parity."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.examples.compositional_larger_problem_tutorial import (
    larger_problem_backends,
    larger_problem_tick_labels,
    larger_problem_time_matrix,
    make_larger_problem_case,
    run_compositional_larger_problem_tutorial,
)
from pymrst.utils.units import DAY


def _check(name: str, ok: bool) -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if ok else 1


def main() -> int:
    case = make_larger_problem_case((3, 3, 2), use_bc=False, include_water=False, use_natural=True)
    bc_case = make_larger_problem_case((3, 3, 2), use_bc=True, include_water=True, use_natural=False)
    run = run_compositional_larger_problem_tutorial(
        dims=(2, 2, 2),
        use_bc=False,
        include_water=False,
        use_natural=False,
        dt=0.01 * DAY,
    )
    rows = run["timings"]
    matrix = larger_problem_time_matrix(rows)
    available_rows = [row for row in rows if row.available and row.total_time is not None]
    labels = [row.backend for row in rows]
    backend_labels = [backend.label() for backend in larger_problem_backends()]

    print("\nPhase 134: larger compositional tutorial timing-table parity")
    failures = 0
    failures += _check("well setup keeps MRST injector/producer corners",
                       case["dims"] == (3, 3, 2)
                       and len(case["wells"]) == 2
                       and case["wells"][0].name == "Inj"
                       and case["wells"][1].name == "Prod")
    failures += _check("water BC setup keeps MRST component streams",
                       bc_case["bc"] is not None
                       and not bc_case["wells"]
                       and bc_case["bc"].components.shape[1] == bc_case["mixture"].n
                       and bc_case["model"].water is True)
    failures += _check("backend rows mirror MRST tutorial plus direct solve",
                       labels[:3] == backend_labels
                       and labels[-1] == "Sparse+Direct solver")
    failures += _check("timing matrix has MRST plotted columns",
                       matrix.shape == (len(available_rows), 3)
                       and np.all(np.isfinite(matrix))
                       and np.all(matrix >= 0.0))
    failures += _check("timing row stores assembly/linear/total split",
                       all(row.assembly_time is not None
                           and row.linear_solve_time is not None
                           and row.total_time is not None
                           for row in available_rows))
    failures += _check("tick labels follow MRST missing-MEX branch",
                       run["tick_labels"] == larger_problem_tick_labels(
                           any(row.backend == "Diagonal (MEX+RowMajor)" and row.available
                               for row in rows)
                       ))
    failures += _check("direct solver row is always available",
                       rows[-1].backend == "Sparse+Direct solver"
                       and rows[-1].available
                       and rows[-1].converged is True)

    if failures:
        print(f"\nPhase 134 failed ({failures} checks).")
    else:
        print("\nLarger compositional tutorial now exposes MRST-shaped timing tables.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
