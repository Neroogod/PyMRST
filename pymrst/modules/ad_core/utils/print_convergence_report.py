"""MRST ``ad-core/utils/printConvergenceReport.m``: one table row per
nonlinear iteration - the iteration number, then each convergence value,
starred where converged - with the header on the first iteration and a
closing rule at the end of the block (by default when every value has
converged after the first iteration)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def print_convergence_report(names: Sequence[str], values: Sequence[float],
                             converged: Sequence[bool], iteration: int,
                             endOfBlock: Optional[Any] = None) -> None:
    """MRST `printConvergenceReport(names, values, converged, iteration, endOfBlock)`."""
    converged = np.atleast_1d(np.asarray(converged, dtype=bool))
    if endOfBlock is None:
        endOfBlock = bool(np.all(converged)) and iteration > 1
    nl = [len(n) for n in names]
    sep = "="*(sum(max(n, 8) for n in nl) + 3*len(nl) + 8)
    out = []
    if iteration == 1:
        out.append(f"{sep}\n")
        out.append("| It # ")
        out.extend(f"| {name:<8s} " for name in names)
        out.append(f"|\n{sep}\n")
    out.append(f"| {iteration:4d} ")
    for i, v in enumerate(np.atleast_1d(values)):
        linen = max(nl[i], 8)
        out.append("|" + ("*" if converged[i] else " ") + f"{_matlab_e(v, linen)} ")
    out.append("|\n")
    if np.all(endOfBlock):
        out.append(f"{sep}\n")
    print("".join(out), end="")


def _matlab_e(v: float, width: int) -> str:
    """MATLAB's `%-<width>.2e`: left-aligned, Inf and NaN spelt as MATLAB does."""
    if np.isnan(v):
        text = "NaN"
    elif np.isinf(v):
        text = "Inf" if v > 0 else "-Inf"
    else:
        text = f"{v:.2e}"
    return f"{text:<{width}s}"


printConvergenceReport = print_convergence_report
__all__ = ["printConvergenceReport", "print_convergence_report"]


if __name__ == "__main__":
    import contextlib
    import io

    def test_header_rows_and_closing_rule():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_convergence_report(["CNV_W", "closureWells (well)"], [1.5e-2, 3e-9], [False, True], 1)
            print_convergence_report(["CNV_W", "closureWells (well)"], [1e-4, 1e-10], [True, True], 2)
        lines = buf.getvalue().splitlines()
        assert lines[0] == "="*(8 + 19 + 6 + 8) and lines[1].startswith("| It # | CNV_W    | closure")
        assert lines[3] == "|    1 | 1.50e-02 |*3.00e-09            |"
        assert lines[4].startswith("|    2 |*1.00e-04 ") and lines[5] == lines[0]
        print("ok test_header_rows_and_closing_rule")

    test_header_rows_and_closing_rule()
    print("\nAll tests passed!")
