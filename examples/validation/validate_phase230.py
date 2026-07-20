"""Phase 230: VLE introduction SPE5 flash diagnostics parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.intro_to_vapor_liquid_equilibrium import (  # noqa: E402
    run_intro_to_vapor_liquid_equilibrium,
    spe5_flash_diagnostics_section,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 230: VLE introduction SPE5 flash diagnostics")
    diagnostics = spe5_flash_diagnostics_section(ns=10)
    all_sections = run_intro_to_vapor_liquid_equilibrium(n_rr=11, n_ternary=8, ns_spe5=8)

    convergence = diagnostics["convergence"]
    timing = diagnostics["timing"]
    failures = 0
    failures += _check("iteration map follows SPE5 pressure-temperature grid",
                       diagnostics["ssi_iteration_map"].shape == (10, 10)
                       and diagnostics["P"].shape == (10, 10)
                       and diagnostics["T"].shape == (10, 10))
    failures += _check("iteration map marks inactive cells as NaN",
                       np.any(np.isnan(diagnostics["ssi_iteration_map"])))
    failures += _check("SSI convergence history is one-dimensional",
                       convergence["active_cells"]["SSI"].ndim == 1
                       and convergence["errors"]["SSI"].shape == convergence["active_cells"]["SSI"].shape)
    failures += _check("SSI active-cell history starts nonnegative and ends no larger",
                       convergence["active_cells"]["SSI"][0] >= 0
                       and convergence["active_cells"]["SSI"][-1] <= convergence["active_cells"]["SSI"][0])
    failures += _check("MRST Newton branch is represented as an available SSI-to-Newton slot",
                       convergence["names"] == ("SSI", "SSI+Newton")
                       and convergence["available"] == ("SSI", "SSI+Newton")
                       and convergence["active_cells"]["SSI+Newton"].ndim == 1
                       and np.all(np.isfinite(convergence["errors"]["SSI+Newton"]))
                       and diagnostics["newton_backend"]["available"])
    failures += _check("timing table has MRST stability/flash rows for both flash paths",
                       timing["rows"] == ("Stability", "Flash")
                       and timing["names"] == ("SSI", "SSI+Newton")
                       and timing["seconds"].shape == (2, 2)
                       and np.all(np.isfinite(timing["seconds"])))
    failures += _check("combined runner returns diagnostics section",
                       "spe5_diagnostics" in all_sections
                       and all_sections["spe5_diagnostics"]["ssi_iteration_map"].shape == (8, 8))

    if failures:
        print(f"\nPhase 230 validation failed ({failures} checks).")
    else:
        print("\nVLE introduction now exposes MRST-shaped SPE5 flash diagnostics.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
