"""Phase 285: VLE introduction SSI-to-Newton diagnostics are active."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.intro_to_vapor_liquid_equilibrium import (  # noqa: E402
    spe5_flash_diagnostics_section,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    diagnostics = spe5_flash_diagnostics_section(ns=9)
    convergence = diagnostics["convergence"]
    timing = diagnostics["timing"]
    ssi_newton_active = convergence["active_cells"]["SSI+Newton"]
    ssi_newton_errors = convergence["errors"]["SSI+Newton"]

    print("\nPhase 285: VLE introduction SSI-to-Newton diagnostics are active")
    failures = 0
    failures += _check(
        "SSI+Newton is now an available diagnostic branch",
        diagnostics["newton_backend"]["available"]
        and convergence["available"] == ("SSI", "SSI+Newton"),
    )
    failures += _check(
        "SSI+Newton active-cell history is finite and decreases or stays bounded",
        ssi_newton_active.ndim == 1
        and ssi_newton_active.size > 0
        and ssi_newton_active[0] >= 0
        and ssi_newton_active[-1] <= ssi_newton_active[0],
    )
    failures += _check(
        "SSI+Newton residual history is finite and shape-aligned",
        ssi_newton_errors.shape == ssi_newton_active.shape
        and np.all(np.isfinite(ssi_newton_errors)),
    )
    failures += _check(
        "timing table reports finite SSI and SSI+Newton flash timings",
        timing["available"] == ("SSI", "SSI+Newton")
        and timing["seconds"].shape == (2, 2)
        and np.all(np.isfinite(timing["seconds"])),
    )

    if failures:
        print(f"\nPhase 285 failed ({failures} checks).")
        return 1
    print("\nintroToVaporLiquidEquilibrium now reports an active AD-linearized SSI-to-Newton flash diagnostic branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
