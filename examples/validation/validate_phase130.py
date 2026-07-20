"""Phase 130: MRST-shaped VLE introduction example sections.

The MATLAB `introToVaporLiquidEquilibrium.m` contains three conceptual
sections: two-component Rachford-Rice, a ternary Peng-Robinson flash map, and
an SPE5 pressure/temperature flash map. This native regression checks that the
PyMRST example exposes the same data-producing sections at scaled sizes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.intro_to_vapor_liquid_equilibrium import (  # noqa: E402
    rachford_rice_section,
    run_intro_to_vapor_liquid_equilibrium,
    spe5_flash_diagnostics_section,
    spe5_phase_diagram_section,
    ternary_flash_section,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    rr = rachford_rice_section(n=51)
    ternary = ternary_flash_section(n=16)
    spe5 = spe5_phase_diagram_section(ns=14)
    diagnostics = spe5_flash_diagnostics_section(ns=10)
    all_sections = run_intro_to_vapor_liquid_equilibrium(n_rr=21, n_ternary=10, ns_spe5=10)

    print("\nPhase 130: VLE introduction example sections")
    failures = 0
    failures += _check("Rachford-Rice uses heavy/light K-values",
                       np.allclose(rr["K"], [0.1, 10.0]))
    failures += _check("Rachford-Rice midpoint solution",
                       np.isclose(rr["vapour_fraction"][25], 0.5, atol=1.0e-10))
    failures += _check("Rachford-Rice objective crosses zero near V=0.5",
                       abs(rr["objective_rr"][np.argmin(np.abs(rr["objective_v"] - 0.5))]) < 0.1)
    failures += _check("ternary grid has MRST composition columns",
                       ternary["composition"].shape[1] == 3
                       and np.allclose(ternary["composition"].sum(axis=1), 1.0))
    failures += _check("ternary map contains two-phase values",
                       np.any(np.isfinite(ternary["liquid_fraction"])))
    failures += _check("ternary invalid triangle is masked",
                       np.any(~ternary["valid"]))
    failures += _check("SPE5 pressure-temperature grid shape",
                       spe5["P"].shape == (14, 14)
                       and spe5["T"].shape == (14, 14))
    failures += _check("SPE5 flash uses six-component benchmark",
                       spe5["mixture"].n == 6
                       and spe5["composition"].shape == (14 * 14, 6))
    failures += _check("SPE5 map records active two-phase cells",
                       np.any(np.isfinite(spe5["active_cells"])))
    failures += _check("SPE5 diagnostics expose MRST convergence tables",
                       diagnostics["ssi_iteration_map"].shape == (10, 10)
                       and diagnostics["convergence"]["names"] == ("SSI", "SSI+Newton")
                       and diagnostics["timing"]["seconds"].shape == (2, 2))
    failures += _check("combined runner returns all MRST sections",
                       set(all_sections) == {"rachford_rice", "ternary", "spe5",
                                             "spe5_diagnostics", "line_flash"})

    if failures:
        print(f"\nPhase 130 failed ({failures} checks).")
        return 1
    print("\nVLE introduction example now mirrors MRST's section structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
