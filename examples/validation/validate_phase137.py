"""Phase 137: runCompositionalSPE1 plotting-data parity."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.examples.run_compositional_spe1 import (
    run_compositional_spe1_comparison,
    spe1_cumulative_iterations,
    spe1_reference_pseudo_component_rates,
    spe1_well_solution_datasets,
)


def _check(name: str, ok: bool) -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if ok else 1


def main() -> int:
    comparison = run_compositional_spe1_comparison(nsteps=1)
    wells = spe1_well_solution_datasets(comparison)
    pseudo = spe1_reference_pseudo_component_rates(comparison.reference)
    its = spe1_cumulative_iterations(comparison)

    print("\nPhase 137: runCompositionalSPE1 plotting-data parity")
    failures = 0
    failures += _check("comparison preserves MRST dataset names",
                       comparison.names == (
                           "MRST-NaturalVariables",
                           "MRST-MolarVariables",
                           "MRST-BlackOil",
                       ))
    failures += _check("well solution datasets expose BHP and phase rates",
                       wells["names"] == comparison.names
                       and set(wells["datasets"]) == set(comparison.names)
                       and all(data["bhp"].shape == (1, 2)
                               and data["qOs"].shape == (1, 2)
                               and data["qGs"].shape == (1, 2)
                               for data in wells["datasets"].values()))
    failures += _check("reference pseudo-component rates mirror MATLAB fields",
                       pseudo["PseudoOil"].shape == (1, 2)
                       and pseudo["PseudoGas"].shape == (1, 2)
                       and pseudo["rhoOS"] > pseudo["rhoGS"] > 0.0
                       and np.all(np.isfinite(pseudo["PseudoOil"]))
                       and np.all(np.isfinite(pseudo["PseudoGas"])))
    failures += _check("cumulative nonlinear iterations have MRST stairs shape",
                       its["names"] == comparison.names
                       and its["iterations"].shape == (1, 3)
                       and its["cumulative"].shape == (1, 3)
                       and np.array_equal(its["iterations"], its["cumulative"]))
    failures += _check("native/reference flags match MRST comparison paths",
                       comparison.natural.native
                       and comparison.overall.native
                       and not comparison.reference.native)

    if failures:
        print(f"\nPhase 137 failed ({failures} checks).")
    else:
        print("\nSPE1 example now exposes MRST-shaped well/rate/iteration plot data.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
