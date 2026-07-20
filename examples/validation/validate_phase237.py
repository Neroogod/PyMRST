"""Phase 237: runCompositionalSPE1 plot-section metadata parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.run_compositional_spe1 import (  # noqa: E402
    run_compositional_spe1_comparison,
    spe1_plot_sections,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    comparison = run_compositional_spe1_comparison(nsteps=1)
    sections = comparison.plot_sections
    helper = spe1_plot_sections(comparison)
    assert sections is not None
    toolbars = sections["state_toolbars"]
    wells = sections["well_solutions"]
    iterations = sections["cumulative_iterations"]

    print("\nPhase 237: runCompositionalSPE1 plot-section metadata parity")
    failures = 0
    failures += _check("comparison carries plot section bundle",
                       set(sections) == {"state_toolbars", "well_solutions",
                                         "cumulative_iterations"})
    failures += _check("state toolbars match MRST titles and views",
                       tuple(t["title"] for t in toolbars) == (
                           "Compositional Natural Variables",
                           "Compositional Molar Variables",
                           "Black-oil",
                       )
                       and all(t["axis"] == "tight" for t in toolbars)
                       and all(t["view"] == (30, 30) for t in toolbars))
    failures += _check("state toolbars follow MRST dataset order",
                       tuple(t["dataset"] for t in toolbars) == comparison.names
                       and [len(t["states"]) for t in toolbars] == [1, 1, 1])
    failures += _check("well solution section carries plotWellSols dataset names",
                       wells["datasetnames"] == comparison.names
                       and set(wells["datasets"]) == set(comparison.names)
                       and all(data["bhp"].shape == (1, 2)
                               for data in wells["datasets"].values()))
    failures += _check("cumulative iteration section mirrors MRST stairs plot",
                       iterations["plot"] == "stairs"
                       and iterations["legend"] == comparison.names
                       and iterations["ylabel"] == "Cumulative nonlinear iterations"
                       and iterations["cumulative"].shape == (1, 3)
                       and np.array_equal(iterations["iterations"], iterations["cumulative"]))
    failures += _check("standalone helper reproduces comparison payload",
                       helper["state_toolbars"][0]["title"] == toolbars[0]["title"]
                       and helper["well_solutions"]["datasetnames"] == wells["datasetnames"]
                       and np.array_equal(helper["cumulative_iterations"]["cumulative"],
                                          iterations["cumulative"]))

    if failures:
        print(f"\nPhase 237 failed ({failures} checks).")
        return 1
    print("\nSPE1 example now exposes MRST toolbar, well-solution, and iteration plot metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
