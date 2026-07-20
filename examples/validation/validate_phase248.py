"""Phase 248: separator example exposes allocation diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.compositional_separator_example import (  # noqa: E402
    evaluate_separator_example_allocation_diagnostics,
    setup_compositional_separator_packed_problems,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    packed = setup_compositional_separator_packed_problems((3, 3, 2), simulate=False)
    out = evaluate_separator_example_allocation_diagnostics((3, 3, 2))
    diagnostics = out["diagnostics"]
    states = out["states"]

    print("\nPhase 248: separator example allocation diagnostics")
    failures = 0
    failures += _check("allocation helper keeps MRST problem ordering",
                       out["names"] == ["Overall-NoSep", "Overall-MultiSep", "Overall-SingleSep"])
    failures += _check("packed setup still exposes diagnostics slot without forcing simulation",
                       packed["separator_allocation_diagnostics"] == [None, None, None])
    failures += _check("diagnostics list follows MRST packed problem order",
                       len(diagnostics) == 3
                       and diagnostics[0] is None
                       and diagnostics[1] is states[1].separator_group_diagnostics
                       and diagnostics[2] is states[2].separator_group_diagnostics)
    failures += _check("multi-stage and single separator cases carry allocation diagnostics",
                       diagnostics[1] is not None and diagnostics[2] is not None)

    for label, diag in zip(("multi-stage", "single"), diagnostics[1:]):
        failures += _check(f"{label} separator records the producer stream",
                           diag["producer_indices"] == [0]
                           and diag["producer_names"] == ["Producer"]
                           and diag["producer_component_mass_streams"].shape == (1, out["mixture"].n)
                           and np.allclose(
                               diag["producer_component_mass_streams"][0],
                               diag["total_component_mass_stream"],
                           ))
        failures += _check(f"{label} separator allocation is conservative",
                           np.isclose(diag["allocation_shares"].sum(), 1.0)
                           and np.all(diag["positive_surface_rates_total"] >= 0.0)
                           and np.allclose(
                               diag["positive_surface_rates_total"],
                               -diag["allocated_surface_rates"].sum(axis=0),
                           ))

    failures += _check("multi-stage and single separators produce different total rates",
                       not np.allclose(
                           diagnostics[1]["positive_surface_rates_total"],
                           diagnostics[2]["positive_surface_rates_total"],
                       ))

    if failures:
        print(f"\nPhase 248 failed ({failures} checks).")
        return 1
    print("\nThe separator example now exposes final allocation diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
