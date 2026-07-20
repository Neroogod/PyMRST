"""Phase 129: MRST-shaped separator example packed problems.

The MATLAB `compositionalSeparatorExample.m` builds one SPE5 depletion schedule
and packs three model alternatives: no separator, a multi-stage separator tree,
and a single separator. This native regression checks that the PyMRST example
now exposes the same setup structure, while Phase 116 remains the MATLAB-backed
numerical validation of the separator flash/tree physics.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.compositional_separator_example import (  # noqa: E402
    setup_compositional_separator_packed_problems,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    out = setup_compositional_separator_packed_problems(dims=(3, 3, 2), simulate=False)
    problems = out["problems"]
    schedule = out["schedule"]

    print("\nPhase 129: separator example packed-problem structure")
    failures = 0
    failures += _check("three MRST separator problems",
                       len(problems) == 3)
    failures += _check("MRST problem ordering",
                       out["names"] == ["Overall-NoSep", "Overall-MultiSep", "Overall-SingleSep"])
    failures += _check("common depletion_spe5 directory",
                       [p.directory for p in problems] == ["depletion_spe5"] * 3)
    failures += _check("all problems share one schedule object",
                       all(p.schedule is schedule for p in problems))
    failures += _check("all problems share one initial state object",
                       all(p.state0 is out["problem"].state0 for p in problems))
    failures += _check("plain model has no separator",
                       out["problem"].model.separator_group is None)
    failures += _check("multi separator attached to middle problem",
                       out["problem_msep"].model.separator_group is out["tree_separator"])
    failures += _check("single separator attached to final problem",
                       out["problem_sep"].model.separator_group is out["single_separator"])
    failures += _check("scaled SPE5 grid",
                       out["grid"].cells.num == 18 and out["mixture"].n == 6)
    failures += _check("one rate-controlled producer",
                       len(out["wells"]) == 1
                       and out["wells"][0].name == "Producer"
                       and out["wells"][0].type == "rate"
                       and out["wells"][0].val < 0.0)
    failures += _check("schedule carries producer control",
                       schedule.step.control.tolist() == [1]
                       and schedule.control[0].wells[0].name == "Producer")
    failures += _check("MRST four-stage tree destination",
                       np.array_equal(
                           out["tree_separator"].phase_destination,
                           np.array([[2, 3], [3, 4], [4, 0], [0, 0]]),
                       ))
    failures += _check("packed problems are not run by default",
                       all(p.result is None for p in problems)
                       and out["states"] == []
                       and out["reports"] == [])

    if failures:
        print(f"\nPhase 129 failed ({failures} checks).")
        return 1
    print("\nSeparator example now exposes MRST-shaped packed problems.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
