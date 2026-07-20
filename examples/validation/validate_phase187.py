"""Phase 187: separator example exposes MRST-style stream diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core import SeparatorStreamResult  # noqa: E402
from pymrst.modules.compositional.examples.book_ii.compositional_separator_example import (  # noqa: E402
    run_compositional_separator_example,
    setup_compositional_separator_packed_problems,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _phase_rows(streams) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in streams])


def validate_separator_example_diagnostics() -> int:
    out = run_compositional_separator_example()
    diagnostics = out["tree_separator_streams"]
    tree = out["tree"]
    final = out["tree_surface_mole_streams"]
    total_final = final[0] + final[1]

    failures = 0
    failures += _check("separator example returns MRST-style diagnostics object",
                       isinstance(diagnostics, SeparatorStreamResult))
    failures += _check("example diagnostics are the tree surface-rate diagnostics",
                       diagnostics is tree.separator_streams)
    failures += _check("example direct and carried surface streams match",
                       np.allclose(_phase_rows(final), _phase_rows(tree.surface_mole_streams)))
    failures += _check("example tree diagnostics include every separator stage",
                       len(diagnostics.molestreams) == len(out["tree_separator"].separators)
                       and len(diagnostics.densities) == len(out["tree_separator"].separators)
                       and len(diagnostics.molefractions) == len(out["tree_separator"].separators))
    failures += _check("example final surface streams conserve input components",
                       np.allclose(total_final.reshape(-1), out["stream_mole"], atol=1.0e-10))
    return failures


def validate_packed_separator_problem_layout_still_matches_mrst() -> int:
    out = setup_compositional_separator_packed_problems()
    failures = 0
    failures += _check("packed separator example keeps MRST problem order",
                       out["names"] == ["Overall-NoSep", "Overall-MultiSep", "Overall-SingleSep"])
    failures += _check("multi-separator packed model uses the tree separator",
                       out["problem_msep"].model.separator_group is out["tree_separator"])
    failures += _check("single-separator packed model uses the single separator",
                       out["problem_sep"].model.separator_group is out["single_separator"])
    failures += _check("base packed model has no separator group",
                       out["problem"].model.separator_group is None)
    return failures


def main() -> int:
    print("\nPhase 187: separator example stream-diagnostics parity")
    failures = validate_separator_example_diagnostics()
    failures += validate_packed_separator_problem_layout_still_matches_mrst()
    if failures:
        print(f"\nPhase 187 failed ({failures} checks).")
        return 1
    print("\nThe separator example now exposes MRST-style intermediate stream diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
