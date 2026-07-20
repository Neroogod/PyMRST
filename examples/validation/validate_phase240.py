"""Phase 240: exact compositional Contents.m catalog parity."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.contents import (  # noqa: E402
    EXAMPLES,
    MRST_CONTENTS_EXAMPLES,
    list_mrst_contents_examples,
)


EXPECTED_CONTENTS = (
    (
        "compositionalBoundaryConditionsExample",
        "Example comparing natural variables and overall composition with boundary conditions",
    ),
    (
        "compositionalExample3DSixComponents",
        "Example demonstrating a three dimensional, six component problem",
    ),
    (
        "compositionalLargerProblemTutorial",
        "Conceptual example on how to run larger compositional problems",
    ),
    (
        "compositionalValidationSimple",
        "Compare MRST results with other simulators",
    ),
    (
        "runCompositionalSPE1",
        "Example demonstrating compositional solvers with K-values",
    ),
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    listed = list_mrst_contents_examples()
    print("\nPhase 240: exact compositional Contents.m catalog parity")
    failures = 0
    failures += _check("Contents.m entries match MRST order and text",
                       listed == EXPECTED_CONTENTS)
    failures += _check("MRST_CONTENTS_EXAMPLES is immutable tuple metadata",
                       isinstance(MRST_CONTENTS_EXAMPLES, tuple)
                       and all(isinstance(item, tuple) for item in MRST_CONTENTS_EXAMPLES))
    failures += _check("public helper returns a defensive tuple copy",
                       listed == MRST_CONTENTS_EXAMPLES
                       and listed is not MRST_CONTENTS_EXAMPLES)
    failures += _check("all Contents.m examples remain in expanded PyMRST catalog",
                       all(name in EXAMPLES for name, _ in listed))
    failures += _check("expanded catalog may include book-ii examples separately",
                       "book-ii/introToVaporLiquidEquilibrium" in EXAMPLES
                       and len(EXAMPLES) > len(listed))

    if failures:
        print(f"\nPhase 240 failed ({failures} checks).")
        return 1
    print("\nCompositional Contents.m now has exact entry/order/description metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
