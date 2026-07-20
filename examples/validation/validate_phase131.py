"""Phase 131: exact single-cell natural/overall example setup.

MRST's `showNaturalOverall.m` and `showNaturalOverall2.m` are single-cell
boundary-driven examples. Earlier PyMRST ports used a two-cell workaround.
This regression checks that the examples now use the same one-cell grid,
boundary-face layout, component boundary streams, and MRST timestep defaults.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.show_natural_overall import (  # noqa: E402
    run_show_natural_overall,
)
from pymrst.modules.compositional.examples.book_ii.show_natural_overall2 import (  # noqa: E402
    run_show_natural_overall2,
)
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _boundary_setup_ok(out, ncomp: int) -> bool:
    grid = out["grid"]
    bc = out["bc"]
    return (
        grid.cells.num == 1
        and bc.face.size == 2
        and np.all(grid.faces.neighbors[bc.face].min(axis=1) < 0)
        and bc.components.shape == (2, ncomp)
        and np.allclose(bc.components.sum(axis=1), 1.0)
        and np.isclose(bc.value[0], 1.0 / DAY)
    )


def main() -> int:
    first = run_show_natural_overall()
    second = run_show_natural_overall2()

    print("\nPhase 131: single-cell natural/overall example parity")
    failures = 0
    failures += _check("showNaturalOverall uses one MRST grid cell",
                       _boundary_setup_ok(first, first["mixture"].n))
    failures += _check("showNaturalOverall solves both formulations",
                       first["iterations"].shape == (2,)
                       and np.all(np.isfinite(first["natural_state"].pressure))
                       and np.all(np.isfinite(first["overall_state"].pressure)))
    failures += _check("showNaturalOverall2 uses one MRST grid cell",
                       _boundary_setup_ok(second, second["mixture"].n))
    failures += _check("showNaturalOverall2 keeps MRST bad initial guess",
                       np.isclose(second["init_guess_overall"].z[0, 0], 1.0, atol=1.0e-6)
                       and np.isclose(second["init_guess_natural"].z[0, 0], 1.0, atol=1.0e-6))
    failures += _check("showNaturalOverall2 solves both formulations",
                       second["iterations"].shape == (2,)
                       and np.all(np.isfinite(second["natural_state"].pressure))
                       and np.all(np.isfinite(second["overall_state"].pressure)))

    if failures:
        print(f"\nPhase 131 failed ({failures} checks).")
        return 1
    print("\nNatural/overall examples now use MRST's single-cell setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
