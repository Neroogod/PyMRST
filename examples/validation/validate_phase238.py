"""Phase 238: setupSimpleCompositionalExample schedule/reference parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.utils.setup_simple_compositional_example import (  # noqa: E402
    setup_simple_compositional_example,
)
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    overall = setup_simple_compositional_example(False, nx=6, nstep=3, timestep=5.0 * DAY)
    natural = setup_simple_compositional_example(True, nx=6, nstep=2, timestep=7.0 * DAY)

    print("\nPhase 238: setupSimpleCompositionalExample schedule/reference parity")
    failures = 0
    failures += _check("setup returns MRST-shaped schedule object",
                       overall.schedule.step.val.shape == (3,)
                       and np.allclose(overall.schedule.step.val, 5.0 * DAY)
                       and np.array_equal(overall.schedule.step.control, np.ones(3, dtype=np.int64))
                       and overall.schedule.control[0].wells is not overall.wells)
    failures += _check("schedule wells mirror MRST component assignment",
                       len(overall.wells) == 2
                       and all(np.allclose(well.components, [0.0, 1.0, 0.0])
                               for well in overall.schedule.control[0].wells)
                       and all(np.allclose(well.compi, [1.0, 0.0])
                               for well in overall.schedule.control[0].wells))
    failures += _check("model-facing producer stream remains numerically stable",
                       np.allclose(overall.wells[0].components, [0.0, 1.0, 0.0])
                       and np.allclose(overall.wells[1].components, overall.info.initial))
    failures += _check("well limits are cleared like MRST utility",
                       all(well.lims is None for well in overall.wells))
    failures += _check("natural branch still selects natural model",
                       isinstance(natural.model, NaturalVariablesCompositionalModel)
                       and natural.schedule.step.val.shape == (2,)
                       and np.allclose(natural.schedule.step.val, 7.0 * DAY))
    failures += _check("initial state and mixture constants remain unchanged",
                       overall.grid.cells.num == 6
                       and overall.mixture.names == ("Methane", "CarbonDioxide", "n-Decane")
                       and np.allclose(overall.info.initial, [0.6, 0.1, 0.3])
                       and np.allclose(overall.state.z.sum(axis=1), 1.0))
    failures += _check("reference comparison payload is explicit unavailable metadata",
                       overall.reference["available"] is False
                       and overall.reference["mrst_file"] == "comparison.mat"
                       and "not bundled" in overall.reference["reason"])

    if failures:
        print(f"\nPhase 238 failed ({failures} checks).")
        return 1
    print("\nsetupSimpleCompositionalExample now returns MRST-shaped schedule/reference metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
