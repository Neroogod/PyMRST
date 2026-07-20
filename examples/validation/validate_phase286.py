"""Phase 286: natural-variable flash bookkeeping fields."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _model(nx: int = 4):
    grid = compute_geometry(cart_grid([nx, 1, 1], [40.0, 10.0, 10.0]))
    rock = make_rock(grid, 100.0 * MILLI * DARCY, 0.25)
    return NaturalVariablesCompositionalModel(grid, rock, simple_mixture(), temperature=350.0)


def main() -> int:
    model = _model()
    state = model.initial_state(np.linspace(55.0 * BARSA, 45.0 * BARSA, 4), [0.5, 0.3, 0.2])
    copied = state.copy()
    forced = state.copy()
    forced.flag = np.ones_like(forced.flag)
    flashed = model.flashPhases(forced, state, None, False, 1)
    flashed_again = model.flashPhases(flashed, state, None, False, 2)

    print("\nPhase 286: natural-variable flash bookkeeping fields")
    failures = 0
    failures += _check(
        "initial natural-variable state carries MRST flash fields",
        state.K.shape == state.z.shape
        and state.L.shape == state.pressure.shape
        and state.Z_L.shape == state.pressure.shape
        and state.Z_V.shape == state.pressure.shape
        and state.switched.shape == state.pressure.shape
        and state.switchCount.shape == state.pressure.shape,
    )
    failures += _check(
        "K, liquid fraction and compressibility factors are finite",
        np.all(np.isfinite(state.K))
        and np.all(np.isfinite(state.L))
        and np.all(np.isfinite(state.Z_L))
        and np.all(np.isfinite(state.Z_V))
        and np.all((state.L >= 0.0) & (state.L <= 1.0)),
    )
    failures += _check(
        "copy preserves independent flash bookkeeping arrays",
        np.allclose(copied.K, state.K)
        and copied.K is not state.K
        and np.allclose(copied.L, state.L)
        and copied.L is not state.L
        and copied.switchCount is not state.switchCount,
    )
    failures += _check(
        "flashPhases marks phase transitions and increments switchCount",
        np.all(flashed.switched)
        and np.allclose(flashed.switchCount, 1.0)
        and np.array_equal(model.getTwoPhaseFlag(flashed), np.ones_like(flashed.switched, dtype=bool)),
    )
    failures += _check(
        "reflashing unchanged flags keeps switchCount stable",
        not np.any(flashed_again.switched)
        and np.allclose(flashed_again.switchCount, flashed.switchCount),
    )

    if failures:
        print(f"\nPhase 286 failed ({failures} checks).")
        return 1
    print("\nNaturalVariablesState now preserves MRST flash bookkeeping: K, L, Z roots, switched mask and switchCount.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
