"""Phase 259: getComponentsTwoPhaseSimpleWater utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.utils import (  # noqa: E402
    getComponentsTwoPhaseSimpleWater,
    get_components_two_phase_simple_water,
)


class _Model:
    def __init__(self, water: bool):
        self.water = water


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _none_at(value) -> bool:
    return value is None


def main() -> int:
    xM = [np.array([0.2, 0.3]), np.array([0.8, 0.7])]
    yM = [np.array([0.6, 0.4]), np.array([0.4, 0.6])]
    sT = np.array([0.1, 0.25])

    no_water = getComponentsTwoPhaseSimpleWater(
        _Model(False),
        [700.0, 4.0],
        sT,
        xM,
        yM,
    )
    water = get_components_two_phase_simple_water(
        _Model(True),
        [1000.0, 650.0, 3.5],
        sT,
        xM,
        yM,
    )

    print("\nPhase 259: getComponentsTwoPhaseSimpleWater utility parity")
    failures = 0
    failures += _check("no-water branch returns one liquid/vapour pair per component",
                       len(no_water) == 2
                       and all(len(comp) == 2 for comp in no_water)
                       and np.allclose(no_water[0][0], 700.0 * xM[0])
                       and np.allclose(no_water[0][1], 4.0 * yM[0])
                       and np.allclose(no_water[1][0], 700.0 * xM[1])
                       and np.allclose(no_water[1][1], 4.0 * yM[1]))
    failures += _check("water branch places hydrocarbon components in oil/gas phases",
                       len(water) == 3
                       and all(len(comp) == 3 for comp in water)
                       and _none_at(water[0][0])
                       and np.allclose(water[0][1], 650.0 * xM[0])
                       and np.allclose(water[0][2], 3.5 * yM[0])
                       and _none_at(water[1][0])
                       and np.allclose(water[1][1], 650.0 * xM[1])
                       and np.allclose(water[1][2], 3.5 * yM[1]))
    failures += _check("water branch appends immiscible water component",
                       np.allclose(water[2][0], 1000.0 * sT)
                       and _none_at(water[2][1])
                       and _none_at(water[2][2]))

    if failures:
        print(f"\nPhase 259 failed ({failures} checks).")
        return 1
    print("\ngetComponentsTwoPhaseSimpleWater now mirrors MRST cell-array layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
