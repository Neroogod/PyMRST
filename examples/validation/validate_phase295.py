"""Phase 295: compositional utils/units layout."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import utils  # noqa: E402
from pymrst.modules.compositional.utils.units import (  # noqa: E402
    fromCelsius,
    fromFahrenheit,
    from_celsius,
    from_fahrenheit,
    list_mrst_contents_units,
    list_units,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 295: compositional utils/units layout")
    c = np.array([-40.0, 0.0, 100.0])
    f = np.array([-40.0, 32.0, 212.0])
    failures = 0
    failures += _check("fromCelsius follows MRST T + 273.15",
                       np.allclose(fromCelsius(c), np.array([233.15, 273.15, 373.15]))
                       and np.allclose(from_celsius(25.0), 298.15))
    failures += _check("fromFahrenheit follows MRST (T + 459.67)*Rankine",
                       np.allclose(fromFahrenheit(f), np.array([233.15, 273.15, 373.15]), atol=1e-12)
                       and np.allclose(from_fahrenheit(68.0), 293.15))
    failures += _check("units Contents metadata lists MRST helpers",
                       list_mrst_contents_units() == (("fromCelsius", ""), ("fromFahrenheit", ""))
                       and set(list_units()) == {"fromCelsius", "fromFahrenheit"})
    failures += _check("helpers are re-exported from compositional.utils",
                       np.allclose(utils.fromCelsius(0.0), 273.15)
                       and np.allclose(utils.fromFahrenheit(32.0), 273.15))
    if failures:
        print(f"\nPhase 295 failed ({failures} checks).")
        return 1
    print("\nCompositional utils/units helpers match MRST layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
