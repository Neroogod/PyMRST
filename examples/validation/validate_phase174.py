"""Phase 174: MRST default WCON producer/injector BHP targets."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import apply_deck_well_controls  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402
from pymrst.utils.units import ATM, BARSA  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, *, sign: int = 0) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=1.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        compi=np.array([0.5, 0.5]),
        refDepth=0.0,
        sign=sign,
        lims={},
    )


def validate_producer_default_bhp() -> int:
    wells = [_well("P1", sign=-1), _well("P2", sign=-1), _well("P3", sign=-1)]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONPROD": [
                ["P1", "OPEN", "BHP"],
                ["P2", "OPEN", "ORAT", 10.0],
                ["P3", "OPEN", "ORAT", 20.0, np.inf, np.inf, np.inf, np.inf, np.inf],
            ],
        },
    )
    by_name = {w.name: w for w in updated}

    failures = 0
    failures += _check("missing producer BHP control defaults to one atmosphere",
                       by_name["P1"].type == "bhp"
                       and np.isclose(by_name["P1"].val, ATM)
                       and np.isclose(by_name["P1"].lims["bhp"], ATM))
    failures += _check("missing producer BHP limit is inserted for rate controls",
                       by_name["P2"].type == "orat"
                       and np.isclose(by_name["P2"].val, -10.0)
                       and np.isclose(by_name["P2"].lims["bhp"], ATM))
    failures += _check("explicit infinite producer BHP remains unlimited",
                       by_name["P3"].type == "orat"
                       and np.isclose(by_name["P3"].val, -20.0)
                       and "bhp" not in by_name["P3"].lims)
    return failures


def validate_injector_default_bhp() -> int:
    wells = [_well("I1", sign=1), _well("I2", sign=1), _well("I3", sign=1)]
    default_bhp = 6895.0 * BARSA
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONINJE": [
                ["I1", "GAS", "OPEN", "BHP"],
                ["I2", "GAS", "OPEN", "RATE", 12.0],
                ["I3", "GAS", "OPEN", "RATE", 13.0, np.inf, np.inf],
            ],
        },
    )
    by_name = {w.name: w for w in updated}

    failures = 0
    failures += _check("missing injector BHP control defaults to MRST high-pressure value",
                       by_name["I1"].type == "bhp"
                       and np.isclose(by_name["I1"].val, default_bhp)
                       and np.isclose(by_name["I1"].lims["bhp"], default_bhp))
    failures += _check("missing injector BHP limit is inserted for rate controls",
                       by_name["I2"].type == "rate"
                       and np.isclose(by_name["I2"].val, 12.0)
                       and np.isclose(by_name["I2"].lims["bhp"], default_bhp))
    failures += _check("explicit infinite injector BHP remains unlimited",
                       by_name["I3"].type == "rate"
                       and np.isclose(by_name["I3"].val, 13.0)
                       and "bhp" not in by_name["I3"].lims)
    return failures


def main() -> int:
    print("\nPhase 174: MRST default WCON producer/injector BHP targets")
    failures = validate_producer_default_bhp()
    failures += validate_injector_default_bhp()
    if failures:
        print(f"\nPhase 174 failed ({failures} checks).")
        return 1
    print("\nMissing WCON BHP columns now receive MRST-style default pressure targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
