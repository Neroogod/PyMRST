"""Phase 141: deck-imported compositional well controls."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import apply_deck_well_controls  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


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


def validate_wconprod_and_weltarg() -> int:
    wells = [_well("P1", sign=-1), _well("I1", sign=1)]
    control = {
        # WCONPROD: name, status, type, ORAT, WRAT, GRAT, LRAT, RESV, BHP, THP
        "WCONPROD": [["P1", "OPEN", "ORAT", 100.0, 200.0, 300.0, 400.0, 500.0, 250.0, 150.0]],
        # WELTARG mutates the parsed control row before it becomes a Well.
        "WELTARG": [["P1", "BHP", 225.0]],
    }
    updated = apply_deck_well_controls(wells, control)
    p1 = updated[0]

    failures = 0
    failures += _check("producer sign is preserved as production",
                       p1.sign == -1)
    failures += _check("producer active control type follows WCONPROD",
                       p1.type == "orat")
    failures += _check("producer active rate target is negative like MRST W.val",
                       np.isclose(p1.val, -100.0))
    failures += _check("producer rate limits are stored as production-signed values",
                       p1.lims is not None
                       and np.isclose(p1.lims["orat"], -100.0)
                       and np.isclose(p1.lims["grat"], -300.0))
    failures += _check("WELTARG updates producer BHP target",
                       p1.lims is not None and np.isclose(p1.lims["bhp"], 225.0))
    failures += _check("original wells are not mutated by default",
                       wells[0].type == "bhp" and wells[0].val == 1.0)
    return failures


def validate_wconinje_and_status() -> int:
    wells = [_well("P1", sign=-1), _well("I1", sign=1)]
    control = {
        # WCONINJE: name, phase, status, type, RATE, RESV, BHP, THP
        "WCONINJE": [["I1", "GAS", "SHUT", "RESV", 10.0, 20.0, 300.0, 200.0]],
        "WELTARG": [["I1", "RATE", 12.5]],
    }
    updated = apply_deck_well_controls(wells, control)
    inj = updated[1]

    failures = 0
    failures += _check("injector sign is positive",
                       inj.sign == 1)
    failures += _check("RESV maps to PyMRST vrat control",
                       inj.type == "vrat" and np.isclose(inj.val, 20.0))
    failures += _check("WELTARG updates injector rate limit",
                       inj.lims is not None and np.isclose(inj.lims["rate"], 12.5))
    failures += _check("injector status follows WCONINJE",
                       inj.status is False)
    failures += _check("gas injector compi selects the gas slot",
                       np.allclose(inj.compi, [0.0, 1.0]))
    return failures


def validate_missing_wells_are_ignored() -> int:
    wells = [_well("P1", sign=-1)]
    updated = apply_deck_well_controls(
        wells,
        {"WCONPROD": [["UNKNOWN", "OPEN", "BHP", 1.0, 1.0, 1.0, 1.0, 1.0, 99.0]]},
    )
    failures = 0
    failures += _check("unknown deck well names do not create geometry",
                       len(updated) == 1 and updated[0].name == "P1")
    failures += _check("unknown controls leave existing well unchanged",
                       updated[0].type == "bhp" and np.isclose(updated[0].val, 1.0))
    return failures


def main() -> int:
    print("\nPhase 141: deck-imported compositional well controls")
    failures = validate_wconprod_and_weltarg()
    failures += validate_wconinje_and_status()
    failures += validate_missing_wells_are_ignored()
    if failures:
        print(f"\nPhase 141 failed ({failures} checks).")
        return 1
    print("\nDeck well-control records update native PyMRST wells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
