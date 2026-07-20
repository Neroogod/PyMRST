"""Phase 164: MRST-style whole-well WELOPEN status updates."""

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


def _well(name: str, *, compi=None) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=1.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        compi=np.asarray([0.5, 0.5] if compi is None else compi, dtype=float),
        refDepth=0.0,
        lims={},
    )


def validate_producer_welopen_shuts_whole_well() -> int:
    well = apply_deck_well_controls(
        [_well("P1")],
        {
            "WCONPROD": [["P1", "OPEN", "ORAT", 10.0, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P1", "SHUT"]],
        },
    )[0]

    return _check(
        "WELOPEN shuts surviving producer control row",
        well.sign == -1 and well.status is False and well.type == "orat" and np.isclose(well.val, -10.0),
    )


def validate_injector_welopen_opens_whole_well() -> int:
    well = apply_deck_well_controls(
        [_well("I1")],
        {
            "WCONINJE": [["I1", "GAS", "SHUT", "RATE", 20.0, np.inf, 200.0]],
            "WELOPEN": [["I1", "OPEN"]],
        },
    )[0]

    failures = 0
    failures += _check("WELOPEN opens surviving injector control row",
                       well.sign == 1 and well.status is True and well.type == "rate")
    failures += _check("WELOPEN preserves injector target and stream",
                       np.isclose(well.val, 20.0) and np.allclose(well.compi, [0.0, 1.0]))
    return failures


def validate_welopen_after_keyword_replacement() -> int:
    well = apply_deck_well_controls(
        [_well("MIXED")],
        {
            "WCONPROD": [["MIXED", "OPEN", "ORAT", 10.0, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WCONINJE": [["MIXED", "WATER", "SHUT", "RATE", 15.0, np.inf, 180.0]],
            "WELOPEN": [["mixed", "OPEN"]],
        },
    )[0]

    return _check(
        "WELOPEN updates only the active post-replacement keyword table",
        well.sign == 1 and well.status is True and well.type == "rate" and np.isclose(well.val, 15.0),
    )


def validate_completion_welopen_is_conservative_noop() -> int:
    well = apply_deck_well_controls(
        [_well("P1")],
        {
            "WCONPROD": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P1", "SHUT", 1, -1, -1, -1, -1]],
        },
    )[0]

    return _check(
        "positive-I/J/K WELOPEN rows are ignored when no grid is supplied",
        well.sign == -1 and well.status is True and well.type == "bhp" and np.isclose(well.val, 100.0),
    )


def validate_mapping_welopen_indices_are_not_whole_well() -> int:
    well = apply_deck_well_controls(
        [_well("P2")],
        {
            "WCONPROD": [{"well": "P2", "status": "OPEN", "control": "BHP", "bhp": 120.0}],
            "WELOPEN": [{"well": "P2", "status": "SHUT", "i": 2}],
        },
    )[0]

    return _check(
        "mapping-style completion WELOPEN is not mistaken for whole-well",
        well.status is True and well.type == "bhp" and np.isclose(well.val, 120.0),
    )


def validate_unknown_well_is_ignored() -> int:
    well = apply_deck_well_controls(
        [_well("P3")],
        {
            "WCONPROD": [["P3", "OPEN", "ORAT", 10.0]],
            "WELOPEN": [["OTHER", "SHUT"]],
        },
    )[0]

    return _check("WELOPEN for unknown wells leaves known controls unchanged", well.status is True)


def main() -> int:
    print("\nPhase 164: MRST-style whole-well WELOPEN status updates")
    failures = validate_producer_welopen_shuts_whole_well()
    failures += validate_injector_welopen_opens_whole_well()
    failures += validate_welopen_after_keyword_replacement()
    failures += validate_completion_welopen_is_conservative_noop()
    failures += validate_mapping_welopen_indices_are_not_whole_well()
    failures += validate_unknown_well_is_ignored()
    if failures:
        print(f"\nPhase 164 failed ({failures} checks).")
        return 1
    print("\nDeck whole-well WELOPEN records now update active MRST-style control tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
