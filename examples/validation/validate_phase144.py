"""Phase 144: MRST deck controls become native compositional schedules."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    run_schedule,
    schedule_controls_from_deck_controls,
    simple_schedule_from_deck_controls,
)
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, *, sign: int, compi=None) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=100.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=sign,
        compi=np.asarray([0.5, 0.5] if compi is None else compi, dtype=float),
        refDepth=0.0,
        lims={},
    )


def validate_control_conversion() -> int:
    base_wells = [_well("P1", sign=-1), _well("I1", sign=1)]
    controls = [
        {
            "WELSPECS": [["P1", "FIELD"], ["I1", "INJECTORS"]],
            "GRUPTREE": [["INJECTORS", "FIELD"]],
            "WCONPROD": [["P1", "OPEN", "ORAT", 10.0, np.inf, np.inf, np.inf, np.inf, 200.0]],
            "WCONINJE": [["I1", "GAS", "OPEN", "RATE", 3.0, np.inf, 300.0]],
            "GCONPROD": [["FIELD", "ORAT", 12.0, np.inf, np.inf, np.inf, "NONE", "YES"]],
            "GRUPNET": [["FIELD", 4.0, 0, 0, "YES", "NO", "NODE_A"]],
        },
        {
            "WELSPECS": [["P1", "FIELD"], ["I1", "INJECTORS"]],
            "GRUPTREE": [["INJECTORS", "FIELD"]],
            "WCONHIST": [["P1", "OPEN", "BHP", 1.0, 2.0, 3.0, 4.0, 5.0, 150.0]],
            "WCONINJH": [["I1", "WATER", "OPEN", 2.0]],
            "WELTARG": [["P1", "BHP", 175.0], ["I1", "RATE", 9.0]],
            "GECON": [["FIELD", 1.0, 2.0, 0.8, 300.0, 0.1, "NONE", "NO", 0]],
        },
    ]

    converted = schedule_controls_from_deck_controls(base_wells, controls)
    first_wells = {w.name: w for w in converted[0].wells}
    second_wells = {w.name: w for w in converted[1].wells}
    first_groups = {g.name: g for g in converted[0].group_controls}
    second_groups = {g.name: g for g in converted[1].group_controls}

    failures = 0
    failures += _check("conversion preserves one control per deck block",
                       len(converted) == 2 and all(isinstance(c, ScheduleControl) for c in converted))
    failures += _check("first producer/injector controls are applied",
                       first_wells["P1"].type == "orat"
                       and np.isclose(first_wells["P1"].val, -10.0)
                       and first_wells["I1"].type == "rate"
                       and np.isclose(first_wells["I1"].val, 3.0))
    failures += _check("second control accepts WCONHIST/WCONINJH and WELTARG",
                       second_wells["P1"].type == "bhp"
                       and np.isclose(second_wells["P1"].lims["bhp"], 175.0)
                       and second_wells["I1"].type == "rate"
                       and np.isclose(second_wells["I1"].lims["rate"], 9.0))
    failures += _check("group limits and network metadata follow controls",
                       np.isclose(first_groups["FIELD"].lims["orat"], 12.0)
                       and first_groups["FIELD"].extra["GRUPNET"]["terminal"] == "NODE_A")
    failures += _check("economics-only metadata survives later control",
                       "GECON" in second_groups["FIELD"].extra
                       and np.isclose(second_groups["FIELD"].extra["GECON"]["max_water_cut"], 0.8))
    failures += _check("base wells are not mutated during conversion",
                       base_wells[0].type == "bhp"
                       and np.isclose(base_wells[0].val, 100.0)
                       and base_wells[1].type == "bhp")
    return failures


def validate_schedule_runtime_switching() -> int:
    class DummyModel:
        def __init__(self):
            self.wells = [_well("P1", sign=-1), _well("I1", sign=1)]
            self.bc = {"original": True}
            self.src = {"original": True}
            self.separator_group = {"original": True}
            self.group_controls = [{"original": True}]
            self.seen = []

        def step(self, state, dt, bc=None, src=None):
            self.seen.append(
                (
                    [w.name for w in self.wells],
                    [w.type for w in self.wells],
                    [float(w.val) for w in self.wells],
                    [g.name for g in self.group_controls],
                    bc,
                    src,
                )
            )
            self.wells[0].type = "mutated"
            self.group_controls[0].name = "MUTATED"
            return state + 1, True, 1

    base_wells = [_well("P1", sign=-1), _well("I1", sign=1)]
    sched = simple_schedule_from_deck_controls(
        [1.0, 1.0, 1.0],
        base_wells,
        [
            {
                "WELSPECS": [["P1", "FIELD"], ["I1", "FIELD"]],
                "WCONPROD": [["P1", "OPEN", "ORAT", 5.0]],
                "WCONINJE": [["I1", "GAS", "OPEN", "RATE", 2.0]],
                "GCONPROD": [["FIELD", "ORAT", 6.0]],
                "bc": {"name": "bc-a"},
            },
            {
                "WELSPECS": [["P1", "SATELLITE"], ["I1", "SATELLITE"]],
                "WCONHIST": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 125.0]],
                "WCONINJH": [["I1", "WATER", "OPEN", 2.5]],
                "GECON": [["SATELLITE", 1.0, 0, 0, 0, 0, "NONE", "NO", 0]],
                "src": {"name": "src-b"},
            },
        ],
        step_control=[1, 2, 1],
    )

    model = DummyModel()
    result = run_schedule(model, 0, sched)

    failures = 0
    failures += _check("schedule runs all deck-selected controls",
                       result.final_state == 3 and result.reports is not None)
    failures += _check("report controls remain MRST one-based",
                       [r["Control"] for r in result.reports] == [1, 2, 1]
                       and [r["ControlChanged"] for r in result.reports] == [True, True, True])
    failures += _check("first and third report steps reuse first deck control",
                       model.seen[0][1] == ["orat", "rate"]
                       and np.allclose(model.seen[0][2], [-5.0, 2.0])
                       and model.seen[2][1] == ["orat", "rate"])
    failures += _check("second report step switches wells and groups",
                       model.seen[1][1] == ["bhp", "rate"]
                       and np.allclose(model.seen[1][2], [125.0, 2.5])
                       and model.seen[1][3] == ["SATELLITE"])
    failures += _check("bc and src remain attached to their deck controls",
                       model.seen[0][4] == {"name": "bc-a"}
                       and model.seen[1][5] == {"name": "src-b"}
                       and model.seen[2][4] == {"name": "bc-a"})
    failures += _check("model facility state is restored after schedule",
                       [w.type for w in model.wells] == ["bhp", "bhp"]
                       and model.bc == {"original": True}
                       and model.src == {"original": True}
                       and model.separator_group == {"original": True}
                       and model.group_controls == [{"original": True}])
    return failures


def main() -> int:
    print("\nPhase 144: MRST deck controls become native schedules")
    failures = validate_control_conversion()
    failures += validate_schedule_runtime_switching()
    if failures:
        print(f"\nPhase 144 failed ({failures} checks).")
        return 1
    print("\nDeck-style control blocks now feed native PyMRST schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
