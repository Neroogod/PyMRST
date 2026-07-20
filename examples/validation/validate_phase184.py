"""Phase 184: deck VFP table conversion and schedule binding parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.vfp import VFPTable  # noqa: E402
from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    run_schedule,
    schedule_controls_from_deck_controls,
    simple_schedule,
    simple_schedule_from_deck_controls,
    vfp_tables_from_deck_control,
)
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _producer_deck_table(offset: float = 0.0) -> dict:
    flo = np.array([0.0, 10.0], dtype=float)
    thp = np.array([100.0, 200.0], dtype=float)
    wfr = np.array([0.0, 1.0], dtype=float)
    gfr = np.array([0.0, 2.0], dtype=float)
    q = np.zeros((2, 2, 2, 2), dtype=float)
    for i, f in enumerate(flo):
        for j, p in enumerate(thp):
            for k, wr in enumerate(wfr):
                for m, gr in enumerate(gfr):
                    q[i, j, k, m] = offset + p + 2.0 * f + 5.0 * wr + 7.0 * gr
    return {
        "FLO": flo,
        "THP": thp,
        "WFR": wfr,
        "GFR": gfr,
        "Q": q,
        "FLOID": "LIQ",
        "WFRID": "WFR",
        "GFRID": "GOR",
    }


def _injector_deck_table(offset: float = 0.0) -> dict:
    flo = np.array([0.0, 10.0], dtype=float)
    thp = np.array([100.0, 200.0], dtype=float)
    bhp = np.zeros((2, 2), dtype=float)
    for i, f in enumerate(flo):
        for j, p in enumerate(thp):
            bhp[i, j] = offset + p + 3.0 * f
    return {"FLO": flo, "THP": thp, "BHP": bhp, "FLOID": "RATE"}


def _well(name: str = "P1", *, sign: int = -1, vfp_index: int = 1) -> Well:
    return Well(
        cells=np.array([0]),
        type="thp",
        val=150.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=sign,
        compi=np.array([0.0, 1.0, 0.0]),
        refDepth=0.0,
        lims={},
        vfp_index=vfp_index,
    )


class DummyModel:
    def __init__(self, wells):
        self.wells = list(wells)
        self.bc = None
        self.src = None
        self.vfp_tables_injector = []
        self.vfp_tables_producer = []
        self.bound_in_step = []
        self.bound_table_value = []

    def _bind_vfp_tables(self):
        for w in self.wells:
            ix = getattr(w, "vfp_index", None)
            if ix is None or int(ix) <= 0:
                continue
            tables = self.vfp_tables_injector if w.sign > 0 else self.vfp_tables_producer
            pos = int(ix) - 1
            if pos < 0 or pos >= len(tables) or tables[pos] is None:
                raise IndexError("dummy missing VFP table")
            w.vfp_table = tables[pos]

    def step(self, state, dt, bc=None, src=None):
        table = getattr(self.wells[0], "vfp_table", None)
        self.bound_in_step.append(isinstance(table, VFPTable))
        if table is not None:
            self.bound_table_value.append(float(table.evaluate_bhp(5.0, 150.0, 0.5, 1.0)))
        return state + float(dt), True, 1


def validate_vfp_table_conversion() -> int:
    prod = _producer_deck_table()
    inj = _injector_deck_table(1000.0)
    inj_tables, prod_tables = vfp_tables_from_deck_control(
        {"VFPPROD": [prod], "VFPINJ": {2: inj}}
    )
    controls = schedule_controls_from_deck_controls(
        [_well()],
        [{"VFPPROD": [prod], "VFPINJ": {2: inj}}],
    )
    simple = simple_schedule(
        [1.0],
        control=[{"wells": [_well()], "VFPPROD": prod}],
    )

    failures = 0
    failures += _check("VFPPROD list converts to native VFPTable",
                       isinstance(prod_tables[0], VFPTable))
    failures += _check("VFPINJ sparse table number preserves one-based slot",
                       len(inj_tables) == 2 and inj_tables[0] is None and isinstance(inj_tables[1], VFPTable))
    failures += _check("deck control conversion carries producer VFP tables",
                       isinstance(controls[0].vfp_tables_producer[0], VFPTable))
    failures += _check("simple_schedule mapping control carries producer VFP tables",
                       isinstance(simple.control[0].vfp_tables_producer[0], VFPTable))
    failures += _check("producer table interpolation survived conversion",
                       np.isclose(prod_tables[0].evaluate_bhp(5.0, 150.0, 0.5, 1.0),
                                  150.0 + 10.0 + 2.5 + 7.0))
    failures += _check("injector table interpolation survived conversion",
                       np.isclose(inj_tables[1].evaluate_bhp(5.0, 150.0),
                                  1000.0 + 150.0 + 15.0))
    return failures


def validate_schedule_binding_and_restore() -> int:
    prod = _producer_deck_table(200.0)
    well = _well(vfp_index=2)
    model = DummyModel([])
    schedule = simple_schedule_from_deck_controls(
        [1.0],
        [],
        [{"wells": [well], "VFPPROD": {2: prod}}],
    )
    out = run_schedule(model, 0.0, schedule)

    failures = 0
    failures += _check("run_schedule binds sparse producer table before model.step",
                       model.bound_in_step == [True])
    failures += _check("run_schedule uses one-based VFP table number during binding",
                       np.isclose(model.bound_table_value[0], 200.0 + 150.0 + 10.0 + 2.5 + 7.0))
    failures += _check("run_schedule restores original wells after schedule",
                       model.wells == [])
    failures += _check("run_schedule restores original VFP table lists after schedule",
                       model.vfp_tables_producer == [] and model.vfp_tables_injector == [])
    failures += _check("schedule run still emits normal report metadata",
                       out.reports[0]["Control"] == 1 and out.iterations.tolist() == [1])
    return failures


def validate_empty_sparse_slot_fails() -> int:
    model = DummyModel([])
    schedule = simple_schedule(
        [1.0],
        control=[ScheduleControl(wells=[_well(vfp_index=2)], vfp_tables_producer=[VFPTable.from_deck(_producer_deck_table())])],
    )
    try:
        run_schedule(model, 0.0, schedule)
    except IndexError:
        failed = True
    else:
        failed = False
    return _check("missing sparse VFP table slot fails loudly", failed)


def main() -> int:
    print("\nPhase 184: deck VFP table conversion and schedule binding parity")
    failures = validate_vfp_table_conversion()
    failures += validate_schedule_binding_and_restore()
    failures += validate_empty_sparse_slot_fails()
    if failures:
        print(f"\nPhase 184 failed ({failures} checks).")
        return 1
    print("\nDeck VFPINJ/VFPPROD controls now become native VFP tables and bind through schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
