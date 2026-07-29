"""Validate ``convertDeckScheduleToMRST``/``makeScheduleConsistent`` against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils import convert_deck_schedule_to_mrst
from pymrst.params import make_rock
from pymrst.utils.units import DARCY, MILLI


def _inputs():
    grid = compute_geometry(cart_grid([2, 1, 2], [20.0, 10.0, 20.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    model = SimpleNamespace(
        G=grid, rock=rock, gravity=0.0,
        fluid=SimpleNamespace(phases=("water", "oil", "gas")),
    )
    specs = [
        ["I", "FIELD", 1, 1, 0.0, "GAS", 0, "STD", "SHUT", "YES", 0, "SEG", 0],
        ["P", "FIELD", 2, 1, 0.0, "OIL", 0, "STD", "SHUT", "YES", 0, "SEG", 0],
    ]
    completions = [
        ["I", 1, 1, 1, 2, "OPEN", 1, 5.0, 0.2, -1, 0, "Default", "Z", -1],
        ["P", 2, 1, 1, 2, "OPEN", 1, 6.0, 0.2, -1, 0, "Default", "Z", -1],
    ]
    controls = [
        {
            "WELSPECS": specs, "COMPDAT": completions,
            "WCONINJE": [["I", "GAS", "OPEN", "RATE", 100.0, 0, 300e5, 0, 0, 0, 0, 0, 0, 0]],
            "WCONPROD": [["P", "OPEN", "ORAT", 200.0, 0, 0, 0, 0, 100e5, 0, 0, 0]],
        },
        {
            "WELSPECS": specs, "COMPDAT": completions,
            "WCONINJE": [["I", "GAS", "SHUT", "RATE", 50.0, 0, 300e5, 0, 0, 0, 0, 0, 0, 0]],
            "WCONPROD": [["P", "OPEN", "BHP", 0, 0, 0, 0, 0, 150e5, 0, 0, 0]],
        },
    ]
    return model, {
        "step": {"val": [10.0, 20.0], "control": [1, 2]},
        "control": controls,
    }


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_raw_deck_schedule.mat",
        squeeze_me=True,
    )
    model, deck = _inputs()
    schedule = convert_deck_schedule_to_mrst(model, deck)
    names = np.asarray(ref["names"], dtype=object)
    cells = np.asarray(ref["cells"], dtype=object)
    types = np.asarray(ref["types"], dtype=object)
    compi = np.asarray(ref["compi"], dtype=object)
    wi = np.asarray(ref["WI"], dtype=object)
    failed = 0

    def check(name, actual, expected, atol=1e-12):
        nonlocal failed
        actual, expected = np.asarray(actual), np.asarray(expected)
        ok = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-12, atol=atol)
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok

    check("step values", schedule.step_val, ref["step_val"])
    check("step controls", schedule.step_control + 1, ref["step_control"], 0)
    for c, ctrl in enumerate(schedule.controls):
        for w, well in enumerate(ctrl.W):
            expected_name = str(names[c, w])
            ok = well.name == expected_name and well.type == str(types[c, w])
            print(f"{'ok  ' if ok else 'FAIL'} control {c} well {w}: {well.name}/{well.type}")
            failed += not ok
            check(f"control {c} {well.name} cells", well.cells + 1, cells[c, w], 0)
            check(f"control {c} {well.name} target", [well.val], [ref["vals"][c, w]])
            check(f"control {c} {well.name} compi", well.compi, compi[c, w])
            check(f"control {c} {well.name} WI", well.WI, wi[c, w])
            ok = well.status == bool(ref["status"][c, w])
            print(f"{'ok  ' if ok else 'FAIL'} control {c} {well.name} status")
            failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
