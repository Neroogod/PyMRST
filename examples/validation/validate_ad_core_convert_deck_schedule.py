"""Validate ``convertDeckScheduleToMRST`` using an MRST-processed schedule."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pymrst.io.mrst_mat import load_mrst_mat
from pymrst.modules.ad_core.utils import convert_deck_schedule_to_mrst


def _items(value):
    return value if isinstance(value, list) else [value]


def main() -> int:
    here = Path(__file__).parent
    ref = load_mrst_mat(here / "ref_ad_core_convert_deck_schedule.mat")
    raw = ref["schedule"]
    # The MRST result is already past processWells. Mark its MATLAB cell
    # indices for the typed conversion boundary.
    controls = _items(raw["control"])
    for ctrl in controls:
        wells = _items(ctrl["W"])
        for well in wells:
            well["one_based"] = True
        ctrl["W"] = wells
    raw["control"] = controls

    model = SimpleNamespace(
        fluid=SimpleNamespace(phases=("water", "oil", "gas"))
    )
    schedule = convert_deck_schedule_to_mrst(model, raw)
    expected_val = np.atleast_1d(raw["step"]["val"]).astype(float)
    expected_ctrl = np.atleast_1d(raw["step"]["control"]).astype(int) - 1
    failed = 0

    def check(name, actual, expected, atol=0.0):
        nonlocal failed
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        ok = actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=1e-12)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok

    check("step values", schedule.step_val, expected_val)
    check("step controls", schedule.step_control, expected_ctrl)
    for ci, (actual_ctrl, expected_ctrl_raw) in enumerate(zip(schedule.controls, controls)):
        expected_wells = _items(expected_ctrl_raw["W"])
        actual_active = {well.name: well for well in actual_ctrl.W if well.status}
        for expected in expected_wells:
            name = str(expected["name"])
            actual = actual_active[name]
            check(f"control {ci} {name} cells", actual.cells,
                  np.atleast_1d(expected["cells"]).astype(int) - 1)
            check(f"control {ci} {name} target", [actual.val], [expected["val"]])
            check(f"control {ci} {name} compi", actual.compi,
                  np.atleast_1d(expected["compi"]).astype(float))
    print(f"controls: {len(schedule.controls)}, steps: {schedule.n_steps}")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
