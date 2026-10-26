"""Validate ``initWellSolAD`` against MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_core.utils import (
    convert_deck_schedule_to_mrst_ad,
    init_well_sol_ad,
)
from pymrst.utils.units import BARSA
from validate_ad_core_raw_deck_schedule import _inputs


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_init_well_sol.mat",
        squeeze_me=True,
    )
    model, deck = _inputs()
    schedule = convert_deck_schedule_to_mrst_ad(model, deck)
    rows = init_well_sol_ad(
        schedule.controls[0].W, model,
        {"pressure": np.array([200.0, 190.0, 180.0, 170.0]) * BARSA},
    )
    failed = 0

    def check(name, actual, expected, atol=1e-12):
        nonlocal failed
        actual, expected = np.asarray(actual), np.asarray(expected)
        ok = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-12, atol=atol)
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok

    check("status", [row.status for row in rows], ref["status"], 0)
    check("targets", [row.val for row in rows], ref["vals"])
    check("signs", [row.sign for row in rows], ref["signs"], 0)
    check("bhp", [row.bhp for row in rows], ref["bhp"])
    check("qWs", [row.qWs for row in rows], ref["qWs"])
    check("qOs", [row.qOs for row in rows], ref["qOs"])
    check("qGs", [row.qGs for row in rows], ref["qGs"])
    for index, row in enumerate(rows):
        check(f"well {index} mixs", row.mixs, ref["mixs"][index])
        check(f"well {index} cstatus", row.cstatus, ref["cstatus"][index], 0)
        check(f"well {index} cdp", row.cdp, ref["cdp"][index])
        check(f"well {index} cqs", row.cqs, ref["cqs"][index])
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
