"""Validate ``setWellSign`` and ``assignWellValuesFromControl`` against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.ad_core.utils import (
    assign_well_values_from_control,
    set_well_sign,
)
from pymrst.params.wells_and_bc import Well


def _well(name, kind, value, sign, compi):
    return Well(
        cells=np.array([0]), type=kind, val=value, r=np.array([0.1]),
        dir="z", WI=np.array([1.0]), dZ=np.array([0.0]), name=name,
        compi=np.asarray(compi, float), refDepth=0.0, sign=sign,
    )


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_well_control_assignment.mat",
        squeeze_me=True,
    )
    model = SimpleNamespace(fluid=SimpleNamespace(phases=("water", "oil", "gas")))
    wells = [
        _well("I", "rate", 100.0, 1, [0.2, 0.3, 0.5]),
        _well("P", "rate", -80.0, -1, [0, 1, 0]),
        _well("B", "bhp", 150.0, -1, [0, 1, 0]),
        _well("O", "orat", -40.0, -1, [0, 1, 0]),
    ]
    rows = [
        SimpleNamespace(status=True, type=w.type, val=w.val, sign=w.sign,
                        bhp=0.0, qWs=1.0, qOs=2.0, qGs=3.0)
        for w in wells
    ]
    rows = assign_well_values_from_control(model, rows, wells)
    sign_wells = [
        _well("P", "rate", -10.0, None, [0, 1, 0]),
        _well("I", "rate", 10.0, None, [1, 0, 0]),
        _well("B", "bhp", 100.0, None, [0, 1, 0]),
        _well("K", "bhp", 100.0, -1, [0, 1, 0]),
    ]
    signs = [well.sign for well in set_well_sign(sign_wells)]
    checks = {
        "bhp": ([row.bhp for row in rows], ref["bhp"]),
        "qWs": ([row.qWs for row in rows], ref["qWs"]),
        "qOs": ([row.qOs for row in rows], ref["qOs"]),
        "qGs": ([row.qGs for row in rows], ref["qGs"]),
        "signs": (signs, ref["signs"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual - expected)))
        ok = np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
