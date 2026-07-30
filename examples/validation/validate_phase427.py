"""Phase 427: compare getEquilPC with MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.utils import get_equil_pc

REF = Path(__file__).with_name("ref_p427_equil_pc.mat")


class _PC:
    def hasJFunctionScaler(self, pair):
        return False


class _Model:
    water = oil = gas = True
    fluid = SimpleNamespace(
        pcOW=[lambda s: 10 * (1 - s), lambda s: 20 * (1 - s)],
        pcOG=[lambda s: 30 * s, lambda s: 40 * s],
    )
    FlowPropertyFunctions = SimpleNamespace(CapillaryPressure=_PC())

    def getActivePhases(self):
        return np.ones(3, dtype=bool)

    def getPhaseIndex(self, phase):
        return {"W": 0, "O": 1, "G": 2}[phase]


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase427_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    cells = ref["cells"].ravel().astype(np.int64) - 1
    region = int(ref["satnum"].item()) - 1
    samples = ref["samples"].ravel()
    curves, signs, scalers = get_equil_pc(_Model(), region, cells)
    evaluated = np.column_stack([curve(samples) for curve in curves])

    failed = 0
    for name, actual, expected in (
        ("curves", evaluated, ref["out_curves"]),
        ("signs", signs.reshape(1, -1), ref["out_sign"]),
        ("scalers", scalers, ref["out_scalers"]),
    ):
        error = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
        ok = np.asarray(actual).shape == np.asarray(expected).shape and error < 1e-12
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d|={error:.3e}")
    print("\ngetEquilPC matches MRST!" if not failed else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
