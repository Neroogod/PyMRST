"""Phase 300: natural-variable sequential utility helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.utils import setup_simple_compositional_example  # noqa: E402
from pymrst.modules.compositional.models.natvars.sequential.utils import (  # noqa: E402
    correctSequentialBHP,
    correct_sequential_bhp,
    getUpdatesFromPressure,
    get_updates_from_pressure,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_updates_from_pressure() -> int:
    setup = setup_simple_compositional_example(use_natural=True, nx=5, nstep=1)
    model = setup.model
    state = setup.state
    state.flag = np.zeros(state.pressure.shape, dtype=np.int8)
    dp = np.full(state.pressure.shape, 1.0e4)
    dx, dy, ds, dL, two_phase, w = get_updates_from_pressure(model, state, dp)
    dx2, dy2, ds2, dL2, two_phase2, w2 = getUpdatesFromPressure(model, state, dp)
    n = state.pressure.size
    nc = state.z.shape[1]
    failures = 0
    failures += _check("get_updates_from_pressure returns MRST-shaped increments",
                       dx.shape == (n, nc)
                       and dy.shape == (n, nc)
                       and ds.shape == (n, 2)
                       and dL.shape == (n,)
                       and two_phase.shape == (n,)
                       and w.shape == (n,))
    failures += _check("getUpdatesFromPressure camelCase alias matches",
                       np.allclose(dx2, dx)
                       and np.allclose(dy2, dy)
                       and np.allclose(ds2, ds)
                       and np.allclose(dL2, dL)
                       and np.array_equal(two_phase2, two_phase)
                       and np.allclose(w2, w))
    failures += _check("pressure update weights are bounded",
                       np.all(w >= 0.0) and np.all(w <= 1.0))
    return failures


def validate_correct_sequential_bhp() -> int:
    well = SimpleNamespace(type="rate", cells=np.array([0, 1]), WI=np.array([2.0, 4.0]))
    well_bhp = SimpleNamespace(type="bhp", cells=np.array([0]), WI=np.array([1.0]))
    control = SimpleNamespace(W=[well, well_bhp])
    schedule = SimpleNamespace(step=SimpleNamespace(control=np.array([1])), control=[control])
    wellsol = SimpleNamespace(
        flux=np.array([[2.0, 1.0], [1.0, 1.0]]),
        cdp=np.array([0.5, 1.0]),
        bhp=0.0,
    )
    bhp_wellsol = SimpleNamespace(flux=np.array([[1.0, 0.0]]), cdp=np.array([0.0]), bhp=123.0)
    state = SimpleNamespace(
        pressure=np.array([100.0, 90.0]),
        mob=np.array([[3.0, 1.0], [2.0, 2.0]]),
        wellSol=[wellsol, bhp_wellsol],
    )
    modelseq = SimpleNamespace(transportModel=SimpleNamespace(extraStateOutput=False))
    out = correct_sequential_bhp(modelseq, [state], schedule)
    q = np.array([3.0, 2.0])
    perf = q / (well.WI * np.array([4.0, 4.0])) - wellsol.cdp + state.pressure
    expected = float(np.sum(perf * np.abs(q)) / np.sum(np.abs(q)))
    out2 = correctSequentialBHP(modelseq, [state], schedule)
    failures = 0
    failures += _check("correct_sequential_bhp updates non-BHP well bhp",
                       out[0] is state and np.allclose(wellsol.bhp, expected))
    failures += _check("correct_sequential_bhp leaves BHP-controlled wells unchanged",
                       bhp_wellsol.bhp == 123.0)
    failures += _check("correctSequentialBHP camelCase alias returns states",
                       out2[0] is state and modelseq.transportModel.extraStateOutput is True)
    return failures


def main() -> int:
    print("\nPhase 300: natural-variable sequential utility helpers")
    failures = validate_updates_from_pressure()
    failures += validate_correct_sequential_bhp()
    if failures:
        print(f"\nPhase 300 failed ({failures} checks).")
        return 1
    print("\nNatural-variable sequential utility helpers are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
