"""Validate the SPE1 tutorial example against MRST ad-blackoil.

Run ``dump_phase5b_reference.m`` in MATLAB to refresh ``ref_p5b_spe1.mat``.
This validator intentionally imports and runs the public tutorial module
(``blackoil_tutorial_spe1.py``), so it checks the presentation/demo path rather
than a separate validation-only implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    run_blackoil_tutorial_spe1,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p5b_spe1.mat"
N_STEPS = 30


def _compare(name: str, py, ref, rtol: float) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    if py.shape != ref.shape:
        print(f"  FAIL {name}: shape {py.shape} != {ref.shape}")
        return 1
    scale = max(float(np.abs(ref).max(initial=0.0)), 1e-300)
    diff = float(np.abs(py - ref).max(initial=0.0))
    ok = diff <= rtol * scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = "
        f"{diff:.3e} (rel {diff / scale:.2e}, rtol {rtol:.0e})"
    )
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase5b_reference.m in MATLAB)")
        return 1

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    order = ("PRODUCER", "INJECTOR") if bool(ref["isProdFirst"]) else (
        "INJECTOR",
        "PRODUCER",
    )
    result = run_blackoil_tutorial_spe1(nsteps=N_STEPS, well_order=order)
    hist = result.history

    wi_py = np.concatenate([np.asarray(w.WI, dtype=float).ravel() for w in result.model.wells])
    failed = 0
    failed += _compare("timesteps", np.diff(np.r_[0.0, result.time]), ref["dts"], 1e-14)
    failed += _compare("WI", wi_py, ref["WI"], 1e-10)
    failed += _compare("pressure", hist["pressure"], ref["p_hist"], 2e-5)
    failed += _compare("sW", hist["sW"], ref["sw_hist"], 2e-5)
    failed += _compare("sG", hist["sG"], ref["sg_hist"], 2e-5)
    failed += _compare("rs", hist["rs"], ref["rs_hist"], 2e-5)
    failed += _compare("qWs", hist["qWs"], ref["qWs"], 2e-5)
    failed += _compare("qOs", hist["qOs"], ref["qOs"], 2e-5)
    failed += _compare("qGs", hist["qGs"], ref["qGs"], 2e-5)
    failed += _compare("bhp", hist["bhp"], ref["bhp"], 2e-5)

    print(
        "\nSPE1 tutorial reproduces the MRST ad-blackoil reference!"
        if failed == 0
        else f"\n{failed} SPE1 tutorial comparisons FAILED"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
