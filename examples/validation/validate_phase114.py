"""Phase 114: native SPE1 K-value compositional solve vs MATLAB MRST.

This validator compares the two-report-step native PyMRST compositional SPE1
run against ``dump_phase114_reference.m``.  The EOS/K-table layer is already
strictly validated in phase 111; this phase checks end-to-end schedule behavior
with tolerances that expose the remaining facility-model differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.examples.run_compositional_spe1 import (
    run_native_compositional_spe1,
)
from pymrst.utils.units import BARSA

HERE = Path(__file__).parent
REF = HERE / "ref_p114_spe1_native_compositional.mat"


def _saturations(out):
    model = out.compositional_model
    rows = []
    for st in out.compositional_states:
        if hasattr(st, "sO") and st.sO is not None:
            s_o, s_g = st.sO, st.sG
        else:
            fp = backend_flow_props(model.mix, model.eos_model, st.pressure, model.T, st.z)
            s_o = (1.0 - st.sW) * fp.sL
            s_g = (1.0 - st.sW) * fp.sV
        rows.append(np.column_stack([st.sW, s_o, s_g]))
    return np.stack(rows)


def _pack(out):
    return {
        "pressure": np.vstack([st.pressure for st in out.compositional_states]),
        "components": np.stack([st.z for st in out.compositional_states]),
        "s": _saturations(out),
        "bhp": np.vstack([st.bhp for st in out.compositional_states]),
        "qOs": np.vstack([st.qOs for st in out.compositional_states]),
        "qGs": np.vstack([st.qGs for st in out.compositional_states]),
        "iterations": np.asarray(out.iterations, dtype=float),
    }


def compare(name, py, ref, rtol=0.0, atol=0.0, scale=1.0, unit="") -> int:
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if py.shape != ref.shape:
        diff = np.inf
        ok = False
    else:
        diff = float(np.max(np.abs(py - ref))) if py.size else 0.0
        ref_scale = max(float(np.max(np.abs(ref))), 1.0)
        ok = diff <= atol + rtol * ref_scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff / scale:.3e} {unit} "
        f"(rtol {rtol:.1e}, atol {atol / scale:.1e} {unit})"
    )
    return 0 if ok else 1


def _validate_model(label: str, natural: bool, ref: dict) -> int:
    out = run_native_compositional_spe1(
        nsteps=2,
        natural=natural,
        min_dt=6.0 * 3600.0,
        interpolation="linear",
    )
    py = _pack(out)
    prefix = "natural" if natural else "overall"
    print(f"\n{label}")
    failures = 0
    failures += compare("pressure", py["pressure"], ref[f"{prefix}_pressure"],
                        rtol=0.0, atol=5.0 * BARSA, scale=BARSA, unit="bar")
    failures += compare("overall pseudo components", py["components"], ref[f"{prefix}_components"],
                        rtol=0.0, atol=1.0e-3)
    failures += compare("phase saturations", py["s"], ref[f"{prefix}_s"],
                        rtol=0.0, atol=5.0e-3)
    failures += compare("well BHP", py["bhp"], ref[f"{prefix}_bhp"],
                        rtol=0.0, atol=3.0 * BARSA, scale=BARSA, unit="bar")
    failures += compare("well oil surface rate", py["qOs"], ref[f"{prefix}_qOs"],
                        rtol=0.0, atol=1.2e-3)
    failures += compare("well gas surface rate", py["qGs"], ref[f"{prefix}_qGs"],
                        rtol=0.0, atol=1.0e-6)
    failures += compare("Newton iterations", py["iterations"], ref[f"{prefix}_iterations"],
                        rtol=0.0, atol=1.0)
    return failures


def main() -> int:
    if not REF.exists():
        print("ref_p114_spe1_native_compositional.mat missing (run dump_phase114_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    print("\nPhase 114: native SPE1 K-value compositional solve vs MATLAB")
    failures = 0
    failures += _validate_model("Overall-composition model", False, ref)
    failures += _validate_model("Natural-variable model", True, ref)
    if failures:
        print(f"\nPhase 114 validation failed ({failures} checks).")
    else:
        print("\nNative SPE1 compositional run matches MATLAB within the current parity envelope.")
        print("Remaining nonzero differences are concentrated in pseudo-component composition interpolation.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
