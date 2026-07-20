"""Phase 115: compositional boundary-condition schedule vs MATLAB MRST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (
    run_compositional_boundary_conditions_example,
)
from pymrst.utils.units import BARSA

HERE = Path(__file__).parent
REF = HERE / "ref_p115_compositional_bc_schedule.mat"


def _saturations(out, natural: bool):
    model = out["natural_model"] if natural else out["overall_model"]
    result = out["natural_result"] if natural else out["result"]
    rows = []
    for st in result.states:
        if natural and getattr(st, "sO", None) is not None:
            rows.append(np.column_stack([st.sO, st.sG]))
        else:
            fp = backend_flow_props(model.mix, model.eos_model, st.pressure, model.T, st.z)
            rows.append(np.column_stack([fp.sL, fp.sV]))
    return np.stack(rows)


def _pack(out, natural: bool):
    result = out["natural_result"] if natural else out["result"]
    return {
        "pressure": np.vstack([st.pressure for st in result.states]),
        "components": np.stack([st.z for st in result.states]),
        "s": _saturations(out, natural),
        "iterations": result.iterations.astype(float),
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


def _validate_model(label: str, out, natural: bool, ref: dict) -> int:
    py = _pack(out, natural)
    prefix = "natural" if natural else "overall"
    print(f"\n{label}")
    failures = 0
    failures += compare("pressure", py["pressure"], ref[f"{prefix}_pressure"],
                        rtol=0.0, atol=1.0e-6 * BARSA, scale=BARSA, unit="bar")
    failures += compare("overall components", py["components"], ref[f"{prefix}_components"],
                        rtol=0.0, atol=1.0e-7)
    failures += compare("phase saturations", py["s"], ref[f"{prefix}_s"],
                        rtol=0.0, atol=1.0e-7)
    failures += compare("Newton iterations", py["iterations"], ref[f"{prefix}_iterations"],
                        rtol=0.0, atol=1.0)
    return failures


def main() -> int:
    if not REF.exists():
        print("ref_p115_compositional_bc_schedule.mat missing (run dump_phase115_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    permeability_md = np.asarray(ref["permeability_md"], dtype=float).ravel()
    out = run_compositional_boundary_conditions_example(
        nstep=2,
        permeability_md=permeability_md,
    )

    print("\nPhase 115: compositional boundary-condition schedule vs MATLAB")
    failures = 0
    failures += compare("schedule timesteps", out["timesteps"], ref["tsteps"], atol=0.0)
    failures += compare("bc components", out["bc"].components, ref["bc_components"], atol=0.0)
    failures += compare("bc saturations", out["bc"].sat, ref["bc_sat"], atol=0.0)
    failures += compare("bc values", out["bc"].value, ref["bc_values"], rtol=0.0, atol=1.0e-20)
    failures += _validate_model("Overall-composition model", out, False, ref)
    failures += _validate_model("Natural-variable model", out, True, ref)
    if failures:
        print(f"\nPhase 115 validation failed ({failures} checks).")
    else:
        print("\nCompositional boundary-condition schedule matches MATLAB within the current parity envelope.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
