"""Phase 118: compositional boundary-condition setup vs MATLAB MRST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.examples._shared import ScheduleControl, run_schedule, simple_schedule
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (
    make_boundary_conditions_case,
)
from pymrst.params.wells_and_bc import fluxside, pside

HERE = Path(__file__).parent
REF = HERE / "ref_p118_compositional_bc.mat"


def _strings(raw) -> list[str]:
    arr = np.asarray(raw, dtype=object).ravel()
    return [str(x[0] if isinstance(x, np.ndarray) else x) for x in arr]


def compare(name, py, ref, rtol=0.0, atol=0.0) -> int:
    py = np.asarray(py)
    ref = np.asarray(ref)
    if py.dtype.kind in {"U", "S", "O"} or ref.dtype.kind in {"U", "S", "O"}:
        ok = list(py.ravel()) == list(ref.ravel())
        diff = 0.0 if ok else np.inf
    elif py.shape != ref.shape:
        diff = np.inf
        ok = False
    else:
        diff = float(np.max(np.abs(py.astype(float) - ref.astype(float)))) if py.size else 0.0
        scale = max(float(np.max(np.abs(ref.astype(float)))), 1.0)
        ok = diff <= atol + rtol * scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
        f"(rtol {rtol:.1e}, atol {atol:.1e})"
    )
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p118_compositional_bc.mat missing (run dump_phase118_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    case = make_boundary_conditions_case(full_size=True)
    mix = case["mixture"]
    bc = case["bc"]

    print("\nPhase 118: compositional boundary-condition setup vs MATLAB")
    failures = 0
    failures += compare("grid dims", case["grid"].cartDims, ref["dims"], atol=0.0)
    failures += compare("cell count", case["grid"].cells.num, ref["cells"], atol=0.0)
    failures += compare("face count", case["grid"].faces.num, ref["faces"], atol=0.0)
    failures += compare("critical temperatures", mix.Tcrit, ref["Tcrit"], atol=1e-12)
    failures += compare("critical pressures", mix.Pcrit, ref["Pcrit"], atol=1e-6)
    failures += compare("critical volumes", mix.Vcrit, ref["Vcrit"], atol=1e-14)
    failures += compare("acentric factors", mix.acentric, ref["acentricFactors"], atol=1e-12)
    failures += compare("molar masses", mix.molarMass, ref["molarMass"], atol=1e-14)
    failures += compare("initial composition", case["initial"], ref["initial"], atol=1e-14)
    failures += compare("injection composition", case["injection"], ref["injection"], atol=1e-14)
    failures += compare("temperature", case["temperature"], ref["temperature"], atol=1e-12)
    failures += compare("minimum pressure", case["min_pressure"], ref["minP"], atol=1e-8)
    failures += compare("total time", case["total_time"], ref["totTime"], atol=1e-5)
    failures += compare("total pore volume", case["bc"].value[:50].sum() * case["total_time"],
                        ref["pv_total"], rtol=1e-12, atol=1e-12)
    failures += compare("bc faces", bc.face + 1, ref["bc_faces"], atol=0.0)
    failures += compare("bc values", bc.value, ref["bc_values"], rtol=1e-10, atol=1e-18)
    failures += compare("bc saturations", bc.sat, ref["bc_sat"], atol=0.0)
    failures += compare("bc components", bc.components, ref["bc_components"], atol=1e-14)
    failures += compare("bc types", bc.type, _strings(ref["bc_types"]))
    failures += compare("timesteps", case["schedule"].step.val, ref["timesteps"],
                        rtol=1e-14, atol=1e-8)

    small = make_boundary_conditions_case(nx=4, ny=2)
    bc_b = fluxside(None, small["grid"], "xmin", small["bc"].value[:2].sum(), sat=np.array([1.0, 0.0]))
    bc_b = pside(bc_b, small["grid"], "xmax", small["min_pressure"], sat=np.array([1.0, 0.0]))
    bc_b.components = np.tile([0.8, 0.2, 0.0], (bc_b.face.size, 1))
    schedule = simple_schedule(
        small["schedule"].step.val[:2],
        control=[ScheduleControl(bc=small["bc"]), ScheduleControl(bc=bc_b)],
        step_control=[1, 2],
    )
    result = run_schedule(small["overall_model"], small["initial_state"], schedule)
    failures += compare("two-control schedule indices", schedule.step.control, [1, 2], atol=0.0)
    failures += compare("two-control switched composition", schedule.control[1].bc.components[0],
                        [0.8, 0.2, 0.0], atol=0.0)
    if not (result.final_state.z[0, 0] > small["initial_state"].z[0, 0]):
        print("  FAIL switched schedule changes inlet composition response")
        failures += 1
    else:
        print("  ok   switched schedule changes inlet composition response")

    if failures:
        print(f"\nPhase 118 validation failed ({failures} checks).")
    else:
        print("\nCompositional boundary-condition setup matches MATLAB and schedule switching is active.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
