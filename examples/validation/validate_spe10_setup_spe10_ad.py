"""Validate MRST setupSPE10_AD, the port in `spe10/setup_spe10_ad.py`.

The reference is MRST 2024a (`dump_setup_spe10_ad_reference.m`): layers 1 and
85 of SPE10 Model 2 with rows 1:110 as 2D models, as `compositeModelTutorial`
builds them through the test case 'spe10_wo'. Well cells identical; well
indices, controls, pore volumes, transmissibilities and the initial state to
`RTOL`; time steps identical; and the first five steps simulated: Newton
iterations identical, well curves and the final pressure and saturation to
`RUN_RTOL`.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-12
RUN_RTOL = 1e-9


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def main() -> int:
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.schedule_types import Schedule
    from pymrst.modules.spe10.setup_spe10_ad import setup_spe10_ad
    from pymrst.params.gravity.gravity import isolated_gravity
    ref = loadmat(HERE/"ref_setup_spe10_ad.mat", squeeze_me=True, struct_as_record=False)
    ok = True
    for layer in (1, 85):
        r = ref[f"layer{layer}"]
        tag = f"layer {layer}:"
        with isolated_gravity():
            state0, model, schedule = setup_spe10_ad(layers=layer, J=np.arange(1, 111),
                                                     make2D=True)
            W = schedule.controls[0].W
            cells = np.concatenate([np.atleast_1d(w.cells) for w in W]) + 1
            ok &= report(np.array_equal(cells, np.atleast_1d(r.cells)), f"{tag} well cells")
            for name, mine, theirs in (
                    ("well indices", np.concatenate([np.atleast_1d(w.WI) for w in W]), r.WI),
                    ("well controls", [w.val for w in W], r.val),
                    ("pore volumes", model.operators.pv, r.pv),
                    ("transmissibilities", model.operators.T, r.T),
                    ("initial pressure", state0.pressure, r.p0),
                    ("initial saturation", state0.s, r.s0)):
                err = _rel(mine, theirs)
                ok &= report(err < RTOL, f"{tag} {name} rel. diff {err:.1e}")
            ok &= report(np.array_equal(schedule.step_val, r.dt),
                         f"{tag} {schedule.step_val.size} time steps identical")
            sub = Schedule(step_val=schedule.step_val[:5], step_control=schedule.step_control[:5],
                           controls=schedule.controls)
            with contextlib.redirect_stdout(io.StringIO()):
                ws, states, rep = simulate_schedule_ad(state0, model, sub)
        its = [c["Iterations"] for c in rep["ControlstepReports"]]
        ok &= report(its == list(np.atleast_1d(r.its)), f"{tag} iterations {its}")
        for field in ("bhp", "qWs", "qOs"):
            err = _rel([[getattr(w, field) for w in row] for row in ws], getattr(r, field))
            ok &= report(err < RUN_RTOL, f"{tag} well {field} rel. diff {err:.1e}")
        err = max(_rel(states[-1]["pressure"], r.p), _rel(np.asarray(states[-1]["s"])[:, 0], r.sw))
        ok &= report(err < RUN_RTOL, f"{tag} final pressure and saturation rel. diff {err:.1e}")
    print("\nsetupSPE10_AD matches MRST" if ok else "\nsetupSPE10_AD does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
