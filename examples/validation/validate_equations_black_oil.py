"""Validate MRST's equationsBlackOil, the equations of ThreePhaseBlackOilModel.

The reference is `dump_equations_black_oil_reference.m` (MRST 2024a): SPE9 as
a `ThreePhaseBlackOilModel` with dissolved gas and the first control's 26
wells, on a state where every seventh cell has free gas (status 3, `rs` on
its saturation curve) and the rest are undersaturated (status 1), with the
pressure perturbed by up to 1 %. Both the old and the current state are
MRST's own arrays. Every residual, and the assembled Jacobian and
right-hand side, to `RTOL`.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    ThreePhaseBlackOilModel,
)
from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
    convert_deck_schedule_to_mrst,
)
from pymrst.modules.test_suite.dataset_functions.spe9 import setup_spe9
from pymrst.params.gravity.gravity import isolated_gravity
from pymrst.utils.adi import value
from pymrst.utils.units import DAY

HERE = Path(__file__).resolve().parent
RTOL = 1e-12


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def main() -> int:
    ref = loadmat(HERE/"ref_equations_black_oil.mat", squeeze_me=True)
    with isolated_gravity():
        G, rock, fluid, deck, state0 = setup_spe9()
        model = ThreePhaseBlackOilModel(G, rock, fluid, inputdata=deck)
        schedule = convert_deck_schedule_to_mrst(model, deck)
        forces = model.get_driving_forces(schedule.controls[0])
        model = model.validate_model(forces)
        st0 = model.validate_state(state0)
        model, st0 = model.update_for_changed_controls(st0, forces)
        st0["pressure"] = np.array(ref["p0"], dtype=float)
        st0["s"] = np.array(ref["s0"], dtype=float)
        st0["rs"] = np.array(ref["rs0"], dtype=float)
        st = copy_state(st0)
        st["pressure"] = np.array(ref["p"], dtype=float)
        st["s"] = np.array(ref["s"], dtype=float)
        st["rs"] = np.array(ref["rs"], dtype=float)
        for i, w in enumerate(st["wellSol"]):
            w.bhp = float(ref["bhp"][i])
            w.qWs, w.qOs, w.qGs = (float(ref[q][i]) for q in ("qWs", "qOs", "qGs"))
        model, st = model.prepare_report_step(st, st0, DAY, forces)
        model, st = model.prepare_timestep(st, st0, DAY, forces)
        problem, _ = model.get_equations(st0, st, DAY, forces, iteration=1)
    names = [str(n) for n in np.ravel(ref["eqnames"])]
    ok = report(list(problem.equation_names) == names, f"equations {names}")
    for name, eq, expected in zip(names, problem.equations, np.ravel(ref["res"])):
        err = _rel(value(eq), expected)
        ok &= report(err < RTOL, f"{name} residual rel. diff {err:.1e}")
    problem = problem.assemble_system()
    A_ref = sp.csr_matrix(ref["A"])
    err = abs(sp.csr_matrix(problem.A) - A_ref).max()/abs(A_ref).max()
    ok &= report(err < RTOL, f"Jacobian rel. diff {err:.1e}")
    # MRST's right-hand side, -r.
    err = _rel(np.ravel(problem.b), ref["b"])
    ok &= report(err < RTOL, f"right-hand side rel. diff {err:.1e}")
    print("\nequationsBlackOil matches MRST" if ok else "\nequationsBlackOil does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
