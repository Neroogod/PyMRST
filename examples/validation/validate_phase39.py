"""Phase 39: well control switching, against MRST.

A real schedule is not "inject at 1 l/s" but "inject at 1 l/s *unless* that
needs more than 300 bar, in which case hold 300 bar". PyMRST's wells had no
limits at all: a rate-controlled injector would demand whatever bottom-hole
pressure its rate required, however unphysical, and a BHP producer would pull
whatever the reservoir would give.

`dump_phase39_reference.m` runs the same case twice through MRST's
`GenericBlackOilModel`:

- **with limits** the injector cannot hold its rate within 300 bar and the
  producer pulls more liquid than it is allowed, so both must switch;
- **with the limits placed out of reach** neither may switch, which checks that
  the machinery is inert when it should be - the failure mode that would
  quietly corrupt every existing validated phase.

What is compared is not just *that* a well switched but *what it switched to*
and *what it then did*: the control mode of every well at every step, and the
pressures and rates that follow from it.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import GenericBlackOilModel
from pymrst.modules.ad_core.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.state_functions import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p39_limits.mat"


def run_python(ref, tight: bool):
    """The case of dump_phase39_reference.m, with limits that bite or do not."""
    nx = int(ref["nx"])
    p_ref = float(ref["pRef"])
    q_inj = float(ref["q_inj"])
    bhp_prod = float(ref["bhp_prod"])

    if tight:
        inj_lims = {"bhp": float(ref["bhp_lim"])}
        prod_lims = {"lrat": float(ref["lrat_lim"])}
    else:
        inj_lims = {"bhp": 1e5 * BARSA}          # unreachable
        prod_lims = {"lrat": -1e3}

    g = compute_geometry(cart_grid([nx, nx, 1], [200.0, 200.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=p_ref,
    )

    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=q_inj,
                 compi=np.array([1.0, 0.0]), sign=1, radius=0.1, name="inj",
                 lims=inj_lims)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=bhp_prod, compi=np.array([0.0, 1.0]), sign=-1,
                 radius=0.1, name="prod", lims=prod_lims)

    model = GenericBlackOilModel(g, rock, fluid, w)
    st = State(
        pressure=np.full(g.cells.num, p_ref),
        sW=np.full(g.cells.num, 0.2),
        qWs=np.array([q_inj, 0.0]), qOs=np.array([0.0, -q_inj]),
        bhp=np.array([p_ref, bhp_prod]),
    )

    solver = NonLinearSolver(max_iterations=40)
    p, sw, bhp, qw, qo, types = [], [], [], [], [], []
    for i in range(6):
        st, rep = solver.solve_ministep(model, st, st.copy(), 30 * DAY)
        assert rep.converged, f"step {i}: {rep.residuals[-1]}"
        p.append(st["pressure"].copy())
        sw.append(st["sW"].copy())
        bhp.append(st["bhp"].copy())
        qw.append(st["qWs"].copy())
        qo.append(st["qOs"].copy())
        types.append([w.type for w in model.wells])
    return (np.vstack(p), np.vstack(sw), np.vstack(bhp), np.vstack(qw),
            np.vstack(qo), np.array(types, dtype=object))


def compare(name, py, ml, rtol) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def compare_types(py, ml) -> int:
    """The control mode of every well at every step - the switch itself."""
    ml = np.asarray(ml, dtype=object)
    same = all(str(py[i, j]).lower() == str(ml[i, j]).lower()
               for i in range(ml.shape[0]) for j in range(ml.shape[1]))
    print(f"  {'ok  ' if same else 'FAIL'} control modes: "
          f"PyMRST inj={list(py[:, 0])}, prod={list(py[:, 1])}")
    if not same:
        print(f"       MATLAB inj={[str(t) for t in ml[:, 0]]}, "
              f"prod={[str(t) for t in ml[:, 1]]}")
    return 0 if same else 1


def main() -> int:
    if not REF.exists():
        print("ref_p39_limits.mat missing (run dump_phase39_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    failed = 0

    print("Limits that bite - both wells must switch:")
    p, sw, bhp, qw, qo, ty = run_python(ref, tight=True)
    failed += compare_types(ty, ref["t_types"])
    failed += compare("pressure", p, ref["t_p"], 1e-9)
    failed += compare("sW", sw, ref["t_sw"], 1e-9)
    failed += compare("bhp", bhp, ref["t_bhp"], 1e-9)
    failed += compare("qWs", qw, ref["t_qw"], 1e-9)
    failed += compare("qOs", qo, ref["t_qo"], 1e-9)

    print("\nLimits out of reach - nothing may switch:")
    p, sw, bhp, qw, qo, ty = run_python(ref, tight=False)
    failed += compare_types(ty, ref["f_types"])
    failed += compare("pressure", p, ref["f_p"], 1e-9)
    failed += compare("sW", sw, ref["f_sw"], 1e-9)
    failed += compare("bhp", bhp, ref["f_bhp"], 1e-9)
    failed += compare("qWs", qw, ref["f_qw"], 1e-9)
    failed += compare("qOs", qo, ref["f_qo"], 1e-9)

    print("\nWell control switching matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
