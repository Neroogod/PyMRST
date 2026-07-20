"""Phase 48b: a *coupled* aquifer simulation, against MATLAB ad-core.

Phase 48 validated the aquifer **formulas** in isolation - it fixed the water
pressure and shrinkage MRST evaluated at the connections and replayed PyMRST's
arithmetic on the identical numbers. That is the right way to test a formula,
and it deliberately leaves one thing untested: the **coupling**.

Two things only exist inside a running simulation:

- `add_contribution`, which subtracts the influx from the water equation - so
  the aquifer enters the *Jacobian*, and the reservoir actually feels it;
- `after_convergence`, which advances the aquifer's own pressure and volume once
  a step is accepted - so the tank actually drains.

A port can get both formulas exactly right and still wire either of them up
wrongly. Flip the sign and the aquifer *drains* the reservoir instead of
supporting it. Hold it explicit at the wrong state and it drains at the wrong
rate. Forget to carry `p_aq`/`V_aq` through a timestep cut and it advances
twice. **None of that shows up in Phase 48**, because none of it is arithmetic -
it is plumbing, and plumbing is what a coupled run tests.

So: a real oil-water reservoir, a producer at one end pulling it down, an
aquifer on the far end holding it up, and twenty steps. Everything the two codes
must agree on is compared - the pressure and saturation fields, the well rates,
and the aquifer's own pressure and volume trajectory. The aquifer is doing real
work here (it gives up 20 bar over the run), so a sign error could not hide.

The schedule is chosen so MRST takes **exactly one ministep per report step**
(the dump asserts it), which is what makes a step-by-step comparison meaningful
at all.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import GenericBlackOilModel
from pymrst.modules.ad_core.models.aquifers.aquifer import AquiferConnections, AquiferModel
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p48b_aquifer.mat"


def build(ref):
    nx, nz = int(ref["nx"]), int(ref["nz"])
    p_ref = float(ref["pRef"])

    g = cart_grid([nx, 1, nz], [500.0, 100.0, 30.0])
    g.nodes.coords[:, 2] += 2000.0
    g = compute_geometry(g)
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)

    fluid = init_simple_adi_fluid(
        mu=np.array([0.5, 2.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[4.0e-5 / BARSA, 1.0e-4 / BARSA], p_ref=p_ref,
    )

    # MATLAB's cell indices are 1-based.
    conncells = np.atleast_1d(np.asarray(ref["conncells"], float)).astype(int) - 1
    prodcells = np.atleast_1d(np.asarray(ref["prodcells"], float)).astype(int) - 1
    nconn = conncells.size

    connections = AquiferConnections(
        aquid=np.zeros(nconn, dtype=int),
        cells=conncells,
        j=np.atleast_1d(ref["Jval"]).astype(float),
        c=np.full(nconn, float(ref["Caq"])),
        alpha=np.atleast_1d(ref["alpha"]).astype(float),
        depth_conn=np.atleast_1d(ref["depthconn"]).astype(float),
        depth_aq=np.full(nconn, float(ref["depthaq"])),
        c_aq=np.array([float(ref["Caq"])]),
    )
    aquifer = AquiferModel(connections,
                           pressures=np.array([250.0 * BARSA]),
                           volumes=np.array([5.0e7]))

    # `gravity` must be handed to `add_well`, not just to the model. MRST's
    # `addWell` reads the *global* gravity and, when it is on, defaults the bhp
    # reference depth to the top of the formation; PyMRST does the same, but
    # only if it is told. Leave it out and the reference depth falls back to
    # zero - the *surface* - so the well carries ~167 bar of extra hydrostatic
    # head, its drawdown changes sign, and the producer silently becomes an
    # injector. It converged; it was just answering a different question.
    w = add_well(None, g, rock, prodcells, well_type="bhp", val=150 * BARSA,
                 compi=np.array([0.0, 1.0]), sign=-1, radius=0.1, name="PROD",
                 gravity=np.array([0.0, 0.0, NORM_GRAVITY]))

    model = GenericBlackOilModel(g, rock, fluid, w, gravity=NORM_GRAVITY,
                                 aquifer=aquifer)

    nc = g.cells.num
    st = State(
        pressure=np.full(nc, p_ref), sW=np.full(nc, 0.2),
        qWs=np.zeros(1), qOs=np.zeros(1), bhp=np.array([150.0 * BARSA]),
    )
    aquifer.init_state(st)
    return model, st


def run(ref):
    model, st = build(ref)
    nstep = int(ref["nstep"])
    dt = 20.0 * DAY
    solver = NonLinearSolver()

    hist = {k: [] for k in ("p", "sw", "aq_p", "aq_V", "aq_q",
                            "qWs", "qOs", "bhp")}
    for i in range(nstep):
        # the aquifer flux MRST records is the one computed at the converged
        # state with the *pre-step* tank pressure - exactly what its own
        # after-convergence update consumes.
        p0, v0 = st["aquifer_p"].copy(), st["aquifer_V"].copy()
        st, rep = solver.solve_ministep(model, st, st.copy(), dt)
        assert rep.converged, f"step {i} did not converge"
        q = np.asarray(value(
            model.aquifer.fluxes(model, model.init_state_ad(st), p0, v0, dt)
        ), dtype=float)
        for key, val in (("p", st["pressure"]), ("sw", st["sW"]),
                         ("aq_p", st["aquifer_p"]), ("aq_V", st["aquifer_V"]),
                         ("aq_q", q), ("qWs", st["qWs"]), ("qOs", st["qOs"]),
                         ("bhp", st["bhp"])):
            hist[key].append(np.asarray(val, float).ravel().copy())
    return model, {k: np.vstack(v) for k, v in hist.items()}


def compare(name, py, ml, rtol) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    if py.size != ml.size:
        print(f"  FAIL {name}: {py.size} values, MATLAB has {ml.size}")
        return 1
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p48b_aquifer.mat missing (run dump_phase48b_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    model, py = run(ref)
    failed = 0

    print("Reservoir, driven by a producer and supported by the aquifer:")
    failed += compare("WI", model.wells[0].WI, ref["WI"], 1e-12)
    failed += compare("pressure history", py["p"], ref["p_hist"], 2e-6)
    failed += compare("sW history", py["sw"], ref["sw_hist"], 2e-6)
    failed += compare("qWs history", py["qWs"], ref["qWs"], 2e-6)
    failed += compare("qOs history", py["qOs"], ref["qOs"], 2e-6)
    failed += compare("bhp history", py["bhp"], ref["bhp"], 2e-6)

    print("\nThe aquifer's own state - it must drain at the right rate:")
    failed += compare("connection flux", py["aq_q"], ref["aq_q"], 2e-6)
    failed += compare("aquifer pressure", py["aq_p"], ref["aq_p"], 2e-6)
    failed += compare("aquifer volume", py["aq_V"], ref["aq_V"], 2e-6)

    # The checks that no reference can make. A sign error in the coupling would
    # still produce *some* trajectory, and it would still match nothing - but
    # these say what the right answer has to look like, so a future change that
    # matches MATLAB for the wrong reason still trips.
    ml_aq = np.asarray(ref["aq_p"], float).ravel()
    ml_p = np.asarray(ref["p_hist"], float)
    conn = np.atleast_1d(np.asarray(ref["conncells"], float)).astype(int) - 1

    drained = float(250.0 * BARSA - ml_aq[-1]) / BARSA
    ok_drain = drained > 1.0
    print(f"\n  {'ok  ' if ok_drain else 'FAIL'} the aquifer is doing real "
          f"work: it gave up {drained:.1f} bar over the run (a decoration "
          f"would give up ~0)")
    failed += 0 if ok_drain else 1

    # ...and it must be *supporting* the reservoir, not draining it: the tank
    # stays above the cells it feeds. Get the sign backwards and this flips.
    supports = bool((py["aq_p"].ravel()
                     > py["p"][:, conn].max(axis=1)).all())
    print(f"  {'ok  ' if supports else 'FAIL'} and supporting, not draining: "
          f"the aquifer stays above its connection cells for every step "
          f"({py['aq_p'].ravel()[-1] / BARSA:.1f} vs "
          f"{py['p'][-1, conn].max() / BARSA:.1f} bar at the end)")
    failed += 0 if supports else 1

    print("\nThe coupled aquifer matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
