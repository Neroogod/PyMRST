"""Phase 49: a MULTI-PERFORATION well, against MATLAB ad-blackoil.

Every well in every MATLAB comparison this port had ever run had **one
perforation**. SPE 1: one. Norne: one. The oil-water quarter spot, the
gas-condensate column, the aquifer case: one each. So the part of the well model
that only exists when a well has *more* than one - the wellbore column - had
never once been checked against MRST.

That is not a corner case. Every real well has dozens of perforations, and the
column is worth bar, not millibar.

**What it gets wrong is subtle, and it is why nothing caught it.** A well is a
chain of nodes: the bhp datum, then each perforation. The fluid in the segment
above a perforation is everything flowing *past* it - the cumulative stream from
every perforation below - so each segment carries its own mixture and its own
density, and the column is their running sum:

    cdp_k = sum_{j<=k} rho_j g (dZ_j - dZ_{j-1})

with `rho_j` the segment's cumulative mass flux over its cumulative volume flux.
PyMRST used to compute a *single* density times the total depth. For one
perforation those are the same thing, exactly - which is why every validated
case passed. For three they are not, and disagreed with MRST at the 1e-3 level.

`dump_phase49_reference.m` runs two genuinely vertical wells, **five
perforations each**, with a heavy density contrast between the phases (water
1050, oil 700) so that *which* fluid sits in *which* segment actually matters.
It dumps the wellbore column itself alongside the states, so the thing under
test is compared directly and not only through its consequences.

The discriminating number is in the producer's column. Its segment increments
come out **unequal** - the stream gets heavier as it collects water on the way
up - and a single-density column can only ever produce equal ones.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import GenericBlackOilModel
from pymrst.modules.ad_core.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.state_functions import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p49_multiperf.mat"


def build(ref):
    nx, ny, nz = int(ref["nx"]), int(ref["ny"]), int(ref["nz"])
    p_ref = float(ref["pRef"])

    g = cart_grid([nx, ny, nz], [400.0, 100.0, 100.0])
    g.nodes.coords[:, 2] += 2000.0
    g = compute_geometry(g)
    nc = g.cells.num
    rock = make_rock(g, 200 * MILLI * DARCY, 0.25)

    fluid = init_simple_adi_fluid(
        mu=np.array([0.4, 3.0]) * CENTI * POISE, rho=[1050.0, 700.0],
        n=[2, 2], c=[4.0e-5 / BARSA, 1.2e-4 / BARSA], p_ref=p_ref,
    )

    inj_cells = np.atleast_1d(np.asarray(ref["inj_cells"], float)).astype(int) - 1
    prod_cells = np.atleast_1d(np.asarray(ref["prod_cells"], float)).astype(int) - 1
    assert inj_cells.size == nz and prod_cells.size == nz, \
        "both wells must be multi-perforation, or this tests nothing"

    grav = np.array([0.0, 0.0, NORM_GRAVITY])
    w = add_well(None, g, rock, inj_cells, well_type="rate", val=4e-3,
                 compi=np.array([1.0, 0.0]), sign=1, radius=0.1, name="INJ",
                 gravity=grav)
    w = add_well(w, g, rock, prod_cells, well_type="bhp", val=200 * BARSA,
                 compi=np.array([0.0, 1.0]), sign=-1, radius=0.1, name="PROD",
                 gravity=grav)

    # MRST's own tolerances are 1e-3/1e-7 and leave a residual of their own; the
    # dump tightens them to 1e-10, so match that rather than compare a converged
    # answer against a half-converged one.
    model = GenericBlackOilModel(g, rock, fluid, w, gravity=NORM_GRAVITY,
                                 tol_cnv=1e-10, tol_mb=1e-13, tol_well=1e-10)

    st = State(pressure=np.full(nc, p_ref), sW=np.full(nc, 0.15),
               qWs=np.array([4e-3, 0.0]), qOs=np.array([0.0, -4e-3]),
               bhp=np.array([p_ref, 200.0 * BARSA]))
    return model, st


def cdp_of(model, st):
    """The wellbore column each well is about to use, at this state.

    MRST computes it once per timestep in `prepareTimestep`, from the state at
    the *start* of the step - so that is what is reproduced here, and it is what
    `wellSol.cdp` holds afterwards.
    """
    fac = model.facility
    ad = model.init_state_ad(st)
    mob = model.get_props(ad, "Mobility")
    b = model.get_props(ad, "ShrinkageFactors")
    rho = model.get_props(ad, "Density")
    flux = st.get("well_flux")
    out = []
    for k, w in enumerate(fac.wells):
        out.append(fac.connection_pressure_drop(
            model, w, ad["pressure"], mob, b, model.n_phases, rho=rho,
            flux=None if flux is None else flux[k]))
    return out


def run(ref):
    model, st = build(ref)
    nstep = int(ref["nstep"])
    dt = 20.0 * DAY
    solver = NonLinearSolver()

    hist = {k: [] for k in ("p", "sw", "qWs", "qOs", "bhp",
                            "cdp_inj", "cdp_prod")}
    for i in range(nstep):
        cdp = cdp_of(model, st)              # the column this step will use
        st, rep = solver.solve_ministep(model, st, st.copy(), dt)
        assert rep.converged, f"step {i} did not converge"
        for key, val in (("p", st["pressure"]), ("sw", st["sW"]),
                         ("qWs", st["qWs"]), ("qOs", st["qOs"]),
                         ("bhp", st["bhp"]),
                         ("cdp_inj", cdp[0]), ("cdp_prod", cdp[1])):
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
        print("ref_p49_multiperf.mat missing (run dump_phase49_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    model, py = run(ref)
    failed = 0

    inj, prod = model.wells
    print(f"Two vertical wells, {len(inj.cells)} and {len(prod.cells)} "
          f"perforations - which no comparison in this port had ever done:")
    failed += compare("INJ  WI", inj.WI, ref["inj_WI"], 1e-12)
    failed += compare("PROD WI", prod.WI, ref["prod_WI"], 1e-12)
    failed += compare("INJ  dZ", inj.dZ, ref["inj_dZ"], 1e-12)
    failed += compare("PROD dZ", prod.dZ, ref["prod_dZ"], 1e-12)

    print("\nThe wellbore column itself - the thing that was wrong:")
    failed += compare("INJ  cdp history", py["cdp_inj"], ref["cdp_inj"], 2e-6)
    failed += compare("PROD cdp history", py["cdp_prod"], ref["cdp_prod"], 2e-6)

    print("\nAnd everything it drives:")
    failed += compare("pressure history", py["p"], ref["p_hist"], 2e-6)
    failed += compare("sW history", py["sw"], ref["sw_hist"], 2e-6)
    failed += compare("qWs history", py["qWs"], ref["qWs"], 2e-6)
    failed += compare("qOs history", py["qOs"], ref["qOs"], 2e-6)
    failed += compare("bhp history", py["bhp"], ref["bhp"], 2e-6)

    # The check no reference can make. A column computed with one density over
    # the whole depth would give increments strictly proportional to the segment
    # lengths - and MRST's producer's are not, because its stream gets heavier
    # as it collects water on the way up. If this ever comes out proportional,
    # the segment-by-segment build has been quietly lost, and the comparison
    # above would still pass on a one-perforation well.
    cdp = py["cdp_prod"][-1]
    dz = np.asarray(ref["prod_dZ"], float).ravel()
    inc = np.diff(np.concatenate([[0.0], cdp]))
    seg = np.diff(np.concatenate([[0.0], dz]))
    rho_eff = inc / seg / NORM_GRAVITY
    spread = float(rho_eff.max() - rho_eff.min())
    ok = spread > 1.0
    print(f"\n  {'ok  ' if ok else 'FAIL'} the segments really do differ: the "
          f"producer's column implies densities "
          f"{np.array2string(rho_eff, precision=1, floatmode='fixed')} kg/m3 "
          f"- a single-density column would make these identical")
    failed += 0 if ok else 1

    print("\nMulti-perforation wells match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
