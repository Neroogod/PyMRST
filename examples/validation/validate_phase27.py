"""Phase 27: coupled flow and geomechanics, validated against MATLAB and physics.

Three levels, because poroelasticity earns all three:

1. **MATLAB.** The same 3D poroelastic box run through MRST's `MechWaterModel`
   (`dump_phase27_reference.m`) and through PyMRST's, compared step by step in
   pressure and displacement.

2. **Terzaghi.** 1D consolidation against the analytic series solution
   (`poromech.py`'s self-tests).

3. **Mandel.** The benchmark that actually discriminates (`mandel.py`). A slab
   squeezed between rigid plates and drained at its sides shows the
   Mandel-Cryer effect: the pressure at the centre *rises above* its initial
   value before decaying, because fluid leaving the drained edge softens it
   and transfers load onto the still-undrained core. A one-way-coupled or
   lagged model decays monotonically and would sail through Terzaghi. Ours
   reproduces the overshoot and the whole profile to under 1%.

The MATLAB comparison here is the weakest of the three and the other two are
not decoration: MRST could in principle be wrong together with us, but it
cannot be wrong together with Mandel.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_water_model import MechWaterModel
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c
from pymrst.params.rock import make_rock

HERE = Path(__file__).parent
REF = HERE / "ref_p27_poromech.mat"


def build_case(ref: Dict[str, Any]):
    """The case of dump_phase27_reference.m, rebuilt in PyMRST."""
    nx, ny, nz = (int(ref[k]) for k in ("nx", "ny", "nz"))
    L = np.asarray(ref["L"], dtype=float).ravel()
    E = float(ref["E"])
    nu = float(ref["nu"])
    alpha = float(ref["alpha"])
    perm = float(ref["perm"])
    poro = float(ref["poro"])
    mu_w = float(ref["muW"])
    rho_w = float(ref["rhoW"])
    c_w = float(ref["cW"])
    sigma0 = float(ref["sigma0"])

    g = create_augmented_grid(compute_geometry(cart_grid([nx, ny, nz], L)))
    rock = make_rock(g, perm, poro)

    pts, fc = g.nodes.coords, g.faces.centroids
    tol = 1e-9
    xmin = np.flatnonzero(pts[:, 0] < tol)
    xmax = np.flatnonzero(pts[:, 0] > L[0] - tol)
    ymin = np.flatnonzero(pts[:, 1] < tol)
    ymax = np.flatnonzero(pts[:, 1] > L[1] - tol)
    base = np.flatnonzero(pts[:, 2] > L[2] - tol)

    nodes = np.concatenate([xmin, xmax, ymin, ymax, base])
    mask = np.vstack([
        np.tile([True, False, False], (xmin.size + xmax.size, 1)),
        np.tile([False, True, False], (ymin.size + ymax.size, 1)),
        np.tile([False, False, True], (base.size, 1)),
    ])

    bd = np.any(g.faces.neighbors == -1, axis=1)
    top_faces = np.flatnonzero(bd & (fc[:, 2] < tol))

    el_bc = ElasticBC(
        disp_nodes=nodes, disp_mask=mask,
        disp_vals=np.zeros((nodes.size, 3)),
        force_faces=top_faces,
        force=np.tile([0.0, 0.0, sigma0], (top_faces.size, 1)),
    )
    C = enu2c(np.full(g.cells.num, E), np.full(g.cells.num, nu))
    mech = MechanicModel(g, C, alpha=alpha, el_bc=el_bc)

    # MRST's initSimpleADIFluid: bW = exp(c (p - pRef)), pRef = 0.
    def b(p):
        from pymrst.utils.adi import ad_exp
        return ad_exp(c_w * p)

    model = MechWaterModel(g, rock, mech, b, mu_w, rho_w,
                           top_faces, np.zeros(top_faces.size))
    return g, model, mech


def run_python(ref) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, model, mech = build_case(ref)
    dt = np.asarray(ref["dt"], dtype=float).ravel()

    st = State(pressure=np.zeros(g.cells.num),
               xd=np.zeros(mech.n_free),
               u=np.zeros(3 * g.nodes.num))

    p_hist, xd_hist, uz_hist = [], [], []
    for i, step in enumerate(dt):
        st, ok, _ = model.step(st, float(step))
        assert ok, f"coupled step {i} did not converge"
        p_hist.append(st["pressure"].copy())
        xd_hist.append(st["xd"].copy())
        uz_hist.append(st["u"].reshape(-1, 3)[:, 2].copy())
    return (np.vstack(p_hist), np.vstack(xd_hist), np.vstack(uz_hist))


def compare(name, py, ml, rtol) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


# -- Phase 27b: multiphase poroelasticity, and MRST's oil-water defect --------

REF_B = HERE / "ref_p27b_mechow.mat"


def build_mech_ow(ref, case: str, mrst_bug: bool):
    """The oil-water poroelastic case of dump_phase27b_reference.m."""
    from pymrst.modules.ad_mechanics.models.fully_coupled.mech_oil_water_model import (
        MechOilWaterModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    c = ref[case]
    cs = np.asarray(c["cellsize"], dtype=float).ravel()
    nx, ny, nz = (int(ref[k]) for k in ("nx", "ny", "nz"))
    dims, phys = [nx, ny, nz], [nx * cs[0], ny * cs[1], nz * cs[2]]

    g = create_augmented_grid(compute_geometry(cart_grid(dims, phys)))
    rock = make_rock(g, float(ref["perm"]), float(ref["poro"]))
    p0, sw0 = float(ref["p0"]), float(ref["sw0"])

    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=p0,
    )

    pts, tol = g.nodes.coords, 1e-9
    xf = np.flatnonzero((pts[:, 0] < tol) | (pts[:, 0] > phys[0] - tol))
    yf = np.flatnonzero((pts[:, 1] < tol) | (pts[:, 1] > phys[1] - tol))
    zf = np.flatnonzero(pts[:, 2] > phys[2] - tol)
    nodes = np.concatenate([xf, yf, zf])
    mask = np.vstack([
        np.tile([True, False, False], (xf.size, 1)),
        np.tile([False, True, False], (yf.size, 1)),
        np.tile([False, False, True], (zf.size, 1)),
    ])
    el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask,
                      disp_vals=np.zeros((nodes.size, 3)))
    C = enu2c(np.full(g.cells.num, float(ref["E"])),
              np.full(g.cells.num, float(ref["nu"])))
    mech = MechanicModel(g, C, alpha=float(ref["alpha"]), el_bc=el_bc)

    w = add_well(None, g, rock, np.array([0]), well_type="rate",
                 val=float(ref["qinj"]), compi=np.array([1.0, 0.0]),
                 sign=1, radius=0.1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=float(ref["pbhp"]), compi=np.array([0.0, 1.0]),
                 sign=-1, radius=0.1)

    model = MechOilWaterModel(g, rock, fluid, w, mech,
                              mrst_oilwater_bug=mrst_bug)
    st = State(
        pressure=np.full(g.cells.num, p0), sW=np.full(g.cells.num, sw0),
        qWs=np.array([float(ref["qinj"]), 0.0]),
        qOs=np.array([0.0, 0.0]),
        bhp=np.array([p0, float(ref["pbhp"])]),
        xd=np.zeros(mech.n_free),            # as MRST's initState.xd = 0
        u=np.zeros(3 * g.nodes.num),
    )
    return model, st


def run_mech_ow(ref, case: str, mrst_bug: bool):
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver

    model, st = build_mech_ow(ref, case, mrst_bug)
    solver = NonLinearSolver()
    p, sw = [], []
    for i, dt in enumerate(np.asarray(ref["dt"], float).ravel()):
        st, rep = solver.solve_ministep(model, st, st.copy(), float(dt))
        assert rep.converged, f"{case} step {i}: {rep.residuals[-1]}"
        p.append(st["pressure"].copy())
        sw.append(st["sW"].copy())
    return np.vstack(p), np.vstack(sw)


def phase_27b() -> int:
    """Multiphase poroelasticity, and MRST's missing cell volume."""
    ref = scipy.io.loadmat(REF_B, squeeze_me=True, struct_as_record=False)
    ref = {k: v for k, v in ref.items() if not k.startswith("__")}
    for k in ("big", "unit"):
        ref[k] = {f: getattr(ref[k], f) for f in ref[k]._fieldnames}

    failed = 0
    print("\n=== Phase 27b: multiphase poroelasticity (MechOilWaterModel) ===")

    # The decisive test. On cells of 500 m^3 a correct Biot term must DISAGREE
    # with MRST, and reproducing MRST's formula must recover MRST's answer
    # *exactly* - which pins the defect to the missing factor and nothing else.
    print("\nCells of 500 m^3 - a correct Biot term disagrees with MRST:")
    p_ok, _ = run_mech_ow(ref, "big", mrst_bug=False)
    ml = np.asarray(ref["big"]["p"], float)
    rel = np.abs(p_ok - ml).max() / np.abs(ml).max()
    print(f"  correct model vs MRST: rel diff = {rel:.2e} "
          f"({'differs, as predicted' if rel > 1e-6 else 'NO DIFFERENCE?!'})")
    if rel <= 1e-6:
        print("  FAIL: the defect should have shown up here")
        failed += 1

    print("\n  ...and putting the missing G.cells.volumes back out recovers")
    print("  MRST's answer to machine precision, so that IS the whole defect:")
    p_bug, sw_bug = run_mech_ow(ref, "big", mrst_bug=True)
    failed += compare("pressure (mrst_oilwater_bug=True)", p_bug,
                      ref["big"]["p"], 1e-8)
    failed += compare("sW (mrst_oilwater_bug=True)", sw_bug,
                      ref["big"]["sw"], 1e-8)

    # Corroboration: on UNIT cells the missing factor is 1, so MRST's formula
    # and the correct one are the same equation and both must match MRST.
    # They do - to 4e-8, which is MRST's own convergence residual on this
    # (stiffer) problem, not a formulation difference: the two PyMRST variants
    # agree with each other to 4e-9 here, exactly as V = 1 requires.
    print("\nUnit cells (volume 1 m^3) - the missing factor is 1, so both")
    print("formulas coincide and both must match MRST (and do, to MRST's own")
    print("convergence residual - see the sixth finding in README.md):")
    p, sw = run_mech_ow(ref, "unit", mrst_bug=False)
    failed += compare("pressure", p, ref["unit"]["p"], 1e-6)
    failed += compare("sW", sw, ref["unit"]["sw"], 1e-6)
    return failed


def main() -> int:
    if not REF.exists():
        print("ref_p27_poromech.mat missing (run dump_phase27_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    p, xd, uz = run_python(ref)

    print("Coupled poroelasticity (MechWaterModel) vs MATLAB ad-mechanics:")
    failed = 0
    failed += compare("pressure history", p, ref["p_hist"], 1e-8)
    failed += compare("displacement (free dofs)", xd, ref["xd_hist"], 1e-8)
    failed += compare("vertical displacement u_z", uz, ref["uz_hist"], 1e-8)

    if REF_B.exists():
        failed += phase_27b()
    else:
        print("\n(ref_p27b_mechow.mat missing - run dump_phase27b_reference.m)")

    print("\nCoupled poroelasticity matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    print("\n(Terzaghi and Mandel are checked in pymrst/modules/ad_mechanics/"
          "{poromech,mandel}.py - Mandel is the one that discriminates.)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
