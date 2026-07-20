"""Phase 54: compositional flow with component-aware wells vs MATLAB.

Checks `OverallCompositionModel` with wells: a 1D methane-rich gas injector
and BHP producer. MRST's `getExtraWellContributions` supplies per-component
mass sources from phase surface rates and injection composition (`W.components`).

`dump_phase54_reference.m` runs MRST's `OverallCompositionCompositionalModel`
with tightened tolerances; PyMRST runs the identical case.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.flow_props import two_phase_flow_props
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p54_compwells.mat"


def build_mixture(ref) -> Mixture:
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def compare(name, py, ml, atol, scale, unit="") -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    diff = float(np.abs(py - ml).max())
    ok = diff <= atol
    tag = "ok  " if ok else "FAIL"
    print(f"  {tag} {name}: max abs diff = {diff / scale:.3e} {unit} "
          f"(atol {atol / scale:.1e} {unit})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p54_compwells.mat missing (run dump_phase54_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = float(ref["p0"])
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])
    nstep = int(ref["nstep"])
    inj_rate = float(ref["injRate"])
    prod_bhp = float(ref["prodBhp"])

    p_ml = np.atleast_2d(ref["p_steps"]).astype(float)
    z_ml = np.asarray(ref["z_steps"], float)
    sO_ml = np.atleast_2d(ref["sO_steps"]).astype(float)
    qOs_ml = np.atleast_2d(ref["qOs"]).astype(float)
    qGs_ml = np.atleast_2d(ref["qGs"]).astype(float)
    bhp_ml = np.atleast_2d(ref["bhp"]).astype(float)

    g = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    inj = add_well(None, g, rock, np.array([0]), well_type="rate",
                   val=inj_rate, compi=np.array([0.0, 1.0]), sign=1)
    inj[0].components = np.array([0.9, 0.05, 0.05])
    wells = add_well(inj, g, rock, np.array([nx - 1]), well_type="bhp",
                     val=prod_bhp, compi=np.array([0.0, 1.0]), sign=-1)
    wells[1].components = z0.copy()
    model = OverallCompositionModel(g, rock, mix, temperature=T0, wells=wells)
    st = model.initial_state(p0, z0)

    p_py = np.zeros((nx, nstep))
    z_py = np.zeros((nx, mix.n, nstep))
    sO_py = np.zeros((nx, nstep))
    qOs_py = np.zeros((nstep, 2))
    qGs_py = np.zeros((nstep, 2))
    bhp_py = np.zeros((nstep, 2))
    for k in range(nstep):
        st, ok, _ = model.step(st, dt)
        assert ok, f"PyMRST did not converge at step {k + 1}"
        p_py[:, k] = st.pressure
        z_py[:, :, k] = st.z
        fp = two_phase_flow_props(mix, st.pressure, model.T, st.z)
        sO_py[:, k] = np.asarray(fp.sL, float)
        qOs_py[k, :] = st.qOs
        qGs_py[k, :] = st.qGs
        bhp_py[k, :] = st.bhp

    print(f"\n{nx} cells, {nstep} steps of {dt / DAY:.0f} days:")

    failed = 0
    failed += compare("pressure (all steps)", p_py, p_ml,
                      5e-3 * BARSA, BARSA, "bar")
    failed += compare("composition z (all steps)", z_py, z_ml, 5e-5, 1.0)
    failed += compare("oil saturation sO (all steps)", sO_py, sO_ml, 5e-5, 1.0)
    failed += compare("qOs (all steps)", qOs_py, qOs_ml, 2e-10, 1.0, "m^3/s")
    failed += compare("qGs (all steps)", qGs_py, qGs_ml, 2e-9, 1.0, "m^3/s")
    failed += compare("bhp (all steps)", bhp_py, bhp_ml, 1e-3 * BARSA, BARSA, "bar")

    # Invariant: injector streams gas at the prescribed rate.
    inj_ok = np.allclose(qGs_py[:, 0], inj_rate, rtol=1e-6)
    tag = "ok  " if inj_ok else "FAIL"
    print(f"  {tag} injector gas rate matches control ({inj_rate:.2e} m^3/s)")
    failed += 0 if inj_ok else 1

    # Invariant: methane enriches at the injector over the run.
    enrich = z_py[0, 0, -1] > z_py[0, 0, 0]
    tag = "ok  " if enrich else "FAIL"
    print(f"  {tag} injector methane mole fraction increases "
          f"({z_py[0, 0, 0]:.4f} -> {z_py[0, 0, -1]:.4f})")
    failed += 0 if enrich else 1

    print("\nCompositional wells match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
