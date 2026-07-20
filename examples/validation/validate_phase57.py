"""Phase 57: compositional water wells vs MATLAB.

Checks `OverallCompositionModel(water=True)` with a water injector and a BHP
producer. Gravity and capillary pressure are off.
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
REF = HERE / "ref_p57_compwaterwells.mat"


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
        print("ref_p57_compwaterwells.mat missing (run dump_phase57_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = float(ref["p0"])
    sw0 = float(ref["sW0"])
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])
    nstep = int(ref["nstep"])
    inj_rate = float(ref["injRate"])
    prod_bhp = float(ref["prodBhp"])

    p_ml = np.atleast_2d(ref["p_steps"]).astype(float)
    sw_ml = np.atleast_2d(ref["sw_steps"]).astype(float)
    z_ml = np.asarray(ref["z_steps"], float)
    sO_ml = np.atleast_2d(ref["sO_steps"]).astype(float)
    sG_ml = np.atleast_2d(ref["sG_steps"]).astype(float)
    qWs_ml = np.atleast_2d(ref["qWs"]).astype(float)
    qOs_ml = np.atleast_2d(ref["qOs"]).astype(float)
    qGs_ml = np.atleast_2d(ref["qGs"]).astype(float)
    bhp_ml = np.atleast_2d(ref["bhp"]).astype(float)

    g = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    inj = add_well(None, g, rock, np.array([0]), well_type="rate",
                   val=inj_rate, compi=np.array([1.0, 0.0, 0.0]), sign=1)
    inj[0].components = z0.copy()
    wells = add_well(inj, g, rock, np.array([nx - 1]), well_type="bhp",
                     val=prod_bhp, compi=np.array([0.2, 0.4, 0.4]), sign=-1)
    wells[1].components = z0.copy()
    model = OverallCompositionModel(g, rock, mix, temperature=T0,
                                    wells=wells, water=True,
                                    n_kr=(2.0, 2.0, 2.0))
    st = model.initial_state(p0, z0, sW=sw0)

    p_py = np.zeros((nx, nstep))
    sw_py = np.zeros((nx, nstep))
    z_py = np.zeros((nx, mix.n, nstep))
    sO_py = np.zeros((nx, nstep))
    sG_py = np.zeros((nx, nstep))
    qWs_py = np.zeros((nstep, 2))
    qOs_py = np.zeros((nstep, 2))
    qGs_py = np.zeros((nstep, 2))
    bhp_py = np.zeros((nstep, 2))
    for k in range(nstep):
        st, ok, _ = model.step(st, dt)
        assert ok, f"PyMRST did not converge at step {k + 1}"
        assert st.sW is not None and st.qWs is not None
        p_py[:, k] = st.pressure
        sw_py[:, k] = st.sW
        z_py[:, :, k] = st.z
        fp = two_phase_flow_props(mix, st.pressure, model.T, st.z)
        sO_py[:, k] = (1.0 - st.sW) * np.asarray(fp.sL, float)
        sG_py[:, k] = (1.0 - st.sW) * np.asarray(fp.sV, float)
        qWs_py[k, :] = st.qWs
        qOs_py[k, :] = st.qOs
        qGs_py[k, :] = st.qGs
        bhp_py[k, :] = st.bhp

    print(f"\n{nx} cells, {nstep} steps of {dt / DAY:.0f} days:")

    failed = 0
    failed += compare("pressure (all steps)", p_py, p_ml,
                      5e-3 * BARSA, BARSA, "bar")
    failed += compare("water saturation sW (all steps)", sw_py, sw_ml,
                      5e-5, 1.0)
    failed += compare("composition z (all steps)", z_py, z_ml, 5e-5, 1.0)
    failed += compare("oil saturation sO (all steps)", sO_py, sO_ml, 5e-5, 1.0)
    failed += compare("gas saturation sG (all steps)", sG_py, sG_ml, 5e-5, 1.0)
    failed += compare("qWs (all steps)", qWs_py, qWs_ml, 2e-9, 1.0, "m^3/s")
    failed += compare("qOs (all steps)", qOs_py, qOs_ml, 2e-9, 1.0, "m^3/s")
    failed += compare("qGs (all steps)", qGs_py, qGs_ml, 2e-9, 1.0, "m^3/s")
    failed += compare("bhp (all steps)", bhp_py, bhp_ml,
                      1e-3 * BARSA, BARSA, "bar")

    inj_ok = np.allclose(qWs_py[:, 0], inj_rate, rtol=1e-6)
    tag = "ok  " if inj_ok else "FAIL"
    print(f"  {tag} injector water rate matches control ({inj_rate:.2e} m^3/s)")
    failed += 0 if inj_ok else 1

    sw_up = sw_py[0, -1] > sw_py[0, 0]
    tag = "ok  " if sw_up else "FAIL"
    print(f"  {tag} injector water saturation increases "
          f"({sw_py[0, 0]:.4f} -> {sw_py[0, -1]:.4f})")
    failed += 0 if sw_up else 1

    print("\nCompositional water wells match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
