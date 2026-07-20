"""Phase 56: compositional flow with water vs MATLAB.

Checks the first `model.water = true` slice of `OverallCompositionModel`: no
wells, no gravity, no capillary pressure.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.flow_props import two_phase_flow_props
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p56_compwater.mat"


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


def masses(model: OverallCompositionModel, state):
    assert state.sW is not None
    fp = two_phase_flow_props(model.mix, state.pressure, model.T, state.z)
    sw = state.sW
    so = (1.0 - sw) * fp.sL
    sg = (1.0 - sw) * fp.sV
    comp = np.zeros(model.mix.n)
    for i in range(model.mix.n):
        comp[i] = np.sum(model.pv * (
            fp.rhoL * so * fp.xM[i] + fp.rhoV * sg * fp.yM[i]))
    water = np.sum(model.pv * sw)
    return water, comp


def main() -> int:
    if not REF.exists():
        print("ref_p56_compwater.mat missing (run dump_phase56_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = np.atleast_1d(ref["p0"]).astype(float)
    sw0 = np.atleast_1d(ref["sW0"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])
    nstep = int(ref["nstep"])

    p_ml = np.atleast_2d(ref["p_steps"]).astype(float)
    sw_ml = np.atleast_2d(ref["sw_steps"]).astype(float)
    z_ml = np.asarray(ref["z_steps"], float)
    sO_ml = np.atleast_2d(ref["sO_steps"]).astype(float)
    sG_ml = np.atleast_2d(ref["sG_steps"]).astype(float)

    g = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = OverallCompositionModel(g, rock, mix, temperature=T0,
                                    water=True, n_kr=(2.0, 2.0, 2.0))
    st = model.initial_state(p0, z0, sW=sw0)
    water0, comp0 = masses(model, st)

    p_py = np.zeros((nx, nstep))
    sw_py = np.zeros((nx, nstep))
    z_py = np.zeros((nx, mix.n, nstep))
    sO_py = np.zeros((nx, nstep))
    sG_py = np.zeros((nx, nstep))
    for k in range(nstep):
        st, ok, _ = model.step(st, dt)
        assert ok, f"PyMRST did not converge at step {k + 1}"
        assert st.sW is not None
        p_py[:, k] = st.pressure
        sw_py[:, k] = st.sW
        z_py[:, :, k] = st.z
        fp = two_phase_flow_props(mix, st.pressure, model.T, st.z)
        sO_py[:, k] = (1.0 - st.sW) * np.asarray(fp.sL, float)
        sG_py[:, k] = (1.0 - st.sW) * np.asarray(fp.sV, float)

    print(f"\n{nx} cells, {nstep} steps of {dt / DAY:.0f} days:")

    failed = 0
    failed += compare("pressure (all steps)", p_py, p_ml,
                      5e-3 * BARSA, BARSA, "bar")
    failed += compare("water saturation sW (all steps)", sw_py, sw_ml,
                      5e-5, 1.0)
    failed += compare("composition z (all steps)", z_py, z_ml, 5e-5, 1.0)
    failed += compare("oil saturation sO (all steps)", sO_py, sO_ml, 5e-5, 1.0)
    failed += compare("gas saturation sG (all steps)", sG_py, sG_ml, 5e-5, 1.0)

    water1, comp1 = masses(model, st)
    water_drift = abs(water1 - water0)
    comp_drift = np.abs(comp1 - comp0).max()
    ok_mass = water_drift <= 1e-8 * max(abs(water0), 1.0)
    ok_mass = ok_mass and comp_drift <= 1e-8 * max(float(np.abs(comp0).max()), 1.0)
    tag = "ok  " if ok_mass else "FAIL"
    print(f"  {tag} water and component masses conserved "
          f"(water {water_drift:.2e} m3, components {comp_drift:.2e} kg)")
    failed += 0 if ok_mass else 1

    print("\nCompositional water flow matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
