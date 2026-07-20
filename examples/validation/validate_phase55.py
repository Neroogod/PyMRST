"""Phase 55: compositional flow with gravity vs MATLAB.

Checks `OverallCompositionModel` in a closed vertical oil/gas column. MRST's
`equationsCompositional` supplies the reference with gravity on and water off.
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
REF = HERE / "ref_p55_compgravity.mat"


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


def component_mass(model: OverallCompositionModel, state) -> np.ndarray:
    fp = two_phase_flow_props(model.mix, state.pressure, model.T, state.z)
    out = np.zeros(model.mix.n)
    for i in range(model.mix.n):
        out[i] = np.sum(model.pv * (
            fp.rhoL * fp.sL * fp.xM[i] + fp.rhoV * fp.sV * fp.yM[i]))
    return out


def main() -> int:
    if not REF.exists():
        print("ref_p55_compgravity.mat missing (run dump_phase55_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nz = int(ref["nz"])
    T0 = float(ref["T0"])
    p0 = float(ref["p0"])
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])
    nstep = int(ref["nstep"])
    gravity = float(ref["g_acc"])

    p_ml = np.atleast_2d(ref["p_steps"]).astype(float)
    z_ml = np.asarray(ref["z_steps"], float)
    sO_ml = np.atleast_2d(ref["sO_steps"]).astype(float)
    sG_ml = np.atleast_2d(ref["sG_steps"]).astype(float)

    g = compute_geometry(cart_grid([1, 1, nz], [10.0, 10.0, 100.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = OverallCompositionModel(g, rock, mix, temperature=T0,
                                    gravity=gravity)
    st = model.initial_state(p0, z0)
    mass0 = component_mass(model, st)

    p_py = np.zeros((nz, nstep))
    z_py = np.zeros((nz, mix.n, nstep))
    sO_py = np.zeros((nz, nstep))
    sG_py = np.zeros((nz, nstep))
    for k in range(nstep):
        st, ok, _ = model.step(st, dt)
        assert ok, f"PyMRST did not converge at step {k + 1}"
        p_py[:, k] = st.pressure
        z_py[:, :, k] = st.z
        fp = two_phase_flow_props(mix, st.pressure, model.T, st.z)
        sO_py[:, k] = np.asarray(fp.sL, float)
        sG_py[:, k] = np.asarray(fp.sV, float)

    print(f"\n{nz} vertical cells, {nstep} steps of {dt / DAY:.0f} days:")

    failed = 0
    failed += compare("pressure (all steps)", p_py, p_ml,
                      2e-3 * BARSA, BARSA, "bar")
    failed += compare("composition z (all steps)", z_py, z_ml, 2e-5, 1.0)
    failed += compare("oil saturation sO (all steps)", sO_py, sO_ml, 2e-5, 1.0)
    failed += compare("gas saturation sG (all steps)", sG_py, sG_ml, 2e-5, 1.0)

    drift = np.abs(component_mass(model, st) - mass0).max()
    ok_mass = drift <= 1e-8 * max(float(np.abs(mass0).max()), 1.0)
    tag = "ok  " if ok_mass else "FAIL"
    print(f"  {tag} component mass conserved (max drift {drift:.2e} kg)")
    failed += 0 if ok_mass else 1

    segregated = z_py[0, 0, -1] > z_py[-1, 0, -1]
    tag = "ok  " if segregated else "FAIL"
    print(f"  {tag} methane enriches upward "
          f"({z_py[0, 0, -1]:.4f} top vs {z_py[-1, 0, -1]:.4f} bottom)")
    failed += 0 if segregated else 1

    print("\nCompositional gravity matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
