"""Phase 58: compositional capillary pressure vs MATLAB.

Checks `OverallCompositionModel(water=True)` with pcOW(sW) and no wells or
gravity.
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
REF = HERE / "ref_p58_compcapillary.mat"


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
        print("ref_p58_compcapillary.mat missing (run dump_phase58_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = float(ref["p0"])
    sw0 = np.atleast_1d(ref["sW0"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    pc0 = float(ref["pc0"])
    dt = float(ref["dt"])
    nstep = int(ref["nstep"])

    p_ml = np.atleast_2d(ref["p_steps"]).astype(float)
    sw_ml = np.atleast_2d(ref["sw_steps"]).astype(float)
    z_ml = np.asarray(ref["z_steps"], float)
    sO_ml = np.atleast_2d(ref["sO_steps"]).astype(float)
    sG_ml = np.atleast_2d(ref["sG_steps"]).astype(float)

    g = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = OverallCompositionModel(
        g, rock, mix, temperature=T0, water=True,
        pc_ow=lambda sw: pc0 * (1.0 - sw),
        n_kr=(2.0, 2.0, 2.0),
    )
    st = model.initial_state(p0, z0, sW=sw0)

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

    moved = np.linalg.norm(sw_py[:, -1] - sw0) > 1e-8
    tag = "ok  " if moved else "FAIL"
    print(f"  {tag} capillary pressure moves water "
          f"(||delta sW||={np.linalg.norm(sw_py[:, -1] - sw0):.2e})")
    failed += 0 if moved else 1

    print("\nCompositional capillary pressure matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
