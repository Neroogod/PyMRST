"""Phase 60: compositional pressure-only Newton step vs MATLAB.

Checks `OverallCompositionModel.pressure_step`, the sequential pressure solve
that uses the Phase 59 reduced `opt.pressure` equation while freezing transport
variables.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p60_comppressurestep.mat"


def build_mixture(ref) -> Mixture:
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def compare(name, py, ml, atol, scale=1.0, unit="") -> int:
    py = np.asarray(py, float)
    ml = np.asarray(ml, float)
    diff = float(np.abs(py - ml).max())
    ok = diff <= atol
    tag = "ok  " if ok else "FAIL"
    print(f"  {tag} {name}: max abs diff = {diff / scale:.3e} {unit} "
          f"(atol {atol / scale:.1e} {unit})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p60_comppressurestep.mat missing (run dump_phase60_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = np.atleast_1d(ref["p0"]).astype(float)
    p_guess = np.atleast_1d(ref["p_guess"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])

    g = compute_geometry(cart_grid([nx, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = OverallCompositionModel(g, rock, mix, temperature=T0)
    st = model.initial_state(p0, z0)
    st.pressure = p_guess.copy()

    stp, ok, its = model.pressure_step(st, dt)
    assert ok, "PyMRST pressure_step did not converge"

    p_ml = np.atleast_1d(ref["p_pressure"]).astype(float)
    z_ml = np.asarray(ref["z_pressure"], float)

    print(f"\n{nx} cells, pressure-only step at {dt / DAY:.0f} days:")
    failed = 0
    failed += compare("pressure", stp.pressure, p_ml,
                      5e-1 * BARSA, BARSA, "bar")
    failed += compare("frozen composition", stp.z, z_ml, 1e-12)
    print(f"  ok   PyMRST pressure solve converged in {its} iterations "
          f"(MRST {int(ref['its'])})")

    print("\nCompositional pressure-only step matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
