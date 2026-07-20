"""Phase 80: natural-variable compositional model vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p80_natvars.mat"


def build_mixture(ref) -> Mixture:
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def compare(name, py, ml, atol, scale=1.0, unit="", rtol=0.0) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    abs_diff = np.abs(py - ml)
    diff = float(abs_diff.max())
    limit = atol + rtol * np.maximum(np.abs(ml), np.abs(py))
    ok = bool(np.all(abs_diff <= limit))
    tag = "ok  " if ok else "FAIL"
    tol_msg = f"atol {atol / scale:.1e} {unit}"
    if rtol:
        rel = float((abs_diff / np.maximum(np.maximum(np.abs(ml), np.abs(py)), 1e-300)).max())
        tol_msg += f", rtol {rtol:.1e}; max rel {rel:.3e}"
    print(f"  {tag} {name}: max abs diff = {diff / scale:.3e} {unit} ({tol_msg})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p80_natvars.mat missing (run dump_phase80_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = float(ref["p0"])
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])
    inj_rate = float(ref["injRate"])
    prod_bhp = float(ref["prodBhp"])

    g = compute_geometry(cart_grid([nx, 1, 1], [40.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=inj_rate, compi=np.array([0.0, 1.0]), sign=1)
    wells[0].components = np.array([0.9, 0.05, 0.05])
    wells = add_well(wells, g, rock, np.array([nx - 1]), well_type="bhp",
                     val=prod_bhp, compi=np.array([0.0, 1.0]), sign=-1)
    wells[1].components = z0.copy()
    model = NaturalVariablesCompositionalModel(g, rock, mix, temperature=T0, wells=wells)
    st = model.initial_state(p0, z0)
    st, ok, iterations = model.step(st, dt)
    assert ok, "PyMRST natural-variable step did not converge"

    print(f"\n{nx} cells, natural-variable step of {dt / DAY:.2f} days:")
    failed = 0
    failed += compare("pressure", st.pressure, ref["p"], 2e-3 * BARSA, BARSA, "bar")
    failed += compare("overall composition z", st.z, ref["z"], 5e-5)
    failed += compare("liquid composition x", st.x, ref["x"], 5e-5)
    failed += compare("vapour composition y", st.y, ref["y"], 5e-5)
    failed += compare("oil saturation sO", st.sO, ref["sO"], 5e-5)
    failed += compare("gas saturation sG", st.sG, ref["sG"], 5e-5)
    failed += compare("qOs", st.qOs, ref["qOs"], 1e-10, 1.0, "m^3/s", rtol=5e-5)
    failed += compare("qGs", st.qGs, ref["qGs"], 1e-10, 1.0, "m^3/s", rtol=5e-5)
    failed += compare("bhp", st.bhp, ref["bhp"], 1e-3 * BARSA, BARSA, "bar")

    same_iters = abs(iterations - int(ref["iterations"])) <= 3
    tag = "ok  " if same_iters else "FAIL"
    print(f"  {tag} Newton iterations comparable "
          f"(PyMRST {iterations}, MRST {int(ref['iterations'])})")
    failed += 0 if same_iters else 1

    print("\nNatural-variable compositional model matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
