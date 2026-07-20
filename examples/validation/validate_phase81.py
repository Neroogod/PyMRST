"""Phase 81: water/oil/gas natural variables vs MATLAB MRST."""

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
REF = HERE / "ref_p81_natvars_water.mat"


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
    msg = f"atol {atol / scale:.1e} {unit}"
    if rtol:
        rel = float((abs_diff / np.maximum(np.maximum(np.abs(ml), np.abs(py)), 1e-300)).max())
        msg += f", rtol {rtol:.1e}; max rel {rel:.3e}"
    print(f"  {tag} {name}: max abs diff = {diff / scale:.3e} {unit} ({msg})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p81_natvars_water.mat missing (run dump_phase81_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = float(ref["p0"])
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    sw0 = float(ref["sw0"])
    dt = float(ref["dt"])

    g = compute_geometry(cart_grid([nx, 1, 1], [40.0, 10.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1.0e-5, compi=np.array([1.0, 0.0, 0.0]), sign=1)
    wells[0].components = z0.copy()
    wells = add_well(wells, g, rock, np.array([nx - 1]), well_type="bhp",
                     val=45.0 * BARSA, compi=np.array([0.0, 0.5, 0.5]), sign=-1)
    wells[1].components = z0.copy()

    model = NaturalVariablesCompositionalModel(g, rock, mix, temperature=T0, wells=wells, water=True)
    st = model.initial_state(p0, z0, sW=sw0)
    st, ok, iterations = model.step(st, dt)
    assert ok, "PyMRST water natural-variable step did not converge"

    print(f"\n{nx} cells, water natural-variable step of {dt / DAY:.2f} days:")
    failed = 0
    failed += compare("pressure", st.pressure, ref["p"], 5e-3 * BARSA, BARSA, "bar")
    failed += compare("overall composition z", st.z, ref["z"], 5e-5)
    failed += compare("liquid composition x", st.x, ref["x"], 5e-5)
    failed += compare("vapour composition y", st.y, ref["y"], 5e-5)
    failed += compare("water saturation sW", st.sW, ref["sW"], 5e-5)
    failed += compare("oil saturation sO", st.sO, ref["sO"], 5e-5)
    failed += compare("gas saturation sG", st.sG, ref["sG"], 5e-5)
    failed += compare("qWs", st.qWs, ref["qWs"], 1e-10, 1.0, "m^3/s", rtol=5e-5)
    failed += compare("qOs", st.qOs, ref["qOs"], 1e-10, 1.0, "m^3/s", rtol=5e-5)
    failed += compare("qGs", st.qGs, ref["qGs"], 1e-10, 1.0, "m^3/s", rtol=5e-5)
    failed += compare("bhp", st.bhp, ref["bhp"], 1e-3 * BARSA, BARSA, "bar")

    ref_iterations = int(ref["iterations"])
    if ref_iterations >= 0:
        same_iters = abs(iterations - ref_iterations) <= 4
        tag = "ok  " if same_iters else "FAIL"
        print(f"  {tag} Newton iterations comparable "
              f"(PyMRST {iterations}, MRST {ref_iterations})")
        failed += 0 if same_iters else 1
    else:
        print(f"  ok   Newton iterations diagnostic skipped "
              f"(PyMRST {iterations}, MRST report did not expose a count)")

    print("\nWater natural-variable compositional model matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
