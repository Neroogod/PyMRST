"""Phase 83: natural-variable compositional boundary-condition schedule vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import make_rock, pore_volume
from pymrst.params.wells_and_bc import fluxside, pside
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI, YEAR

HERE = Path(__file__).parent
REF = HERE / "ref_p83_natural_bc.mat"


def build_mixture(ref) -> Mixture:
    return Mixture(
        names=("Methane", "CarbonDioxide", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def compare(name, py, ml, atol, scale=1.0, unit="") -> int:
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
        print("ref_p83_natural_bc.mat missing (run dump_phase83_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    mix = build_mixture(ref)
    nx = int(ref["nx"])
    ny = int(ref["ny"])
    T0 = float(ref["T0"])
    p0 = float(ref["p0"])
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = np.atleast_1d(ref["dt"]).astype(float)

    g = compute_geometry(cart_grid([nx, ny, 1], [1000.0, 1000.0, 10.0]))
    rock = make_rock(g, 100.0 * MILLI * DARCY, 0.2)
    pv = pore_volume(g, rock)
    bc = fluxside(None, g, "xmin", pv.sum() / (20.0 * YEAR), sat=np.array([1.0, 0.0]))
    bc = pside(bc, g, "xmax", 75.0 * BARSA, sat=np.array([1.0, 0.0]))
    bc.components = np.tile(np.array([0.1, 0.9, 0.0]), (bc.face.size, 1))

    model = NaturalVariablesCompositionalModel(g, rock, mix, temperature=T0, bc=bc, rho_s=(500.0, 500.0))
    st = model.initial_state(p0, z0)
    states = []
    iterations = []
    for step in dt:
        st, ok, it = model.step(st, float(step), bc=bc)
        assert ok, "PyMRST natural-variable BC step did not converge"
        states.append(st.copy())
        iterations.append(it)

    p_py = np.column_stack([s.pressure for s in states])
    z_py = np.stack([s.z for s in states], axis=2)
    x_py = np.stack([s.x for s in states], axis=2)
    y_py = np.stack([s.y for s in states], axis=2)
    s_o_py = np.column_stack([s.sO for s in states])
    s_g_py = np.column_stack([s.sG for s in states])
    flag_py = np.column_stack([s.flag for s in states])

    print(f"\n{nx * ny} cells, {dt.size} natural-variable BC steps ({dt.sum() / DAY:.2f} days):")
    failed = 0
    failed += compare("bc face values", bc.value, ref["bcValues"], 1e-12)
    failed += compare("bc components", bc.components, ref["bcComponents"], 1e-12)
    failed += compare("pressure", p_py, ref["p"], 5e-2 * BARSA, BARSA, "bar")
    failed += compare("overall composition z", z_py, ref["z"], 5e-5)
    failed += compare("liquid composition x", x_py, ref["x"], 5e-5)
    failed += compare("vapour composition y", y_py, ref["y"], 5e-5)
    failed += compare("oil saturation sO", s_o_py, ref["sO"], 5e-5)
    failed += compare("gas saturation sG", s_g_py, ref["sG"], 5e-5)
    failed += compare("phase flag", flag_py, ref["flag"], 0.0)
    print(f"  ok   Newton iterations diagnostic "
          f"(PyMRST {iterations}, MRST relaxed solver {np.asarray(ref['iterations'], int).tolist()})")

    print("\nNatural-variable boundary-condition schedule matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
