"""Phase 59: compositional `opt.pressure` equation vs MATLAB.

This checks the MRST pressure-reduction assembly slice: component conservation
equations are collapsed into one pressure equation with numerical
partial-volume weights from the accumulation Jacobian. It does not yet run a
full pressure-only sequential Newton step.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.utils.adi import combine_equations, init_variables_adi
from pymrst.utils.units import DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p59_comppressure.mat"


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
        print("ref_p59_comppressure.mat missing (run dump_phase59_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = np.atleast_1d(ref["p0"]).astype(float)
    p1 = np.atleast_1d(ref["p1"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    z1 = np.asarray(ref["z1"], float)
    dt = float(ref["dt"])

    g = compute_geometry(cart_grid([nx, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = OverallCompositionModel(g, rock, mix, temperature=T0)
    st0 = model.initial_state(p0, z0)
    st = model.initial_state(p1, z1)

    prim = init_variables_adi(
        st.pressure,
        *[st.z[:, i] for i in range(mix.n - 1)],
        np.zeros(0), np.zeros(0), np.zeros(0),
    )
    eqs, _, _, weights = model.assemble(
        st0, prim[0], list(prim[1:mix.n]),
        prim[mix.n], prim[mix.n + 1], prim[mix.n + 2], dt,
        pressure_reduction=True, return_pressure_weights=True,
    )
    peq = combine_equations(*eqs)
    jac_p = peq.jac[0][:nx, :nx].toarray()
    weights_py = np.column_stack(weights)

    peq_ml = np.atleast_1d(ref["peq"]).astype(float)
    jac_ml = np.asarray(ref["jacP"], float)
    weights_ml = np.asarray(ref["weights"], float)

    print(f"\n{nx} cells, pressure-equation assembly at {dt / DAY:.0f} days:")
    failed = 0
    failed += compare("partial-volume weights", weights_py, weights_ml,
                      5e-2)
    failed += compare("pressure residual", peq.val, peq_ml,
                      1e-1)
    failed += compare("pressure Jacobian block", jac_p, jac_ml,
                      1e-6)

    print("\nCompositional pressure equation matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
