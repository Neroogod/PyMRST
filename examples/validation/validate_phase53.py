"""Phase 53: overall-composition flow, against MATLAB compositional.

This is the end-to-end check of the compositional *flow* model
(`equationsCompositional`): accumulation + flux, solved fully implicitly, with
every underlying piece already validated on its own (flash and its derivatives,
densities, viscosities, mass fractions, saturations - Phases 13/50/51/52).

`dump_phase53_reference.m` runs MRST's `OverallCompositionCompositionalModel` on
a closed 1D column started off a pressure gradient - no wells, no-flow
boundaries - so the fluid redistributes with both accumulation and flux active
and no source to hide an error. PyMRST runs the identical case and the
per-step pressure, composition and saturation trajectories are compared.

Because MRST's default nonlinear tolerance (1e-3) stops before its own
equations are solved, the reference is generated with tightened tolerances;
PyMRST converges tightly too, so the comparison is between two *converged*
answers rather than one converged and one loosely stopped.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture, simple_mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.utils.units import DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p53_compflow.mat"


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
        print("ref_p53_compflow.mat missing (run dump_phase53_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    ship = simple_mixture()
    for attr in ("Tcrit", "Pcrit", "Vcrit", "acentric", "molarMass"):
        assert np.allclose(getattr(ship, attr), getattr(mix, attr)), attr

    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = np.atleast_1d(ref["p0"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])
    nstep = int(ref["nstep"])

    p_ml = np.atleast_2d(ref["p_steps"]).astype(float)     # (nx, nstep)
    z_ml = np.asarray(ref["z_steps"], float)               # (nx, ncomp, nstep)
    sO_ml = np.atleast_2d(ref["sO_steps"]).astype(float)

    # PyMRST: same grid, rock, mixture, drive.
    g = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = OverallCompositionModel(g, rock, mix, temperature=T0)
    st = model.initial_state(p0, z0)

    p_py = np.zeros((nx, nstep))
    z_py = np.zeros((nx, mix.n, nstep))
    sO_py = np.zeros((nx, nstep))
    from pymrst.modules.compositional.flow_props import two_phase_flow_props
    for k in range(nstep):
        st, ok, its = model.step(st, dt)
        assert ok, f"PyMRST did not converge at step {k+1}"
        p_py[:, k] = st.pressure
        z_py[:, :, k] = st.z
        fp = two_phase_flow_props(mix, st.pressure, model.T, st.z)
        sO_py[:, k] = np.asarray(fp.sL, float)

    from pymrst.utils.units import BARSA
    print(f"\n{nx} cells, {nstep} steps of {dt / DAY:.0f} days:")

    failed = 0
    # The simulators equilibrate to the same uniform state; the remaining gap is a
    # small systematic offset (~3.7e-4 bar in pressure, ~8e-7 in z) from step 1,
    # not accumulating drift. Mass is conserved to solver tol. Saturation is
    # checked both against MRST's stored state.s and, independently, by
    # recomputing from MRST's own p and z (which matches to ~1e-11).
    failed += compare("pressure (all steps)", p_py, p_ml,
                      5e-4 * BARSA, BARSA, "bar")
    failed += compare("composition z (all steps)", z_py, z_ml, 1e-6, 1.0)
    failed += compare("oil saturation sO (all steps)", sO_py, sO_ml, 5e-6, 1.0)

    sO_from_ml = np.zeros_like(sO_ml)
    for k in range(nstep):
        fpk = two_phase_flow_props(mix, p_ml[:, k], model.T, z_ml[:, :, k])
        sO_from_ml[:, k] = np.asarray(fpk.sL, float)
    recomp = float(np.abs(sO_from_ml - sO_ml).max())
    tag = "ok  " if recomp < 1e-9 else "FAIL"
    print(f"  {tag} sO recomputed from MRST p,z matches MRST sO "
          f"(max diff {recomp:.2e})")
    failed += 0 if recomp < 1e-9 else 1

    # Independent invariant: with no wells, component mass is conserved.
    def total_mass(pressure, z):
        fp = two_phase_flow_props(mix, pressure, model.T, z)
        return np.array([
            np.sum(model.pv * (fp.rhoL * fp.sL * fp.xM[i]
                               + fp.rhoV * fp.sV * fp.yM[i]))
            for i in range(mix.n)])
    m0 = total_mass(p0, model.initial_state(p0, z0).z)
    m1 = total_mass(p_py[:, -1], z_py[:, :, -1])
    drift = float(np.abs(m1 - m0).max() / m0.max())
    tag = "ok  " if drift < 1e-6 else "FAIL"
    print(f"  {tag} component mass conserved (max drift {drift:.2e})")
    failed += 0 if drift < 1e-6 else 1

    print("\nCompositional flow matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
