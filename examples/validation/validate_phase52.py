"""Phase 52: analytic flash derivatives, against MATLAB compositional.

The flash (Phase 13) solves for the phase split on values; a fully-implicit
compositional flow model, whose primaries are pressure and overall composition
`z`, needs it *differentiated*: `dL/dp, dL/dz, dx/dp, dx/dz, dy/dp, dy/dz`. The
flash is an implicit function of `(p, T, z)`, so those come from the implicit
function theorem - `ds/du = -(dF/ds)^{-1}(dF/du)` - which is what MRST's
`getPhaseFractionDerivativesPTZ` computes and `eos_ad.py` ports.

Per the project's "fix the inputs, compare the algorithm" contract,
`dump_phase52_reference.m` dumps MRST's flash (`x, y, L, Z_L, Z_V`) together
with the derivatives it took at that flash. The Python side replays its own
implicit-function-theorem solve on **MRST's exact flash** and compares the
sensitivities column by column (`[p, T, z_1..z_n]`), so a disagreement is the
derivative machinery and nothing else. The derivatives are additionally checked
against finite differences of the flash in `eos_ad.py`'s own self-tests, an
independent confirmation that needs no reference at all.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.eos import Mixture, simple_mixture
from pymrst.modules.compositional.eos_ad import phase_fraction_derivatives

HERE = Path(__file__).parent
REF = HERE / "ref_p52_flashderiv.mat"


def build_mixture(ref) -> Mixture:
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def compare(name, py, ml, rtol) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    good = np.isfinite(ml) & np.isfinite(py)
    scale = max(np.abs(ml[good]).max(), 1e-300)
    diff = float(np.abs(py[good] - ml[good]).max())
    ok = diff <= rtol * scale
    tag = "ok  " if ok else "FAIL"
    print(f"  {tag} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p52_flashderiv.mat missing (run dump_phase52_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    ship = simple_mixture()
    for attr in ("Tcrit", "Pcrit", "Vcrit", "acentric", "molarMass"):
        assert np.allclose(getattr(ship, attr), getattr(mix, attr)), attr

    two = np.asarray(ref["twoPhase"], bool)
    p = np.atleast_1d(ref["p_all"]).astype(float)[two]
    T = np.atleast_1d(ref["T_all"]).astype(float)[two]
    z = np.atleast_2d(ref["z_all"]).astype(float)[two]
    x = np.atleast_2d(ref["x"]).astype(float)[two]
    y = np.atleast_2d(ref["y"]).astype(float)[two]
    L = np.atleast_1d(ref["L"]).astype(float)[two]
    z_l = np.atleast_1d(ref["Z_L"]).astype(float)[two]
    z_v = np.atleast_1d(ref["Z_V"]).astype(float)[two]
    n = mix.n

    print(f"\n{int(two.sum())} two-phase cells, nprimary = {int(ref['nprimary'])}:")

    dL, dx, dy = phase_fraction_derivatives(mix, p, T, z, x, y, L, z_l, z_v)

    dL_ml = np.atleast_2d(ref["dL"]).astype(float)
    dx_ml = np.asarray(ref["dx"], float)          # (ncomp, N2, nprimary)
    dy_ml = np.asarray(ref["dy"], float)

    failed = 0
    print("\nLiquid mole fraction L:")
    failed += compare("dL/d[p,T,z]", dL, dL_ml, 1e-9)

    print("\nLiquid compositions x_i:")
    for i in range(n):
        failed += compare(f"dx_{i+1}/d[p,T,z]", dx[i], dx_ml[i], 1e-9)
    print("\nVapour compositions y_i:")
    for i in range(n):
        failed += compare(f"dy_{i+1}/d[p,T,z]", dy[i], dy_ml[i], 1e-9)

    print("\nFlash derivatives match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
