"""Phase 50: EOS phase density and viscosity, against MATLAB compositional.

`eos.py` (Phase 13) answers what an equation of state is *for* - given a
pressure, a temperature and an overall composition, which phases exist and what
is in them - and returns compositions and compressibility factors. A **flow**
model cannot use any of that: Darcy's law wants a density and a viscosity, and
neither is in a flash result. `properties.py` adds that missing half - the
density `rho = M(x)/(ZRT/p)` and the Lohrenz-Bray-Clark viscosity - and this is
its MATLAB check.

`dump_phase50_reference.m` flashes a three-component mixture (methane / n-butane
/ n-decane) over a grid of pressures and temperatures spanning single-phase
liquid, single-phase vapour and the two-phase region, then asks MRST's
`CompositionalPropertyModel` for both phases' density and viscosity - with and
without a volume shift.

The comparison follows the project's "fix the inputs, compare the algorithm"
contract: it feeds **MRST's own** flash outputs (`x`, `y`, `Z_L`, `Z_V`) into
the PyMRST property functions, so a disagreement is the density/viscosity
formula and not the flash that fed it (the flash is validated separately in
Phase 13). The compressibility factors are dumped alongside for exactly that
reason. The volume-shift variants are checked too, because a shift is only
legitimate if it moves the density and leaves the equilibrium alone - a port
that applied it in the wrong place would move both, and only the coupled check
would notice.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.eos import Mixture, simple_mixture
from pymrst.modules.compositional.properties import (
    mass_density,
    viscosity_lbc,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p50_eosprops.mat"


def build_mixture(ref) -> Mixture:
    """The mixture from the dumped parameters (identical to `simple_mixture`)."""
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def compare(name, py, ml, rtol) -> int:
    """Max relative difference where MRST's value is finite (single-phase cells
    can carry a NaN for the absent phase's root)."""
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    good = np.isfinite(ml) & np.isfinite(py)
    scale = max(np.abs(ml[good]).max(), 1e-300)
    diff = float(np.abs(py[good] - ml[good]).max())
    ok = diff <= rtol * scale
    skipped = int((~good).sum())
    tag = "ok  " if ok else "FAIL"
    extra = f", {skipped} non-finite skipped" if skipped else ""
    print(f"  {tag} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e}{extra})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p50_eosprops.mat missing (run dump_phase50_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    # The shipped `simple_mixture()` must be the mixture the dump used, or every
    # number below would be comparing two different fluids.
    ship = simple_mixture()
    for attr in ("Tcrit", "Pcrit", "Vcrit", "acentric", "molarMass"):
        assert np.allclose(getattr(ship, attr), getattr(mix, attr)), attr

    p = np.atleast_1d(ref["p_all"]).astype(float)
    T = np.atleast_1d(ref["T_all"]).astype(float)
    x = np.atleast_2d(ref["x"]).astype(float)
    y = np.atleast_2d(ref["y"]).astype(float)
    z_l = np.atleast_1d(ref["Z_L"]).astype(float)
    z_v = np.atleast_1d(ref["Z_V"]).astype(float)
    shift = np.atleast_1d(ref["shift"]).astype(float)
    n2 = int(np.asarray(ref["twoPhase"]).sum())
    print(f"\n{p.size} conditions, {n2} of them two-phase:")

    failed = 0

    print("\nNo volume shift:")
    failed += compare("liquid density", mass_density(mix, p, T, x, z_l),
                      ref["rho_L"], 1e-9)
    failed += compare("vapour density", mass_density(mix, p, T, y, z_v),
                      ref["rho_V"], 1e-9)
    failed += compare("liquid viscosity", viscosity_lbc(mix, p, T, x, z_l),
                      ref["mu_L"], 1e-8)
    failed += compare("vapour viscosity", viscosity_lbc(mix, p, T, y, z_v),
                      ref["mu_V"], 1e-8)

    print("\nWith volume shift (moves the density, not the flash):")
    failed += compare("liquid density", mass_density(mix, p, T, x, z_l, shift),
                      ref["rho_L_shift"], 1e-9)
    failed += compare("vapour density", mass_density(mix, p, T, y, z_v, shift),
                      ref["rho_V_shift"], 1e-9)
    failed += compare("liquid viscosity",
                      viscosity_lbc(mix, p, T, x, z_l, shift),
                      ref["mu_L_shift"], 1e-8)
    failed += compare("vapour viscosity",
                      viscosity_lbc(mix, p, T, y, z_v, shift),
                      ref["mu_V_shift"], 1e-8)

    # The point of the phase: the two phases must be physically distinct, or the
    # densities could match MRST and still be describing one fluid twice.
    tp = np.asarray(ref["twoPhase"], bool)
    if tp.any():
        rho_l = np.asarray(mass_density(mix, p, T, x, z_l), float)[tp]
        rho_v = np.asarray(mass_density(mix, p, T, y, z_v), float)[tp]
        heavier = bool(np.all(rho_l > rho_v))
        print(f"\n  {'ok  ' if heavier else 'FAIL'} the liquid is heavier than "
              f"the vapour in every two-phase cell "
              f"(rho_L in [{rho_l.min():.0f}, {rho_l.max():.0f}], "
              f"rho_V in [{rho_v.min():.0f}, {rho_v.max():.0f}] kg/m3)")
        failed += 0 if heavier else 1

    print("\nEOS density and viscosity match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
