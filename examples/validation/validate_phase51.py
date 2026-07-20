"""Phase 51: flash -> flow properties, against MATLAB compositional.

Phase 13 ported the flash and Phase 50 the phase density and viscosity. What sat
between those and a compositional flow model is `computeTwoPhaseFlowProps` - the
step that turns a flash result into the exact quantities a *per-component*
conservation equation reads:

- the **mass fractions** `xM`, `yM` (`getMassFraction`): the flash speaks in
  moles, the equation is written in mass;
- the **phase saturations** `sL`, `sV` (`computeSaturations`): the flash gives a
  *molar* liquid fraction `L`, and this turns it into a *volume* fraction - with
  no volume shift, `sL = L Z_L / (L Z_L + (1-L) Z_V)`.

`flow_props.py` ports both. This is its MATLAB check.

Following the project's "fix the inputs, compare the algorithm" contract,
`dump_phase51_reference.m` dumps MRST's flash (`x, y, L, Z_L, Z_V`) alongside
the derived `xM, yM, sL, sV`, and the Python side replays its formulas on
**MRST's own** flash - so a disagreement is the mass-fraction or saturation
formula, not the flash that fed it (validated separately in Phase 13). The
saturations are also checked with a volume shift, which must move the volume
split while leaving the molar flash untouched.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.eos import Mixture, simple_mixture
from pymrst.modules.compositional.flow_props import (
    mass_fraction,
    phase_saturations,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p51_flowprops.mat"


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
        print("ref_p51_flowprops.mat missing (run dump_phase51_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    ship = simple_mixture()
    for attr in ("Tcrit", "Pcrit", "Vcrit", "acentric", "molarMass"):
        assert np.allclose(getattr(ship, attr), getattr(mix, attr)), attr

    p = np.atleast_1d(ref["p_all"]).astype(float)
    T = np.atleast_1d(ref["T_all"]).astype(float)
    x = np.atleast_2d(ref["x"]).astype(float)
    y = np.atleast_2d(ref["y"]).astype(float)
    z_l = np.atleast_1d(ref["Z_L"]).astype(float)
    z_v = np.atleast_1d(ref["Z_V"]).astype(float)
    L = np.atleast_1d(ref["L"]).astype(float)
    shift = np.atleast_1d(ref["shift"]).astype(float)
    n2 = int(np.asarray(ref["twoPhase"]).sum())
    print(f"\n{p.size} conditions, {n2} of them two-phase:")

    failed = 0

    # -- mass fractions: getMassFraction, fed MRST's own flash compositions ----
    xM_py = np.column_stack(mass_fraction(mix, [x[:, i] for i in range(mix.n)]))
    yM_py = np.column_stack(mass_fraction(mix, [y[:, i] for i in range(mix.n)]))
    print("\nMass fractions (getMassFraction):")
    failed += compare("liquid mass fraction", xM_py,
                      np.atleast_2d(ref["xM"]).astype(float), 1e-12)
    failed += compare("vapour mass fraction", yM_py,
                      np.atleast_2d(ref["yM"]).astype(float), 1e-12)

    # -- saturations: computeSaturations, no shift and with a volume shift ------
    sL, sV = phase_saturations(mix, p, T, L, x, y, z_l, z_v)
    print("\nSaturations (computeSaturations), no volume shift:")
    failed += compare("liquid saturation", sL, ref["sL"], 1e-12)
    failed += compare("vapour saturation", sV, ref["sV"], 1e-12)

    sL_s, sV_s = phase_saturations(mix, p, T, L, x, y, z_l, z_v,
                                   volume_shift=shift)
    print("\nSaturations with volume shift (moves volumes, not the flash):")
    failed += compare("liquid saturation", sL_s, ref["sL_shift"], 1e-12)
    failed += compare("vapour saturation", sV_s, ref["sV_shift"], 1e-12)

    # The shift must actually have done something, or the check above is empty.
    moved = float(np.nanmax(np.abs(np.asarray(sL_s) - np.asarray(sL))))
    tag = "ok  " if moved > 1e-4 else "FAIL"
    print(f"\n  {tag} the volume shift moved the saturation split "
          f"(max |dsL| = {moved:.3e})")
    failed += 0 if moved > 1e-4 else 1

    print("\nFlash-to-flow-props match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
