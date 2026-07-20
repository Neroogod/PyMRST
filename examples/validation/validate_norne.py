"""Norne coupled geomechanics: PyMRST vs MATLAB MRST, on the real field.

`dump_norne_reference.m` runs the identical case through MRST's
`MechOilWaterModel`. This replays it in PyMRST and compares.

The comparison is not a straight "do the numbers match", because they must
not: MRST's `equationsOilWaterMech` omits `G.cells.volumes` on its Biot term
(README.md, seventh finding), so its mechanical storage is short by a factor
of the cell volume. On Norne's ~70,000 m3 cells that is not a rounding error -
it is the difference between the rock absorbing the injected water and not.

So there are two comparisons, and both must hold:

1. **PyMRST with `mrst_oilwater_bug=True` must reproduce MRST.** That
   validates the port: same grid, same rock, same boundary conditions, same
   wells, same discretisation, same answer.

2. **PyMRST's correct model must differ, and differ the way the missing
   factor predicts** - the mechanical storage comes back, so the pressure has
   somewhere to go.

Plus the check that pins the boundary conditions independently of any of it:
both codes must start in *exact equilibrium*, |u| ~ 1e-15 m, because a
confining pressure of pRef cancels a pore pressure of pRef at alpha = 1.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_core.nonlinear_solver import NonLinearSolver
from pymrst.utils.units import BARSA, DAY

HERE = Path(__file__).parent
REF = HERE / "ref_norne_mech.mat"

# The example is a script, not a package module; load it by path so this
# validator runs from anywhere.
_spec = importlib.util.spec_from_file_location(
    "run_norne_geomech", HERE.parent / "run_norne_geomech.py"
)
norne = importlib.util.module_from_spec(_spec)
sys.modules["run_norne_geomech"] = norne
_spec.loader.exec_module(norne)


def run_python(mrst_bug: bool):
    """The example's case, with the correct or the MRST Biot term."""
    g, model, st = norne.build_case("mini", verbose=False)
    if mrst_bug:
        from pymrst.modules.ad_mechanics.poromech import (
            PoroMechanicalPoreVolume,
        )
        model.PVTPropertyFunctions.set_state_function(
            "PoreVolume",
            PoroMechanicalPoreVolume(model, scale_mech_by_volume=False),
        )
        st = model.seed_state(st)

    dts = [1.0 * DAY] + [5.0 * DAY] * 20
    solver = NonLinearSolver(max_iterations=40)

    p, sw, u, uz, divu, strain, eff_stress, total_stress, bhp = (
        [], [], [], [], [], [], [], [], []
    )
    for n, dt in enumerate(dts):
        st, rep = solver.solve_ministep(model, st, st.copy(), dt)
        assert rep.converged, f"step {n}: {rep.residuals[-1]}"
        diag = model.mech.diagnostics(st)
        p.append(st["pressure"].copy())
        sw.append(st["sW"].copy())
        u.append(st["u"].copy())
        uz.append(st["u"].reshape(-1, 3)[:, 2].copy())
        divu.append(diag["volumetric_strain"].copy())
        strain.append(diag["strain"].copy())
        eff_stress.append(diag["effective_stress"].copy())
        total_stress.append(diag["total_stress"].copy())
        bhp.append(st["bhp"].copy())
    return g, model, (
        np.vstack(p),
        np.vstack(sw),
        np.vstack(u),
        np.vstack(uz),
        np.vstack(divu),
        np.stack(strain),
        np.stack(eff_stress),
        np.stack(total_stress),
        np.vstack(bhp),
    )


def compare(name, py, ml, rtol) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_norne_mech.mat missing (run dump_norne_reference.m "
              "in MATLAB)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    ml_p, ml_sw = ref["p_hist"], ref["sw_hist"]
    ml_uz, ml_uz0 = ref["uz_hist"], ref["uz0"]
    ml_bhp = ref["bhp"]
    have_mech_diagnostics = all(
        k in ref for k in (
            "u_hist", "divu_hist", "strain_hist",
            "eff_stress_hist", "total_stress_hist",
        )
    )
    failed = 0

    print("=" * 70)
    print("Norne, coupled flow and geomechanics: PyMRST vs MRST")
    print("=" * 70)

    # -- 1. reproduce MRST, by reproducing its Biot term --------------------
    print("\n1. PyMRST with MRST's Biot term (mrst_oilwater_bug=True)")
    print("   must reproduce MRST - same grid, rock, BCs, wells:\n")
    g, model, (p, sw, u, uz, divu, strain, eff, total, bhp) = run_python(
        mrst_bug=True
    )
    failed += compare("water saturation", sw, ml_sw, 1e-6)
    if have_mech_diagnostics:
        failed += compare("full displacement u", u[1:], ref["u_hist"][1:], 1e-6)
    failed += compare("displacement u_z", uz[1:], ml_uz[1:], 1e-6)
    if have_mech_diagnostics:
        failed += compare("volumetric strain div(u) (from step 2)", divu[2:],
                          ref["divu_hist"][2:], 1e-6)
        failed += compare("strain tensor (from step 2)", strain[2:],
                          ref["strain_hist"][2:], 1e-6)
        failed += compare("effective stress (from step 2)", eff[2:],
                          ref["eff_stress_hist"][2:], 1e-6)
        failed += compare("total stress (from step 2)", total[2:],
                          ref["total_stress_hist"][2:], 1e-6)
    else:
        print("\n  note: ref_norne_mech.mat does not yet contain full "
              "stress/strain diagnostics.")
        print("        Re-run dump_norne_reference.m in MATLAB to enable "
              "those comparisons.")
    failed += compare("well bhp", bhp[1:], ml_bhp[1:], 1e-6)
    failed += compare("pressure (from step 2)", p[2:], ml_p[2:], 1e-6)

    # -- 1b. the one place the two codes genuinely differ -------------------
    # Not swept under a loose tolerance. The first step differs, the rest
    # inherit a decaying memory of it, and the decay is the evidence that it
    # is one initialisation difference rather than a systematic error.
    scale = np.abs(ml_p).max()
    per_step = np.abs(p - ml_p).max(axis=1) / scale

    prod = model.wells[1]
    rho_s = model.fluid.rho_s[:2]
    rho_oil = float(rho_s[1])                       # the wellbore sits in oil
    rho_mrst = float(np.array([0.5, 0.5]) @ rho_s)  # MRST's zero-flux guess
    dz = float(np.asarray(prod.dZ).ravel()[0])
    d_cdp = (rho_mrst - rho_oil) * model.gravity_magnitude * dz

    print("\n   The first step differs, and it is worth saying exactly why.")
    print("   MRST takes each well's wellbore density from its *connection")
    print("   fluxes*, which are zero before the well has ever flowed. Its")
    print("   guard replaces those zeros with sign*1e-12 - the same value in")
    print("   every phase - so a producer that has not yet produced is modelled")
    print(f"   as holding a 50/50 mixture ({rho_mrst:.0f} kg/m3) rather than the")
    print(f"   oil it is actually sitting in ({rho_oil:.0f} kg/m3). Over this")
    print(f"   well's {dz:.0f} m of column that is {d_cdp / 1e5:.2f} bar of")
    print("   connection pressure drop, on step 0 only. PyMRST uses the")
    print("   physical mixture. From step 1 both wells have flowed, both codes")
    print("   use the real mixture, and the discrepancy decays away:\n")
    print("     step:  " + "  ".join(f"{i:>7d}" for i in (0, 1, 2, 5, 10, 20)))
    print("     rel :  " + "  ".join(f"{per_step[i]:7.1e}"
                                     for i in (0, 1, 2, 5, 10, 20)))
    if not (per_step[0] > per_step[1] > per_step[2] and per_step[-1] < 1e-7):
        print("  FAIL: that discrepancy should decay, and it did not")
        failed += 1

    # -- 2. the boundary conditions, checked independently -----------------
    print("\n2. Both codes start in exact equilibrium (pRef confines pRef),")
    print("   which pins the boundary conditions on a non-flat corner-point")
    print("   surface - the thing easiest to get silently wrong:\n")
    _, model0, st0 = norne.build_case("mini", verbose=False)
    py_u0 = float(np.abs(st0["u"]).max())
    ml_u0 = float(np.abs(ml_uz0).max())
    print(f"  ok   initial |u_z|: PyMRST {py_u0:.2e} m, MRST {ml_u0:.2e} m")
    if max(py_u0, ml_u0) > 1e-9:
        print("  FAIL: the field should start undeformed")
        failed += 1

    # -- 3. the correct model, and what MRST's defect costs on a real field --
    print("\n3. PyMRST's CORRECT Biot term must now disagree with MRST -")
    print("   and the disagreement is what the missing cell volume buys:\n")
    g2, model2, (p2, sw2, u2, uz2, divu2, strain2, eff2, total2,
                 bhp2) = run_python(mrst_bug=False)

    vol = g2.cells.volumes
    print(f"  Norne cell volumes: {vol.min():.0f} - {vol.max():.0f} m3 "
          f"(median {np.median(vol):.0f})")
    print(f"  MRST     : final mean pressure {ml_p[-1].mean() / BARSA:8.1f} "
          f"bar,  max uplift {1000 * np.max(-(ml_uz[-1] - ml_uz0)):8.1f} mm")
    print(f"  PyMRST   : final mean pressure {p2[-1].mean() / BARSA:8.1f} "
          f"bar,  max uplift "
          f"{1000 * np.max(-(uz2[-1] - st0['u'].reshape(-1, 3)[:, 2])):8.1f} mm")

    ratio = ml_p[-1].mean() / p2[-1].mean()
    print(f"\n  MRST over-pressures the field by a factor of {ratio:.1f}.")
    print("  Why: its pore volume is  poro*V + alpha*div(u)  instead of")
    print("  V*(poro + alpha*div(u)), so the *mechanical* storage of a")
    print(f"  {np.median(vol):.0f} m3 cell is not ~{np.median(vol) * 0.01:.0f} m3")
    print("  but ~0.01 m3. The rock stops absorbing the injected water, and")
    print("  with a nearly incompressible fluid the pressure has nowhere to")
    print("  go. On a unit-cell grid the defect is invisible; on a real")
    print("  field it is not.")

    if ratio < 2.0:
        print("\n  FAIL: the defect should have shown up here")
        failed += 1

    print("\n" + "=" * 70)
    if failed == 0:
        print("PyMRST reproduces MRST on Norne, and the divergence from it is\n"
              "exactly the documented defect.")
    else:
        print(f"{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
