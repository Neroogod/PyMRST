"""Phase 355: CNV / material-balance convergence, against MATLAB ad-core (AC19).

PyMRST computed the CNV (max-norm) and MB (material-balance) convergence values
with a copy of the same arithmetic inside every model class. AC19 factored that
into `ad_core/utils/convergence.py`, one kernel matching MRST's
`getConvergenceValuesCNV`, and refactored the five black-oil models onto it. This
checks that kernel against MRST number for number.

The residual is a solver's *first iteration* of SPE 1's step 1: the equations
assembled at the initial state with the wells active. The accumulation term is
zero there (the state has not moved), so what is left is the well-driven
residual - non-trivial in every phase, gas included, whose `b ~ 262` is the
whole reason each phase is weighted by `B_avg = mean(1/b)`.

This follows the "fix the inputs, compare the algorithm" contract. The CNV/MB
values depend on the residual, and PyMRST and MRST initialise the well bhp
differently, so their iteration-1 residuals are *not* the same - that difference
is the well model's, not the convergence criterion's. To test only the
criterion, the dump saves MRST's **own** per-phase residuals, and PyMRST's
kernel is run on those identical numbers. The ingredients (`B_avg`, pore volume)
are cross-checked too, so a criterion mismatch could not hide behind a residual
one.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    initial_state_from_deck,
)
from pymrst.modules.ad_core.utils.convergence import (
    cnv_mb_values,
    formation_volume_factor_avg,
)
from pymrst.modules.ad_props.blackoil_fluid import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.adi import combine_equations, value

HERE = Path(__file__).parent
REF = HERE / "ref_p355_convergence.mat"
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


def build():
    deck = read_spe1_deck(DECK)
    g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(g, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = None
    for name in ("PRODUCER", "INJECTOR"):
        wd = deck["wells"][name]
        wells = add_well(wells, g, rock, wd["cells"], well_type=wd["type"],
                         val=wd["val"], radius=wd["radius"], compi=wd["compi"],
                         sign=wd["sign"], name=name, ref_depth=wd["ref_depth"],
                         gravity=np.array([0.0, 0.0, NORM_GRAVITY]))
    model = GenericBlackOilModel(g, rock, fluid, wells, disgas=True,
                                 gravity=NORM_GRAVITY, tol_well=1e-8)
    st = initial_state_from_deck(model, deck)
    return model, st, deck


def compare(name, py, ml, rtol, atol=0.0) -> int:
    py = float(py)
    ml = float(ml)
    diff = abs(py - ml)
    ok = diff <= rtol * max(abs(ml), 1e-300) + atol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {py:.6e} vs {ml:.6e} "
          f"(|d| {diff:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p355_convergence.mat missing (run dump_phase355_reference.m)")
        return 1
    if not DECK.exists():
        print("SPE1 deck not found; skipping")
        return 0
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    dt = float(ref["dt"])
    ml_values = np.asarray(ref["values"], float).ravel()
    names = [str(n).strip() for n in np.atleast_1d(ref["names"])]
    ml = dict(zip(names, ml_values))
    ml_b = np.atleast_2d(np.asarray(ref["b_mat"], float))
    ml_pv = np.asarray(ref["pv"], float).ravel()
    ml_res = np.atleast_2d(np.asarray(ref["res_surf"], float))   # (nc, nph)

    model, st, deck = build()
    ad = model.init_state_ad(st)

    # 1. the ingredients: PyMRST must compute the *same* B_avg and pore volume
    #    MRST did, or a value mismatch below could be blamed on either.
    b = model.get_props(ad, "ShrinkageFactors")
    pv = np.asarray(value(model.get_props(ad, "PoreVolume")), float)
    b_avg = formation_volume_factor_avg(b)
    ml_b_avg = np.array([float(np.mean(1.0 / ml_b[:, i]))
                         for i in range(ml_b.shape[1])])

    failed = 0
    print("Ingredients (so a criterion mismatch is not blamed on the residual):")
    failed += compare("pore volume (sum)", pv.sum(), ml_pv.sum(), 1e-8)
    for i, ph in enumerate("WOG"):
        failed += compare(f"B_avg[{ph}]", b_avg[i], ml_b_avg[i], 1e-8)

    # 2. the criterion itself, on MRST's *own* residual (input-fixing): PyMRST's
    #    kernel must reproduce the CNV/MB values MRST reported, number for number.
    cell_eqs = [ml_res[:, i] for i in range(model.n_phases)]
    cnv, mb = cnv_mb_values(cell_eqs, ml_b_avg, ml_pv, dt)

    print("\nCNV / MB on MRST's own residual, vs MRST's getConvergenceValues:")
    for i, ph in enumerate("WOG"):
        failed += compare(f"CNV_{ph}", cnv[i], ml[f"CNV_{ph}"], 1e-8, atol=1e-14)
    for i, ph in enumerate("WOG"):
        failed += compare(f"MB_{ph}", mb[i], ml[f"MB_{ph}"], 1e-8, atol=1e-20)

    print("\nCNV / MB convergence matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
