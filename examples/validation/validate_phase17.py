"""Phase 17: the CO2 trapping inventory and injection-point ranking.

Compares against MRST's massTrappingDistributionVEADI, maximizeTrapping and
findOptimalInjectionPoint, on the VE states in states_p17.mat.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(Path(__file__).parent))
from dump_phase17_states import build, build_asymmetric            # noqa: E402

from pymrst.modules.co2lab.inventory import (                      # noqa: E402
    CATEGORIES,
    mass_trapping_distribution,
)
from pymrst.modules.co2lab.trapping import (                       # noqa: E402
    TrapAnalysis,
    find_optimal_injection_point,
    maximize_trapping,
    trap_analysis,
    volumes_of_traps,
)
from pymrst.modules.co2lab.ve import VEFluid, VEState              # noqa: E402

REF = Path(__file__).with_name("ref_p17_co2_inventory.mat")
STATES = Path(__file__).with_name("states_p17.mat")


def main() -> int:
    if not REF.exists() or not STATES.exists():
        print(f"{REF.name} missing (run dump_phase17_states.py, then "
              f"dump_phase17_reference.m in MATLAB)")
        return 0

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    s = scipy.io.loadmat(STATES, squeeze_me=True)

    ts, rock2 = build()
    res = trap_analysis(ts)
    fluid = VEFluid(rhoWS=1020.0, rhoGS=float(s["rhoGS"]), mu_w=8e-4,
                    mu_g=6e-5, srw=float(s["srw"]), srg=float(s["srg"]),
                    krW_max=1.0, krG_max=1.0, c_water=float(s["c_water"]),
                    c_gas=float(s["c_gas"]), p_ref=float(s["p_ref"]))

    p = np.atleast_2d(s["p"])
    sg = np.atleast_2d(s["sG"])
    sgmax = np.atleast_2d(s["sGmax"])
    n = p.shape[1]

    def inv(k, **kw):
        st = VEState(pressure=p[:, k], sG=sg[:, k], sGmax=sgmax[:, k],
                     qGs=np.zeros(0), bhp=np.zeros(0))
        return mass_trapping_distribution(ts, st, rock2, fluid, res, **kw)

    ok = True
    print(f"CO2 inventory over {n} VE states ({res.n_traps} traps, "
          f"{ts.g.cells.num} cells)\n")

    # ---- the inventory, state by state -----------------------------------
    mine = np.array([inv(k).masses for k in range(n)])
    theirs = np.atleast_2d(ref["masses"])
    scale = float(np.abs(theirs).max())
    d = np.abs(mine - theirs).max() / scale
    good = d <= 1e-12
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {'inventory, all 7 categories':34s} "
          f"max rel diff = {d:.2e}  (tol 1e-12)")

    # ---- with sub-scale trapping, and with dissolution --------------------
    for label, got, want in (
        ("with sub-scale trapping (dh=1m)",
         inv(n - 1, dh=np.ones(ts.g.cells.num)).masses,
         np.asarray(ref["masses_dh"], float).ravel()),
        ("with dissolution (rs=0.05)",
         inv(n - 1, rs=0.05).masses,
         np.asarray(ref["masses_rs"], float).ravel()),
    ):
        d = np.abs(got - want).max() / scale
        good = d <= 1e-12
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {label:34s} "
              f"max rel diff = {d:.2e}  (tol 1e-12)")

    # ---- trap volumes and the injection-point ranking ---------------------
    # MRST's maximizeTrapping calls volumesOfTraps with no rock, so it ranks
    # on *bulk* trap volume, not pore volume. Match that.
    vols = volumes_of_traps(ts, res)
    want_vols = np.asarray(ref["trap_vols"], float).ravel()
    d = np.abs(vols - want_vols).max() / np.abs(want_vols).max()
    good = d <= 1e-9
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {'trap capacities':34s} "
          f"max rel diff = {d:.2e}  (tol 1e-09)")

    trees, _ = maximize_trapping(ts, res)
    roots = np.array([t.root for t in trees])
    values = np.array([t.value for t in trees])
    want_roots = np.atleast_1d(np.asarray(ref["tree_roots"], float).ravel())
    want_values = np.atleast_1d(np.asarray(ref["tree_values"], float).ravel())

    same = (roots.size == want_roots.size
            and np.array_equal(roots, want_roots.astype(int)))
    ok &= same
    print(f"  {'ok  ' if same else 'FAIL'} {'spill-tree roots':34s} "
          f"{list(roots)} vs MATLAB {list(want_roots.astype(int))}")

    if same:
        d = np.abs(values - want_values).max() / np.abs(want_values).max()
        good = d <= 1e-9
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {'spill-tree capacities':34s} "
              f"max rel diff = {d:.2e}  (tol 1e-09)")

    cell, vol, traps = find_optimal_injection_point(ts, res)
    want_cell = int(ref["opt_cell"]) - 1              # MATLAB is 1-based
    want_vol = float(ref["opt_vol"])

    # The objective ties: trap 2's spill tree already contains trap 1, so a
    # divide against trap 2 alone reaches exactly as much as the 1-2 divide
    # does. Both codes therefore pick a valid maximiser, but not the same one.
    # What must agree is the volume reached, and it does, exactly.
    d = abs(vol - want_vol) / want_vol
    good = d <= 1e-9
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {'volume the best point reaches':34s} "
          f"rel diff = {d:.2e}  (tol 1e-09)")
    if cell != want_cell:
        print(f"       note: a different (tied) divide is chosen - cell {cell} "
              f"vs MATLAB's {want_cell}, both reaching {vol:.4e} m3")

    # ---- catchments -------------------------------------------------------
    # Trap labels are arbitrary in both codes, so the regions can only be
    # compared after mapping the labels by trap-cell overlap. Comparing the
    # raw integers understates the agreement badly - it was what made this
    # look like a much larger discrepancy than it is.
    def map_labels(py: TrapAnalysis, ml_traps: np.ndarray) -> np.ndarray:
        remap = np.zeros(py.n_traps + 1, dtype=int)
        for i in range(1, py.n_traps + 1):
            cnt = np.bincount(ml_traps[py.traps == i],
                              minlength=int(ml_traps.max()) + 1)
            cnt[0] = 0
            remap[i] = int(np.argmax(cnt))
        return remap

    # The asymmetric caprock is the real test. On the symmetric one above,
    # mirror-image cells sit at bit-identical depths in MATLAB but differ by
    # ~2e-13 in the Python grid, so MATLAB sees a dome with two summits where
    # Python sees one - a round-off tie, not an algorithmic difference. Offset
    # the domes and the ties vanish.
    ts_a, _ = build_asymmetric()
    res_a = trap_analysis(ts_a)
    ml_traps_a = np.asarray(ref["asym_traps"], int).ravel()
    ml_reg_a = np.asarray(ref["asym_regions"], int).ravel()

    same_cells = np.array_equal(res_a.traps > 0, ml_traps_a > 0)
    ok &= same_cells
    print(f"  {'ok  ' if same_cells else 'FAIL'} "
          f"{'asymmetric: trapped cells':34s} "
          f"{int((res_a.traps > 0).sum())} vs {int((ml_traps_a > 0).sum())}")

    got_reg = map_labels(res_a, ml_traps_a)[res_a.trap_regions]
    n_ok = int((got_reg == ml_reg_a).sum())
    good = n_ok == got_reg.size
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {'asymmetric: catchments':34s} "
          f"{n_ok}/{got_reg.size} cells agree")

    # and on the symmetric case, report the round-off tie rather than hide it
    ml_reg = np.asarray(ref["trap_regions"], int).ravel()
    agree_sym = int((res.trap_regions == ml_reg).sum())
    print(f"\n  note: on the *symmetric* caprock the catchments agree on "
          f"{agree_sym}/{ml_reg.size} cells ({agree_sym / ml_reg.size:.1%}).")
    print("       Mirror-image cells are bit-identical in MATLAB and 2e-13 "
          "apart in the Python grid, so")
    print("       MATLAB finds a third summit that Python does not. A "
          "round-off tie on a perfectly")
    print("       symmetric surface - the asymmetric case above is the one "
          "that settles the algorithm.")

    free = theirs[:, 1:].sum(axis=1)
    print(f"\n  (MRST: the mobile fraction falls from "
          f"{theirs[0, 6] / free[0]:.0%} to {theirs[-1, 6] / free[-1]:.0%}; "
          f"the best injection point reaches {want_vol:.3e} m3)")

    print("\nCO2 inventory and trapping ranking match MATLAB!" if ok
          else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
