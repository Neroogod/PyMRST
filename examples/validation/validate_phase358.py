"""Phase 358: ad-core numeric utilities against MRST (AC22).

AC22 ports the genuinely-numeric `ad-core/utils` helpers one file at a time,
each with its own self-test. This pins the ones with non-trivial arithmetic
against MRST's own output (`dump_phase358_reference.m`), number for number:

- `interpolate_idw`          - inverse-distance-weighted scattered interpolation,
- `fast_interp_table`        - linear interpolation with linear extrapolation,
- `phase_densities_to_bfactor` - MRST `phaseDensitiesTobfactor`, dead and live,
- `merge_ordered_arrays`     - the supported "insertion" merge cases.

The `merge_ordered_arrays` cases that MRST itself errors on are covered by the
module self-test (they raise here too, by design) and are not part of this
number-for-number comparison.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.utils.fast_interp_table import fast_interp_table
from pymrst.modules.ad_core.utils.interpolate_idw import interpolate_idw
from pymrst.modules.ad_core.utils.merge_ordered_arrays import merge_ordered_arrays
from pymrst.modules.ad_core.utils.phase_densities_to_bfactor import (
    phase_densities_to_bfactor,
)

REF = Path(__file__).with_name("ref_p358_numeric.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase358_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    failed = 0

    print("interpolate_idw vs MRST:")
    v = interpolate_idw(ref["idw_x"], ref["idw_f"], ref["idw_xq"],
                        float(ref["idw_order"].ravel()[0]))
    err = np.max(np.abs(v - ref["idw_v"]))
    failed += _check("IDW values match", err < 1e-12, f"max|dv| = {err:.3e}")

    print("\nfast_interp_table vs MRST:")
    yi = fast_interp_table(ref["fit_X"].ravel(), ref["fit_Y"].ravel(),
                           ref["fit_xi"].ravel())
    err = np.max(np.abs(yi - ref["fit_yi"].ravel()))
    failed += _check("linear + extrapolation match", err < 1e-12,
                     f"max|dy| = {err:.3e}")

    print("\nphase_densities_to_bfactor vs MRST:")
    rho = ref["pdb_rho"]
    rho_s = ref["pdb_rhoS"].ravel()
    b_dead = phase_densities_to_bfactor([rho[:, 0], rho[:, 1]], rho_s)
    b_dead = np.column_stack(b_dead)
    err = np.max(np.abs(b_dead - ref["pdb_b_dead"]))
    failed += _check("dead-oil b-factors match", err < 1e-12,
                     f"max|db| = {err:.3e}")

    # live: 3 phases, gas (2) dissolved in oil (1); rebuild the dissolved table
    b_live_ref = ref["pdb_b_live"]
    nc = b_live_ref.shape[0]
    # reconstruct inputs from the same numbers the dump used
    rho3 = [np.array([1000.0] * nc), np.array([850.0, 860.0][:nc]),
            np.array([1.2, 1.25][:nc])]
    rho_s3 = [1000.0, 800.0, 1.0]
    rs = np.array([120.0, 110.0][:nc])
    dissolved = [[None, None, None], [None, None, None], [None, rs, None]]
    b_live = np.column_stack(phase_densities_to_bfactor(rho3, rho_s3, dissolved))
    err = np.max(np.abs(b_live - b_live_ref))
    failed += _check("live-oil b-factors match (rs in oil)", err < 1e-12,
                     f"max|db| = {err:.3e}")

    print("\nmerge_ordered_arrays vs MRST (supported insertion cases):")
    for key, old, new in [("merge1", [1, 2, 3, 4], [1, 2, 3, 4]),
                          ("merge2", [1, 2, 3], [1, 2, 4, 3]),
                          ("merge3", [10, 20, 30], [10, 15, 20, 30])]:
        out = merge_ordered_arrays(np.array(old), np.array(new))
        exp = ref[key].ravel()
        failed += _check(f"{key} = [{' '.join(map(str, exp.astype(int)))}]",
                         np.array_equal(out, exp))

    print("\nNumeric utilities match MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
