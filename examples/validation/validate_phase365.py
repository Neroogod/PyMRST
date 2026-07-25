"""Phase 365: processNNC against MRST (Phase 25 core leftovers).

Pins `process_nnc` against MRST's `processNNC` on a 4x4x3 grid with a
deterministic NNC list mixing genuine non-neighbour connections with pairs that
are already faces. Both the keep-all result and the excludeExisting result must
match MRST's resolved cell pairs and transmissibilities - confirming the
column-major logical->linear->active mapping and the existing-face exclusion.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.process_nnc import process_nnc

REF = Path(__file__).with_name("ref_p365_nnc.mat")

NNC = np.array([[1, 1, 1, 4, 4, 3, 0.10],
                [1, 1, 1, 2, 1, 1, 0.20],
                [2, 2, 2, 2, 2, 3, 0.30],
                [1, 2, 1, 3, 4, 2, 0.40],
                [4, 4, 1, 1, 1, 3, 0.50]])


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase365_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    g = compute_geometry(cart_grid([4, 4, 3], [4.0, 4.0, 3.0]))

    failed = 0
    print("processNNC (keep all) vs MRST:")
    out = process_nnc(g, NNC)
    cells_ref = ref["cells_all"].astype(np.int64) - 1        # 1- -> 0-based
    failed += _check("cell pairs match", np.array_equal(out["cells"], cells_ref),
                     f"{out['cells'].shape[0]} connections")
    failed += _check("transmissibilities match",
                     np.allclose(out["trans"], ref["trans_all"].ravel()))

    print("\nprocessNNC (excludeExisting) vs MRST:")
    out_x = process_nnc(g, NNC, exclude_existing=True)
    cells_xref = ref["cells_excl"].astype(np.int64) - 1
    failed += _check("existing faces dropped, pairs match",
                     np.array_equal(out_x["cells"], cells_xref),
                     f"{out_x['cells'].shape[0]} connections (from {NNC.shape[0]})")
    failed += _check("transmissibilities match",
                     np.allclose(out_x["trans"], ref["trans_excl"].ravel()))

    print("\nprocessNNC matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
