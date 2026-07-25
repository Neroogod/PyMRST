"""Phase 366: grdecl2Rock against MRST (Phase 25 core leftovers).

Pins `grdecl2rock` against MRST's `grdecl2Rock` across the three permeability
layouts - isotropic (lone PERMX -> 1 column), diagonal (PERMX/Y/Z -> 3) and full
symmetric (adding PERMXY/XZ/YZ -> 6 in [xx,xy,xz,yy,yz,zz] order) - plus PORO and
NTG. Same deterministic inputs, compared number for number.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.params.rock.grdecl2rock import grdecl2rock

REF = Path(__file__).with_name("ref_p366_grdecl2rock.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase366_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    nc = 5
    base = np.arange(1, nc + 1, dtype=float)
    failed = 0

    print("grdecl2Rock isotropic (lone PERMX) vs MRST:")
    ri = grdecl2rock({"PERMX": 10 + base, "PORO": 0.1 + 0.01 * base,
                      "NTG": 0.9 - 0.02 * base})
    failed += _check("perm matches", np.allclose(ri.perm, ref["perm_iso"]),
                     f"{ri.perm.shape[1]} column(s)")
    failed += _check("poro matches", np.allclose(ri.poro, ref["poro_iso"]))
    failed += _check("ntg matches", np.allclose(ri.ntg, ref["ntg_iso"]))

    print("\ngrdecl2Rock diagonal (PERMX/Y/Z) vs MRST:")
    rd = grdecl2rock({"PERMX": 10 + base, "PERMY": 20 + base,
                      "PERMZ": 1 + 0.1 * base, "PORO": 0.2 + 0.01 * base})
    failed += _check("perm matches", np.allclose(rd.perm, ref["perm_diag"]),
                     f"{rd.perm.shape[1]} columns")

    print("\ngrdecl2Rock full symmetric (PERMXY/XZ/YZ) vs MRST:")
    rf = grdecl2rock({"PERMX": 10 + base, "PERMY": 20 + base,
                      "PERMZ": 1 + 0.1 * base, "PERMXY": 0.5 + 0.01 * base,
                      "PERMXZ": 0.3 + 0.01 * base, "PERMYZ": 0.2 + 0.01 * base})
    failed += _check("perm matches (column order xx,xy,xz,yy,yz,zz)",
                     np.allclose(rf.perm, ref["perm_full"]),
                     f"{rf.perm.shape[1]} columns")

    print("\ngrdecl2Rock matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
