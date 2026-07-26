"""Phase 386: permeabilityConverter against MRST (Phase 25 core leftovers).

Pins `permeability_converter` against MRST's `permeabilityConverter` on a tilted
corner-point deck (sloped layers -> non-global cell axes -> off-diagonal terms).
PyMRST rebuilds the identical dumped input deck, rotates the diagonal PERMX/Y/Z
to the global frame in the 'cell' system, and compares all six tensor components
number for number.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.params.rock.permeability_converter import permeability_converter

REF = Path(__file__).with_name("ref_p386_permconv.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase386_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    grdecl = {
        "cartDims": ref["in_cartDims"].ravel().astype(np.int64),
        "COORD": ref["in_COORD"].ravel(),
        "ZCORN": ref["in_ZCORN"].ravel(),
        "PERMX": ref["in_PERMX"].ravel(),
        "PERMY": ref["in_PERMY"].ravel(),
        "PERMZ": ref["in_PERMZ"].ravel(),
    }
    out = permeability_converter(grdecl, "cell")

    failed = 0
    print("permeabilityConverter (tilted deck, 'cell' system) vs MRST:")
    for key in ("PERMX", "PERMY", "PERMZ", "PERMXY", "PERMYZ", "PERMZX"):
        err = np.max(np.abs(np.asarray(out[key]).ravel()
                            - ref[f"out_{key}"].ravel()))
        failed += _check(f"{key} matches", err < 1e-9, f"max|d| = {err:.3e}")

    print("\npermeabilityConverter matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
