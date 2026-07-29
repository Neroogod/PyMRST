"""Phase 417: DG basis functions against MRST (dgBasis).

Pins PyMRST's `dg_basis` against MRST's `dgBasis` for several
(dim, degree, type) combinations: the multi-indices, and the values of every
basis function and its gradient at a set of test points, must match.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.dg.discretizations.dg_basis import dg_basis

REF = Path(__file__).with_name("ref_p417_dg_basis.mat")

# Same order as the MATLAB dump: (dim, degree, type).
CASES = [(2, 2, "legendre"), (2, 3, "poly"), (3, 2, "legendre"), (2, 2, "chebyshev")]


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase417_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    failed = 0
    print("dgBasis vs MRST:")
    for c, (dim, degree, typ) in enumerate(CASES, start=1):
        b = dg_basis(dim, degree, typ)
        pre = f"c{c}_"
        k_ref = np.asarray(ref[pre + "k"], dtype=np.int64)
        x = np.asarray(ref[pre + "x"], dtype=float)
        psi_ref = np.asarray(ref[pre + "psi"], dtype=float)
        grad_ref = np.asarray(ref[pre + "grad"], dtype=float)
        if grad_ref.ndim == 2:                          # MATLAB squeezes dim==1
            grad_ref = grad_ref[:, :, None]

        tag = f"dim={dim} deg={degree} {typ}"
        failed += _check(f"{tag}: multi-indices match", np.array_equal(b.k, k_ref))
        pe = np.max(np.abs(b.evaluate(x) - psi_ref))
        failed += _check(f"{tag}: basis values match", pe < 1e-12,
                         f"max|dpsi| = {pe:.2e}")
        ge = np.max(np.abs(b.evaluate_gradient(x) - grad_ref))
        failed += _check(f"{tag}: gradients match", ge < 1e-12,
                         f"max|dgrad| = {ge:.2e}")

    print("\nDG basis matches MRST!" if failed == 0 else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
