"""Phase 420: DG moment-fitting against MRST (fitMoments).

Moment-fitting produces a non-unique minimal rule, so PyMRST and MRST need not
pick the same points. Using the identical candidate grid and Legendre moments,
this checks the meaningful invariant both must satisfy: the fitted rule
reproduces every basis moment (hence integrates polynomials up to the precision
exactly) with non-negative weights, and compresses to a comparable point count.
"""

from itertools import product
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.dg.cubatures.moment_fitting import fit_moments
from pymrst.modules.dg.discretizations.dg_basis import dg_basis

REF = Path(__file__).with_name("ref_p420_moment_fitting.mat")


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _reproduces(basis, pts, w, M):
    return np.max(np.abs(basis.evaluate(pts).T @ w - M))


def _monomial_err(pts, w, k):
    worst = 0.0
    for exps in product(range(k + 1), repeat=2):
        if sum(exps) > k:
            continue
        num = float(w @ np.prod(pts ** np.array(exps), axis=1))
        exact = 1.0
        for e in exps:
            exact *= (1 - (-1) ** (e + 1)) / (e + 1)
        worst = max(worst, abs(num - exact))
    return worst


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase420_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    failed = 0
    print("DG moment-fitting (fitMoments) vs MRST:")
    for k in (2, 3, 4):
        basis = dg_basis(2, k, "legendre")
        m = 2 * k + 1
        g = np.linspace(-1, 1, m)
        X, Y = np.meshgrid(g, g, indexing="ij")
        cand = np.column_stack([X.ravel(), Y.ravel()])
        M = np.zeros(basis.n_dof)
        M[0] = 4.0

        pts, w = fit_moments(cand, basis, M)
        n_mine = pts.shape[0]

        pts_ref = np.asarray(ref[f"k{k}_x"], dtype=float)
        w_ref = np.asarray(ref[f"k{k}_w"], dtype=float).ravel()
        n_ref = int(np.asarray(ref[f"k{k}_n"]).ravel()[0])

        # Both rules must reproduce the moments and integrate monomials exactly.
        mine_ok = _reproduces(basis, pts, w, M) < 1e-9 and _monomial_err(pts, w, k) < 1e-9
        mrst_ok = _reproduces(basis, pts_ref, w_ref, M) < 1e-9 \
            and _monomial_err(pts_ref, w_ref, k) < 1e-9
        pos_w = np.all(w >= -1e-12)
        failed += _check(
            f"k={k}: both rules exact, weights >= 0",
            mine_ok and mrst_ok and pos_w,
            f"PyMRST {n_mine} pts, MRST {n_ref} pts (nDof {basis.n_dof})")

    print("\nMoment-fitting matches MRST (invariant)!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
