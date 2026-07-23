"""Concrete AD linear solvers (MRST ad-core/solvers).

Ports MRST's `BackslashSolverAD.m`, `NoOpSolverAD.m` and
`GMRES_ILUSolverAD.m` - the direct, no-op, and Krylov siblings of
`LinearSolverAD`. PyMRST already has `CPRSolver` (the scalable one) and a
`DirectSolver`; these complete the family on the MRST-shaped base so a model can
be handed any of them by class.

- `BackslashSolverAD` - a direct solve (scipy `spsolve`), MRST's ``A\\b``.
  Exact, the reference every iterative solver is checked against.
- `NoOpSolverAD` - returns a zero increment. Used to freeze a block, or as a
  placeholder in a composite/segregated solve.
- `GMRES_ILUSolverAD` - restarted GMRES preconditioned by an incomplete LU
  factorisation, for systems too large for a direct factorisation but not
  needing CPR's two-stage structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD


class BackslashSolverAD(LinearSolverAD):
    """Direct solve via ``spsolve`` (MRST's mldivide ``A\\b``)."""

    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        self.last_iterations = 1
        return spla.spsolve(A.tocsc(), b)

    def get_description(self) -> str:
        return "BackslashSolverAD (spsolve direct solve)"


class NoOpSolverAD(LinearSolverAD):
    """Returns a zero increment - solves nothing (MRST `NoOpSolverAD`)."""

    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        self.last_iterations = 0
        return np.zeros(b.size)

    def get_description(self) -> str:
        return "NoOpSolverAD (zero increment)"


class GMRES_ILUSolverAD(LinearSolverAD):
    """Restarted GMRES with an ILU preconditioner (MRST `GMRES_ILUSolverAD`)."""

    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 200,
                 restart: int = 30, ilu_drop_tol: float = 1e-4, **kw):
        super().__init__(tolerance=tolerance, max_iterations=max_iterations,
                         **kw)
        self.restart = restart
        self.ilu_drop_tol = ilu_drop_tol

    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        A = A.tocsc()
        ilu = spla.spilu(A, drop_tol=self.ilu_drop_tol)
        M = spla.LinearOperator(A.shape, ilu.solve)
        counter = _IterCounter()
        try:
            x, info = spla.gmres(A, b, M=M, rtol=self.tolerance,
                                 restart=self.restart,
                                 maxiter=self.max_iterations,
                                 callback=counter, callback_type="pr_norm")
        except TypeError:                                  # older scipy: `tol`
            x, info = spla.gmres(A, b, M=M, tol=self.tolerance,
                                 restart=self.restart,
                                 maxiter=self.max_iterations, callback=counter)
        self.last_iterations = counter.n
        return x

    def get_description(self) -> str:
        return "GMRES_ILUSolverAD (GMRES + ILU)"


class _IterCounter:
    def __init__(self):
        self.n = 0

    def __call__(self, *args):
        self.n += 1


__all__ = ["BackslashSolverAD", "GMRES_ILUSolverAD", "NoOpSolverAD"]


if __name__ == "__main__":
    def _system(n=60, seed=1):
        rng = np.random.default_rng(seed)
        A = sp.random(n, n, density=0.15, random_state=rng, format="csr")
        A = A + sp.eye(n) * (np.abs(A).sum(axis=1).A1.max() + 1.0)
        x = rng.standard_normal(n)
        return A.tocsr(), A @ x, x

    def test_backslash_is_exact():
        A, b, x = _system()
        dx = BackslashSolverAD().solve(A, b)
        assert np.allclose(dx, x)
        assert np.max(np.abs(A @ dx - b)) < 1e-10
        print("ok test_backslash_is_exact")

    def test_noop_returns_zero():
        A, b, x = _system()
        assert np.allclose(NoOpSolverAD().solve(A, b), 0.0)
        print("ok test_noop_returns_zero")

    def test_gmres_ilu_matches_direct():
        A, b, x = _system()
        dx = GMRES_ILUSolverAD(tolerance=1e-10).solve(A, b)
        assert np.allclose(dx, x, atol=1e-6), np.max(np.abs(dx - x))
        assert np.max(np.abs(A @ dx - b)) < 1e-6
        print("ok test_gmres_ilu_matches_direct (iters="
              f"{GMRES_ILUSolverAD().last_iterations})")

    test_backslash_is_exact()
    test_noop_returns_zero()
    test_gmres_ilu_matches_direct()
    print("\nAll tests passed!")
