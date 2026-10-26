"""MRST ``ad-core/solvers/GMRES_ILUSolverAD.m``.

GMRES preconditioned by an incomplete LU factorisation, with MRST's
defaults: ILU type 'nofill' (ILU(0)), no drop tolerance, no modified ILU,
zero-pivot replacement on, pivot threshold 1, and equations reordered for a
non-zero diagonal. Solving follows MRST's `solveLinearSystem`:
- optionally reorder the equations (`reorderForILU`);
- factorise `A ~ L*U` with MATLAB's `ilu(A, opts)`;
- run GMRES without restarts for at most `min(maxIterations, n)`
  iterations, preconditioned by `L` and `U`, to relative tolerance
  `tolerance`, warning when it does not converge.

ILU(0) is unique for a matrix's sparsity pattern, so the factors are
MATLAB's to round-off. GMRES is SciPy's, so iterates agree with MATLAB's
only to the solver tolerance. Only 'nofill' is implemented; MATLAB's
'crout' and 'ilutp' raise. `GMRES_ILUSolverADAD` (in `concrete_solvers.py`)
is PyMRST's earlier solver (SuperLU ILUTP, restarted GMRES), kept for its
callers.
"""

from __future__ import annotations

import warnings
from time import perf_counter
from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)


class GMRES_ILUSolverAD(LinearSolverAD):
    """MRST `GMRES_ILUSolverAD`."""

    def __init__(self, **options: Any):
        super().__init__(tolerance=1e-8, max_iterations=25)
        self.ilutype = "nofill"
        self.dropTolerance = 0.0
        self.modifiedIncompleteILU = "off"
        self.udiagReplacement = True
        self.pivotThreshold = 1.0
        self.reorderEquations = True
        for key, item in options.items():
            if key == "maxIterations":
                key = "max_iterations"
            if not hasattr(self, key):
                raise TypeError(f"unknown option {key!r}")
            setattr(self, key, item)

    @property
    def maxIterations(self) -> int:
        return self.max_iterations

    @maxIterations.setter
    def maxIterations(self, value: int) -> None:
        self.max_iterations = value

    def solve_linear_system(self, A: Any, b: np.ndarray) -> np.ndarray:
        """`[result, report] = solveLinearSystem(A, b)`; the report is kept
        as `last_solve_report`."""
        timer = perf_counter()
        A = sp.csr_array(A)
        b = np.asarray(b, dtype=float).ravel()
        nel = A.shape[0]
        if self.reorderEquations:
            from pymrst.modules.ad_core.utils.reorder_for_ilu import reorder_for_ilu
            A, b = reorder_for_ilu(A, b)
            A = sp.csr_array(A)
        L, U = ilu(A, self.get_opts_ilu())
        # MATLAB's gmres(A, b, [], tol, maxit, L, U) preconditions from the
        # left and measures convergence (and `relres`) on the preconditioned
        # residual norm(M\(b - A*x))/norm(M\b). GMRES on M\A x = M\b
        # applies exactly that test.
        Minv_A = spla.LinearOperator(A.shape, lambda v: _lu_solve(L, U, A @ v), dtype=float)
        Minv_b = _lu_solve(L, U, b)
        iterations = [0]

        def count(_):
            iterations[0] += 1

        maxit = int(min(self.max_iterations, nel))
        result, flag = spla.gmres(Minv_A, Minv_b, rtol=self.tolerance, atol=0.0,
                                  restart=maxit, maxiter=1, callback=count,
                                  callback_type="pr_norm")
        res = (np.linalg.norm(Minv_b - Minv_A @ result)
               / max(np.linalg.norm(Minv_b), np.finfo(float).tiny))
        self.last_iterations = iterations[0]
        self.last_solve_report = self.get_solve_report(
            Residual=res, Iterations=iterations[0], Converged=flag == 0,
            LinearSolutionTime=perf_counter() - timer)
        self.last_solve_report["GMRESFlag"] = flag
        if flag > 0:
            warnings.warn(f"Solver did not converge to specified tolerance of "
                          f"{self.tolerance:g}. Reported residual estimate was {res:g}")
        return result

    def get_opts_ilu(self) -> dict:
        return {"type": self.ilutype, "droptol": self.dropTolerance,
                "milu": self.modifiedIncompleteILU, "udiag": self.udiagReplacement,
                "thresh": self.pivotThreshold}

    def get_description(self) -> Tuple[str, str]:
        return ("GMRES with incomplete LU-factorization (ILU(0)) as preconditioner",
                "Matlab-GMRES-ILU(0)")

    solveLinearSystem = solve_linear_system
    getOptsILU = get_opts_ilu
    getDescription = get_description


def ilu(A: Any, opts: dict):
    """MATLAB `[L, U] = ilu(A, opts)` for type 'nofill' (ILU(0)): `L` unit
    lower and `U` upper triangular, with `A`'s sparsity pattern."""
    if opts.get("type", "nofill") != "nofill":
        raise NotImplementedError(f"ilu type {opts['type']!r}: only 'nofill' is ported")
    if opts.get("milu", "off") != "off":
        raise NotImplementedError("modified ILU (milu) is not ported")
    A = sp.csr_array(A)
    A.sort_indices()
    n = A.shape[0]
    data = A.data.astype(np.float64).copy()
    indptr, indices = A.indptr.astype(np.int64), A.indices.astype(np.int64)
    status = _ilu0(indptr, indices, data, n)
    if status >= 0:
        raise ZeroDivisionError(f"ILU(0) met a zero pivot in row {status + 1}")
    F = sp.csr_array((data, indices, indptr), shape=A.shape)
    L = sp.tril(F, k=-1, format="csr") + sp.eye_array(n, format="csr")
    U = sp.triu(F, format="csr")
    return sp.csr_array(L), sp.csr_array(U)


@njit(cache=True)
def _ilu0(indptr, indices, data, n):
    """In-place ILU(0) of a CSR matrix with sorted columns (IKJ order).
    Returns -1, or the row whose pivot is zero or missing."""
    diag = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            if indices[p] == i:
                diag[i] = p
    marker = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if diag[i] < 0:
            return i
        for p in range(indptr[i], indptr[i + 1]):
            marker[indices[p]] = p
        for p in range(indptr[i], diag[i]):
            k = indices[p]
            pivot = data[diag[k]]
            if pivot == 0.0:
                return k
            data[p] /= pivot
            lik = data[p]
            for q in range(diag[k] + 1, indptr[k + 1]):
                pos = marker[indices[q]]
                if pos >= 0:
                    data[pos] -= lik*data[q]
        for p in range(indptr[i], indptr[i + 1]):
            marker[indices[p]] = -1
        if data[diag[i]] == 0.0:
            return i
    return -1


def _lu_solve(L: sp.csr_array, U: sp.csr_array, r: np.ndarray) -> np.ndarray:
    y = _forward(L.indptr.astype(np.int64), L.indices.astype(np.int64), L.data,
                 np.asarray(r, dtype=float).ravel())
    return _backward(U.indptr.astype(np.int64), U.indices.astype(np.int64), U.data, y)


@njit(cache=True)
def _forward(indptr, indices, data, b):
    """Solve `L y = b`, `L` unit lower triangular in CSR."""
    n = b.size
    y = b.copy()
    for i in range(n):
        acc = y[i]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j < i:
                acc -= data[p]*y[j]
        y[i] = acc
    return y


@njit(cache=True)
def _backward(indptr, indices, data, y):
    """Solve `U x = y`, `U` upper triangular in CSR."""
    n = y.size
    x = y.copy()
    for i in range(n - 1, -1, -1):
        acc = x[i]
        d = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j > i:
                acc -= data[p]*x[j]
            elif j == i:
                d = data[p]
        x[i] = acc/d
    return x


__all__ = ["GMRES_ILUSolverAD", "ilu"]


if __name__ == "__main__":
    def _system(n=40, seed=3):
        rng = np.random.default_rng(seed)
        A = sp.random_array((n, n), density=0.1, random_state=rng, format="csr")
        A = A + sp.diags_array(np.abs(A).sum(axis=1) + 1.0)
        return sp.csr_array(A), rng.standard_normal(n)

    def test_ilu0_matches_a_on_its_pattern():
        """ILU(0) is the unique L*U that equals A wherever A is non-zero."""
        A, _ = _system()
        L, U = ilu(A, {"type": "nofill"})
        LU = (L @ U).toarray()
        mask = A.toarray() != 0
        np.testing.assert_allclose(LU[mask], A.toarray()[mask], rtol=1e-12, atol=1e-12)
        assert np.allclose(np.diag(L.toarray()), 1.0)
        assert sp.tril(U, k=-1).nnz == 0
        print("ok test_ilu0_matches_a_on_its_pattern")

    def test_ilu0_of_a_tridiagonal_matrix_is_exact():
        """No fill is lost for a tridiagonal matrix, so L*U = A."""
        n = 20
        A = sp.diags_array([-np.ones(n - 1), 4*np.ones(n), -np.ones(n - 1)], offsets=[-1, 0, 1])
        L, U = ilu(A, {"type": "nofill"})
        np.testing.assert_allclose((L @ U).toarray(), A.toarray(), atol=1e-13)
        print("ok test_ilu0_of_a_tridiagonal_matrix_is_exact")

    def test_gmres_solves_to_the_tolerance():
        """Convergence is judged on the preconditioned residual, as MATLAB's
        left-preconditioned gmres does."""
        A, b = _system()
        solver = GMRES_ILUSolverAD(tolerance=1e-10, maxIterations=100, reorderEquations=False)
        x = solver.solve_linear_system(A, b)
        L, U = ilu(A, {"type": "nofill"})
        r = _lu_solve(L, U, b - A @ x)
        assert np.linalg.norm(r) <= 1e-10*np.linalg.norm(_lu_solve(L, U, b))
        assert solver.last_solve_report["Converged"]
        assert np.linalg.norm(A @ x - b) <= 1e-8*np.linalg.norm(b)
        print("ok test_gmres_solves_to_the_tolerance")

    def test_mrsts_defaults():
        s = GMRES_ILUSolverAD()
        assert (s.ilutype, s.dropTolerance, s.modifiedIncompleteILU, s.udiagReplacement,
                s.pivotThreshold, s.reorderEquations) == ("nofill", 0.0, "off", True, 1.0, True)
        assert (s.tolerance, s.maxIterations) == (1e-8, 25)
        print("ok test_mrsts_defaults")

    test_ilu0_matches_a_on_its_pattern()
    test_ilu0_of_a_tridiagonal_matrix_is_exact()
    test_gmres_solves_to_the_tolerance()
    test_mrsts_defaults()
    print("\nAll tests passed!")
