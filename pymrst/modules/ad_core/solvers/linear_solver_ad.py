"""Linear-solver base and linearized problem (MRST ad-core/solvers).

Ports MRST's `LinearSolverAD.m` (the base contract every AD linear solver
programs against) and `LinearizedProblem.m` (the container that carries a
linearized Newton system - its matrix, right-hand side, and the names/sizes of
the primary-variable blocks it is ordered by).

PyMRST's concrete solvers (`CPRSolver`, `DirectSolver`) already expose
``solve(jac, rhs, model)``. This adds the MRST-shaped base so those and the new
siblings (`BackslashSolverAD`, `NoOpSolverAD`, `GMRES_ILUSolverAD`) share one
interface: subclasses implement ``solve_linear_system(A, b)``, and the base
provides the ``solve(jac, rhs, model)`` entry the model driver calls, an
optional diagonal scaling, and a solve report. `LinearizedProblem` additionally
splits a solved increment back into the per-variable blocks the model updates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


class LinearizedProblem:
    """A linearized Newton system ``A dx = -b`` with its variable layout.

    ``var_names`` / ``var_sizes`` describe the primary-variable blocks the
    columns of ``A`` (and entries of ``b``) are ordered by, so a solved
    increment can be split back into named pieces via `store_increments`.
    """

    def __init__(self, A: sp.spmatrix, b: np.ndarray,
                 var_names: Optional[Sequence[str]] = None,
                 var_sizes: Optional[Sequence[int]] = None,
                 state: Any = None, dt: Optional[float] = None):
        self.A = A
        self.b = np.asarray(b, dtype=float).ravel()
        self.var_names = list(var_names) if var_names is not None else None
        self.var_sizes = list(var_sizes) if var_sizes is not None else None
        self.state = state
        self.dt = dt

    def num_dof(self) -> int:
        return self.b.size

    def store_increments(self, dx: np.ndarray) -> Dict[str, np.ndarray]:
        """Split a solved increment into the named primary-variable blocks."""
        dx = np.asarray(dx, dtype=float).ravel()
        if self.var_names is None or self.var_sizes is None:
            return {"dx": dx}
        out: Dict[str, np.ndarray] = {}
        off = 0
        for name, sz in zip(self.var_names, self.var_sizes):
            out[name] = dx[off:off + sz]
            off += sz
        return out


class LinearSolverAD:
    """Base class for linear solvers of the AD Newton system."""

    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 100,
                 apply_scaling: bool = False, verbose: bool = False):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.apply_scaling = apply_scaling
        self.verbose = verbose
        self.last_iterations = 0
        self.last_solve_report: Dict[str, Any] = {}

    # subclasses implement this
    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LinearSolverAD is abstract")

    def get_description(self) -> str:
        return type(self).__name__

    # -- optional diagonal (Jacobi) scaling --------------------------------
    def _apply_scaling(self, A: sp.spmatrix, b: np.ndarray):
        if not self.apply_scaling:
            return A, b, None
        d = np.asarray(np.abs(A.diagonal()), float)
        d[d == 0] = 1.0
        s = 1.0 / d
        S = sp.diags(s)
        return (S @ A).tocsr(), s * b, s

    # -- the entry the model driver calls ----------------------------------
    def solve(self, jac: sp.spmatrix, rhs: np.ndarray, model: Any = None,
              **kwargs) -> np.ndarray:
        """Solve ``jac dx = rhs`` (PyMRST-compatible signature)."""
        A, b, scaling = self._apply_scaling(sp.csr_matrix(jac),
                                            np.asarray(rhs, float).ravel())
        dx = self.solve_linear_system(A, b)
        return np.asarray(dx, float).ravel()

    def solve_linear_problem(self, problem: LinearizedProblem,
                             model: Any = None) -> np.ndarray:
        """Solve a `LinearizedProblem`'s ``A dx = -b`` and return ``dx``."""
        return self.solve(problem.A, -problem.b, model)


__all__ = ["LinearSolverAD", "LinearizedProblem"]


if __name__ == "__main__":
    def _system(n=30, seed=0):
        rng = np.random.default_rng(seed)
        A = sp.random(n, n, density=0.2, random_state=rng, format="csr")
        A = A + sp.eye(n) * (np.abs(A).sum(axis=1).A1.max() + 1.0)  # diag-dom
        x = rng.standard_normal(n)
        return A.tocsr(), A @ x, x

    class _Direct(LinearSolverAD):
        def solve_linear_system(self, A, b):
            import scipy.sparse.linalg as spla
            return spla.spsolve(A.tocsc(), b)

    def test_base_is_abstract():
        try:
            LinearSolverAD().solve_linear_system(sp.eye(2), np.zeros(2))
            raise SystemExit("base must be abstract")
        except NotImplementedError:
            pass
        print("ok test_base_is_abstract")

    def test_solve_and_scaling_agree():
        """A concrete solver recovers x; diagonal scaling gives the same x."""
        A, b, x = _system()
        assert np.allclose(_Direct().solve(A, b), x)
        assert np.allclose(_Direct(apply_scaling=True).solve(A, b), x)
        print("ok test_solve_and_scaling_agree")

    def test_linearized_problem_splits_increment():
        """store_increments splits a solution into named variable blocks."""
        prob = LinearizedProblem(sp.eye(5), np.zeros(5),
                                 var_names=["p", "sW"], var_sizes=[3, 2])
        blocks = prob.store_increments(np.array([1., 2., 3., 4., 5.]))
        assert np.allclose(blocks["p"], [1, 2, 3])
        assert np.allclose(blocks["sW"], [4, 5])
        print("ok test_linearized_problem_splits_increment")

    def test_solve_linear_problem_uses_minus_b():
        """solve_linear_problem solves A dx = -b."""
        A, b, x = _system()
        prob = LinearizedProblem(A, -b)               # so -b = +A x
        assert np.allclose(_Direct().solve_linear_problem(prob), x)
        print("ok test_solve_linear_problem_uses_minus_b")

    test_base_is_abstract()
    test_solve_and_scaling_agree()
    test_linearized_problem_splits_increment()
    test_solve_linear_problem_uses_minus_b()
    print("\nAll tests passed!")
