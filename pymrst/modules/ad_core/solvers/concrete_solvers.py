"""Concrete AD linear solvers (MRST ad-core/solvers).

MRST's `BackslashSolverAD.m` and `NoOpSolverAD.m` are ported in their own
files and re-exported here, with PyMRST's `GMRES_ILUSolverADAD` - the direct, no-op, and Krylov siblings of
`LinearSolverAD`. PyMRST already has `CPRSolver` (the scalable one) and a
`DirectSolver`; these complete the family on the MRST-shaped base so a model can
be handed any of them by class.

- `BackslashSolverAD` - a direct solve (scipy `spsolve`), MRST's ``A\\b``.
  Exact, the reference every iterative solver is checked against.
- `NoOpSolverAD` - returns a zero increment. Used to freeze a block, or as a
  placeholder in a composite/segregated solve.
- `GMRES_ILUSolverADAD` - restarted GMRES preconditioned by an incomplete LU
  factorisation, for systems too large for a direct factorisation but not
  needing CPR's two-stage structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.solvers.backslash_solver_ad import BackslashSolverAD
from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD
from pymrst.modules.ad_core.solvers.no_op_solver_ad import NoOpSolverAD


class GMRES_ILUSolverADAD(LinearSolverAD):
    """Restarted GMRES with an ILU preconditioner (MRST `GMRES_ILUSolverADAD`)."""

    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 200,
                 restart: int = 30, ilu_drop_tol: float = 1e-4, **kw):
        super().__init__(tolerance=tolerance, max_iterations=max_iterations,
                         **kw)
        self.restart = restart
        self.ilu_drop_tol = ilu_drop_tol
        # MRST's ILU knobs, under its own names, so `get_opts_ilu` reports the
        # settings actually in force rather than a second set that drifts.
        self.ilutype = "ilutp"
        self.modifiedIncompleteILU = False
        self.udiagReplacement = False
        # `None` means "SuperLU's own choice", and it is the default rather
        # than MATLAB's 1.0 deliberately. Setting 1.0 here *did* change the
        # factorisation - a 2.5e-16 drift in the SPE1 well rates, caught by
        # the linear-solver qualification graph - and a port that reports
        # MRST's option names must not silently retune the solver to reach
        # them. `get_opts_ilu` therefore reports what is in force, not what
        # MATLAB would have defaulted to.
        self.pivotThreshold = None

    def get_opts_ilu(self) -> dict:
        """The ILU settings, in MRST's `ilu` option layout.

        MATLAB's `ilu` takes these as a struct; SciPy's `spilu` takes separate
        keyword arguments and supports a subset. Reporting them in MRST's
        shape keeps a script that reads `getOptsILU` working, and
        `_scipy_ilu_kwargs` is where the translation - and the part SciPy
        cannot express - is made explicit rather than silently dropped.
        """
        return {"type": self.ilutype,
                "droptol": self.ilu_drop_tol,
                "milu": self.modifiedIncompleteILU,
                "udiag": self.udiagReplacement,
                "thresh": self.pivotThreshold}

    getOptsILU = get_opts_ilu

    def _scipy_ilu_kwargs(self) -> dict:
        """MRST's options translated to `spilu`'s, dropping what it lacks.

        `udiag` (replace zero diagonal entries) has no `spilu` equivalent and
        is not emulated: silently ignoring it would make a solver configured
        for a singular diagonal quietly behave as though it were not.
        """
        if self.udiagReplacement:
            raise NotImplementedError(
                "udiagReplacement has no scipy.sparse.linalg.spilu "
                "equivalent; MATLAB's ilu replaces zero diagonal entries and "
                "spilu offers no such option")
        kwargs = {"drop_tol": self.ilu_drop_tol}
        if self.ilutype == "ilutp" and self.pivotThreshold is not None:
            # `thresh` is MATLAB's pivot threshold on [0, 1]; spilu spells the
            # same idea `diag_pivot_thresh`. Passed only when set, so leaving
            # it alone leaves SuperLU's factorisation exactly as it was.
            kwargs["diag_pivot_thresh"] = self.pivotThreshold
        return kwargs

    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        A = A.tocsc()
        ilu = spla.spilu(A, **self._scipy_ilu_kwargs())
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
        return "GMRES_ILUSolverADAD (GMRES + ILU)"


class _IterCounter:
    def __init__(self):
        self.n = 0

    def __call__(self, *args):
        self.n += 1


__all__ = ["BackslashSolverAD", "GMRES_ILUSolverADAD", "NoOpSolverAD"]


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
        dx = GMRES_ILUSolverADAD(tolerance=1e-10).solve(A, b)
        assert np.allclose(dx, x, atol=1e-6), np.max(np.abs(dx - x))
        assert np.max(np.abs(A @ dx - b)) < 1e-6
        print("ok test_gmres_ilu_matches_direct (iters="
              f"{GMRES_ILUSolverADAD().last_iterations})")

    test_backslash_is_exact()
    test_noop_returns_zero()
    test_gmres_ilu_matches_direct()
    print("\nAll tests passed!")
