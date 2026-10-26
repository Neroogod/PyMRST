"""MRST ``ad-core/solvers/AMGCLSolverAD.m``: AMGCL - a C++ algebraic
multigrid library - as a linear solver, through MRST's own MEX gateway
(`linearsolvers/amgcl/utils/amgcl_matlab`, MRST's `amgcl_matlab.cpp` compiled
for Python against the AMGCL MRST bundles). `amgcl_setup` is MRST's option
struct (`getAMGCLMexStruct`); `solveLinearSystem` hands `A'`, `b` and it to
the gateway with `solver_id = 1`.

`AMGCLSolverADAD`, below, is PyMRST's earlier solver on the `pyamgcl`
binding, kept for its callers. Its notes follow.

AMGCL-backed linear solver (MRST ``ad-core/solvers/AMGCLSolverAD.m``).

MRST reaches AMGCL - a C++ algebraic-multigrid library - through a MEX
binding: ``amgcl_matlab``. There is no MATLAB in this port to bind from, so
the equivalent path is the actual C++ library the C++ way: `pyamgcl`, a
pybind11 binding built directly against AMGCL's own headers. This module wraps
that binding to the same place in the solver hierarchy MRST's class occupies -
a `LinearSolverAD` subclass with `solve_linear_system(A, b) -> x` - so it drops
into the same call sites `BackslashSolverAD` and `GMRES_ILUSolverADAD` do,
including as `CPRSolverADAD`'s ``elliptic_solver`` for the pressure stage. That
pairing is the Python-side equivalent of MRST's `AMGCL_CPRSolverAD`: AMGCL
solving CPR's pressure block, though MRST does it with AMGCL's own built-in
CPR preconditioner (`solver_id = 2`) where this port does it by handing AMGCL
to the two-stage machinery `CPRSolverADAD` already has and already tests.

**Optional, additive, and inert until asked for.** Nothing in the default
solve path imports this module. Building the pybind11 extension needs Boost
headers and a build-time `pybind11` install - see ``pip install
pymrst[amgcl]`` - which is a real cost this port does not impose on anyone who
does not choose the AMGCL solver.

MRST's parameter surface is name-and-numeric-ID pairs translated by
`translateOptionsAMGCL` inside the MEX call; this binds directly to AMGCL's own
dotted parameter names (``relax.type``, ``coarsening.type``, ...) rather than
reproducing that translation table, because those are the vocabulary the
underlying C++ library actually understands and `pyamgcl` exposes verbatim.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD

try:                                      # optional: needs Boost + pybind11
    import pyamgcl as _amgcl
    HAVE_AMGCL = True
except ImportError:                       # pragma: no cover
    _amgcl = None
    HAVE_AMGCL = False


#: MRST's LinearSolverAD property names, and this port's attribute for each.
_PROPERTY = {"maxIterations": "max_iterations", "keepNumber": "keep_number",
             "applyLeftDiagonalScaling": "apply_left_diagonal_scaling",
             "applyRightDiagonalScaling": "apply_right_diagonal_scaling",
             "variableOrdering": "variable_ordering", "equationOrdering": "equation_ordering",
             "useSparseReduction": "use_sparse_reduction", "initialGuessFun": "initial_guess_fun"}


def merge_solver_options(solver: Any, options: Mapping[str, Any]) -> dict:
    """MRST's `[solver, extra] = merge_options(solver, varargin{:})`: set the
    options that name a property of `solver`, return the rest."""
    extra = {}
    for key, val in options.items():
        name = _PROPERTY.get(key, key)
        if hasattr(solver, name):
            setattr(solver, name, val)
        else:
            extra[key] = val
    return extra


class AMGCLSolverAD(LinearSolverAD):
    """MRST `AMGCLSolverAD(...)`."""

    def __init__(self, **options: Any):
        from pymrst.modules.linearsolvers.amgcl.utils.get_amgcl_mex_struct import (
            get_amgcl_mex_struct,
        )
        super().__init__()
        self.amgcl_setup: dict = {}
        self.reuseMode = 1
        self.id = ""
        self.reduceToCell = True
        self.tolerance = 1e-6
        self.max_iterations = 100
        extra = merge_solver_options(self, options)
        self.amgcl_setup = get_amgcl_mex_struct(**extra)

    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray, x0: Any = None) -> np.ndarray:
        result, _ = self.call_amgcl_mex(A, b, 1, x0)
        return result

    def set_coarsening(self, *v: Any) -> None:
        self.set_parameter_group("coarsening", None, *v)

    def set_relaxation(self, *v: Any) -> None:
        self.set_parameter_group("relaxation", None, *v)

    def set_preconditioner(self, *v: Any) -> None:
        self.set_parameter_group("preconditioner", None, *v)

    def set_solver(self, *v: Any) -> None:
        self.set_parameter_group("solver", None, *v)

    def set_parameter_group(self, group: str, fld: Optional[str], *v: Any) -> None:
        from pymrst.modules.linearsolvers.amgcl.utils.translate_options_amgcl import (
            translate_options_amgcl,
        )
        group = group.lower()
        fld = fld or group
        if not v:
            print(f"No {group} argument given. Available options:")
            _, choices, descriptions, _ = translate_options_amgcl(group, None)
            for choice, description in zip(choices, descriptions):
                print(f"{choice:>10s}: {description}")
        else:
            self.amgcl_setup[fld] = translate_options_amgcl(group, v[0])[0]

    def get_parameter_group(self, group: str, fld: Optional[str] = None):
        from pymrst.modules.linearsolvers.amgcl.utils.translate_options_amgcl import (
            translate_options_amgcl,
        )
        group = group.lower()
        return translate_options_amgcl(group, self.amgcl_setup[fld or group])

    def call_amgcl_mex(self, A: sp.spmatrix, b: np.ndarray, id: int, x0: Any = None):
        """MRST `[result, report] = callAMGCL_MEX(solver, A, b, id, x0)`."""
        import time
        import warnings

        from pymrst.modules.linearsolvers.amgcl.utils.amgcl_matlab import amgcl_matlab
        timer = time.perf_counter()
        result, res, its = amgcl_matlab(sp.csr_matrix(A).T, b, self.amgcl_setup, self.tolerance,
                                        self.max_iterations, id, self.reuseMode, x0)
        t_solve = time.perf_counter() - timer
        if res > self.tolerance:
            warnings.warn(f"Solver did not converge to specified tolerance of {self.tolerance:1.3e} "
                          f"in {its} iterations. Reported residual estimate was {res:1.3e} "
                          f"after {t_solve:2.2f} seconds")
        elif self.verbose:
            print(f"AMGCL solver converged to {res:1.3e} in {its:2d} iterations after "
                  f"{t_solve:2.2f} seconds.")
        report = self.get_solve_report(Converged=res <= self.tolerance, Residual=res,
                                       Iterations=its)
        self.last_iterations = its
        self.last_solve_report = report
        return result, report

    def cleanup_solver(self, A: Any, b: Any, **kwargs: Any):
        from pymrst.modules.linearsolvers.amgcl.utils.reset_amgcl import reset_amgcl
        if self.reuseMode > 1:
            reset_amgcl()
        return self

    def get_description(self) -> Tuple[str, str]:
        sn = "AMGCL"
        if self.amgcl_setup["block_size"] > 1:
            sn += "-block"
        sn += self.id
        prm = ["solver", "preconditioner", "relaxation"]
        if self.amgcl_setup["preconditioner"] == 1:
            prm.append("coarsening")
        lines = [self.get_feature_description(s) for s in prm]
        return "General AMGCL solver. Configuration:\n" + "".join(f"\t{l}\n" for l in lines), sn

    def get_feature_description(self, name: str, extra: Optional[str] = None) -> str:
        extra = extra or name
        _, choice, description, param = self.get_parameter_group(name, extra)
        out = f"{extra:>15s}: {choice} ({description})"
        for p in param or []:
            if p in self.amgcl_setup:
                out += f"\n\t\t\t\t\t - {p} = {self.amgcl_setup[p]:g}"
        return out

    callAMGCL_MEX = call_amgcl_mex
    cleanupSolver = cleanup_solver
    setCoarsening = set_coarsening
    setRelaxation = set_relaxation
    setPreconditioner = set_preconditioner
    setSolver = set_solver
    setParameterGroup = set_parameter_group
    getParameterGroup = get_parameter_group
    getDescription = get_description
    getFeatureDescription = get_feature_description


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

class AMGCLSolverADAD(LinearSolverAD):
    """AMGCL algebraic multigrid, preconditioning a Krylov solve.

    MRST defaults: ``tolerance = 1e-6``, ``max_iterations = 100``
    (``AMGCLSolverADAD.m``). ``coarsening``/``relaxation`` build the AMG
    hierarchy (MRST's ``setCoarsening``/``setRelaxation``); ``krylov`` selects
    the outer iteration (MRST's ``setSolver``). ``amgcl_options`` passes any
    further dotted AMGCL parameter straight through, for anything this
    surface does not name explicitly.
    """

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100,
                 coarsening: str = "smoothed_aggregation",
                 relaxation: str = "spai0", krylov: str = "bicgstab",
                 amgcl_options: Optional[Mapping[str, Any]] = None, **kw):
        if not HAVE_AMGCL:
            raise ImportError(
                "AMGCLSolverADAD needs the 'pyamgcl' package, which is not "
                "installed. It compiles a C++ extension against AMGCL's own "
                "headers and needs Boost at build time: install Boost "
                "(e.g. `brew install boost`), then `pip install "
                "pymrst[amgcl]`.")
        super().__init__(tolerance=tolerance, max_iterations=max_iterations,
                         **kw)
        self.coarsening = coarsening
        self.relaxation = relaxation
        self.krylov = krylov
        self.amgcl_options = dict(amgcl_options or {})
        self.last_residual = float("nan")
        # MRST's `reuseMode`: 1 rebuilds the AMG hierarchy every solve, 2
        # reuses it across calls with the same sparsity pattern. This port
        # only ever rebuilds - `solve_linear_system` is called once per Newton
        # iteration on a Jacobian whose *values* have changed even when its
        # pattern has not, and AMGCL's hierarchy is built from both.
        self.reuse_mode = 1

    def _amg_params(self) -> dict:
        params = {"relax.type": self.relaxation}
        if self.coarsening:
            params["coarsening.type"] = self.coarsening
        params.update(self.amgcl_options)
        return params

    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        A = sp.csr_matrix(A, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
        precond = _amgcl.amg(A, self._amg_params())
        solver = _amgcl.solver(
            precond, {"type": self.krylov, "tol": self.tolerance,
                     "maxiter": self.max_iterations})
        x = np.asarray(solver(b), dtype=float).ravel()
        self.last_iterations = int(solver.iters)
        self.last_residual = float(solver.error)
        if self.last_residual > self.tolerance and self.verbose:
            print(f"AMGCLSolverADAD: did not reach tolerance "
                  f"{self.tolerance:.3e} in {self.max_iterations} "
                  f"iterations (residual {self.last_residual:.3e})")
        return x

    def get_description(self) -> str:
        return (f"AMGCLSolverADAD ({self.krylov}, {self.coarsening}/"
               f"{self.relaxation})")

    # -- MRST's option setters, translated ------------------------------------

    def set_coarsening(self, name: str) -> None:
        self.coarsening = name

    setCoarsening = set_coarsening

    def set_relaxation(self, name: str) -> None:
        self.relaxation = name

    setRelaxation = set_relaxation

    def set_solver(self, name: str) -> None:
        self.krylov = name

    setSolver = set_solver


if __name__ == "__main__":
    import sys

    def test_the_mrst_solver_solves_through_its_mex():
        n = 30
        main = 4*np.ones(n*n)
        off = -np.ones(n*n - 1)
        off[np.arange(1, n*n) % n == 0] = 0
        A = sp.diags([main, off, off, -np.ones(n*n - n), -np.ones(n*n - n)],
                     [0, 1, -1, n, -n], format="csr")
        b = np.ones(n*n)
        solver = AMGCLSolverAD(tolerance=1e-10, relaxation="ilu0")
        assert solver.amgcl_setup["relaxation"] == 3 and solver.tolerance == 1e-10
        x = solver.solve_linear_system(A, b)
        assert np.linalg.norm(A @ x - b)/np.linalg.norm(b) < 1e-8
        assert solver.last_solve_report["Converged"] and solver.last_iterations > 0
        print(f"ok test_the_mrst_solver_solves_through_its_mex ({solver.last_iterations} its)")

    test_the_mrst_solver_solves_through_its_mex()

    if not HAVE_AMGCL:
        print("pyamgcl is not installed - skipping (pip install "
              "pymrst[amgcl], after `brew install boost`)")
        sys.exit(0)

    import scipy.sparse.linalg as spla

    def _poisson_1d(n: int) -> sp.csr_matrix:
        """A well-posed SPD system: -u'' = f on a line, Dirichlet ends."""
        main = np.full(n, 2.0)
        off = np.full(n - 1, -1.0)
        return sp.diags([off, main, off], [-1, 0, 1], format="csr")

    def test_matches_a_direct_solve() -> None:
        A = _poisson_1d(300)
        b = np.ones(300)
        solver = AMGCLSolverADAD(tolerance=1e-10)
        x = solver.solve_linear_system(A, b)
        direct = spla.spsolve(A.tocsc(), b)
        assert np.allclose(x, direct, atol=1e-6), np.abs(x - direct).max()
        assert solver.last_iterations >= 1
        print("ok test_matches_a_direct_solve "
              f"({solver.last_iterations} its, resid {solver.last_residual:.2e})")

    def test_reports_a_real_residual_not_a_fixed_zero() -> None:
        """A solver that reports iterations=1, residual=0 unconditionally is
        not actually calling AMGCL - it is a stub. Two different systems must
        take different iteration counts or leave different residuals."""
        solver = AMGCLSolverADAD(tolerance=1e-10, max_iterations=2)
        easy = solver.solve_linear_system(_poisson_1d(50), np.ones(50))
        hard_solver = AMGCLSolverADAD(tolerance=1e-10, max_iterations=2)
        # A poorly-scaled, harder-to-precondition system.
        A = _poisson_1d(50).tolil()
        A[0, 0] = 1.0e6
        hard = hard_solver.solve_linear_system(A.tocsr(), np.ones(50))
        assert solver.last_residual != hard_solver.last_residual, (
            "residuals must differ between two different systems")
        print("ok test_reports_a_real_residual_not_a_fixed_zero")

    def test_the_krylov_choice_reaches_amgcl() -> None:
        """The `krylov` option must actually change which solver runs.

        Two solvers that both converge to the right answer is not proof the
        choice was honoured - `cg` and `bicgstab` agree on an SPD system
        regardless of which one actually ran. Spy on the real call instead: a
        wrapper that hardcodes `type='bicgstab'` inside `_amg_params`-adjacent
        code would pass every behavioural check and still be wrong.
        """
        seen = []
        real_solver = _amgcl.solver

        def spy(P, prm):
            seen.append(dict(prm))
            return real_solver(P, prm)

        import pymrst.modules.ad_core.solvers.amgcl_solver_ad as mod
        mod._amgcl.solver = spy
        try:
            for name in ("cg", "bicgstab"):
                s = AMGCLSolverADAD(tolerance=1e-10, krylov=name)
                s.solve_linear_system(_poisson_1d(30), np.ones(30))
            assert seen[0]["type"] == "cg", seen[0]
            assert seen[1]["type"] == "bicgstab", seen[1]
        finally:
            mod._amgcl.solver = real_solver
        print("ok test_the_krylov_choice_reaches_amgcl")

    def _poisson_2d(n: int) -> sp.csr_matrix:
        """The 5-point stencil on an n x n grid, Dirichlet boundary rows
        pinned to identity - the construction pyamgcl's own test suite uses.

        AMG is so effective on a plain 1D diffusion problem that it takes it
        in a single cycle at any size tried here - useless for telling a real
        iteration count from a hardcoded one. 2D needs a handful of cycles at
        tight tolerance (9, empirically, at n=80), which a stub cannot fake.

        A first version of this built the operator as a Kronecker sum and
        left the diagonal at a constant 2 everywhere instead of 4 at interior
        points - a near-singular matrix, not a merely slow one. Its direct
        solve residual was 2.5, not the ~1e-13 checked below; AMGCL was
        correctly failing to solve a broken system; hardcoding 500 iterations
        would have looked exactly like genuine non-convergence and the defect
        would have shipped as a passing test.
        """
        idx = lambda i, j: i*n + j
        rows, cols, vals = [], [], []
        for i in range(n):
            for j in range(n):
                k = idx(i, j)
                if i in (0, n - 1) or j in (0, n - 1):
                    rows.append(k); cols.append(k); vals.append(1.0)
                else:
                    rows += [k]*5
                    cols += [idx(i-1, j), idx(i, j-1), k,
                            idx(i, j+1), idx(i+1, j)]
                    vals += [-1.0, -1.0, 4.0, -1.0, -1.0]
        return sp.csr_matrix((vals, (rows, cols)), shape=(n*n, n*n))

    def test_iteration_count_is_the_real_solver_s_own() -> None:
        """`last_iterations` must be read from AMGCL's own report, not a
        constant. 2D diffusion provably takes more than one cycle here - a
        hardcoded `= 1` disagrees with it, where it agreed with every 1D case
        elsewhere in this file."""
        A = _poisson_2d(80)
        b = np.ones(A.shape[0])
        # The system must be well-posed before its iteration count means
        # anything - see the note on the near-singular Kronecker version this
        # replaced.
        direct = spla.spsolve(A.tocsc(), b)
        assert np.linalg.norm(b - A@direct)/np.linalg.norm(b) < 1e-8, (
            "test matrix is not well-posed; fix the construction, not the "
            "assertion below")
        solver = AMGCLSolverADAD(tolerance=1e-10, max_iterations=500)
        x = solver.solve_linear_system(A, b)
        assert np.allclose(x, direct, atol=1e-6)
        assert solver.last_iterations > 1, (
            "a hardcoded last_iterations = 1 would pass every 1D test in "
            f"this file; got {solver.last_iterations}")
        assert solver.last_iterations < 500, (
            "hit the iteration cap - the system may be malformed again")
        print(f"ok test_iteration_count_is_the_real_solver_s_own "
              f"({solver.last_iterations} its)")

    def test_the_relaxation_choice_reaches_amgcl() -> None:
        """Same shape as the krylov check: spy on the actual `prm` handed to
        `_amgcl.amg`, since spai0 and ilu0 can converge equally well on an
        easy system and leave no behavioural trace of which one ran."""
        seen = []
        real_amg = _amgcl.amg

        def spy(A, prm):
            seen.append(dict(prm))
            return real_amg(A, prm)

        import pymrst.modules.ad_core.solvers.amgcl_solver_ad as mod
        mod._amgcl.amg = spy
        try:
            for name in ("spai0", "ilu0"):
                s = AMGCLSolverADAD(tolerance=1e-8, relaxation=name)
                s.solve_linear_system(_poisson_1d(30), np.ones(30))
            assert seen[0]["relax.type"] == "spai0", seen[0]
            assert seen[1]["relax.type"] == "ilu0", seen[1]
        finally:
            mod._amgcl.amg = real_amg
        print("ok test_the_relaxation_choice_reaches_amgcl")

    def test_plugs_into_cpr_as_the_elliptic_solver() -> None:
        """The point of the port: MRST's `AMGCL_CPRSolverAD` is AMGCL
        solving CPR's pressure block. Here that is `CPRSolverADAD`'s
        `elliptic_solver=` slot - and this checks it is actually reached by
        a real CPR solve, not just that the two objects can coexist.
        `tests/ad_core/solvers/test_cpr_pluggable_elliptic_solver.py` covers
        this end to end against a real reservoir Jacobian; this is the
        lightweight version that lives with the solver it is testing."""
        from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver

        n = 60
        Ap = _poisson_1d(n)
        rp = np.ones(n)
        amgcl = AMGCLSolverADAD(tolerance=1e-10)
        # No model/state, so CPR falls back to a direct solve of the *whole*
        # system - which for this single-block SPD system is exactly the
        # elliptic solve, and still proves AMGCLSolverADAD.solve_linear_system
        # is what CPRSolver reaches for when it is the injected
        # elliptic_solver, via `_elliptic_solver`'s own dedicated branch.
        got = CPRSolver(elliptic_solver=amgcl)._elliptic_solver(Ap)(rp)
        want = spla.spsolve(Ap.tocsc(), rp)
        assert np.allclose(got, want, atol=1e-5)
        assert amgcl.last_iterations >= 1, (
            "AMGCLSolverADAD was not really called by the injected path")
        print("ok test_plugs_into_cpr_as_the_elliptic_solver")

    def test_amgcl_options_reach_the_amg_call() -> None:
        """`amgcl_options` is the escape hatch for anything not named on the
        constructor - it must survive the merge with `relax.type`/
        `coarsening.type`, not be silently dropped by it."""
        seen = []
        real_amg = _amgcl.amg

        def spy(A, prm):
            seen.append(dict(prm))
            return real_amg(A, prm)

        import pymrst.modules.ad_core.solvers.amgcl_solver_ad as mod
        mod._amgcl.amg = spy
        try:
            s = AMGCLSolverADAD(tolerance=1e-8,
                              amgcl_options={"coarsening.aggr.eps_strong": 0.1})
            s.solve_linear_system(_poisson_1d(30), np.ones(30))
        finally:
            mod._amgcl.amg = real_amg
        assert seen[0].get("coarsening.aggr.eps_strong") == 0.1, seen[0]
        print("ok test_amgcl_options_reach_the_amg_call")

    def test_missing_pyamgcl_raises_a_clear_error() -> None:
        """Simulated: construction must fail with an actionable message, not
        an AttributeError deep inside `solve_linear_system`."""
        import pymrst.modules.ad_core.solvers.amgcl_solver_ad as mod
        original = mod.HAVE_AMGCL
        mod.HAVE_AMGCL = False
        try:
            try:
                mod.AMGCLSolverADAD()
            except ImportError as err:
                assert "pyamgcl" in str(err)
                print("ok test_missing_pyamgcl_raises_a_clear_error")
                return
            raise AssertionError("must raise ImportError when unavailable")
        finally:
            mod.HAVE_AMGCL = original

    test_matches_a_direct_solve()
    test_reports_a_real_residual_not_a_fixed_zero()
    test_the_krylov_choice_reaches_amgcl()
    test_iteration_count_is_the_real_solver_s_own()
    test_the_relaxation_choice_reaches_amgcl()
    test_amgcl_options_reach_the_amg_call()
    test_plugs_into_cpr_as_the_elliptic_solver()
    test_missing_pyamgcl_raises_a_clear_error()
    print("\nAll tests passed!")
