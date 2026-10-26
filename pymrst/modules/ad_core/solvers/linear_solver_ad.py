"""Linear-solver base (MRST ad-core/solvers).

Ports MRST's `LinearSolverAD.m`, the base contract every AD linear solver
programs against. MRST's `LinearizedProblem.m` is ported in full in
`linearized_problem.py`; the solvers here take it, with MRST's `b = -residual`.
`LinearizedProblemAD`, below, is PyMRST's earlier problem class, kept for its
callers: it is built from an assembled `(A, b)` with `b` the residual itself,
and the solvers recognise it and negate.

PyMRST's concrete solvers (`CPRSolver`, `DirectSolver`) already expose
``solve(jac, rhs, model)``. This adds the MRST-shaped base so those and the new
siblings (`BackslashSolverAD`, `NoOpSolverAD`, `GMRES_ILUSolverADAD`) share one
interface: subclasses implement ``solve_linear_system(A, b)``, and the base
provides the ``solve(jac, rhs, model)`` entry the model driver calls, an
optional diagonal scaling, and a solve report. `LinearizedProblem` additionally
splits a solved increment back into the per-variable blocks the model updates.
"""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


from pymrst.modules.ad_core.solvers.linearized_problem import LinearizedProblem


def _sparse_block(block):
    return block.to_sparse() if hasattr(block, "to_sparse") else block


def _assembled(problem):
    """The assembled problem: MRST's returns a copy, the earlier class
    assembles itself."""
    out = problem.assemble_system()
    return problem if out is None else out


def _rhs(problem) -> np.ndarray:
    """MRST's right-hand side (minus the residual) of an assembled problem.
    `LinearizedProblemAD` stores the residual itself."""
    b = np.asarray(problem.b, dtype=float)
    return -b if isinstance(problem, LinearizedProblemAD) else b


def _set_rhs(problem, b: np.ndarray) -> None:
    problem.b = -b if isinstance(problem, LinearizedProblemAD) else b


class LinearizedProblemAD:
    """A linearized Newton system ``A dx = -b`` with its variable layout.

    ``var_names`` / ``var_sizes`` describe the primary-variable blocks the
    columns of ``A`` (and entries of ``b``) are ordered by, so a solved
    increment can be split back into named pieces via `store_increments`.
    """

    def __init__(self, A: Optional[sp.spmatrix] = None,
                 b: Optional[np.ndarray] = None,
                 var_names: Optional[Sequence[str]] = None,
                 var_sizes: Optional[Sequence[int]] = None,
                 state: Any = None, dt: Optional[float] = None):
        self.A = A
        self.b = (np.asarray(b, dtype=float).ravel() if b is not None
                  else np.zeros(0))
        self.var_names = list(var_names) if var_names is not None else None
        self.var_sizes = list(var_sizes) if var_sizes is not None else None
        self.num_vars: Optional[List[int]] = None     # MRST's getNumVars
        self.state = state
        self.dt = dt
        # MRST's equation-oriented view (Phase 2). Empty for a problem built
        # straight from an assembled `(A, b)`.
        self.equations: List[Any] = []
        self.types: List[str] = []
        self.equation_names: List[str] = []
        self.primary_variables: List[str] = list(var_names or [])
        self.iteration_no: Optional[int] = None
        self.driving_forces: Any = None

    # -- MRST's equation-oriented construction ------------------------------
    #
    # MRST's `LinearizedProblem` holds the *equations* - AD residuals, each
    # carrying a Jacobian block per primary variable - and assembles `(A, b)`
    # on demand. PyMRST's took `(A, b)` already assembled. Both are useful, so
    # both exist, but they are different enough that overloading one
    # constructor on argument types would be a trap: `setup_linearized_problem`
    # did exactly that by accident and passed MRST's argument order into the
    # matrix constructor, putting equations in `A` and type strings in `b`.
    # It raised on the first call, and nothing called it. Hence a named
    # alternative constructor rather than type sniffing.

    @classmethod
    def from_equations(cls, equations, types, names, primary_vars,
                       state: Any = None, dt: float = 0.0
                       ) -> "LinearizedProblem":
        """MRST's `LinearizedProblem(eqs, types, names, primaryVars, ...)`."""
        equations, types, names = cls.check_inputs(equations, types, names)
        problem = cls(None, None, None, None, state, dt)
        problem.equations = list(equations)
        problem.types = list(types)
        problem.equation_names = list(names)
        problem.primary_variables = ([primary_vars]
                                     if isinstance(primary_vars, str)
                                     else list(primary_vars))
        problem.var_names = list(problem.primary_variables)
        sizes = state.get("primaryVariableSizes") if isinstance(state, dict) else None
        problem.num_vars = (list(sizes) if sizes is not None
                            and len(sizes) == len(problem.primary_variables) else None)
        return problem

    @staticmethod
    def check_inputs(equations, types, names):
        """Wrap a single equation, and refuse a mismatched set.

        A count mismatch here becomes a silent misalignment later - equation
        three reported under equation four's name - so it is caught at
        construction.
        """
        if not isinstance(equations, (list, tuple)):
            return [equations], [types], [names]
        equations, types, names = list(equations), list(types), list(names)
        if not (len(types) == len(names) == len(equations)):
            raise ValueError(
                f"inconsistent counts: {len(equations)} equations, "
                f"{len(types)} types, {len(names)} names")
        return equations, types, names

    checkInputs = check_inputs

    def num_eq(self) -> int:
        return len(self.equations)

    def norm(self, *args) -> np.ndarray:
        """MRST `norm(problem, ...)`: the norm of each equation's value."""
        from pymrst.utils.adi import value as _value
        return np.array([np.linalg.norm(np.ravel(_value(eq)), *args)
                         if np.size(_value(eq)) else 0.0
                         for eq in self.equations])

    numeq = num_eq

    def assemble_system(self) -> "LinearizedProblem":
        """Build `(A, b)` from the equations' Jacobian blocks.

        Block row `i`, block column `j` is equation `i`'s Jacobian with respect
        to primary variable `j`, which is the layout `store_increments` splits
        back apart.
        """
        if self.A is not None and self.b.size:
            return self
        if not self.equations:
            raise ValueError("problem carries no equations to assemble")
        # A diagonal-backend Jacobian block is converted the way MRST's
        # `sparse(...)` converts a DiagonalJacobian.
        blocks = [[_sparse_block(eq.jac[j]) for j in range(len(eq.jac))]
                  for eq in self.equations]
        self.A = sp.bmat(blocks, format="csr")
        self.b = np.concatenate(
            [np.asarray(eq.val, dtype=float).ravel() for eq in self.equations])
        self.var_sizes = [np.asarray(eq.val).size for eq in self.equations]
        return self

    assembleSystem = assemble_system

    def get_linear_system(self):
        self.assemble_system()
        return self.A, self.b

    getLinearSystem = get_linear_system

    def clear_system(self) -> "LinearizedProblem":
        """Drop the assembled matrix, so the next request rebuilds it.

        Called by everything that changes the equations. An assembled `A` that
        outlives the equations it came from is the worst possible state: it
        still solves, and it solves the previous problem.
        """
        self.A = None
        self.b = np.zeros(0)
        return self

    clearSystem = clear_system

    def append_equations(self, equations, types, names) -> "LinearizedProblem":
        equations, types, names = self.check_inputs(equations, types, names)
        self.equations += list(equations)
        self.types += list(types)
        self.equation_names += list(names)
        return self.clear_system()

    appendEquations = append_equations

    def prepend_equations(self, equations, types, names) -> "LinearizedProblem":
        equations, types, names = self.check_inputs(equations, types, names)
        self.equations = list(equations) + self.equations
        self.types = list(types) + self.types
        self.equation_names = list(names) + self.equation_names
        return self.clear_system()

    prependEquations = prepend_equations

    def count_of_type(self, name: str) -> int:
        return sum(1 for t in self.types if str(t).lower() == str(name).lower())

    countOfType = count_of_type

    def index_of_equation_name(self, name: str) -> Optional[int]:
        lowered = [str(n).lower() for n in self.equation_names]
        key = str(name).lower()
        return lowered.index(key) if key in lowered else None

    indexOfEquationName = index_of_equation_name

    def index_of_primary_variable(self, name: str) -> Optional[int]:
        lowered = [str(n).lower() for n in self.primary_variables]
        key = str(name).lower()
        return lowered.index(key) if key in lowered else None

    indexOfPrimaryVariable = index_of_primary_variable

    def get_equation_var_num(self, index=None):
        """How many rows each equation contributes."""
        equations = (self.equations if index is None
                     else [self.equations[i] for i in np.atleast_1d(index)])
        return np.array([np.asarray(eq.val).size for eq in equations])

    getEquationVarNum = get_equation_var_num

    def reorder_equations(self, order: Sequence[int]) -> "LinearizedProblem":
        order = list(order)
        if len(order) != self.num_eq():
            raise ValueError(
                f"{len(order)} indices for {self.num_eq()} equations")
        self.equations = [self.equations[i] for i in order]
        if self.types:
            self.types = [self.types[i] for i in order]
        if self.equation_names:
            self.equation_names = [self.equation_names[i] for i in order]
        return self.clear_system()

    reorderEquations = reorder_equations

    def pop_increment(self, dx, name: str, nargout: int = 2):
        """Take one named block out of an increment list.

        Returns `(the block, the rest)`. The block is `None` when the variable
        is not present, which is how a caller asks "did this problem solve for
        that?" without knowing the layout. With `nargout=3`, as MRST's
        `[dvar, dx, problem] = popIncrement(...)`, also a copy of the problem
        without that primary variable.
        """
        index = self.index_of_primary_variable(name)
        if index is None:
            return (None, dx) if nargout < 3 else (None, dx, self)
        rest = [d for i, d in enumerate(dx) if i != index]
        if nargout < 3:
            return dx[index], rest
        from copy import copy
        problem = copy(self)
        keep = [i for i in range(len(self.primary_variables)) if i != index]
        problem.primary_variables = [self.primary_variables[i] for i in keep]
        if getattr(self, "num_vars", None) is not None:
            problem.num_vars = [self.num_vars[i] for i in keep]
        return dx[index], rest, problem

    popIncrement = pop_increment

    def index_of_type(self, name: str) -> np.ndarray:
        """Boolean mask of the equations of a given type (MRST `indexOfType`).

        Case-insensitive, as MRST's `strcmpi` is. Returned as a mask rather
        than indices because every caller uses it to split the system in two,
        and a mask makes both halves and their complement one expression.
        """
        key = str(name).lower()
        return np.array([str(t).lower() == key for t in self.types], dtype=bool)

    indexOfType = index_of_type

    def reduce_to_single_variable_type(self, type_name: str):
        """Eliminate every equation not of `type_name`. Returns `(self, elim)`.

        The usual call is `type_name='cell'`: the well and control equations
        are Schur-complemented into the reservoir equations, leaving a system
        the size of the grid. That is exact, so the cell increments are the
        same as a full solve - the wells are recovered afterwards from
        `recover_from_single_variable_type`.
        """
        keep = self.index_of_type(type_name)
        self.clear_system()
        eliminated = []
        for name in [n for n, k in zip(list(self.equation_names), keep)
                     if not k]:
            eliminated.append(self.eliminate_variable(name))
        return self, eliminated

    reduceToSingleVariableType = reduce_to_single_variable_type

    @staticmethod
    def recover_from_single_variable_type(reduced_problem, original_problem,
                                          increments_reduced, eliminated):
        """Rebuild the full increment list after a single-type reduction.

        Two things are load-bearing and both are MRST's.

        The eliminated blocks are recovered **in reverse**: each elimination
        was formed against the ones already removed, so recovering forwards
        would use a Schur complement that has not been applied yet.

        And the kept variables must be *contiguous and first*. MRST asserts
        this, and so does this: `recover_vars` addresses the eliminated
        equation's Jacobian by position, so a kept block appearing after an
        eliminated one would be read from the wrong column. The assertion is
        cheap and the failure it prevents is a silently mismatched increment.
        """
        from pymrst.modules.ad_core.utils.recover_vars import recover_vars

        type_name = reduced_problem.types[0] if reduced_problem.types else "cell"
        current = original_problem.index_of_type(type_name)
        cell_index = np.flatnonzero(current)
        not_cell_index = np.flatnonzero(~current)

        if cell_index.size and np.any(np.diff(cell_index) != 1):
            raise ValueError(
                "the kept variables must be contiguous; got positions "
                f"{cell_index.tolist()}")
        if cell_index.size and cell_index[0] != 0:
            raise ValueError(
                "the kept variables must come first in the ordering; they "
                f"start at position {int(cell_index[0])}")

        dx: List[Any] = [None]*len(current)
        for slot, value in zip(cell_index, increments_reduced):
            dx[slot] = value
        n_cell = cell_index.size
        for i in range(len(eliminated) - 1, -1, -1):
            position = int(not_cell_index[i])
            # MRST `dx(recovered)`: the known blocks in position order - the
            # cells, then the blocks eliminated after this one.
            known = [dx[k] for k in range(len(dx)) if dx[k] is not None]
            dx[position] = recover_vars(eliminated[i], n_cell, known)
        return dx

    recoverFromSingleVariableType = recover_from_single_variable_type

    def eliminate_variable(self, variable) -> Any:
        """Schur-complement one equation out of the system.

        The remaining equations absorb the eliminated one exactly:

            J_ij <- J_ij - J_in * (J_nn \\ J_nj)
            r_i  <- r_i  - J_in * (J_nn \\ r_n)

        so the reduced system has the same solution for the variables it keeps.
        That is the property worth testing, and it is exact rather than
        approximate - this is a rearrangement, not a preconditioner.
        """
        import scipy.sparse.linalg as spla

        if isinstance(variable, str):
            n = self.index_of_equation_name(variable)
            if n is None:
                raise KeyError(f"no equation named {variable!r}")
        else:
            n = int(variable)

        # MATLAB eliminates on value copies: the AD objects may be shared with
        # the caller's problem (a solver keeps the original to recover the
        # eliminated increments), so each is copied before it is changed.
        equations = [type(eq)(np.array(eq.val, copy=True), list(eq.jac))
                     for eq in self.equations]
        eliminated = equations[n]
        keep = [i for i in range(len(equations)) if i != n]

        pivot = sp.csc_matrix(eliminated.jac[n])
        solve = spla.factorized(pivot)

        def _apply(rhs):
            rhs = sp.csc_matrix(rhs)
            return np.column_stack(
                [solve(np.asarray(rhs[:, c].todense()).ravel())
                 for c in range(rhs.shape[1])]) if rhs.shape[1] else rhs

        for i in keep:
            eq = equations[i]
            for j in keep:
                if eq.jac[j] is None or eliminated.jac[j] is None:
                    continue
                correction = eq.jac[n] @ sp.csr_matrix(
                    _apply(eliminated.jac[j]))
                eq.jac[j] = eq.jac[j] - correction
            eq.val = eq.val - eq.jac[n] @ solve(
                np.asarray(eliminated.val, dtype=float).ravel())

        for i in keep:
            equations[i].jac = [equations[i].jac[j] for j in keep]

        self.equations = [equations[i] for i in keep]
        self.equation_names = [self.equation_names[i] for i in keep]
        self.types = [self.types[i] for i in keep]
        self.primary_variables = [self.primary_variables[i] for i in keep]
        self.var_names = list(self.primary_variables)
        if getattr(self, "num_vars", None) is not None:
            self.num_vars = [self.num_vars[i] for i in keep]
        self.clear_system()
        return eliminated

    eliminateVariable = eliminate_variable

    def num_dof(self) -> int:
        if self.b.size:
            return self.b.size
        return int(sum(np.asarray(eq.val).size for eq in self.equations))

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
        # MRST LinearSolverAD's preparation machinery (Phase 2). All inert at
        # these defaults: no reduction, no diagonal scaling, no reordering, so
        # a solver that ignores them behaves exactly as it did.
        self.keep_number = None          # rows kept by the Schur reduction
        self.use_sparse_reduction = False
        self.apply_left_diagonal_scaling = False
        self.apply_right_diagonal_scaling = False
        self.variable_ordering = None
        self.equation_ordering = None
        self.initial_guess_fun = None
        # MRST LinearSolverAD properties read by `solve_linear_problem`.
        self.reduceToCell = False
        self.replaceNaN = False
        self.replacementNaN = 0.0
        self.replaceInf = False
        self.replacementInf = 0.0

    # -- MRST LinearSolverAD.m preparation/recovery, ported (Phase 2) -------
    #
    # A production linear solve is rarely `A \ b`. It is: reduce the system to
    # the block you can solve well, scale it so the solver sees sensible
    # magnitudes, reorder it to suit the preconditioner, solve, then undo all
    # three in reverse. Each step has an exact inverse, and getting one of
    # those inverses wrong produces an answer that is close enough to look
    # plausible - which is why they are tested as exact round trips.

    def get_diagonal_inverse(self, A) -> sp.spmatrix:
        """`diag(1/|a_ii|)`, with a zero diagonal entry left as 1.

        A zero on the diagonal gives an infinite scale factor, which turns one
        bad row into an entirely NaN system. MRST substitutes 1 - the row is
        then unscaled rather than destroyed.
        """
        A = sp.csr_matrix(A)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"matrix must be square, got {A.shape}")
        n = A.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            d = 1.0/np.abs(A.diagonal())
        d[~np.isfinite(d)] = 1.0
        return sp.diags(d, format="csr")

    getDiagonalInverse = get_diagonal_inverse

    def apply_diagonal_scaling(self, A, b, x0=None):
        """Row- or column-scale the system. Returns `(A, b, scaling, x0)`.

        Named `apply_diagonal_scaling` rather than MRST's `applyScaling`
        because this class already carries an `apply_scaling` *attribute* - a
        boolean for PyMRST's own scaling path - and a method of that name would
        be shadowed by it on every instance. `applyScaling` is aliased below,
        so MRST's spelling still resolves.
        """
        scaling: Dict[str, Any] = {}
        left, right = (self.apply_left_diagonal_scaling,
                       self.apply_right_diagonal_scaling)
        if not (left or right):
            return A, b, scaling, x0
        if left and right:
            raise ValueError(
                "cannot apply both left and right diagonal scaling")

        M = self.get_diagonal_inverse(A)
        if left:
            A, b = M @ A, M @ b
        else:
            A = A @ M
            if x0 is not None:
                x0 = spla.spsolve(sp.csc_matrix(M), x0)
        scaling["M"] = M
        return A, b, scaling, x0

    applyScaling = apply_diagonal_scaling

    def undo_scaling(self, x, scaling):
        """Map a solution back through right scaling.

        Only right scaling changes what the unknown *means* - it solves for
        `M^-1 x`, so the answer must be multiplied back. Left scaling multiplies
        both sides and leaves the unknown alone, which is why undoing it would
        be wrong.
        """
        if self.apply_right_diagonal_scaling and scaling.get("M") is not None:
            return scaling["M"] @ x
        return x

    undoScaling = undo_scaling

    def reduce_linear_system(self, A, b, is_adjoint: bool = False, x0=None):
        """Schur-complement the system down to its first `keep_number` rows.

            A' = B - C E^-1 D        b' = f - C E^-1 h

        Exact, not approximate: solving the reduced system gives the same first
        `keep_number` entries as solving the whole thing. `recover_linear_system`
        reconstructs the rest.
        """
        keep = self.keep_number
        system: Dict[str, Any] = {}
        if keep is None or keep >= A.shape[0]:
            return A, b, system, x0

        A = sp.csr_matrix(A)
        B, C = A[:keep, :keep], A[:keep, keep:]
        D, E = A[keep:, :keep], A[keep:, keep:]
        f, h = b[:keep], b[keep:]
        solve_E = spla.factorized(sp.csc_matrix(E))

        # Only the columns of D that couple to E (the well cells) give a
        # nonzero E^-1 D column; an empty one solves to exact zeros, so it
        # is skipped rather than solved once per reservoir unknown.
        D_csc = sp.csc_matrix(D)
        coupled = np.flatnonzero(np.diff(D_csc.indptr))
        E_inv_D = np.zeros((E.shape[0], D.shape[1]))
        for j in coupled:
            E_inv_D[:, j] = solve_E(D_csc[:, [j]].toarray().ravel())
        reduced_A = B - C @ sp.csr_matrix(E_inv_D)
        if is_adjoint:
            reduced_b = f - D.T @ spla.spsolve(sp.csc_matrix(E).T, h)
        else:
            reduced_b = f - C @ solve_E(h)

        system.update(B=B, C=C, D=D, E=E, f=f, h=h, solve_E=solve_E)
        if x0 is not None:
            x0 = x0[:keep]
        return reduced_A, reduced_b, system, x0

    reduceLinearSystem = reduce_linear_system

    def recover_linear_system(self, x, system):
        """Rebuild the eliminated block: `s = E^-1 (h - D x)`."""
        if not system or system.get("solve_E") is None:
            return x
        s = system["solve_E"](system["h"] - system["D"] @ x)
        return np.concatenate([x, s])

    recoverLinearSystem = recover_linear_system

    def reorder_linear_system(self, A, b, order=None, x0=None):
        """Permute rows and/or columns, padding a short ordering.

        A permutation shorter than the system leaves the tail in place rather
        than truncating it - MRST's behaviour, and the safe one: a truncated
        ordering would silently drop equations.
        """
        n = A.shape[0]

        def _prepare(spec):
            if spec is None:
                return None
            if callable(spec):
                spec = spec(A, b)
            spec = np.asarray(spec, dtype=np.int64).ravel()
            if spec.size < n:
                spec = np.concatenate([spec, np.arange(spec.size, n)])
            return spec[:n]

        variables = _prepare(self.variable_ordering)
        equations = _prepare(self.equation_ordering)
        if equations is None and self.equation_ordering is None and \
                variables is not None and self.equation_ordering is not None:
            equations = variables

        if variables is not None and equations is not None:
            A, b = A[equations, :][:, variables], b[equations]
        elif variables is not None:
            A = A[:, variables]
        elif equations is not None:
            A, b = A[equations, :], b[equations]
        if x0 is not None and variables is not None:
            x0 = x0[variables]
        return A, b, {"variable_ordering": variables,
                      "equation_ordering": equations}, x0

    reorderLinearSystem = reorder_linear_system

    def deorder_linear_system(self, x, order=None):
        """Undo a variable permutation on the solution."""
        ordering = (order or {}).get("variable_ordering")
        if ordering is None:
            return x
        out = np.empty_like(x)
        out[ordering] = x[:ordering.size]
        return out

    deorderLinearSystem = deorder_linear_system

    # -- the adjoint half of the preparation stack -------------------------
    #
    # The adjoint solves `A' lam = b`, so every step of the forward stack has a
    # mirrored partner: the roles of rows and columns swap, and with them left
    # scaling swaps with right, and the variable ordering swaps with the
    # equation ordering. Getting one of those mirrors wrong produces a
    # multiplier vector of the right shape and the wrong values - and an
    # adjoint gradient that is quietly incorrect, which is the failure this
    # whole subsystem exists to avoid.

    def reduce_linear_system_adjoint(self, A, b, x0=None):
        """Schur reduction for the transposed system.

        The same split as the forward case; only the right-hand side differs,
        which `reduce_linear_system` already handles through `is_adjoint`.
        """
        return self.reduce_linear_system(A, b, is_adjoint=True, x0=x0)

    reduceLinearSystemAdjoint = reduce_linear_system_adjoint

    def recover_linear_system_adjoint(self, x, system):
        """Rebuild the eliminated block for the transposed system.

        `s = E^-T (h - C^T x)`, against the forward `s = E^-1 (h - D x)`. Both
        the matrix that is transposed *and* the block that multiplies `x` change
        - using `D` here, or forgetting the transpose on `E`, gives a plausible
        vector that solves nothing.
        """
        if not system or system.get("E") is None:
            return x
        E = sp.csc_matrix(system["E"]).T
        C = sp.csr_matrix(system["C"]).T
        s = spla.spsolve(E, system["h"] - C @ x)
        return np.concatenate([x, np.atleast_1d(s)])

    recoverLinearSystemAdjoint = recover_linear_system_adjoint

    def apply_scaling_adjoint(self, A, b, x0=None):
        """Diagonal scaling for the transposed system.

        Mirrors `apply_diagonal_scaling`: there, left scaling multiplies `A`
        and `b`; here it multiplies `A` and the *initial guess*, because in the
        transposed system the left factor acts on the unknown rather than on
        the residual. Right scaling mirrors the other way.
        """
        scaling: Dict[str, Any] = {}
        left, right = (self.apply_left_diagonal_scaling,
                       self.apply_right_diagonal_scaling)
        if not (left or right):
            return A, b, scaling, x0
        if left and right:
            raise ValueError(
                "cannot apply both left and right diagonal scaling")

        M = self.get_diagonal_inverse(A)
        if left:
            A = M @ A
            if x0 is not None:
                x0 = spla.spsolve(sp.csc_matrix(M).T, x0)
        else:
            A = A @ M
            b = sp.csr_matrix(M).T @ b
        scaling["M"] = M
        return A, b, scaling, x0

    applyScalingAdjoint = apply_scaling_adjoint

    def undo_scaling_adjoint(self, x, scaling):
        """Undo *left* scaling - the mirror of the forward case, which undoes
        right. Undoing the wrong one leaves the multipliers scaled."""
        if self.apply_left_diagonal_scaling and scaling.get("M") is not None:
            return sp.csr_matrix(scaling["M"]).T @ x
        return x

    undoScalingAdjoint = undo_scaling_adjoint

    def deorder_linear_system_adjoint(self, x, order=None):
        """Undo the *equation* permutation - the forward case undoes the
        variable permutation. In the transposed system the unknown is indexed
        by equations, so this is the one that applies."""
        ordering = (order or {}).get("equation_ordering")
        if ordering is None:
            return x
        out = np.empty_like(x)
        out[ordering] = x[:ordering.size]
        return out

    deorderLinearSystemAdjoint = deorder_linear_system_adjoint

    # -- eliminating whole equations ---------------------------------------

    def reduce_to_variable(self, problem, keep):
        """Eliminate every equation not marked `keep`, in order.

        Returns `(problem, eliminated)`. The eliminated equations are kept so
        `recover_result` can rebuild their increments; discarding them makes
        the reduction one-way.
        """
        keep = np.asarray(keep, dtype=bool)
        names = list(problem.equation_names)
        problem = problem.clear_system() or problem
        eliminated = []
        for name in [n for n, k in zip(names, keep) if not k]:
            out = problem.eliminate_variable(name)
            if isinstance(out, tuple):
                problem, out = out
            eliminated.append(out)
        return problem, eliminated

    reduceToVariable = reduce_to_variable

    def recover_result(self, dx_kept, eliminated, keep):
        """MRST `dx = recoverResult(solver, dxElim, eliminatedEqs, keep)`: the
        eliminated increments from their equations (`recoverVars`), last
        eliminated first, each from the increments recovered so far."""
        from pymrst.modules.ad_core.utils.recover_vars import recover_vars
        keep = np.asarray(keep, dtype=bool)
        kept = np.flatnonzero(keep)
        left = np.flatnonzero(~keep)
        kept_no = kept.size
        dx: List[Any] = [None]*keep.size
        recovered = keep.copy()
        for slot, value in zip(kept, dx_kept):
            dx[slot] = value
        for i in range(len(eliminated) - 1, -1, -1):
            pos = left[i]
            sol = [dx[j] for j in np.flatnonzero(recovered)]
            dx[pos] = recover_vars(eliminated[i], kept_no, sol)
            recovered[pos] = True
        return dx

    recoverResult = recover_result

    def preconditioner_inverse(self, M, x):
        """Apply a preconditioner, given as a callable or a matrix."""
        if M is None:
            return x
        if callable(M):
            return M(x)
        return spla.spsolve(sp.csc_matrix(M), x)

    preconditionerInverse = preconditioner_inverse

    def get_initial_guess(self, problem):
        return self.initial_guess_fun(problem) if self.initial_guess_fun else None

    getInitialGuess = get_initial_guess

    def setup_solver(self, A, b, **kwargs):
        """Hook before a solve - where a subclass factorises or allocates."""
        return self

    setupSolver = setup_solver

    def cleanup_solver(self, A, b, **kwargs):
        """Hook after a solve, paired with `setup_solver`."""
        return self

    cleanupSolver = cleanup_solver

    def get_solve_report(self, **kwargs) -> Dict[str, Any]:
        """MRST's solve report, with the timing split it distinguishes.

        Preparation and post-processing are separate from the solve itself
        because reduction and scaling can cost more than the solve they were
        meant to speed up, and a single total hides that.
        """
        report = {"Iterations": 0, "Residual": 0.0, "SolverTime": 0.0,
                  "LinearSolutionTime": 0.0, "PreparationTime": 0.0,
                  "PostProcessTime": 0.0, "Converged": True}
        report.update(kwargs)
        return report

    getSolveReport = get_solve_report

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
                             model: Any = None):
        """MRST `solveLinearProblem`: `(dx, result, report)`.

        The preparation stack runs in MRST's order - local elimination when
        reducing to the cell equations, then reorder, Schur reduction and
        diagonal scaling - and is undone in reverse after the solve. `dx` is
        the increment split into one block per primary variable.

        The system solved is MRST's `A dx = b`, `b` minus the residual
        (`LinearizedProblemAD` stores the residual, and is negated).
        """
        import time

        timer = time.perf_counter()
        lsys = None
        eliminated: List[Any] = []
        keep = None
        keep_number0 = self.keep_number
        initial_guess = self.get_initial_guess(problem)
        reorder_first = not (self.reduceToCell and self.keep_number is None)
        if not reorder_first:
            problem, eliminated, keep, lsys, initial_guess = self.eliminate_local(
                problem, initial_guess)
        problem = _assembled(problem)
        if not np.all(np.isfinite(problem.b)):
            raise ValueError("Linear system rhs must have finite entries.")
        A, b = problem.A, _rhs(problem)
        x0 = (np.concatenate([np.ravel(g) for g in initial_guess])
              if initial_guess else None)
        ordering = None
        if reorder_first:
            A, b, ordering, x0 = self.reorder_linear_system(A, b, None, x0)
        if lsys is None:
            A, b, lsys, x0 = self.reduce_linear_system(A, b, False, x0)
        if not reorder_first:
            A, b, ordering, x0 = self.reorder_linear_system(A, b, None, x0)
        A, b, scaling, x0 = self.apply_diagonal_scaling(A, b, x0)
        t_prepare = time.perf_counter() - timer
        result = np.asarray(self.solve_linear_system(A, b), dtype=float).ravel()
        t_solve = time.perf_counter() - timer - t_prepare
        result = self.undo_scaling(result, scaling)
        if not reorder_first:
            result = self.deorder_linear_system(result, ordering)
        result = self.recover_linear_system(result, lsys)
        if reorder_first:
            result = self.deorder_linear_system(result, ordering)
        report = self.get_solve_report(
            Iterations=int(self.last_iterations),
            SolverTime=time.perf_counter() - timer,
            LinearSolutionTime=t_solve, PreparationTime=t_prepare)
        report["PostProcessTime"] = (report["SolverTime"] - t_solve
                                     - t_prepare)
        if self.replaceNaN:
            result[np.isnan(result)] = self.replacementNaN
        if self.replaceInf:
            result[np.isinf(result)] = self.replacementInf
        dx = self.store_increments(problem, result)
        if eliminated:
            dx = self.recover_result(dx, eliminated, keep)
        self.keep_number = keep_number0
        return dx, result, report

    solveLinearProblem = solve_linear_problem

    def store_increments(self, problem: LinearizedProblem, result) -> List[Any]:
        """MRST `storeIncrements`: one block per primary variable.

        The block widths are the columns of the first AD equation's Jacobian
        blocks, which is MRST's `getNumVars`. A problem built straight from an
        assembled `(A, b)` has no AD equations; it splits by its declared
        `var_sizes` when it has them and returns the vector whole otherwise.
        """
        result = np.asarray(result, dtype=float).ravel()
        sample = next((eq for eq in problem.equations if hasattr(eq, "jac")),
                      None)
        num_vars = getattr(problem, "num_vars", None)
        if sample is not None and num_vars is not None \
                and sum(num_vars) == sum(block.shape[1] for block in sample.jac):
            widths = list(num_vars)
        elif sample is not None:
            widths = [block.shape[1] for block in sample.jac]
        elif getattr(problem, "var_sizes", None) and getattr(problem, "var_names", None):
            widths = list(problem.var_sizes)
        elif result.size == 0:
            return []
        else:
            return [result]
        bounds = np.concatenate([[0], np.cumsum(widths)])
        return [result[bounds[i]:bounds[i + 1]] for i in range(len(widths))]

    storeIncrements = store_increments

    # -- reduced and adjoint problem solves ---------------------------------

    def solve_cell_reduced_linear_problem(self, problem, model=None):
        """Solve after Schur-complementing everything but the cell equations.

        Returns `(dx, result, report)`, where `dx` is the *full* increment
        list and `result` the reduced solve's raw vector.

        The reduction is exact, so this is not an approximation of the full
        solve - it is the same answer obtained from a system the size of the
        grid rather than the grid plus the wells. What it buys is that the
        reduced matrix has the reservoir's own structure, which is what the
        AMG and CPR machinery is built for; the well rows would otherwise sit
        in it as dense rows touching every perforated cell.
        """
        import time

        keep = problem.index_of_type("cell")
        problem, eliminated = self.reduce_to_variable(problem, keep)
        problem = _assembled(problem)

        started = time.perf_counter()
        # `-problem.b`, where MRST writes `problem.b`. The two ports store
        # opposite signs: MRST's `assembleSystem` sets `b = -eqs.val`, PyMRST's
        # stores the residual itself and negates at solve time (see
        # `solve_linear_problem`, which does the same). Transcribing MRST's
        # call here returned `-dx` - the right magnitude with every sign
        # flipped, which a Newton loop turns into divergence rather than an
        # error.
        result = self.solve_linear_system(problem.A, _rhs(problem))
        report = self.get_solve_report(SolverTime=time.perf_counter() - started)

        dx_cell = problem.store_increments(result)
        if isinstance(dx_cell, dict):
            dx_cell = list(dx_cell.values())
        dx = self.recover_result(dx_cell, eliminated, keep)
        return dx, result, report

    solveCellReducedLinearProblem = solve_cell_reduced_linear_problem

    def eliminate_local(self, problem, initial_guess=None):
        """Split the system into its kept/eliminated blocks.

        Returns `(problem, eliminated, keep, lsys, initial_guess)` where
        `lsys` holds the four blocks of

            [B  C] [x_keep] = [f]
            [D  E] [x_elim]   [h]

        and the reduced system is the Schur complement `B - C E^-1 D`.

        Named for MRST's `eliminateLocal`: the eliminated variables are
        *local* - well and control unknowns, which couple to a handful of
        cells each rather than to their neighbours - so `E` is block-diagonal
        by well and inverting it is cheap. That is the whole reason this
        elimination is worth doing rather than handing the coupled system to
        the solver.
        """
        keep = np.asarray(problem.index_of_type("cell"), dtype=bool)
        eliminated: List[Any] = []
        lsys = None
        if not bool(np.all(keep)):
            sample = next((eq for eq in problem.equations if hasattr(eq, "jac")), None)
            if type(sample).__name__ == "GenericAD":
                n_keep = int(keep.sum())
                if not (np.all(keep[:n_keep]) and not np.any(keep[n_keep:])):
                    raise ValueError("Cell variables must all combine first in the ordering "
                                     "for this AutodiffBackend.")
                # MATLAB's problem is a value: the caller's keeps its own system.
                problem = copy(_assembled(copy(problem)))
                A = sp.csr_matrix(problem.A)
                sizes = np.asarray(problem.get_equation_var_num()).ravel()
                split = int(sizes[:n_keep].sum())
                B, C = A[:split, :split], A[:split, split:]
                D, E = A[split:, :split], A[split:, split:]
                rhs = _rhs(problem)
                f, h = rhs[:split], rhs[split:]
                solve_E = spla.factorized(sp.csc_matrix(E))
                D_csc = sp.csc_matrix(D)
                E_inv_D = np.zeros((E.shape[0], D.shape[1]))
                for j in np.flatnonzero(np.diff(D_csc.indptr)):
                    E_inv_D[:, j] = solve_E(D_csc[:, [j]].toarray().ravel())
                lsys = {"B": B, "C": C, "D": D, "E": E, "f": f, "h": h, "solve_E": solve_E}
                problem.A = (B - C @ sp.csr_matrix(E_inv_D)).tocsr()
                _set_rhs(problem, f - C @ solve_E(h))
            else:
                problem, eliminated = self.reduce_to_variable(problem, keep)
            if initial_guess and initial_guess[0] is not None and np.size(initial_guess[0]):
                initial_guess = [g for g, k in zip(initial_guess, keep) if k]
        return problem, eliminated, keep, lsys, initial_guess

    eliminateLocal = eliminate_local

    def solve_adjoint_problem(self, problem_prev, problem_curr,
                              next_lambda_vec, objective, model=None,
                              **kwargs):
        """One backward step of the adjoint recursion.

        Solves, for the Lagrange multiplier at this step,

            (dF_i/dx_i)^T lambda_i = -(dg_i/dx_i)^T
                                     - (dF_{i+1}/dx_i)^T lambda_{i+1}

        Two things are easy to get wrong and both are pinned by tests.

        The system is the **transpose** of the forward Jacobian. The forward
        step asks "how does the state respond to a perturbation"; the adjoint
        asks "which perturbations does this objective respond to", and those
        are adjoint questions in the linear-algebra sense.

        The coupling term uses `problem_prev`, whose Jacobian is `dF_{i+1}/dx_i`
        - the *next* step's residual differentiated against *this* step's
        state. It is the only place the recursion carries information
        backwards in time, and dropping it produces a gradient that is right
        for a single step and wrong for a schedule, which reads as an
        optimiser that stalls rather than as a bug.
        """
        problem_curr = _assembled(problem_curr)
        A = sp.csr_matrix(problem_curr.A)

        rhs = np.zeros(A.shape[0]) if objective is None else \
            -np.asarray(_objective_gradient(objective), dtype=float).ravel()

        if next_lambda_vec is not None and problem_prev is not None:
            prev = problem_prev
            if getattr(prev, "A", None) is None:
                prev = _assembled(prev)
            rhs = rhs - sp.csr_matrix(prev.A).T @ np.asarray(
                next_lambda_vec, dtype=float).ravel()

        lambda_vec = self.solve_linear_system(A.T.tocsr(), rhs)
        lambda_blocks = problem_curr.store_increments(lambda_vec)
        return lambda_blocks, lambda_vec, self.get_solve_report(
            Adjoint=True, **kwargs)

    solveAdjointProblem = solve_adjoint_problem


def _objective_gradient(objective) -> np.ndarray:
    """`dg/dx` from an AD objective, or a plain vector as given."""
    if hasattr(objective, "jac"):
        jac = objective.jac[0] if isinstance(objective.jac, (list, tuple)) \
            else objective.jac
        return np.asarray(sp.csr_matrix(jac).todense()).ravel()
    return np.asarray(objective, dtype=float).ravel()


__all__ = ["LinearSolverAD", "LinearizedProblem", "LinearizedProblemAD"]


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
        prob = LinearizedProblemAD(sp.eye(5), np.zeros(5),
                                   var_names=["p", "sW"], var_sizes=[3, 2])
        blocks = prob.store_increments(np.array([1., 2., 3., 4., 5.]))
        assert np.allclose(blocks["p"], [1, 2, 3])
        assert np.allclose(blocks["sW"], [4, 5])
        print("ok test_linearized_problem_splits_increment")

    def test_solve_linear_problem_uses_minus_b():
        """solve_linear_problem solves A dx = -b."""
        A, b, x = _system()
        prob = LinearizedProblemAD(A, -b)             # so -b = +A x
        dx, result, report = _Direct().solve_linear_problem(prob)
        assert np.allclose(result, x) and np.allclose(dx[0], x)
        assert report["SolverTime"] >= report["LinearSolutionTime"]
        print("ok test_solve_linear_problem_uses_minus_b")

    test_base_is_abstract()
    def test_schur_reduction_with_few_coupled_columns_is_exact():
        # Two "well" unknowns coupled to two of 20 reservoir unknowns.
        A, b, x = _system(22, seed=3)
        A = A.tolil()
        A[20:, :20] = 0
        A[20, 4] = 1.5
        A[21, 11] = -2.0
        A = A.tocsr()
        b = A @ x
        solver = _Direct()
        solver.keep_number = 20
        reduced_A, reduced_b, system, _ = solver.reduce_linear_system(A, b)
        x_keep = spla.spsolve(sp.csc_matrix(reduced_A), reduced_b)
        np.testing.assert_allclose(solver.recover_linear_system(x_keep, system), x,
                                   rtol=1e-10)
        print("ok test_schur_reduction_with_few_coupled_columns_is_exact")

    import scipy.sparse.linalg as spla
    test_schur_reduction_with_few_coupled_columns_is_exact()
    test_solve_and_scaling_agree()
    test_linearized_problem_splits_increment()
    test_solve_linear_problem_uses_minus_b()
    print("\nAll tests passed!")
