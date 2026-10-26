"""MRST ``ad-core/solvers/LinearizedProblem.m``: a linearized nonlinear
problem - the equations (AD residuals, one Jacobian block per primary
variable), their types and names, the primary variables, and the state, time
step, iteration and driving forces they came from.

As in MRST the linear system `A x = b` is assembled on demand with
`b = -residual`, and the problem is a value: every method that changes it
returns a changed copy, so the caller's problem is never altered - the solvers
rely on keeping the original to recover eliminated increments. `indexOf...`
return a logical mask, one entry per equation or variable.

Constructed as MRST's `LinearizedProblem(eqs, types, names, primaryVars,
state[, dt])` or `LinearizedProblem(primaryVars, state[, dt])`; the classmethod
`from_equations` is the first form by name. The attributes also answer to the
names the rest of this port uses (`equation_names`, `primary_variables`,
`iteration_no`, `driving_forces`). PyMRST additions: `num_vars`, the primary
variable widths when the state records them, and `store_increments`.
"""

from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


class LinearizedProblem:
    """MRST's `LinearizedProblem`."""

    def __init__(self, *args: Any) -> None:
        if len(args) in (5, 6):
            eqs, eqtypes, eqnames, primary_vars, state = args[:5]
            timestep = args[5] if len(args) == 6 else 0
        elif len(args) in (2, 3):
            eqs, eqtypes, eqnames = [], [], []
            primary_vars, state = args[:2]
            timestep = args[2] if len(args) == 3 else 0
        else:
            raise TypeError("Bad number of input arguments")
        eqs, eqtypes, eqnames = self.check_inputs(eqs, eqtypes, eqnames)
        if isinstance(primary_vars, str):
            primary_vars = [primary_vars]
        self.equations = list(eqs)
        self.types = list(eqtypes)
        self.equationNames = list(eqnames)
        self.primaryVariables = list(primary_vars)
        assert len(self.types) == len(self.equationNames) == len(self.equations), (
            "Inconsistent number of types/names/equation numbers.")
        self.A = None
        self.b = None
        self.dt = timestep
        self.iterationNo = np.nan
        self.state = state
        self.drivingForces = None
        sizes = state.get("primaryVariableSizes") if isinstance(state, dict) else None
        self.num_vars = (list(sizes) if sizes is not None
                         and len(sizes) == len(self.primaryVariables) else None)

    def __copy__(self) -> "LinearizedProblem":
        """MATLAB's value copy: the cell arrays are copied with the object, so
        replacing an equation of the copy leaves the original alone."""
        problem = object.__new__(type(self))
        problem.__dict__.update(self.__dict__)
        for name in ("equations", "types", "equationNames", "primaryVariables"):
            problem.__dict__[name] = list(self.__dict__[name])
        if self.num_vars is not None:
            problem.num_vars = list(self.num_vars)
        return problem

    @classmethod
    def from_equations(cls, equations, types, names, primary_vars, state: Any = None,
                       dt: float = 0.0) -> "LinearizedProblem":
        """MRST's `LinearizedProblem(eqs, types, names, primaryVars, state, dt)`."""
        return cls(equations, types, names, primary_vars, state, dt)

    # The names this port's models use for MRST's properties.
    equation_names = property(lambda self: self.equationNames,
                              lambda self, v: setattr(self, "equationNames", v))
    primary_variables = property(lambda self: self.primaryVariables,
                                 lambda self, v: setattr(self, "primaryVariables", v))
    iteration_no = property(lambda self: self.iterationNo,
                            lambda self, v: setattr(self, "iterationNo", v))
    driving_forces = property(lambda self: self.drivingForces,
                              lambda self, v: setattr(self, "drivingForces", v))

    def prepend_equations(self, equations, types, names) -> "LinearizedProblem":
        equations, types, names = self.check_inputs(equations, types, names)
        problem = copy(self)
        problem.equations = list(equations) + self.equations
        problem.types = list(types) + self.types
        problem.equationNames = list(names) + self.equationNames
        problem.A = problem.b = None
        return problem

    def append_equations(self, equations, types, names) -> "LinearizedProblem":
        equations, types, names = self.check_inputs(equations, types, names)
        problem = copy(self)
        problem.equations = self.equations + list(equations)
        problem.types = self.types + list(types)
        problem.equationNames = self.equationNames + list(names)
        problem.A = problem.b = None
        return problem

    @staticmethod
    def check_inputs(equations, types, names) -> Tuple[List[Any], List[Any], List[Any]]:
        if isinstance(equations, (list, tuple)):
            assert len(types) == len(equations) and len(names) == len(equations), (
                "Inconsistent number of types/names/equation numbers.")
            return list(equations), list(types), list(names)
        return [equations], [types], [names]

    def assemble_system(self) -> "LinearizedProblem":
        """The problem with `A` and `b = -residual` assembled (if not already)."""
        if self.A is not None or self.b is not None:
            return self
        from pymrst.modules.ad_core.utils.combine_equations import combine_equations
        nonempty = [eq for eq in self.equations if _nonempty(eq)]
        eqs = combine_equations(*nonempty)
        problem = copy(self)
        if hasattr(eqs, "jac"):
            problem.A = sp.csr_matrix(_sparse_block(eqs.jac[0]))
            problem.b = -np.asarray(eqs.val, dtype=float).ravel()
        else:
            problem.b = -np.asarray(eqs, dtype=float).ravel()
        return problem

    def process_result_after_solve(self, result: Any, report: Any) -> Tuple[Any, Any]:
        return result, report

    def clear_system(self) -> "LinearizedProblem":
        problem = copy(self)
        problem.A = problem.b = None
        return problem

    def get_linear_system(self) -> Tuple[Any, np.ndarray]:
        problem = self.assemble_system()
        return problem.A, problem.b

    def norm(self, *args: Any) -> np.ndarray:
        """MATLAB's `norm(value(eq), ...)` of each equation (2-norm by default)."""
        from pymrst.utils.adi import value
        return np.array([np.linalg.norm(np.ravel(value(eq)), *args) for eq in self.equations])

    def numeq(self) -> int:
        return len(self.equations)

    def reorder_equations(self, new_indices: Sequence[int]) -> "LinearizedProblem":
        new_indices = list(new_indices)
        assert self.numeq() == len(new_indices)
        problem = copy(self)
        problem.equations = [self.equations[i] for i in new_indices]
        if self.types:
            problem.types = [self.types[i] for i in new_indices]
        if self.equationNames:
            problem.equationNames = [self.equationNames[i] for i in new_indices]
        return problem

    def get_equation_var_num(self, n: Any = None) -> np.ndarray:
        eqs = self.equations if n is None else [self.equations[i] for i in np.atleast_1d(n)]
        return np.array([np.size(eq.val) for eq in eqs])

    def index_of_type(self, name: str) -> np.ndarray:
        return np.array([str(t).lower() == str(name).lower() for t in self.types], dtype=bool)

    def count_of_type(self, name: str) -> int:
        return int(self.index_of_type(name).sum())

    def index_of_primary_variable(self, name: str) -> np.ndarray:
        return np.array([str(v).lower() == str(name).lower() for v in self.primaryVariables],
                        dtype=bool)

    def index_of_equation_name(self, name: str) -> np.ndarray:
        return np.array([str(v).lower() == str(name).lower() for v in self.equationNames],
                        dtype=bool)

    def pop_increment(self, dx: Sequence[Any], name: Any, nargout: int = 1):
        """MRST `[dvar, dx, problem] = popIncrement(problem, dx, name)`;
        `name` is a variable name or a logical mask."""
        if isinstance(name, str):
            pos = self.index_of_primary_variable(name)
        else:
            pos = np.asarray(name)
            assert pos.dtype == bool
        dx = list(dx)
        problem = self
        if pos.any():
            dvar = dx[int(np.flatnonzero(pos)[0])]
            if nargout > 1:
                dx = [d for d, p in zip(dx, pos) if not p]
                if nargout > 2:
                    problem = copy(self)
                    problem.primaryVariables = [v for v, p in zip(self.primaryVariables, pos)
                                                if not p]
                    if self.num_vars is not None:
                        problem.num_vars = [w for w, p in zip(self.num_vars, pos) if not p]
        else:
            dvar = None
        if nargout == 1:
            return dvar
        return (dvar, dx) if nargout == 2 else (dvar, dx, problem)

    def eliminate_variable(self, variable: Any) -> Tuple["LinearizedProblem", Any]:
        """MRST `[problem, eliminatedEquation] = eliminateVariable(problem, variable)`:
        the equation (by name, index or mask) Schur-complemented out of the
        others, its own primary variable with it."""
        import scipy.sparse.linalg as spla
        if isinstance(variable, str):
            n = int(np.flatnonzero(self.index_of_equation_name(variable))[0])
        elif isinstance(variable, np.ndarray) and variable.dtype == bool:
            n = int(np.flatnonzero(variable)[0])
        else:
            n = int(variable)
        eqs = [type(eq)(np.array(eq.val, copy=True), list(eq.jac)) for eq in self.equations]
        solve_inx = [i for i in range(len(eqs)) if i != n]
        eliminated = eqs[n]
        pivot = spla.factorized(sp.csc_matrix(_sparse_block(eliminated.jac[n])))

        def backslash(rhs):
            rhs = _sparse_block(rhs)
            if sp.issparse(rhs):
                rhs = rhs.toarray()
            rhs = np.asarray(rhs, dtype=float)
            if rhs.ndim == 1:
                return pivot(rhs)
            return np.column_stack([pivot(rhs[:, c]) for c in range(rhs.shape[1])]) \
                if rhs.shape[1] else rhs

        for eq_num in solve_inx:
            eq = eqs[eq_num]
            jn = _sparse_block(eq.jac[n])
            for jac_num in solve_inx:
                if _numel(eq.jac[jac_num]) != 0 and _numel(eliminated.jac[jac_num]) != 0:
                    eq.jac[jac_num] = (_sparse_block(eq.jac[jac_num])
                                       - jn @ sp.csr_matrix(backslash(eliminated.jac[jac_num])))
            if np.size(eq.val) and np.size(eliminated.val):
                eq.val = eq.val - jn @ backslash(np.asarray(eliminated.val, dtype=float).ravel())
        eqs = [eqs[i] for i in solve_inx]
        for eq in eqs:
            eq.jac = [eq.jac[j] for j in solve_inx]
        problem = copy(self)
        problem.equations = eqs
        problem.equationNames = [self.equationNames[i] for i in solve_inx]
        problem.types = [self.types[i] for i in solve_inx]
        problem.primaryVariables = [self.primaryVariables[i] for i in solve_inx]
        if self.num_vars is not None:
            problem.num_vars = [self.num_vars[i] for i in solve_inx]
        problem.A = problem.b = None
        return problem, eliminated

    def reduce_to_single_variable_type(self, type_name: str) -> Tuple["LinearizedProblem", List[Any]]:
        """MRST `[problem, eliminated] = reduceToSingleVariableType(problem, type)`."""
        is_current = self.index_of_type(type_name)
        problem = self.clear_system()
        elim_names = [self.equationNames[i] for i in np.flatnonzero(~is_current)]
        eliminated = []
        for name in elim_names:
            problem, eq = problem.eliminate_variable(name)
            eliminated.append(eq)
        return problem, eliminated

    def recover_from_single_variable_type(self, original_problem: "LinearizedProblem",
                                          increments_reduced: Sequence[Any],
                                          eliminated: Sequence[Any]) -> List[Any]:
        """MRST `dx = recoverFromSingleVariableType(reduced, original, dxReduced, eliminated)`."""
        from pymrst.modules.ad_core.utils.recover_vars import recover_vars
        current = original_problem.index_of_type(self.types[0])
        not_cell_index = np.flatnonzero(~current)
        cell_index = np.flatnonzero(current)
        cell_eq_no = cell_index.size
        n = current.size
        dx: List[Any] = [None]*n
        recovered = np.zeros(n, dtype=bool)
        recovered[cell_index] = True
        for slot, value in zip(cell_index, increments_reduced):
            dx[slot] = value
        assert np.all(np.diff(cell_index) == 1), (
            "This solver currently assumes that the cell variables comes first!")
        for i in range(len(eliminated) - 1, -1, -1):
            pos = int(not_cell_index[i])
            known = [dx[k] for k in np.flatnonzero(recovered)]
            dx[pos] = recover_vars(eliminated[i], cell_eq_no, known)
            recovered[pos] = True
        return dx

    def store_increments(self, result: np.ndarray) -> List[np.ndarray]:
        """A solution split into one block per primary variable (the widths of
        the first equation's Jacobian blocks, or `num_vars`)."""
        result = np.asarray(result, dtype=float).ravel()
        sample = next((eq for eq in self.equations if hasattr(eq, "jac")), None)
        if sample is None:
            return [result]
        widths = [_sparse_block(block).shape[1] for block in sample.jac]
        if self.num_vars is not None and sum(self.num_vars) == sum(widths):
            widths = list(self.num_vars)
        bounds = np.concatenate([[0], np.cumsum(widths)])
        return [result[bounds[i]:bounds[i + 1]] for i in range(len(widths))]

    prependEquations = prepend_equations
    appendEquations = append_equations
    checkInputs = check_inputs
    assembleSystem = assemble_system
    processResultAfterSolve = process_result_after_solve
    clearSystem = clear_system
    getLinearSystem = get_linear_system
    num_eq = numeq
    reorderEquations = reorder_equations
    getEquationVarNum = get_equation_var_num
    indexOfType = index_of_type
    countOfType = count_of_type
    indexOfPrimaryVariable = index_of_primary_variable
    indexOfEquationName = index_of_equation_name
    popIncrement = pop_increment
    eliminateVariable = eliminate_variable
    reduceToSingleVariableType = reduce_to_single_variable_type
    recoverFromSingleVariableType = recover_from_single_variable_type
    storeIncrements = store_increments


def _sparse_block(block: Any) -> Any:
    return block.to_sparse() if hasattr(block, "to_sparse") else block


def _numel(block: Any) -> int:
    if block is None:
        return 0
    block = _sparse_block(block)
    return int(np.prod(block.shape)) if hasattr(block, "shape") else int(np.size(block))


def _nonempty(eq: Any) -> bool:
    from pymrst.utils.adi import value
    return eq is not None and np.size(value(eq)) > 0


__all__ = ["LinearizedProblem"]


if __name__ == "__main__":
    import scipy.sparse.linalg as spla

    from pymrst.utils.adi import init_variables_adi, value

    def _problem():
        """Two cell unknowns and one well unknown, coupled."""
        p, q = init_variables_adi(np.array([1.0, 2.0]), np.array([0.5]))
        cell = p*p - 3 + 0.5*q
        well = q*4 + p[0] - 1
        return LinearizedProblem([cell, well], ["cell", "well"], ["cell", "well"],
                                 ["pressure", "q"], {}, 1.0)

    def test_assemble_uses_the_negative_residual_and_leaves_the_caller_alone():
        problem = _problem()
        assembled = problem.assemble_system()
        assert problem.A is None and assembled.A.shape == (3, 3)
        assert np.allclose(assembled.b, -np.concatenate([value(e) for e in problem.equations]))
        A, b = problem.get_linear_system()
        assert np.allclose(A.toarray(), assembled.A.toarray())
        print("ok test_assemble_uses_the_negative_residual_and_leaves_the_caller_alone")

    def test_reduction_gives_the_full_solution():
        problem = _problem()
        A, b = problem.get_linear_system()
        full = spla.spsolve(sp.csc_matrix(A), b)
        reduced, eliminated = problem.reduce_to_single_variable_type("cell")
        assert problem.numeq() == 2 and reduced.numeq() == 1
        Ar, br = reduced.get_linear_system()
        dx_cell = spla.spsolve(sp.csc_matrix(Ar), br)
        dx = reduced.recover_from_single_variable_type(problem, [dx_cell], eliminated)
        assert np.allclose(np.concatenate([np.ravel(d) for d in dx]), full)
        print("ok test_reduction_gives_the_full_solution")

    def test_masks_and_pop_increment():
        problem = _problem()
        assert np.array_equal(problem.index_of_type("CELL"), [True, False])
        assert np.array_equal(problem.index_of_primary_variable("q"), [False, True])
        dvar, rest, smaller = problem.pop_increment([np.ones(2), np.zeros(1)], "pressure", 3)
        assert np.array_equal(dvar, [1, 1]) and len(rest) == 1
        assert smaller.primaryVariables == ["q"] and problem.primaryVariables == ["pressure", "q"]
        assert problem.pop_increment([1, 2], "missing") is None
        print("ok test_masks_and_pop_increment")

    def test_short_constructor_and_appending():
        problem = LinearizedProblem("pressure", {"x": 1}, 2.0)
        assert problem.numeq() == 0 and problem.primaryVariables == ["pressure"] and problem.dt == 2.0
        grown = problem.append_equations(np.ones(2), "cell", "c")
        assert grown.numeq() == 1 and problem.numeq() == 0
        assert np.isnan(problem.iterationNo) and problem.equation_names == []
        print("ok test_short_constructor_and_appending")

    test_assemble_uses_the_negative_residual_and_leaves_the_caller_alone()
    test_reduction_gives_the_full_solution()
    test_masks_and_pop_increment()
    test_short_constructor_and_appending()
    print("\nAll tests passed!")
