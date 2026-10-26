"""MRST ``ad-core/solvers/AMGCL_CPRSolverAD.m``: AMGCL's own constrained
pressure residual (CPR) solver, through MRST's MEX gateway with
`solver_id = 2`.

Before AMGCL sees the system, MRST prepares it in MATLAB: the cell equations
are weighted by the model's true-IMPES CPR factors, the variables and
equations are ordered cell by cell (`getCellMajorReordering`), the pressure
column is scaled by the mean pressure, and - for the default 'mrst' strategy
- each cell's pressure row is replaced by the (weighted) sum of its rows,
with AMGCL told to take those rows as they are (`drs_row_weights`). All of
that is ported here; the CPR iteration itself is AMGCL's C++ code, the same
as MRST runs.

Indices are 0-based here (MRST's `psub = 1:bz:...` is `0:bz:...`).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.solvers.amgcl_solver_ad import AMGCLSolverAD, merge_solver_options
from pymrst.utils.adi import value


class AMGCL_CPRSolverAD(AMGCLSolverAD):
    """MRST `AMGCL_CPRSolverAD(...)`."""

    def __init__(self, **options: Any):
        from pymrst.modules.linearsolvers.amgcl.utils.get_amgcl_mex_struct import (
            get_amgcl_mex_struct,
        )
        super().__init__()
        self.doApplyScalingCPR = True
        self.useSYMRCMOrdering = False
        self.pressureScaling: Any = None
        self.diagonalTol = 0.2
        self.couplingTol = 0.02
        self.decoupling = "trueIMPES"
        self.strategy = "mrst"
        self.reduceToCell = True
        self.tolerance = 1e-6
        extra = merge_solver_options(self, options)
        self.amgcl_setup = get_amgcl_mex_struct(**extra)
        self.amgcl_setup["solver_id"] = 2

    def solve_linear_system(self, A: sp.spmatrix, b: np.ndarray, x0: Any = None) -> np.ndarray:
        result, _ = self.call_amgcl_mex(A, b, self.amgcl_setup["solver_id"], x0)
        return result

    def solve_linear_problem(self, problem: Any, model: Any = None):
        problem = self.prepare_problem_cpr(problem, model)
        return super().solve_linear_problem(problem, model)

    def get_description(self):
        sn = "AMGCL-CPR"
        if self.amgcl_setup["cpr_blocksolver"]:
            sn += "-block"
        sn += self.id
        prm = ["solver", "preconditioner", "relaxation"]
        if self.amgcl_setup["preconditioner"] == 1:
            prm.append("coarsening")
        lines = [self.get_feature_description(s) for s in prm]
        lines.append(self.get_feature_description("relaxation", "s_relaxation"))
        return ("AMGCL constrained-pressure-residual (CPR) solver. Configuration:\n"
                + "".join(f"\t{l}\n" for l in lines), sn)

    def set_s_relaxation(self, *v: Any) -> None:
        self.set_parameter_group("relaxation", "s_relaxation", *v)

    def prepare_problem_cpr(self, problem: Any, model: Any) -> Any:
        """MRST `[problem, scale] = prepareProblemCPR(solver, problem, model)`
        (the scale is only needed by the adjoint, not ported here)."""
        from pymrst.modules.ad_core.utils.get_cell_major_reordering import (
            get_cell_major_reordering,
        )
        from pymrst.modules.ad_core.utils.get_grid_symrcm_ordering import (
            get_grid_symrcm_ordering,
        )
        from pymrst.modules.linearsolvers.amgcl.utils.reset_amgcl import reset_amgcl
        n = model.G.cells.num
        self.pressureScaling = float(np.mean(value(problem.state["pressure"])))
        setup = self.amgcl_setup
        if setup["update_sprecond"] or setup["update_ptransfer"]:
            self.reuseMode = 2
            if problem.iteration_no == 1:
                reset_amgcl()
        else:
            self.reuseMode = 1
        if setup["block_size"] == 0:
            is_cell = np.asarray(problem.index_of_type("cell"))
            nv = np.array([_num_vars(eq) for eq in problem.equations], dtype=float)
            setup["block_size"] = int(round(np.sum(nv[is_cell]/n)))
        setup["cell_num"] = n
        setup["cell_size"] = n
        if self.doApplyScalingCPR and self.decoupling.lower() == "trueimpes":
            if getattr(problem, "A", None) is not None:
                problem = problem.clear_system()
                if self.verbose:
                    print("System already assembled. CPR will re-assemble!")
            scale = model.get_scaling_factors_cpr(problem, problem.equation_names, self.decoupling)
            for i in range(len(scale)):
                if problem.types[i] != "cell":
                    continue
                ds = np.atleast_1d(value(scale[i]))
                if ds.size > 1 or np.any(ds != 0):
                    problem.equations[i] = problem.equations[i]*scale[i]
        m = setup["block_size"]
        assert m > 0
        if self.keep_number is None:
            if self.reduceToCell:
                ndof = n*m
            else:
                problem = problem.assemble_system()
                ndof = problem.A.shape[0]
                if setup["active_rows"] == 0:
                    setup["active_rows"] = n*m
        else:
            ndof = self.keep_number
        if self.variable_ordering is None:
            sym_ordering = get_grid_symrcm_ordering(model.G) if self.useSYMRCMOrdering else None
            ordering = get_cell_major_reordering(n, m, ndof=ndof, cell_ordering=sym_ordering)
            self.variable_ordering = ordering
            if self.equation_ordering is None:
                self.equation_ordering = ordering
        strategy = self.strategy.lower()
        if strategy in ("mrst", "mrst_drs"):
            setup["use_drs"] = True
            setup["drs_eps_ps"] = -1e8
            setup["drs_eps_dd"] = -1e8
        elif strategy == "amgcl":
            setup["use_drs"] = False
        elif strategy == "amgcl_drs":
            setup["use_drs"] = True
            setup["drs_eps_ps"] = self.couplingTol if np.isfinite(self.couplingTol) else -1e8
            setup["drs_eps_dd"] = self.diagonalTol if np.isfinite(self.diagonalTol) else -1e8
        else:
            raise ValueError(f"Unknown CPR strategy {self.strategy}")
        return problem

    def apply_diagonal_scaling(self, A: sp.spmatrix, b: np.ndarray, x0: Any = None):
        """MRST `[A, b, scaling, x0] = applyScaling(solver, A, b, x0)`."""
        bz = self.amgcl_setup["block_size"]
        nc = self.amgcl_setup["cell_size"]
        psub = np.arange(0, nc*bz - bz + 1, bz)
        scaling: dict = {}
        if self.apply_left_diagonal_scaling or self.apply_right_diagonal_scaling:
            A, b, scaling, x0 = super().apply_diagonal_scaling(A, b, x0)
        elif np.size(self.pressureScaling) != 1 or self.pressureScaling != 1:
            n = A.shape[0]
            d = np.ones(n)
            d[psub] = self.pressureScaling
            M = sp.diags(d, format="csr")
            A = (sp.csr_matrix(A) @ M).tocsr()
            if x0 is not None:
                x0 = np.array(x0, dtype=float)
                x0[psub] = x0[psub]/d[psub]
            scaling["M"] = M
        strategy = self.strategy.lower()
        if strategy in ("mrst", "mrst_drs"):
            w = _get_scaling_internal_cpr(self, A, b)
            ncv = bz*nc
            ndof = b.shape[0]
            I = np.repeat(np.arange(0, ncv, bz), bz)
            J = np.arange(ncv)
            Id = np.concatenate([np.delete(J, np.arange(0, ncv - bz + 1, bz)), np.arange(ncv, ndof)])
            D = sp.csr_matrix((np.concatenate([w[:ncv], np.ones(Id.size)]),
                               (np.concatenate([I, Id]), np.concatenate([J, Id]))),
                              shape=(ndof, ndof))
            A = (D @ A).tocsr()
            b = D @ b
            scaling["D"] = D
            w_override = np.zeros(ncv)
            w_override[psub] = 1
            self.amgcl_setup["drs_row_weights"] = w_override
        elif strategy == "amgcl_drs":
            self.amgcl_setup["drs_row_weights"] = _get_scaling_internal_cpr(self, A, b)
        return A, b, scaling, x0

    def undo_scaling(self, x: np.ndarray, scaling: Any) -> np.ndarray:
        if isinstance(scaling, dict) and scaling.get("M") is not None:
            x = scaling["M"] @ x
        return x

    def get_diagonal_inverse(self, A: sp.spmatrix) -> sp.spmatrix:
        if self.apply_right_diagonal_scaling:
            n = A.shape[0]
            assert A.shape[0] == A.shape[1], "Matrix must be square!"
            bz = self.amgcl_setup["block_size"]
            nc = self.amgcl_setup["cell_size"]
            with np.errstate(divide="ignore"):
                d = 1/np.abs(A.diagonal())
            d[~np.isfinite(d)] = 1
            d[np.arange(0, nc*bz - bz + 1, bz)] = self.pressureScaling
            return sp.diags(d, format="csr")
        return super().get_diagonal_inverse(A)

    prepareProblemCPR = prepare_problem_cpr
    applyScaling = apply_diagonal_scaling
    undoScaling = undo_scaling
    getDiagonalInverse = get_diagonal_inverse
    setSRelaxation = set_s_relaxation


def _num_vars(eq: Any) -> int:
    """MRST's `getSampleAD(...).getNumVars()` for one equation: its Jacobian's
    total column count."""
    jac = getattr(eq, "jac", None)
    if jac is None:
        return 0
    return int(sum(np.shape(j)[1] for j in jac)) if isinstance(jac, (list, tuple)) else int(np.shape(jac)[1])


def _get_scaling_internal_cpr(solver: AMGCL_CPRSolverAD, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
    """MRST's `getScalingInternalCPR`: the row weights of each cell block."""
    import scipy.sparse.linalg as spla
    bz = solver.amgcl_setup["block_size"]
    nc = solver.amgcl_setup["cell_size"]
    ndof = bz*nc
    p_inx = np.arange(0, ndof - bz + 1, bz)
    A = sp.coo_matrix(A)
    ii, jj, vv = A.row, A.col, A.data
    if solver.decoupling.lower() == "quasiimpes":
        block_i, block_j = ii//bz, jj//bz
        keep = (block_j >= block_i) & (block_j < block_i + 1) & (ii < ndof) & (jj < ndof)
        Ap = sp.csc_matrix((vv[keep], (jj[keep], ii[keep])), shape=(ndof, ndof))
        q = np.zeros(ndof)
        q[p_inx] = 1
        w = spla.spsolve(Ap, q)
        w = np.concatenate([w, np.ones(A.shape[0] - ndof)])
    else:
        w = np.ones(b.shape[0])
    if solver.strategy.lower() == "mrst_drs":
        isp = np.zeros(ndof, dtype=bool)
        isp[p_inx] = True
        block_no = ii//bz
        block_conn = jj//bz
        isdp = np.zeros(ii.size, dtype=bool)
        inside = jj < ndof
        isdp[inside] = isp[jj[inside]]
        isdiag = (block_conn == block_no) & isdp
        is_offdiag_p = isdp & ~isdiag
        pd = np.zeros(ndof)
        pd[ii[isdiag]] = vv[isdiag]
        cellno = np.arange(ndof)//bz
        if np.isfinite(solver.diagonalTol):
            sum_offdiag = np.bincount(ii[is_offdiag_p], np.abs(vv[is_offdiag_p]), minlength=ndof)
            ok_dd = pd >= sum_offdiag*solver.diagonalTol
        else:
            ok_dd = np.ones(ndof, dtype=bool)
        if np.isfinite(solver.couplingTol):
            is_other_p = isdp & (block_conn != block_no)
            sum_other = np.bincount(ii[is_other_p], np.abs(vv[is_other_p]), minlength=ndof)
            ok_ps = sum_other >= pd*solver.couplingTol
        else:
            ok_ps = np.ones(ndof, dtype=bool)
        ok = ok_dd & ok_ps
        ok_count = np.bincount(cellno, ok.astype(float), minlength=nc)
        bad = np.flatnonzero(ok_count == 0)
        ok[bad*bz] = True
        ok[p_inx] = True
        w[:ndof][~ok] = 0
    return w


__all__ = ["AMGCL_CPRSolverAD"]


if __name__ == "__main__":
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _case():
        from pymrst.gridprocessing import cart_grid, compute_geometry
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
        from pymrst.modules.ad_props import init_simple_adi_fluid
        from pymrst.params import make_rock
        from pymrst.params.wells_and_bc.vertical_well import vertical_well
        G = compute_geometry(cart_grid([8, 8, 2], [100.0, 100.0, 10.0]))
        rock = make_rock(G, 0.1*DARCY, 0.3)
        W = vertical_well([], G, rock, 0, 0, None, compi=np.array([1.0, 0, 0]), well_type="rate", val=1e-3)
        W = vertical_well(W, G, rock, 7, 7, None, compi=np.array([1.0, 0, 0]), well_type="bhp", val=100*BARSA)
        fluid = init_simple_adi_fluid(n=[2, 3, 1], rho=[1000, 700, 10],
                                      mu=np.array([1, 5, 0.1])*CENTI*POISE,
                                      c=np.array([0, 1e-6, 1e-4])/BARSA)
        model = GenericBlackOilModel(G, rock, fluid, disgas=False, vapoil=False)
        return model, W

    def test_amgcl_solves_the_system_cpr_prepares():
        """MRST weights the cell equations by its true-IMPES factors as AD
        values, so the system CPR hands AMGCL is MRST's prepared one, not the
        Newton system itself; AMGCL must solve that one to its tolerance."""
        import scipy.sparse.linalg as spla
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        from pymrst.modules.ad_core.utils.schedule import simple_schedule
        from pymrst.solvers.init_res_sol import init_res_sol

        class _Direct(AMGCL_CPRSolverAD):
            def solve_linear_system(self, A, b, x0=None):
                return spla.spsolve(sp.csc_matrix(A), b)

        results = []
        for cls in (AMGCL_CPRSolverAD, _Direct):
            model, W = _case()
            schedule = simple_schedule(np.array([DAY]), W=W)
            forces = model.get_driving_forces(schedule.controls[0])
            model = model.validate_model(forces)
            state = model.validate_state(init_res_sol(model.G, 150*BARSA, np.array([0.2, 0.7, 0.1])))
            model, state = model.update_for_changed_controls(state, forces)
            problem, _ = model.get_equations(state, copy_state(state), DAY, forces, iteration=1)
            solver = cls(tolerance=1e-12, block_size=3, coarsening="aggregation", relaxation="ilu0")
            dx, _, _ = solver.solve_linear_problem(problem, model)
            results.append(np.concatenate([np.ravel(np.asarray(d, dtype=float)) for d in dx]))
            if cls is AMGCL_CPRSolverAD:
                assert solver.amgcl_setup["use_drs"]
                assert solver.amgcl_setup["drs_row_weights"].size == 3*128
        np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-9*np.abs(results[1]).max())
        print("ok test_amgcl_solves_the_system_cpr_prepares")

    test_amgcl_solves_the_system_cpr_prepares()
    print("\nAll tests passed!")
