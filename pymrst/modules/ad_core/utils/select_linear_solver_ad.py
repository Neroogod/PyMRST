"""MRST ``ad-core/utils/selectLinearSolverAD.m``.

`select_linear_solver_ad` is the port. Up to `BackslashThreshold` unknowns
(components times cells) it returns a direct solver. Above it, in MRST's
order: AMGCL-CPR, CPR with an AGMG/AMGCL/direct pressure solver, AMGCL, then
GMRES-ILU, with cell-major ordering for the block solvers. AGMG is not
available to PyMRST, so CPR's pressure solver is AMGCL (when `checkAMGCL`
passes) or a direct solve. The AMGCL-CPR branch needs MRST's
`AMGCL_CPRSolverAD`, which is not ported yet, and says so.

`select_linear_solver_ad_ad`, below, is PyMRST's earlier selector for the
legacy models, kept for its callers. Its notes follow.

Pick a linear solver for a model (legacy).

MRST's version walks a list of optional external libraries - AMGCL, AGMG - and
falls back through them to a direct solve. Those libraries are not in this
port, so what survives is the part that is a *decision* rather than a
dependency check, and it is the one that matters:

    ndof = ncomp * G.cells.num;
    lsolve = BackslashSolverAD();
    if ndof <= opt.BackslashThreshold, return; end
    ... otherwise CPR ...

**Below the threshold a direct solve wins, above it CPR does.** That is not a
preference. A sparse LU on a small system is exact and costs less than
assembling a preconditioner; on a large one its fill-in makes it super-linear
in the cell count, while CPR's cost stays close to linear. MRST puts the
crossover at 10000 degrees of freedom - `ncomp * cells`, not cells, because a
three-phase model has three unknowns per cell and it is the *system* size that
decides.

Getting the count wrong in the safe direction is not free either: a direct
solve on a large system does converge, it just takes long enough that the run
looks hung.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: MRST's `BackslashThreshold`: the system size above which CPR is chosen.
BACKSLASH_THRESHOLD = 10000

#: MRST's `'tolerance'` default for the outer solver.
DEFAULT_TOLERANCE = 1e-4


def select_linear_solver_ad_ad(model: Any, *,
                            use_cpr: bool = True,
                            backslash_threshold: int = BACKSLASH_THRESHOLD,
                            tolerance: float = DEFAULT_TOLERANCE,
                            max_iterations: int = 50,
                            **kwargs: Any):
    """Return a linear solver suited to ``model``'s system size."""
    from pymrst.modules.ad_core.solvers.concrete_solvers import (
        BackslashSolverAD,
    )

    n_dof = get_component_count(model)*int(model.G.cells.num)
    if n_dof <= int(backslash_threshold) or not use_cpr:
        return BackslashSolverAD(**kwargs)

    from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD
    return CPRSolverADAD(tolerance=float(tolerance),
                       max_iterations=int(max_iterations), **kwargs)


def select_linear_solver_ad(model: Any, *, useAMGCL: bool = True, useAGMG: bool = True,
                            useILU: bool = True, useSYMRCMOrdering: bool = False,
                            useCPR: bool = True, useAMGCLCPR: bool = True,
                            BackslashThreshold: int = BACKSLASH_THRESHOLD,
                            tolerance: float = DEFAULT_TOLERANCE, **solver_arg: Any):
    """MRST `selectLinearSolverAD`: a linear solver for `model`'s system."""
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    from pymrst.modules.ad_core.solvers.gmres_ilu_solver_ad import GMRES_ILUSolverAD
    opt = dict(useAMGCL=useAMGCL, useAGMG=useAGMG, useILU=useILU,
               useSYMRCMOrdering=useSYMRCMOrdering, useCPR=useCPR,
               useAMGCLCPR=useAMGCLCPR)
    solver_arg = {"tolerance": tolerance, **solver_arg}
    lsolve = BackslashSolverAD()
    ncomp = _component_count(model)
    ndof = ncomp*model.G.cells.num
    if ndof <= BackslashThreshold:
        return lsolve
    if opt["useAMGCL"]:
        opt["useAMGCL"] = _check_amgcl()
    is_diagonal = type(getattr(model, "AutoDiffBackend", None)).__name__ == \
        "DiagonalAutoDiffBackend"
    if opt["useAMGCLCPR"] and opt["useAMGCL"] and opt["useCPR"]:
        raise NotImplementedError(
            "selectLinearSolverAD picks AMGCL_CPRSolverAD here; it is not "
            "ported yet.")
    elif opt["useCPR"] and not is_diagonal:
        from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverAD
        parg = {"tolerance": 1e-3, "max_iterations": 25}
        if opt["useAGMG"] and _check_agmg():
            raise NotImplementedError("AGMGSolverAD is not ported")
        elif opt["useAMGCL"] and _check_amgcl():
            from pymrst.modules.ad_core.solvers.amgcl_solver_ad import AMGCLSolverAD
            psolve = AMGCLSolverAD(**parg)
        else:
            psolve = BackslashSolverAD()
        lsolve = CPRSolverAD(ellipticSolver=psolve, **solver_arg)
    elif opt["useAMGCL"]:
        from pymrst.modules.ad_core.solvers.amgcl_solver_ad import AMGCLSolverAD
        lsolve = AMGCLSolverAD(relaxation="ilu0", max_iterations=50,
                               amgcl_options={"preconditioner": "relaxation",
                                              "block_size": ncomp},
                               tolerance=solver_arg.pop("tolerance"), **solver_arg)
        _set_solver_ordering_reduction(model, lsolve, ncomp, opt)
    elif opt["useILU"]:
        lsolve = GMRES_ILUSolverAD(maxIterations=100, reorderEquations=False, **solver_arg)
        _set_solver_ordering_reduction(model, lsolve, ncomp, opt)
    return lsolve


def _set_solver_ordering_reduction(model: Any, lsolve: Any, ncomp: int, opt) -> None:
    from pymrst.modules.ad_core.utils.get_cell_major_reordering import (
        get_cell_major_reordering,
    )
    G = model.G
    ndof = ncomp*G.cells.num
    sym_ordering = None
    if opt["useSYMRCMOrdering"]:
        from pymrst.modules.ad_core.utils.get_grid_symrcm_ordering import (
            get_grid_symrcm_ordering,
        )
        sym_ordering = get_grid_symrcm_ordering(G)
    ordering = get_cell_major_reordering(G.cells.num, ncomp, ndof=ndof,
                                         cell_ordering=sym_ordering)
    # MRST's variableOrdering/equationOrdering; the base class reads them
    # under these names.
    lsolve.variable_ordering = ordering
    lsolve.equation_ordering = ordering
    if type(getattr(model, "AutoDiffBackend", None)).__name__ == \
            "DiagonalAutoDiffBackend" or type(lsolve).__name__ == "AMGCLSolverAD":
        lsolve.reduceToCell = False
        lsolve.keep_number = ndof
    else:
        lsolve.reduceToCell = True


def _check_amgcl() -> bool:
    """MRST `checkAMGCL`: AMGCL is used only with the `useMEX` setting on
    and the library callable (pyamgcl here)."""
    from pymrst.modules.ad_core.solvers.amgcl_solver_ad import HAVE_AMGCL
    from pymrst.utils.settings_manager.mrst_settings import mrst_settings
    return bool(mrst_settings("get", "useMEX")) and bool(HAVE_AMGCL)


def _check_agmg() -> bool:
    """MRST `checkAGMG`: whether the AGMG library is on the path. PyMRST has
    no AGMG binding."""
    return False


def _component_count(model: Any) -> int:
    """MRST `getComponentCount`."""
    model = model.validate_model()
    names = model.get_component_names()
    if getattr(model, "isGenericReservoirModel", False):
        ncomp = len(names)
    elif any(c.__name__ == "ThreePhaseBlackOilModel" for c in type(model).__mro__):
        ncomp = len(names) + int(np.count_nonzero(model.get_active_phases()))
    else:
        ncomp = len(names) + int(model.water)
    if hasattr(model, "thermal"):
        ncomp += int(model.thermal)
    return ncomp


selectLinearSolverAD = select_linear_solver_ad


def get_component_count(model: Any) -> int:
    """Unknowns per cell (MRST ``getComponentCount``).

    The phase count for a black-oil model; the component count plus water for
    a compositional one. It multiplies the cell count into the *system* size,
    which is what the threshold is measured in - a three-phase model reaches
    10000 unknowns at 3334 cells, not at 10000.
    """
    eos = getattr(model, "EOSModel", None)
    if eos is not None:
        mixture = getattr(eos, "CompositionalMixture", None)
        n_components = (len(getattr(mixture, "names", ())) if mixture is not None
                        else 0)
        if n_components:
            return n_components + int(bool(getattr(model, "water", True)))

    for name in ("getNumberOfPhases", "n_phases"):
        found = getattr(model, name, None)
        if callable(found):
            return int(found())
        if found is not None:
            return int(found)
    return len(model.get_phase_names())


getComponentCount = get_component_count

__all__ = ["BACKSLASH_THRESHOLD", "DEFAULT_TOLERANCE", "getComponentCount",
           "get_component_count", "selectLinearSolverAD",
           "select_linear_solver_ad", "select_linear_solver_ad_ad"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model(n_cells):
        side = int(round(n_cells ** (1/3))) or 1
        dims = [side, side, max(1, n_cells // (side*side))]
        g = compute_geometry(cart_grid(dims, [100.0, 100.0, 10.0]))
        rock = make_rock(g, 100*MILLI*DARCY, 0.3)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0])*CENTI*POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6/BARSA, 1e-6/BARSA], p_ref=200*BARSA)
        return GenericBlackOilModelAD(g, rock, fluid, [])

    def test_a_small_system_gets_a_direct_solve():
        """Exact, and cheaper than assembling a preconditioner."""
        from pymrst.modules.ad_core.solvers.concrete_solvers import (
            BackslashSolverAD,
        )
        model = _model(1000)
        assert isinstance(select_linear_solver_ad_ad(model), BackslashSolverAD)
        print("ok test_a_small_system_gets_a_direct_solve")

    def test_a_large_system_gets_cpr():
        """Above the crossover the direct solve's fill-in makes it
        super-linear in the cell count; it still converges, it just takes long
        enough that the run looks hung."""
        from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD
        model = _model(27000)
        assert isinstance(select_linear_solver_ad_ad(model), CPRSolverADAD)
        print("ok test_a_large_system_gets_cpr")

    def test_the_threshold_counts_unknowns_not_cells():
        """A two-phase model reaches 10000 unknowns at 5000 cells. Counting
        cells instead leaves a 10000-cell three-phase model - 30000 unknowns -
        on a direct solve."""
        model = _model(8000)                     # 8000 cells, two phases
        assert get_component_count(model) == 2
        assert 8000 <= BACKSLASH_THRESHOLD < 2*8000
        from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD
        assert isinstance(select_linear_solver_ad_ad(model), CPRSolverADAD)
        print("ok test_the_threshold_counts_unknowns_not_cells")

    def test_cpr_can_be_declined():
        from pymrst.modules.ad_core.solvers.concrete_solvers import (
            BackslashSolverAD,
        )
        model = _model(27000)
        solver = select_linear_solver_ad_ad(model, use_cpr=False)
        assert isinstance(solver, BackslashSolverAD)
        print("ok test_cpr_can_be_declined")

    def test_the_threshold_is_mrsts():
        assert BACKSLASH_THRESHOLD == 10000
        print("ok test_the_threshold_is_mrsts")

    from types import SimpleNamespace

    def _generic(n_cells):
        """A generic three-component model, as `getComponentCount` sees it."""
        model = SimpleNamespace(G=SimpleNamespace(cells=SimpleNamespace(num=n_cells)),
                                isGenericReservoirModel=True, AutoDiffBackend=None,
                                get_component_names=lambda: ["water", "oil", "gas"])
        model.validate_model = lambda: model
        return model

    def test_the_port_solves_small_systems_directly():
        from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
        assert type(select_linear_solver_ad(_generic(3333))) is BackslashSolverAD
        print("ok test_the_port_solves_small_systems_directly")

    def test_a_sparse_backend_gets_cpr_with_a_direct_pressure_solve():
        from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
        from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverAD
        solver = select_linear_solver_ad(_generic(3334), useAMGCL=False)
        assert type(solver) is CPRSolverAD
        assert type(solver.ellipticSolver) is BackslashSolverAD
        assert (solver.tolerance, solver.maxIterations) == (1e-4, 25)
        print("ok test_a_sparse_backend_gets_cpr_with_a_direct_pressure_solve")

    def test_without_cpr_or_amgcl_the_port_falls_back_to_gmres_ilu():
        from pymrst.modules.ad_core.solvers.gmres_ilu_solver_ad import GMRES_ILUSolverAD
        solver = select_linear_solver_ad(_generic(4000), useCPR=False, useAMGCL=False)
        assert type(solver) is GMRES_ILUSolverAD and solver.reduceToCell
        assert (solver.maxIterations, solver.tolerance, solver.reorderEquations) == (100, 1e-4, False)
        assert solver.variable_ordering[:3].tolist() == [0, 4000, 8000], "cell-major"
        assert solver.equation_ordering is solver.variable_ordering
        print("ok test_without_cpr_or_amgcl_the_port_falls_back_to_gmres_ilu")

    test_the_port_solves_small_systems_directly()
    test_a_sparse_backend_gets_cpr_with_a_direct_pressure_solve()
    test_without_cpr_or_amgcl_the_port_falls_back_to_gmres_ilu()
    test_a_small_system_gets_a_direct_solve()
    test_a_large_system_gets_cpr()
    test_the_threshold_counts_unknowns_not_cells()
    test_cpr_can_be_declined()
    test_the_threshold_is_mrsts()
    print("\nAll tests passed!")
