"""Sweep AMGCL's coarsening/relaxation/Krylov options on a real system.

Ported from MRST ``linearsolvers/examples/showOptionsAMGCL.m``. That script
is interactive: three ``listdlg`` menus let a user pick a preconditioner,
Krylov accelerator, and coarsening/relaxation scheme, then solves once with
the chosen combination and prints ``getAMGCLMexStruct()`` to show the rest of
AMGCL's option surface.

A menu has no equivalent in a non-interactive port, so this keeps the
informational content - the same option lists, straight from AMGCL's own
documentation, as MRST's comment points at
(https://amgcl.readthedocs.io/en/latest/runtime.html) - and turns "let the
user pick one" into "try every coarsening x relaxation combination and report
which converged and how fast," which is more useful unattended than a menu
would be. The Krylov and preconditioner choices are fixed (bicgstab, amg) so
the sweep stays a coarsening/relaxation comparison, matching what
``getAMGCLMexStruct()``'s printed struct was for: showing the option surface,
not exhaustively combining every axis of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, remove_cells
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.deckformat import load_spe10_model1
from pymrst.params import make_rock
from pymrst.utils.adi import init_variables_adi
from pymrst.utils.units import FT

try:                                      # optional: needs Boost + pybind11
    from pymrst.modules.ad_core.solvers.amgcl_solver_ad import (
        AMGCLSolverAD,
        HAVE_AMGCL,
    )
except ImportError:                       # pragma: no cover
    AMGCLSolverAD = None
    HAVE_AMGCL = False

# MRST's own lists, verbatim (showOptionsAMGCL.m) - the vocabulary AMGCL's
# C++ library understands and `pyamgcl` exposes without translation.
COARSENING_SCHEMES = (
    "ruge_stuben",           # "standard" coarsening
    "smoothed_aggregation",  # aggregation with smoothed iterates
    "aggregation",           # very fast, simple constant interpolation
    "smoothed_aggr_emin",    # aggregation with energy-minimising smoothing
)
RELAXATION_SCHEMES = (
    "spai0",          # sparse approximate inverse, degree 0
    "spai1",          # sparse approximate inverse
    "gauss_seidel",
    "ilu0",           # incomplete LU, no fill-in
    "iluk",           # level-based ILU
    "ilut",           # thresholded ILU
    "damped_jacobi",
    "chebyshev",
)
KRYLOV_SOLVERS = (
    "bicgstab", "cg", "bicgstabl", "gmres", "lgmres", "fgmres", "idrs",
)


@dataclass
class SweepResult:
    coarsening: str
    relaxation: str
    converged: bool
    iterations: int
    residual: float
    seconds: float


def _spe10_pressure_system(*, layers=None, active=None, data_root=None):
    """Same construction as `test_amgcl.py`'s - see its docstring for why
    this is the real TPFA operator, not a synthetic stand-in."""
    dataset = load_spe10_model1(layers=layers, active=active,
                                data_root=data_root)
    full = cart_grid(dataset["cartDims"],
                     np.asarray(dataset["cartDims"])*np.array([25.0, 25.0, 2.5])*FT)
    grid = remove_cells(full, ~dataset["active"])
    grid = compute_geometry(grid)
    rock = make_rock(grid, dataset["perm"], dataset["poro"][:, None])
    ops = setup_operators_tpfa(grid, rock)

    nc = grid.cells.num
    (p,) = init_variables_adi(np.ones(nc))
    res = ops.div(-ops.T * ops.grad(p))
    a = res.jac[0].tolil()
    a[0, 0] = 2.0*a[0, 0]
    return a.tocsr(), np.ones(nc)


def run_show_options_amgcl(*, layers=None, verbose: bool = True
                           ) -> list[SweepResult]:
    """Solve SPE10's pressure system with every coarsening x relaxation
    combination AMGCL offers, `bicgstab`/`amg`-preconditioned throughout."""
    if not HAVE_AMGCL:
        raise ImportError(
            "run_show_options_amgcl needs the 'pyamgcl' package - install "
            "Boost (e.g. `brew install boost`), then `pip install "
            "pymrst[amgcl]`.")

    a, b = _spe10_pressure_system(layers=layers)
    results = []
    for coarsening in COARSENING_SCHEMES:
        for relaxation in RELAXATION_SCHEMES:
            solver = AMGCLSolverAD(
                tolerance=1e-8, max_iterations=200, krylov="bicgstab",
                coarsening=coarsening, relaxation=relaxation)
            t0 = perf_counter()
            try:
                x = solver.solve_linear_system(a, b)
                elapsed = perf_counter() - t0
                residual = float(np.linalg.norm(a @ x - b)/np.linalg.norm(b))
                converged = residual < 1e-4
            except Exception:
                elapsed = perf_counter() - t0
                residual = float("inf")
                converged = False
            results.append(SweepResult(
                coarsening, relaxation, converged, solver.last_iterations,
                residual, elapsed))
            if verbose:
                r = results[-1]
                status = "ok" if r.converged else "FAILED"
                print(f"{r.coarsening:22s} / {r.relaxation:14s}: {status:6s} "
                     f"{r.iterations:4d} its, {r.seconds:.4f}s, "
                     f"residual {r.residual:.2e}")
    return results


if __name__ == "__main__":
    if not HAVE_AMGCL:
        print("pyamgcl is not installed - skipping (pip install "
              "pymrst[amgcl], after `brew install boost`)")
        raise SystemExit(0)

    def test_every_combination_is_tried_exactly_once() -> None:
        results = run_show_options_amgcl(layers=[1], verbose=False)
        assert len(results) == len(COARSENING_SCHEMES)*len(RELAXATION_SCHEMES)
        seen = {(r.coarsening, r.relaxation) for r in results}
        assert len(seen) == len(results), "a combination was tried more than once"
        print(f"ok test_every_combination_is_tried_exactly_once ({len(results)} combos)")

    def test_most_combinations_converge_on_a_well_posed_system() -> None:
        """Not all - some relaxation/coarsening pairs are known to struggle
        on particular matrices, which is exactly the kind of thing this sweep
        exists to surface. The bar is "most", not "all"."""
        results = run_show_options_amgcl(layers=[1], verbose=False)
        converged = sum(1 for r in results if r.converged)
        assert converged >= 0.7*len(results), (
            f"only {converged}/{len(results)} combinations converged")
        print(f"ok test_most_combinations_converge_on_a_well_posed_system "
             f"({converged}/{len(results)})")

    test_every_combination_is_tried_exactly_once()
    test_most_combinations_converge_on_a_well_posed_system()
    print("\nAll tests passed!")
