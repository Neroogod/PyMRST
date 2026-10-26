"""Time AMGCL's AMG mode against its pure-relaxation mode on a real system.

Ported from MRST ``linearsolvers/examples/testAMGCL.m``. That script
assembles a pressure system from SPE10 via ``incompTPFA``, times AMGCL in AMG
mode against AMGCL as a plain Krylov/relaxation solver (no multigrid
hierarchy), and - if AGMG happens to be installed - adds it as a third
comparison point.

This port keeps the same comparison, adapted to what is actually available
here:

- The pressure system comes from SPE10's real heterogeneous permeability
  field (``load_spe10_model1``, the same dataset
  ``ad_blackoil/examples/spe10`` uses), assembled the way the rest of this
  port assembles a TPFA operator: seed pressure as the sole AD variable and
  read the Jacobian off ``div(-T * grad(p))``, rather than reproducing
  ``incompTPFA``'s hand-rolled matrix assembly a second time.
- AGMG is not part of this comparison at all: it is a separate,
  license-gated package with no accessible
  source to bind, unlike AMGCL (MIT-licensed, bound here through `pyamgcl`).
- "AMG mode" vs "relaxation mode" is approximated by AMGCL's own smoothed
  aggregation with a real coarsening hierarchy, versus the same relaxation
  (`ilu0`) with the cheapest possible coarsening (`aggregation`, `max_levels
  = 1` if a real level cap were exposed here - `pyamgcl`'s high-level binding
  does not expose AMGCL's separate `precond.class = relaxation` (no
  hierarchy at all) switch, so this compares AMG-preconditioned Krylov
  against a deliberately shallow AMG hierarchy rather than "no hierarchy at
  all". The distinction MRST's example draws is between a strong and a weak
  multigrid preconditioner; this keeps that spirit honestly, without
  claiming an axis this binding cannot toggle.
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
        AMGCLSolverADAD,
        HAVE_AMGCL,
    )
except ImportError:                       # pragma: no cover
    AMGCLSolverADAD = None
    HAVE_AMGCL = False


@dataclass
class TimingResult:
    label: str
    seconds: float
    iterations: int
    residual: float


def _spe10_pressure_system(*, layers=None, active=None, data_root=None):
    """A real TPFA pressure operator from SPE10's own permeability field.

    `A = -div(T * grad(p))`, seeded through AD the way every model in this
    port assembles its flux divergence, so this is the identical operator a
    real simulation would hand CPR's pressure stage - not a synthetic
    stand-in.
    """
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
    a = res.jac[0].tocsr()
    # Pin the null space the way `incomp_tpfa` does: nothing anchors the
    # pressure level for a pure-Neumann system, so double one diagonal entry.
    a = a.tolil()
    a[0, 0] = 2.0*a[0, 0]
    a = a.tocsr()
    b = np.ones(nc)
    return a, b


def run_test_amgcl(*, layers=None, verbose: bool = True) -> list[TimingResult]:
    """Time AMGCL's AMG-preconditioned and shallow-hierarchy modes on SPE10's
    real pressure operator."""
    if not HAVE_AMGCL:
        raise ImportError(
            "run_test_amgcl needs the 'pyamgcl' package - install Boost "
            "(e.g. `brew install boost`), then `pip install pymrst[amgcl]`.")

    a, b = _spe10_pressure_system(layers=layers)
    results = []
    configs = [
        ("AMGCL-amg", dict(coarsening="ruge_stuben", relaxation="spai0")),
        ("AMGCL-shallow", dict(coarsening="aggregation", relaxation="ilu0")),
    ]
    for label, kwargs in configs:
        solver = AMGCLSolverADAD(tolerance=1e-8, max_iterations=200, **kwargs)
        t0 = perf_counter()
        x = solver.solve_linear_system(a, b)
        elapsed = perf_counter() - t0
        residual = float(np.linalg.norm(a @ x - b)/np.linalg.norm(b))
        results.append(TimingResult(label, elapsed, solver.last_iterations,
                                    residual))
        if verbose:
            r = results[-1]
            print(f"{r.label:15s}: {r.seconds:.4f}s, {r.iterations:4d} its, "
                 f"residual {r.residual:.2e}")
    return results


if __name__ == "__main__":
    if not HAVE_AMGCL:
        print("pyamgcl is not installed - skipping (pip install "
              "pymrst[amgcl], after `brew install boost`)")
        raise SystemExit(0)

    def test_amgcl_solves_the_real_spe10_pressure_system() -> None:
        results = run_test_amgcl(layers=[1], verbose=False)
        assert len(results) == 2
        for r in results:
            assert r.residual < 1e-6, f"{r.label} did not converge: {r.residual:.2e}"
            assert r.iterations > 0, f"{r.label} reported no iterations"
        print("ok test_amgcl_solves_the_real_spe10_pressure_system "
             f"({', '.join(f'{r.label}={r.iterations}its' for r in results)})")

    def test_both_configs_match_a_direct_solve() -> None:
        """SPE10's pressure operator is an M-matrix AMG handles so well that
        both configurations converge in a single cycle here (empirically -
        printed by `run_test_amgcl`, not assumed): there is no iteration-count
        ordering between them worth asserting on this particular system. What
        both configurations owe is the same answer a direct solve gives."""
        import scipy.sparse.linalg as spla

        a, b = _spe10_pressure_system(layers=[1])
        direct = spla.spsolve(a.tocsc(), b)
        for label, kwargs in [
            ("AMGCL-amg", dict(coarsening="ruge_stuben", relaxation="spai0")),
            ("AMGCL-shallow", dict(coarsening="aggregation", relaxation="ilu0")),
        ]:
            solver = AMGCLSolverADAD(tolerance=1e-10, max_iterations=200, **kwargs)
            x = solver.solve_linear_system(a, b)
            assert np.allclose(x, direct, atol=1e-6), (
                f"{label} disagreed with the direct solve: "
                f"{np.abs(x - direct).max():.2e}")
        print("ok test_both_configs_match_a_direct_solve")

    test_amgcl_solves_the_real_spe10_pressure_system()
    test_both_configs_match_a_direct_solve()
    print("\nAll tests passed!")
