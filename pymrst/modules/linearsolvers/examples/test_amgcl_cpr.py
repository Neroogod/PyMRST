"""AMGCL as CPR's pressure-stage solver on a field-scale case.

Ported from MRST ``linearsolvers/examples/testAMGCL_cpr.m``. That script
builds a plain Schur-reduced pressure system by hand first (to show what CPR
automates), then runs the SPE9 black-oil schedule through
``AMGCL_CPRSolverAD`` and plots the linear-solver-vs-assembly time split.

This port takes the same two-part shape, adapted to what this port's own
solver stack actually exposes:

- MRST's ``AMGCL_CPRSolverAD`` is AMGCL's own *internal* CPR preconditioner
  mode (``solver_id = 2``). This port instead hands ``AMGCLSolverADAD`` to
  ``CPRSolverADAD``'s ``elliptic_solver`` slot, so AMGCL solves the pressure
  block inside this port's own two-stage CPR machinery
  (``pymrst/modules/ad_core/solvers/linear_solvers.py``) - same physical
  role, different plumbing underneath it.
- SPE10 Model I (``pymrst.modules.ad_blackoil.examples.spe10``) stands in for
  MRST's SPE9 case: it is the field-scale black-oil example already ported
  with a working schedule, wells, and tolerances, so this reuses real
  infrastructure instead of assembling a new field case just for this demo.
- No plot: this prints the same per-step assembly-vs-linear-solver time
  breakdown ``testAMGCL_cpr.m``'s bar chart shows, as a table, so the example
  runs headless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.ad_blackoil.examples.spe10.tenth_csp_model_i import (
    setup_tenth_csp_model_i_ad,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD

try:                                      # optional: needs Boost + pybind11
    from pymrst.modules.ad_core.solvers.amgcl_solver_ad import (
        AMGCLSolverADAD,
        HAVE_AMGCL,
    )
except ImportError:                       # pragma: no cover
    AMGCLSolverADAD = None
    HAVE_AMGCL = False


@dataclass
class StepTiming:
    step: int
    wall_seconds: float
    assembly_seconds: float
    linear_seconds: float
    linear_iterations: int


def run_test_amgcl_cpr(nsteps: int = 3, *, layers=None, verbose: bool = True
                       ) -> list[StepTiming]:
    """Run a few SPE10 Model I steps with AMGCL solving CPR's pressure
    stage, and report the assembly-vs-linear-solver time split per step."""
    if not HAVE_AMGCL:
        raise ImportError(
            "run_test_amgcl_cpr needs the 'pyamgcl' package - install Boost "
            "(e.g. `brew install boost`), then `pip install pymrst[amgcl]`.")

    dataset, model, schedule, state0 = setup_tenth_csp_model_i_ad(layers=layers)

    lsolve = CPRSolverADAD(max_iterations=200, tolerance=1e-6)
    lsolve.elliptic_solver = AMGCLSolverADAD(
        tolerance=1e-3, coarsening="aggregation", relaxation="ilu0",
        krylov="bicgstab")
    solver = NonLinearSolver(linear_solver=lsolve, max_iterations=30,
                             max_timestep_cuts=8, use_line_search=True,
                             use_relaxation=True)

    timings: list[StepTiming] = []
    state = state0.copy()
    count = min(int(nsteps), schedule.n_steps)
    for step_index in range(count):
        dt = float(schedule.step_val[step_index])
        state, report = solver.solve_timestep(model, state, dt)
        if not report.converged:
            raise RuntimeError(f"step {step_index + 1} failed: {report}")
        linear_seconds = sum(s.wall_seconds for s in report.linear_solves)
        linear_its = sum(s.iterations for s in report.linear_solves)
        timings.append(StepTiming(
            step=step_index + 1, wall_seconds=report.wall_seconds,
            assembly_seconds=report.assembly_seconds,
            linear_seconds=linear_seconds, linear_iterations=linear_its,
        ))
        if verbose:
            t = timings[-1]
            print(f"step {t.step}/{count}: {t.wall_seconds:.3f}s total, "
                 f"{t.assembly_seconds:.3f}s assembly, "
                 f"{t.linear_seconds:.3f}s linear solve "
                 f"({t.linear_iterations} AMGCL its)")
    return timings


if __name__ == "__main__":
    if not HAVE_AMGCL:
        print("pyamgcl is not installed - skipping (pip install "
              "pymrst[amgcl], after `brew install boost`)")
        raise SystemExit(0)

    def test_amgcl_cpr_solves_a_field_scale_schedule() -> None:
        timings = run_test_amgcl_cpr(nsteps=2, layers=[1], verbose=False)
        assert len(timings) == 2
        for t in timings:
            assert t.linear_iterations > 0, (
                "AMGCL reported no iterations - the pressure stage was not "
                "really solved by it")
            assert t.linear_seconds > 0.0
            assert t.assembly_seconds >= 0.0
        print("ok test_amgcl_cpr_solves_a_field_scale_schedule "
             f"({sum(t.linear_iterations for t in timings)} total AMGCL its)")

    def test_amgcl_cpr_matches_the_default_pressure_stage() -> None:
        """Same schedule, same two steps, AMGCL swapped out for this port's
        default pyamg pressure stage - the two must reach the same states,
        since both are solving the same equations to a tight tolerance."""
        from pymrst.modules.ad_blackoil.examples.spe10.tenth_csp_model_i import (
            setup_tenth_csp_model_i_ad as _setup,
        )

        dataset, model, schedule, state0 = _setup(layers=[1])
        lsolve = CPRSolverADAD(max_iterations=200, tolerance=1e-8)
        solver = NonLinearSolver(linear_solver=lsolve, max_iterations=30,
                                 max_timestep_cuts=8, use_line_search=True,
                                 use_relaxation=True)
        state_default = state0.copy()
        for step_index in range(2):
            state_default, report = solver.solve_timestep(
                model, state_default, float(schedule.step_val[step_index]))
            assert report.converged

        dataset2, model2, schedule2, state0_2 = _setup(layers=[1])
        lsolve2 = CPRSolverADAD(max_iterations=200, tolerance=1e-8)
        lsolve2.elliptic_solver = AMGCLSolverADAD(tolerance=1e-6)
        solver2 = NonLinearSolver(linear_solver=lsolve2, max_iterations=30,
                                  max_timestep_cuts=8, use_line_search=True,
                                  use_relaxation=True)
        state_amgcl = state0_2.copy()
        for step_index in range(2):
            state_amgcl, report = solver2.solve_timestep(
                model2, state_amgcl, float(schedule2.step_val[step_index]))
            assert report.converged

        assert np.allclose(state_default["pressure"], state_amgcl["pressure"],
                           rtol=1e-4)
        print("ok test_amgcl_cpr_matches_the_default_pressure_stage")

    def test_amgcl_cpr_converges_on_a_badly_scaled_pressure_block() -> None:
        """Regression test for a real convergence failure on SPE10 AD.

        `CPRSolver.solve`'s elliptic-stage matrix `a_p` inherits whatever
        raw physical units the model's equations were assembled in - the
        local stage's matrix is always Ruiz-equilibrated first, but `a_p`
        was not, before this fix. On SPE10 AD's field-scale case that left
        `a_p`'s diagonal spanning `6.5e-15` to `3.1e-9`; solving it
        unscaled returned a pressure correction inflated to `~1.2e7`
        against a residual of `~1e-3`, which then poisoned `r - a @ x` with
        catastrophic cancellation and GMRES never recovered (`invalid
        value encountered in scalar multiply`, non-convergence regardless
        of which AMG coarsening/relaxation was tried).

        A tiny 30x30x1 SPE10 AD slice (900 cells) reproduces the same
        pathology fast: confirmed (by running this exact case against the
        pre-fix code) to fail with the identical RuntimeError signature
        there, and to converge here. Below 30x30 (e.g. 20x20, 400 cells)
        the permeability contrast is not extreme enough to trigger it -
        this specific size is not an arbitrary choice.
        """
        from pymrst.modules.ad_blackoil.examples.spe10.setup_spe10_ad import (
            setup_spe10_ad_ad, run_spe10_ad,
        )
        import numpy as _np

        case = setup_spe10_ad_ad(layers=[1], i_range=_np.arange(1, 31),
                              j_range=_np.arange(1, 31))
        lsolve = CPRSolverADAD(max_iterations=200, tolerance=1e-6)
        lsolve.elliptic_solver = AMGCLSolverADAD(
            tolerance=1e-3, coarsening="smoothed_aggregation",
            relaxation="ilu0", krylov="bicgstab")
        case.model.linear_solver = lsolve
        states, its = run_spe10_ad(case, nsteps=2, verbose=False)
        assert len(states) == 2
        assert _np.all(_np.isfinite(states[-1].pressure))
        print(f"ok test_amgcl_cpr_converges_on_a_badly_scaled_pressure_block "
             f"({case.grid.cells.num} cells, {its} Newton its)")

    test_amgcl_cpr_solves_a_field_scale_schedule()
    test_amgcl_cpr_matches_the_default_pressure_stage()
    test_amgcl_cpr_converges_on_a_badly_scaled_pressure_block()
    print("\nAll tests passed!")
