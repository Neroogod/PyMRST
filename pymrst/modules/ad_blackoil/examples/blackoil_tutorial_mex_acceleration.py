"""MRST ``ad-blackoil/examples/blackoilTutorialMexAcceleration.m``: accelerated
assembly and a compiled CPR linear solver on a 3D quarter-five-spot.

A 70x70x30 box with uniform rock, water injected at one corner and produced
at a bhp at the other, a three-phase fluid with compressibility, five 30-day
steps. The model gets the diagonal autodiff backend with its accelerated
operators (`useMex`; numba kernels here, C++ MEX in MRST), the wells take
`bhp` as their primary variable, the operators are benchmarked
(`testMexDiagonalOperators`), and the run uses AMGCL's CPR solver
(`AMGCL_CPRSolverAD`, MRST's own AMGCL C++ code compiled for Python) - once
single-threaded and once with four threads (the build here has no OpenMP, so
both run serially) - then the time is broken down with `getReportTimings`.

`dims`/`pdims` are the script's grid variables ("Obviously, you can adjust
this"). `LinearSolver` (a PyMRST addition) replaces the AMGCL solver, so the
run can be compared with MRST's where AMGCL cannot be built. `plot=False`
leaves out the figures.

`build_blackoil_tutorial_mex_acceleration_ad` and
`run_blackoil_tutorial_mex_acceleration_ad`, below, are PyMRST's earlier
version on the legacy models, kept for their callers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Sequence


def blackoil_tutorial_mex_acceleration(plot: bool = True, *, dims: Sequence[int] = (70, 70, 30),
                                       pdims: Sequence[float] = (1000.0, 1000.0, 100.0),
                                       LinearSolver: Any = None) -> SimpleNamespace:
    """MRST `blackoilTutorialMexAcceleration`."""
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.examples.utils.get_report_timings import get_report_timings
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
    from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.backends.diagonal.diagonal_auto_diff_backend import (
        DiagonalAutoDiffBackend,
    )
    from pymrst.modules.ad_core.backends.diagonal.operators.mex.test_mex_diagonal_operators import (
        test_mex_diagonal_operators,
    )
    from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
        get_initialization_regions_black_oil,
    )
    from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
        init_state_black_oil_ad,
    )
    from pymrst.modules.ad_core.solvers.amgcl_cpr_solver_ad import AMGCL_CPRSolverAD
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock, pore_volume
    from pymrst.params.gravity.gravity import gravity
    from pymrst.params.wells_and_bc.vertical_well import vertical_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, METER, POISE, YEAR

    # Define a 70x70x30 grid with uniform properties
    dims = list(dims)
    pdims = np.asarray(pdims, dtype=float)*METER
    G = compute_geometry(cart_grid(dims, pdims))
    rock = make_rock(G, 0.1*DARCY, 0.3)
    # Set up QFS water injection scenario
    gravity("reset", "on")
    time = 10*YEAR
    irate = 0.3*np.sum(pore_volume(G, rock)/time)
    W = vertical_well([], G, rock, 0, 0, None, compi=np.array([1.0, 0, 0]),
                      well_type="rate", val=irate)
    W = vertical_well(W, G, rock, dims[0] - 1, dims[1] - 1, [0], compi=np.array([1.0, 0, 0]),
                      well_type="bhp", val=100*BARSA)
    # Three-phase fluid model with compressibility and nonlinear flux
    fluid = init_simple_adi_fluid(n=[2, 3, 1], rho=[1000, 700, 10],
                                  mu=np.array([1, 5, 0.1])*CENTI*POISE,
                                  c=np.array([0, 1e-6, 1e-4])/BARSA)
    # Do 5 time-steps of 30 days each to test assembly speed
    dt = np.full(5, 30*DAY)
    schedule = simple_schedule(dt, W=W)
    model = GenericBlackOilModel(G, rock, fluid, disgas=False, vapoil=False)
    # Set up mex-accelerated backend and reduced variable set for wells
    model.AutoDiffBackend = DiagonalAutoDiffBackend(use_mex=True)
    model = model.validate_model()
    model.FacilityModel.primaryVariableSet = "bhp"
    # Run assembly tests
    operator_tests = test_mex_diagonal_operators(model, block_size=3, print=plot)
    # Set up a compiled linear solver
    ncomp = model.get_number_of_components()
    solver = AMGCL_CPRSolverAD(tolerance=1e-3, block_size=ncomp, coarsening="aggregation",
                               relaxation="ilu0")
    if LinearSolver is not None:
        solver = LinearSolver
    nls = NonLinearSolver(LinearSolver=solver)
    # Set up initial state
    region = get_initialization_regions_black_oil(model, [70, 30], datum_pressure=200*BARSA)
    state0, _ = init_state_black_oil_ad(model, region)
    if plot:
        _plot_saturations(G, W, state0, "Initial saturations")
    # Run a serial simulation
    if hasattr(solver, "amgcl_setup"):
        solver.amgcl_setup["nthreads"] = 1
    ws0, states0, rep0 = simulate_schedule_ad(state0, model, schedule, NonLinearSolver=nls)
    # Run a parallel simulation with four threads
    if hasattr(solver, "amgcl_setup"):
        solver.amgcl_setup["nthreads"] = 4
    ws, states, rep = simulate_schedule_ad(state0, model, schedule, NonLinearSolver=nls)
    N = 4
    if plot:
        _plot_saturations(G, W, states[-1], "Final saturations")
    # Plot timings
    serial = get_report_timings(rep0)
    par = get_report_timings(rep)
    total = lambda t, k: sum(x[k] for x in t)
    a = [total(serial, k) for k in ("Total", "Assembly", "LinearSolve", "LinearSolvePrep")]
    b = [total(par, k) for k in ("Total", "Assembly", "LinearSolve", "LinearSolvePrep")]
    a.append(a[0] - sum(a[1:]))
    b.append(b[0] - sum(b[1:]))
    if plot:
        _plot_timings(a, b, N)
    return SimpleNamespace(G=G, rock=rock, W=W, fluid=fluid, schedule=schedule, model=model,
                           solver=solver, nls=nls, state0=state0, ws0=ws0, states0=states0,
                           rep0=rep0, ws=ws, states=states, rep=rep, its0=rep0["Iterations"],
                           its=rep["Iterations"], serial=serial, par=par, a=a, b=b,
                           operator_tests=operator_tests)


def _plot_saturations(G: Any, W: Any, state: Any, title: str) -> None:
    import numpy as np
    from pymrst.modules.ad_blackoil.examples.utils.figures import cell_data_figure
    s = np.asarray(state["s"])[:, [1, 2, 0]]
    cell_data_figure(G, s, title, view=(30, 30))


def _plot_timings(a: Sequence[float], b: Sequence[float], N: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    x = np.arange(2)
    width = 0.15
    labels = ["Total time", "Assembly", "Linear solve", "Linear solve preparation",
              "Other functions"]
    for i, label in enumerate(labels):
        plt.bar(x + (i - 2)*width, [a[i], b[i]], width, label=label)
    plt.xticks(x, ["Serial", f"{N} threads"])
    plt.legend()


blackoilTutorialMexAcceleration = blackoil_tutorial_mex_acceleration


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.examples.utils.get_report_timings import (
    Timing, get_report_timings_ad,
)
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
    get_initialization_regions_black_oil_ad,
)
from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
    init_state_black_oil_ad_ad,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.utils.schedule import Schedule, simple_schedule
from pymrst.modules.ad_core.utils.select_linear_solver_ad import (
    select_linear_solver_ad_ad,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock, pore_volume
from pymrst.params.wells_and_bc.vertical_well import vertical_well
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, POISE, YEAR

#: MRST's `gravity reset on`.
NORM_GRAVITY = 9.80665

#: The published grid: 147000 cells, three phases, 441000 unknowns.
DIMS = (70, 70, 30)
PDIMS = (1000.0, 1000.0, 100.0)

#: `getInitializationRegionsBlackOil(model, [70, 30], ...)` - the oil-water and
#: gas-oil contacts, in metres below the top of the model.
CONTACTS = (70.0, 30.0)
DATUM_PRESSURE = 200*BARSA


@dataclass
class MexAccelerationResult:
    model: Any
    state0: Any
    schedule: Any
    states: List[Any]
    report: Any
    timings: List[Timing]
    total: Timing
    linear_solver: Any

    def breakdown(self) -> Dict[str, float]:
        """MRST's five bars: total, assembly, linear solve, prep, other."""
        return {"total": self.total.total,
                "assembly": self.total.assembly,
                "linear_solve": self.total.linear_solve,
                "linear_solve_prep": self.total.linear_solve_prep,
                "other": self.total.other}


def build_blackoil_tutorial_mex_acceleration_ad(dims: Sequence[int] = DIMS,
                                             pdims: Sequence[float] = PDIMS):
    """MRST's case, parameter for parameter."""
    dims = list(dims)
    G = compute_geometry(cart_grid(dims, list(pdims)))
    rock = make_rock(G, 0.1*DARCY, 0.3)

    time = 10*YEAR
    inj_rate = 0.3*float(np.sum(pore_volume(G, rock)))/time

    # `verticalWell` is 1-based in MATLAB. The injector takes every layer
    # (`[]`); the producer only the first (`1`).
    W: List[Any] = []
    W = vertical_well(W, G, rock, 0, 0, None,
                      compi=np.array([1.0, 0.0, 0.0]),
                      well_type="rate", val=inj_rate, name="I1")
    W = vertical_well(W, G, rock, dims[0] - 1, dims[1] - 1, [0],
                      compi=np.array([1.0, 0.0, 0.0]),
                      well_type="bhp", val=100*BARSA, name="P1")

    fluid = init_simple_adi_fluid(
        n=[2, 3, 1],
        rho=[1000.0, 700.0, 10.0],
        mu=np.array([1.0, 5.0, 0.1])*CENTI*POISE,
        c=np.array([0.0, 1e-6, 1e-4])/BARSA,
    )
    schedule = simple_schedule(np.full(5, 30.0*DAY), W=W)

    model = GenericBlackOilModelAD(G, rock, fluid, W, disgas=False, vapoil=False,
                                 gravity=NORM_GRAVITY)
    region = get_initialization_regions_black_oil_ad(
        model, list(CONTACTS), datum_pressure=DATUM_PRESSURE)
    state0 = init_state_black_oil_ad_ad(model, region)
    _seed_well_unknowns(model, state0, W)
    return model, state0, schedule


def run_blackoil_tutorial_mex_acceleration_ad(
    nsteps: Optional[int] = None, *,
    dims: Sequence[int] = DIMS, pdims: Sequence[float] = PDIMS,
    verbose: bool = True,
) -> MexAccelerationResult:
    """Run the case and break its wall time down as MRST's plot does."""
    model, state0, schedule = build_blackoil_tutorial_mex_acceleration_ad(
        dims, pdims)
    if nsteps is not None:
        count = min(int(nsteps), int(schedule.n_steps))
        schedule = Schedule(schedule.step_val[:count],
                            schedule.step_control[:count], schedule.controls)

    linear_solver = select_linear_solver_ad_ad(model)
    solver = NonLinearSolver(linear_solver=linear_solver)
    solver.verbose = bool(verbose)
    states, report = solver.simulate_schedule(model, state0, schedule)

    return MexAccelerationResult(
        model=model, state0=state0, schedule=schedule, states=states,
        report=report, timings=get_report_timings_ad(report),
        total=get_report_timings_ad(report, total=True),
        linear_solver=linear_solver,
    )


def run_thread_comparison(*_args, **_kwargs):
    """MRST's serial-vs-parallel bar - refused rather than faked."""
    raise NotImplementedError(
        "the serial/parallel comparison measures MRST's MEX-compiled diagonal "
        "AD backend at one thread against four. PyMRST assembles in NumPy and "
        "solves through SciPy, with no threaded assembly to compare against; "
        "timing the same single-threaded code twice would present measurement "
        "noise as a speed-up. `run_blackoil_tutorial_mex_acceleration` gives "
        "the breakdown the comparison is read from")


#: MRST's own spelling. MRST files each example as a script; this port gives it
#: a `run_` entry point, and the alias keeps the MATLAB name callable - the
#: convention the rest of the package follows.


def _seed_well_unknowns(model: Any, state: Any, wells) -> None:
    """The well primary variables the black-oil models read off the state."""
    wells = list(wells or ())
    phases = [str(name).upper()[0] for name in model.get_phase_names()]
    n = len(wells)
    p_mean = float(np.mean(np.asarray(state["pressure"], dtype=float)))
    rates = {f"q{phase}s": np.zeros(n) for phase in phases}
    bhp = np.zeros(n)
    for index, well in enumerate(wells):
        composition = np.atleast_1d(np.asarray(well.compi, dtype=float))
        if str(well.type).lower() == "bhp":
            bhp[index] = float(well.val)
        else:
            bhp[index] = p_mean + 10*BARSA*int(well.sign or 1)
            for position, phase in enumerate(phases):
                if position < composition.size:
                    rates[f"q{phase}s"][index] = (float(well.val)
                                                  * composition[position])
    for name, values in rates.items():
        state[name] = values
    state["bhp"] = bhp


__all__ = ["MexAccelerationResult", "blackoilTutorialMexAcceleration",
           "blackoil_tutorial_mex_acceleration", "build_blackoil_tutorial_mex_acceleration_ad",
           "run_blackoil_tutorial_mex_acceleration_ad"]


if __name__ == "__main__":
    def test_the_tutorial_on_a_small_grid():
        import numpy as np
        out = blackoil_tutorial_mex_acceleration(plot=False, dims=(8, 8, 4), pdims=(100, 100, 10))
        assert len(out.states0) == 5 and len(out.states) == 5
        assert out.model.FacilityModel.primaryVariableSet == "bhp"
        assert type(out.solver).__name__ == "AMGCL_CPRSolverAD"
        np.testing.assert_array_equal(np.asarray(out.its0), np.asarray(out.its))
        assert out.operator_tests["div"]["jacobian_error"] < 1e-14
        assert out.a[0] > 0 and abs(sum(out.a[1:]) - out.a[0]) < 1e-9
        print("ok test_the_tutorial_on_a_small_grid")

    test_the_tutorial_on_a_small_grid()
    print("\nAll tests passed!")
