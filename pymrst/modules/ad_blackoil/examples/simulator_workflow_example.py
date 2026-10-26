"""MRST ``ad-blackoil/examples/simulatorWorkflowExample.m``: the whole
MRST-AD workflow on one self-contained case.

A faulted corner-point model (`makeModel3([50 50 5], [1000 1000 5])`,
three permeability layers, `K_z = K_x/10`) with three bhp producers and a
rate injector is simulated with `TwoPhaseOilWaterModel`, oil compressibility
`bO = exp((p - 300 bar) 0.001/bar)` and gravity on, from an equilibrium
initial state. The model is then coarsened to the `partitionUI` blocks split
over the faults (`processPartition` on `makeInternalBoundary`) twice: by
`upscaleModelTPFA`/`upscaleSchedule`/`upscaleState` with harmonic
permeabilities, and with `upscaleTrans`'s flow-based transmissibilities and
well indices from the problem's own wells. All three are simulated and
compared through `computeTOFandTracer`. `plot=False` leaves out the figures.

`makeModel3` perturbs the grid with `rand` and `randn` draws. The script, run
in a fresh MATLAB session, takes them from MATLAB's default stream; `randn`
(Ziggurat) cannot be generated in NumPy, so those draws are
`MATLAB_DEFAULT_DRAWS` below and are used unless `draws` gives others. Other
realisations are not interchangeable: on NumPy's own normal draws the
injector's top cell lies below the contact, the first Newton update sends its
bhp to 1e11 Pa, `bO` overflows, and MRST's script fails at step 1 exactly as
the port does. MRST's last section opens
`interactiveDiagnostics`, an interactive viewer that is not ported; with
`plot` the static diagnostics figures before it are drawn.

`run_simulator_workflow_example_ad` and the functions after it are PyMRST's
earlier version, kept for their callers. Its notes follow.

Port of MRST ``ad-blackoil/examples/simulatorWorkflowExample.m``.

MRST's script walks one case through a whole workflow: build a faulted
corner-point model, simulate it, coarsen it two ways, simulate both, and
compare all three with flow diagnostics. This is that, parameter for
parameter, minus the plotting.

The previous port shared none of it. It built a 12x12x1 `cart_grid` with
uniform 100 mD rock and two wells, where MRST processes a **faulted
corner-point grid** from `makeModel3([50 50 5], [1000 1000 5])` - 10020 cells,
1209 fault faces, three permeability layers and four wells. The fault is not
decoration: it is the reason the script partitions with `processPartition`
rather than `partitionUI` alone, and a Cartesian grid has nothing to split.

Three parameters need a word, because each is a place the port could quietly
diverge:

* **`bO` is overridden after the fluid is built.** `initSimpleADIFluid` gives
  an oil `b` from `cR`; the script then replaces it with
  `exp((p - 300 bar) * 0.001/bar)`, a compressibility a hundred times larger.
  Keeping the constructed one makes the oil nearly incompressible and the
  pressure transient far too stiff;
* **the model is gravity-driven and equilibrated.** `TwoPhaseOilWaterModelAD`
  takes no `gravity`, so this uses `GenericBlackOilModelAD`, which is PyMRST's
  two-phase-capable model that does - `TwoPhaseOilWaterModel` in MRST is a
  restriction of `ThreePhaseBlackOilModel`, and the restriction that matters
  here is the phase set, not the gravity;
* **logical well positions are 1-based in MATLAB.** MRST's `offset = 10` is
  PyMRST's `9`. Off by one puts every well in the neighbouring column.

**The flow-based upscaling is not ported.** `upscaleTrans` lives in MRST's
`upscaling` module, requires `incomp` and `agglom`, and is 738 lines of
transmissibility fitting. `run_flow_upscaling` refuses with that named rather
than approximating it - a wrong coarse transmissibility gives a coarse model
that runs and answers differently, which is exactly what an upscaling
comparison is supposed to detect.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pymrst.gridprocessing import compute_geometry, process_grdecl
from pymrst.gridprocessing.testgrids.make_model3 import make_model3
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
    get_initialization_regions_black_oil_ad,
)
from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
    init_state_black_oil_ad_ad,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa_ad as upscale_model_tpfa
from pymrst.modules.ad_core.upscale.upscale_schedule import upscale_schedule_ad as upscale_schedule
from pymrst.modules.ad_core.upscale.upscale_state import upscale_state_ad as upscale_state
from pymrst.modules.ad_core.utils.schedule import simple_schedule
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock, pore_volume
from pymrst.params.wells_and_bc.vertical_well import vertical_well
from pymrst.utils.adi import ad_exp
from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE, YEAR


#: MATLAB's `rand(6, 1)` and `randn(6, 2)` from the default stream of a fresh
#: session (`rng(0)`): the draws MRST's `makeModel3([50 50 5], ...)` takes.
MATLAB_DEFAULT_DRAWS = (
    np.array([0.8147236863931789, 0.9057919370756192, 0.12698681629350606,
              0.9133758561390194, 0.6323592462254095, 0.09754040499940952]),
    np.array([[-0.43359202230568356, 0.7254042249461056],
              [0.3426244665386499, -0.06305487318965619],
              [3.5783969397257605, 0.7147429038260958],
              [2.769437029884877, -0.20496605829977463],
              [-1.3498869401565212, -0.12414434821631191],
              [3.0349234663318545, 1.489697607785465]]),
)


def simulator_workflow_example(plot: bool = True,
                               draws: Optional[Tuple[Any, Any]] = None) -> SimpleNamespace:
    """MRST `simulatorWorkflowExample`: every stage's grid, model, schedule,
    states, well solutions, reports and diagnostics."""
    from pymrst.gridprocessing.make_internal_boundary import make_internal_boundary
    from pymrst.modules.ad_blackoil.models import TwoPhaseOilWaterModel
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
        get_initialization_regions_black_oil,
    )
    from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
        init_state_black_oil_ad,
    )
    from pymrst.modules.ad_core.upscale import (
        upscale_model_tpfa as upscale_model_tpfa_mrst,
        upscale_schedule as upscale_schedule_mrst,
        upscale_state as upscale_state_mrst,
    )
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_ui import partition_ui
    from pymrst.modules.coarsegrid.process_partition import process_partition
    from pymrst.modules.diagnostics.utils.compute_tof_and_tracer import compute_tof_and_tracer
    from pymrst.modules.upscaling.upscale_trans import upscale_trans
    from pymrst.params.gravity.gravity import gravity
    from pymrst.utils.units import KILOGRAM, METER

    # Reservoir geometry and petrophysical properties
    draws = MATLAB_DEFAULT_DRAWS if draws is None else draws
    grdecl = make_model3([50, 50, 5], np.array([1000, 1000, 5])*METER, draws=draws)
    G = process_grdecl(grdecl)
    G = compute_geometry(G)
    I, J, K = grid_logical_indices(G)[:3]
    K = K + 1
    top = K < G.cartDims[2]/3
    lower = K > 2*G.cartDims[2]/3
    middle = ~(lower | top)
    px = np.ones(G.cells.num)
    px[lower] = 300*MILLI*DARCY
    px[middle] = 100*MILLI*DARCY
    px[top] = 500*MILLI*DARCY
    perm = np.column_stack([px, px, 0.1*px])
    rock = make_rock(G, perm, 0.3)

    # Wells and simulation schedule (MRST's 1-based logical indices, less one)
    simTime = 10*YEAR
    nstep = 25
    refine = 5
    pv = pore_volume(G, rock)
    injRate = 1*np.sum(pv)/simTime
    offset = 10
    common = dict(compi=np.array([1.0, 0.0]), ref_depth=-8)
    W = vertical_well([], G, rock, offset - 1, offset - 1, None, name="P1",
                      val=250*BARSA, well_type="bhp", **common)
    W = vertical_well(W, G, rock, offset - 1, G.cartDims[0]//2 + 3 - 1, None, name="P2",
                      val=250*BARSA, well_type="bhp", **common)
    W = vertical_well(W, G, rock, offset - 1, int(G.cartDims[1] - offset/2) - 1, None,
                      name="P3", val=250*BARSA, well_type="bhp", **common)
    W = vertical_well(W, G, rock, G.cartDims[0] - 5 - 1, offset - 1, [0], name="I1",
                      val=injRate, well_type="rate", **common)
    startSteps = np.full(refine, (simTime/(nstep + 1))/refine)
    restSteps = np.full(nstep, simTime/(nstep + 1))
    timesteps = np.concatenate([startSteps, restSteps])
    schedule = simple_schedule(timesteps, W=W)

    # Simulation model
    fluid = init_simple_adi_fluid(mu=np.array([1, 5])*CENTI*POISE,
                                  rho=np.array([1000, 700])*KILOGRAM/METER**3,
                                  n=[2, 2], c_r=1e-8/BARSA, phases="wo")
    c = 0.001/BARSA
    p_ref = 300*BARSA
    fluid.bO = lambda p, *varargin: ad_exp((p - p_ref)*c)
    gravity("reset", "on")
    model = TwoPhaseOilWaterModel(G, rock, fluid)

    # Initial state and the base case
    region = get_initialization_regions_black_oil(model, 5, datum_pressure=p_ref)
    state0, _ = init_state_black_oil_ad(model, region)
    wellSols, states, report = simulate_schedule_ad(state0, model, schedule)

    # Coarse partition, split over the fault lines
    cdims = [5, 5, 2]
    p0 = partition_ui(G, cdims)
    CG = generate_coarse_grid(G, p0)
    G_fault = make_internal_boundary(G, np.flatnonzero(np.asarray(G.faces.tag) > 0))[0]
    p = process_partition(G_fault, p0)

    # Upscaled model, schedule and state, and the coarse run
    model_c = upscale_model_tpfa_mrst(model, p)
    G_c = model_c.G
    rock_c = model_c.rock
    schedule_c = upscale_schedule_mrst(model_c, schedule)
    state0_c = upscale_state_mrst(model_c, model, state0)
    wellSols_c, states_c, report_c = simulate_schedule_ad(state0_c, model_c, schedule_c)

    # Flow-based transmissibility upscaling
    _, T_c, W_c, _ = upscale_trans(G_c, model.operators.t_all, wells=W, bc_method="wells",
                                   fix_trans=True)
    model_c2 = upscale_model_tpfa_mrst(model, p, transCoarse=T_c)
    schedule_c2 = copy.deepcopy(schedule)
    for control in schedule_c2.controls:
        control.W = W_c
    wellSols_c2, states_c2, report_c2 = simulate_schedule_ad(state0_c, model_c2, schedule_c2)

    # Flow diagnostics
    D = compute_tof_and_tracer(states[-1], G, rock, wells=schedule.controls[0].W)
    D_c = compute_tof_and_tracer(states_c[-1], G_c, rock_c, wells=schedule_c.controls[0].W)
    D_c2 = compute_tof_and_tracer(states_c2[-1], G_c, rock_c, wells=schedule_c.controls[0].W)

    out = SimpleNamespace(
        G=G, rock=rock, W=W, timesteps=timesteps, schedule=schedule, fluid=fluid, model=model,
        state0=state0, wellSols=wellSols, states=states, report=report,
        p0=p0, CG=CG, G_fault=G_fault, p=p,
        model_c=model_c, G_c=G_c, rock_c=rock_c, schedule_c=schedule_c, state0_c=state0_c,
        wellSols_c=wellSols_c, states_c=states_c, report_c=report_c,
        T_c=T_c, W_c=W_c, model_c2=model_c2, schedule_c2=schedule_c2,
        wellSols_c2=wellSols_c2, states_c2=states_c2, report_c2=report_c2,
        D=D, D_c=D_c, D_c2=D_c2)
    if plot:
        _plot_workflow(out, p_ref, c)
    return out


def _plot_workflow(out: SimpleNamespace, p_ref: float, c: float) -> None:
    """MRST's figures; coarse quantities are drawn on the fine cells of each
    block, as `plotCellData` does for a coarse grid."""
    import matplotlib.pyplot as plt

    from pymrst.modules.ad_blackoil.examples.utils.figures import cell_data_figure
    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    view = (50, 50)
    G, p = out.G, np.asarray(out.p)
    cell_data_figure(G, out.rock.perm[:, 0]/(MILLI*DARCY), "K_x [mD]", view=view)
    p0 = np.arange(100, 501, 10)*BARSA
    plt.figure()
    plt.plot(p0/BARSA, np.exp((p0 - p_ref)*c))
    plt.xlabel("Pressure (bar)")
    plt.ylabel("Ratio")
    plt.title("Reciprocal formation volume factor for oil (bO)")
    cell_data_figure(G, np.asarray(out.state0["s"])[:, 0], "Initial water saturation", view=view)
    cell_data_figure(G, np.asarray(out.p0, dtype=float), "Straightforward index partition",
                     view=(125, 55))
    cell_data_figure(G, (p != np.asarray(out.p0)).astype(float), "Splitting over fault lines",
                     view=(125, 55))
    cell_data_figure(G, np.asarray(out.states_c[-1]["s"])[:, 0][p],
                     "Water saturation, simple upscaling", view=view)
    cell_data_figure(G, np.asarray(out.states_c2[-1]["s"])[:, 0][p],
                     "Water saturation, flow upscaling", view=view)
    plot_well_sols([out.wellSols, out.wellSols_c, out.wellSols_c2],
                   np.cumsum(out.timesteps),
                   DatasetNames=["Fine scale", "Simple upscaling", "Flow upscaling"], Field="qOs")
    total = np.log10(out.D.tof.sum(axis=1))
    plotter = cell_data_figure(G, total, "Log of total travel time, fine model", view=view)
    clim = (total.min(), total.max())
    cell_data_figure(G, np.log10(out.D_c.tof.sum(axis=1))[p],
                     "Log of total travel time, coarse model", view=view, clim=clim)
    cell_data_figure(G, np.log10(out.D_c2.tof.sum(axis=1))[p],
                     "Log of total travel time, coarse model", view=view, clim=clim)
    cell_data_figure(G, out.D.ppart + 1.0, "Drainage regions, fine model", view=view)
    cell_data_figure(G, (out.D_c.ppart + 1.0)[p], "Drainage regions, simple upscaled model",
                     view=view)
    cell_data_figure(G, (out.D_c2.ppart + 1.0)[p], "Drainage regions, flow upscaled model",
                     view=view)
    del plotter
    plt.show()


simulatorWorkflowExample = simulator_workflow_example


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

#: MRST's `gravity reset on`.
NORM_GRAVITY = 9.80665

#: The reference pressure the oil `b` is written against.
P_REF = 300*BARSA


@dataclass
class WorkflowCase:
    """What the first third of the script builds."""

    G: Any
    rock: Any
    fluid: Any
    model: Any
    state0: Any
    schedule: Any
    W: List[Any]


@dataclass
class WorkflowResult:
    fine: Dict[str, Any] = field(default_factory=dict)
    coarse: Dict[str, Any] = field(default_factory=dict)
    partition: Optional[np.ndarray] = None


def build_simulator_workflow_example_ad() -> WorkflowCase:
    """MRST's model, rock, fluid, wells and schedule, parameter for parameter."""
    grdecl = make_model3([50, 50, 5], [1000.0, 1000.0, 5.0])
    G = compute_geometry(process_grdecl(grdecl))

    # MRST indexes layers from one; `grid_logical_indices` from zero.
    _I, _J, K = grid_logical_indices(G)
    k = np.asarray(K, dtype=np.int64) + 1
    nz = int(G.cartDims[2])
    top = k < nz/3
    lower = k > 2*nz/3
    middle = ~(lower | top)

    px = np.ones(G.cells.num)
    px[lower] = 300*MILLI*DARCY
    px[middle] = 100*MILLI*DARCY
    px[top] = 500*MILLI*DARCY
    rock = make_rock(G, np.column_stack([px, px, 0.1*px]), 0.3)

    sim_time = 10*YEAR
    nstep, refine = 25, 5
    pv = pore_volume(G, rock)
    inj_rate = 1*float(np.sum(pv))/sim_time
    offset = 10

    # `verticalWell` is 1-based in MATLAB; subtract one from each index.
    W: List[Any] = []
    W = vertical_well(W, G, rock, offset - 1, offset - 1, None,
                      name="P1", compi=np.array([1.0, 0.0]),
                      val=250*BARSA, well_type="bhp", ref_depth=-8)
    W = vertical_well(W, G, rock, offset - 1,
                      int(np.floor(G.cartDims[0]/2)) + 3 - 1, None,
                      name="P2", compi=np.array([1.0, 0.0]),
                      val=250*BARSA, well_type="bhp", ref_depth=-8)
    W = vertical_well(W, G, rock, offset - 1,
                      int(G.cartDims[1] - offset/2) - 1, None,
                      name="P3", compi=np.array([1.0, 0.0]),
                      val=250*BARSA, well_type="bhp", ref_depth=-8)
    W = vertical_well(W, G, rock, int(G.cartDims[0]) - 5 - 1, offset - 1, [0],
                      name="I1", compi=np.array([1.0, 0.0]),
                      val=inj_rate, well_type="rate", ref_depth=-8)

    start_steps = np.full(refine, (sim_time/(nstep + 1))/refine)
    rest_steps = np.full(nstep, sim_time/(nstep + 1))
    timesteps = np.concatenate([start_steps, rest_steps])
    schedule = simple_schedule(timesteps, W=W)

    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0])*CENTI*POISE,
        rho=[1000.0, 700.0],
        n=[2, 2],
        c=[1e-8/BARSA, 1e-8/BARSA],       # MRST's `cR`
        p_ref=P_REF,
    )
    # MRST overrides the constructed oil `b` outright. `c` here is
    # 0.001/barsa - a hundred times the `cR` above - so keeping the
    # constructed curve makes the oil nearly incompressible.
    # `ad_exp`, not `np.exp`: this is evaluated on the pressure the Newton
    # solver differentiates, and `np.exp` on an `ADI` raises outright.
    c = 0.001/BARSA
    fluid.b_o = lambda p, *args, c=c, p_ref=P_REF: ad_exp((p - p_ref)*c)

    # `GenericBlackOilModelAD`, not `TwoPhaseOilWaterModelAD`: the latter takes
    # no gravity, and without gravity the equilibration below is vacuous.
    model = GenericBlackOilModelAD(G, rock, fluid, W, gravity=NORM_GRAVITY)

    region = get_initialization_regions_black_oil_ad(model, [5.0],
                                                  datum_pressure=P_REF)
    state0 = init_state_black_oil_ad_ad(model, region)
    _seed_well_unknowns(model, state0, W)

    return WorkflowCase(G=G, rock=rock, fluid=fluid, model=model,
                        state0=state0, schedule=schedule, W=W)


def run_simulator_workflow_example_ad(nsteps: Optional[int] = None, *,
                                   coarse: bool = True,
                                   verbose: bool = True) -> WorkflowResult:
    """Simulate the fine model, then the simply upscaled coarse one."""
    case = build_simulator_workflow_example_ad()
    schedule = _truncate(case.schedule, nsteps)

    solver = NonLinearSolver()
    solver.verbose = bool(verbose)
    states, report = solver.simulate_schedule(case.model, case.state0, schedule)
    out = WorkflowResult(fine={"model": case.model, "states": states,
                               "report": report, "schedule": schedule,
                               "state0": case.state0, "W": case.W})
    if not coarse:
        return out

    partition = coarse_partition_ad(case.G, [5, 5, 2])
    out.partition = partition
    model_c = upscale_model_tpfa(case.model, partition)
    schedule_c = upscale_schedule(model_c, schedule)
    state0_c = upscale_state(model_c, case.model, case.state0)
    _seed_well_unknowns(model_c, state0_c, schedule_c.controls[0].W)

    states_c, report_c = solver.simulate_schedule(model_c, state0_c, schedule_c)
    out.coarse = {"model": model_c, "states": states_c, "report": report_c,
                  "schedule": schedule_c, "state0": state0_c}
    return out


def coarse_partition_ad(G: Any, cdims=(5, 5, 2)) -> np.ndarray:
    """A uniform partition, split along the faults (MRST's second step).

    `partitionUI` alone puts cells from either side of a fault in one block, so
    the coarse model lets flow cross a barrier the fine model blocks.
    `processPartition` against a grid with the faults made into internal
    boundaries is what separates them.
    """
    from pymrst.gridprocessing.make_internal_boundary import (
        make_internal_boundary,
    )
    from pymrst.modules.coarsegrid.partition_ui import partition_ui
    from pymrst.modules.coarsegrid.process_partition import process_partition

    p0 = partition_ui(G, list(cdims))
    tag = np.asarray(getattr(G.faces, "tag", np.zeros(G.faces.num)))
    faults = np.flatnonzero(tag > 0)
    if faults.size == 0:
        return p0
    # `make_internal_boundary` returns `(grid, N)`; MRST's returns the grid
    # alone. Passing the tuple on reaches `process_partition` as a tuple and
    # the fault split silently never happens.
    G_fault, _replaced = make_internal_boundary(G, faults)
    return process_partition(G_fault, p0)


def run_flow_upscaling_ad(*_args, **_kwargs):
    """The third of MRST's three models - refused rather than approximated."""
    raise NotImplementedError(
        "flow-based transmissibility upscaling needs `upscaleTrans`, which is "
        "in MRST's `upscaling` module (and requires `incomp` and `agglom`); "
        "it is not ported. An approximate coarse transmissibility would give "
        "a coarse model that runs and answers differently, which is the very "
        "thing this comparison exists to measure")


def flow_diagnostics_ad(model: Any, state: Any, wells) -> Any:
    """Time-of-flight and tracer partition (MRST `computeTOFandTracer`)."""
    from pymrst.modules.diagnostics.utils.compute_tof_and_tracer import (
        compute_diagnostics,
    )
    return compute_diagnostics(model.G, model.rock, list(wells), state)




def _truncate(schedule: Any, nsteps: Optional[int]):
    if nsteps is None:
        return schedule
    from pymrst.modules.ad_core.utils.schedule import Schedule
    count = min(int(nsteps), int(schedule.n_steps))
    return Schedule(schedule.step_val[:count], schedule.step_control[:count],
                    schedule.controls)


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


__all__ = ["NORM_GRAVITY", "P_REF", "WorkflowCase", "WorkflowResult",
           "build_simulator_workflow_example_ad", "coarse_partition_ad",
           "flow_diagnostics_ad", "run_flow_upscaling_ad",
           "run_simulator_workflow_example_ad", "simulatorWorkflowExample",
           "simulator_workflow_example", "MATLAB_DEFAULT_DRAWS"]


if __name__ == "__main__":
    CASE = build_simulator_workflow_example_ad()

    def test_the_grid_is_the_faulted_corner_point_model():
        """`makeModel3` + `processGRDECL`, not a Cartesian box. The previous
        port used `cart_grid([12, 12, 1])` - 144 cells and no fault."""
        assert CASE.G.cells.num == 10020, CASE.G.cells.num
        assert tuple(CASE.G.cartDims) == (50, 50, 5)
        assert int(np.sum(np.asarray(CASE.G.faces.tag) > 0)) > 0, "no fault"
        print(f"ok test_the_grid_is_the_faulted_corner_point_model "
              f"({CASE.G.cells.num} cells, "
              f"{int(np.sum(np.asarray(CASE.G.faces.tag) > 0))} fault faces)")

    def test_the_permeability_is_three_layers_not_one_value():
        """500 mD on top, 100 in the middle, 300 below - the contrast the
        upscaling comparison is about. A uniform field upscales exactly and
        the whole exercise measures nothing."""
        kx = np.asarray(CASE.rock.perm)[:, 0]/(MILLI*DARCY)
        assert sorted(set(np.round(kx, 6))) == [100.0, 300.0, 500.0], set(kx)
        kz = np.asarray(CASE.rock.perm)[:, 2]/(MILLI*DARCY)
        assert np.allclose(kz, 0.1*kx, atol=0.0)
        print("ok test_the_permeability_is_three_layers_not_one_value")

    def test_the_layers_are_assigned_by_matlabs_one_based_index():
        """`K < cartDims(3)/3` with K starting at one selects layer 1 alone;
        starting at zero selects layers 1 and 2, and the top of the model
        gets the wrong permeability."""
        _I, _J, K = grid_logical_indices(CASE.G)
        k = np.asarray(K) + 1
        kx = np.asarray(CASE.rock.perm)[:, 0]/(MILLI*DARCY)
        assert np.allclose(kx[k == 1], 500.0)
        assert np.allclose(kx[(k == 2) | (k == 3)], 100.0)
        assert np.allclose(kx[k >= 4], 300.0)
        print("ok test_the_layers_are_assigned_by_matlabs_one_based_index")

    def test_the_four_wells_are_three_producers_and_one_injector():
        assert [w.name for w in CASE.W] == ["P1", "P2", "P3", "I1"]
        assert [w.type for w in CASE.W] == ["bhp", "bhp", "bhp", "rate"]
        assert all(np.isclose(w.val, 250*BARSA) for w in CASE.W[:3])
        assert all(np.isclose(w.refDepth, -8.0) for w in CASE.W)
        print("ok test_the_four_wells_are_three_producers_and_one_injector")

    def test_the_injector_is_perforated_in_the_top_layer_only():
        """MRST passes `1` for the layer; the producers pass `[]`, which is
        every layer. Perforating the injector through the whole column
        injects into the wrong part of the model."""
        _I, _J, K = grid_logical_indices(CASE.G)
        k = np.asarray(K) + 1
        assert set(k[np.asarray(CASE.W[3].cells)]) == {1}
        assert len(CASE.W[0].cells) > 1
        print("ok test_the_injector_is_perforated_in_the_top_layer_only")

    def test_the_injection_rate_fills_the_pore_volume_in_ten_years():
        """`injRate = sum(pv)/simTime`: one pore volume over the run."""
        pv = float(np.sum(pore_volume(CASE.G, CASE.rock)))
        assert np.isclose(CASE.W[3].val*10*YEAR, pv, rtol=1e-12)
        print("ok test_the_injection_rate_fills_the_pore_volume_in_ten_years")

    def test_the_schedule_refines_its_first_step_five_ways():
        """Five short steps then twenty-five long ones - the transient at
        startup is what needs them. Thirty equal steps is a different run.

        The five refined steps together are one ordinary step, so the total
        is `26 * simTime/26` - exactly the ten years asked for.
        """
        dt = np.asarray(CASE.schedule.step_val)
        assert dt.size == 30, dt.size
        assert np.allclose(dt[:5], dt[5]/5)
        assert np.isclose(dt[:5].sum(), dt[5], rtol=1e-12)
        assert np.isclose(dt.sum(), 10*YEAR, rtol=1e-12), dt.sum()/YEAR
        print("ok test_the_schedule_refines_its_first_step_five_ways")

    def test_the_oil_b_is_the_overridden_exponential():
        """MRST replaces `bO` after building the fluid, with a
        compressibility a hundred times the constructed one."""
        p = np.array([200*BARSA, 400*BARSA])
        assert np.allclose(CASE.fluid.b_o(p),
                           np.exp((p - P_REF)*(0.001/BARSA)))
        assert not np.isclose(CASE.fluid.b_o(np.array([400*BARSA]))[0], 1.0)
        print("ok test_the_oil_b_is_the_overridden_exponential")

    def test_the_state_is_equilibrated_about_the_contact_at_five_metres():
        z = np.asarray(CASE.G.cells.centroids)[:, 2]
        sw = np.asarray(CASE.state0["sW"])
        assert sw.min() < sw.max(), "a uniform state is not equilibrated"
        assert sw[z > 5.0].mean() > sw[z < 5.0].mean()
        print(f"ok test_the_state_is_equilibrated_about_the_contact_at_five_metres "
              f"(sW {sw.min():.2f}-{sw.max():.2f})")

    def test_the_partition_is_split_along_the_faults():
        """`partitionUI` alone puts cells from either side of a fault in one
        block, and the coarse model then lets flow cross a barrier the fine
        model blocks."""
        from pymrst.modules.coarsegrid.partition_ui import partition_ui
        plain = partition_ui(CASE.G, [5, 5, 2])
        split = coarse_partition_ad(CASE.G, [5, 5, 2])
        assert split.max() > plain.max(), (plain.max(), split.max())
        print(f"ok test_the_partition_is_split_along_the_faults "
              f"({plain.max() + 1} -> {split.max() + 1} blocks)")

    def test_the_flow_upscaling_is_refused_and_names_its_module():
        try:
            run_flow_upscaling_ad()
        except NotImplementedError as refused:
            assert "upscaleTrans" in str(refused)
            assert "upscaling" in str(refused)
            print("ok test_the_flow_upscaling_is_refused_and_names_its_module")
            return
        raise AssertionError("an unported upscaling must not look done")

    def test_the_fine_and_coarse_models_both_run():
        """Three report steps of each - enough to show the coarse model is a
        model and not a data structure."""
        out = run_simulator_workflow_example_ad(3, verbose=False)
        assert len(out.fine["states"]) == 3
        assert all(s.converged for s in out.fine["report"].steps)
        assert len(out.coarse["states"]) == 3
        assert all(s.converged for s in out.coarse["report"].steps)
        fine_cells = out.fine["model"].G.cells.num
        coarse_cells = out.coarse["model"].G.cells.num
        assert coarse_cells < fine_cells
        print(f"ok test_the_fine_and_coarse_models_both_run "
              f"({fine_cells} -> {coarse_cells} cells)")

    def test_the_faithful_workflow_runs_every_stage():
        """MRST's script end to end, without figures: the fault-split partition
        of 70 blocks, both upscalings, three runs over the 30 report steps and
        three sets of diagnostics. Against MRST itself see
        examples/validation/validate_ad_oo_simulator_workflow_example.py."""
        from pymrst.params.gravity.gravity import isolated_gravity
        with isolated_gravity():
            out = simulator_workflow_example(plot=False)
        assert out.G.cells.num == 10020 and out.G_c.cells.num == 70
        for states in (out.states, out.states_c, out.states_c2):
            assert len(states) == 30
        assert np.all(np.isfinite(out.T_c)) and len(out.W_c) == 4
        assert np.all(out.W_c[3].WI > 0) and out.W_c[3].type == "rate"
        for D, n in ((out.D, 10020), (out.D_c, 70), (out.D_c2, 70)):
            assert D.tof.shape == (n, 2) and np.array_equal(D.inj, [3])
            assert np.array_equal(D.prod, [0, 1, 2]) and set(np.unique(D.ppart)) <= {-1, 0, 1, 2}
        print("ok test_the_faithful_workflow_runs_every_stage")

    test_the_grid_is_the_faulted_corner_point_model()
    test_the_permeability_is_three_layers_not_one_value()
    test_the_layers_are_assigned_by_matlabs_one_based_index()
    test_the_four_wells_are_three_producers_and_one_injector()
    test_the_injector_is_perforated_in_the_top_layer_only()
    test_the_injection_rate_fills_the_pore_volume_in_ten_years()
    test_the_schedule_refines_its_first_step_five_ways()
    test_the_oil_b_is_the_overridden_exponential()
    test_the_state_is_equilibrated_about_the_contact_at_five_metres()
    test_the_partition_is_split_along_the_faults()
    test_the_flow_upscaling_is_refused_and_names_its_module()
    test_the_fine_and_coarse_models_both_run()
    test_the_faithful_workflow_runs_every_stage()
    print("\nAll tests passed!")
