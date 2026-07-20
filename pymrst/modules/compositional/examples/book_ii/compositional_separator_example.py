"""Port of MRST `book-ii/compositionalSeparatorExample.m`.

The original MRST example compares a base producer, a single separator, and a
multi-stage separator tree. PyMRST exposes the same separator objects natively;
this compact example applies them to the SPE5 produced component stream.
"""

from __future__ import annotations

import copy
import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.models.facilities.separators import EOSSeparator, SeparatorGroup
from pymrst.modules.compositional.fluids import get_compositional_fluid_case
from pymrst.modules.compositional.examples._shared import (
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    ScheduleControl,
    run_schedule,
    simple_schedule,
    simulate_packed_problem,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (
    make_boundary_conditions_case,
)
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import ATM, BARSA, DARCY, DAY


def run_compositional_separator_example(stream_mole=None):
    """Run SPE5 through a stock-tank separator and MRST's four-stage tree."""
    mixture, info = get_compositional_fluid_case("spe5")
    if stream_mole is None:
        stream_mole = np.asarray(info.initial, dtype=float)
    else:
        stream_mole = np.asarray(stream_mole, dtype=float)

    stock_tank, single_separator, tree_separator = make_separator_tree()

    single = single_separator.compute_surface_rates(mixture, stream_mole)
    tree = tree_separator.compute_surface_rates(mixture, stream_mole)
    tree_surface_mole_streams = tree.surface_mole_streams
    tree_separator_streams = tree.separator_streams

    return {
        "mixture": mixture,
        "info": info,
        "stream_mole": stream_mole,
        "stock_tank": stock_tank,
        "single_separator": single_separator,
        "tree_separator": tree_separator,
        "single": single,
        "tree": tree,
        "tree_surface_mole_streams": tree_surface_mole_streams,
        "tree_separator_streams": tree_separator_streams,
    }


def make_separator_tree():
    """Build the single separator and MRST four-stage separator tree."""
    stock_tank = EOSSeparator(pressure=ATM, T=300.0)
    single_separator = SeparatorGroup(stock_tank)
    tree_separator = SeparatorGroup(
        stock_tank,
        pressure=np.array([200.0, 175.0, 50.0, 10.0]) * BARSA,
        T=np.array([423.0, 400.0, 350.0, 300.0]),
        phase_destination=np.array(
            [
                [2, 3],
                [0, 4],
                [4, 0],
                [0, 0],
            ]
        ),
    )
    return stock_tank, single_separator, tree_separator


def setup_separator_depletion_case(
    dims: tuple[int, int, int] = (3, 3, 2),
    *,
    rate: float = -1.0e-5,
    timestep: float = 0.05 * DAY,
    nstep: int = 1,
):
    """Build a scaled SPE5 depletion case shaped like MRST's separator example."""
    dims = tuple(int(v) for v in dims)
    grid = compute_geometry(cart_grid(dims, [100.0, 100.0, 30.0]))
    rock = make_rock(grid, 0.1 * DARCY, 0.3)
    mixture, info = get_compositional_fluid_case("spe5")
    ix = min(max(dims[0] // 2, 0), dims[0] - 1)
    iy = min(max(dims[1] // 2, 0), dims[1] - 1)
    cell = ix + iy * dims[0]
    wells = add_well(
        None,
        grid,
        rock,
        np.array([cell]),
        well_type="rate",
        val=float(rate),
        compi=np.array([1.0, 0.0]),
        sign=-1,
        name="Producer",
    )
    wells[0].components = info.initial
    model = OverallCompositionModel(
        grid,
        rock,
        mixture,
        temperature=info.T,
        wells=wells,
        rho_s=(100.0, 1.0),
        max_iterations=60,
    )
    state0 = model.initial_state(info.pressure, info.initial)
    schedule = simple_schedule(np.full(int(nstep), float(timestep)), wells=wells)
    return grid, rock, mixture, info, wells, model, state0, schedule


def setup_compositional_separator_packed_problems(
    dims: tuple[int, int, int] = (3, 3, 2),
    *,
    simulate: bool = False,
):
    """Create MRST-shaped packed separator problems.

    Mirrors the MATLAB script's ``packer`` block and problem ordering:
    ``Overall-NoSep``, ``Overall-MultiSep``, and ``Overall-SingleSep``.
    Execution is optional because the compact default validation checks exact
    setup parity; separator flash/tree physics is MATLAB-backed by Phase 116.
    """
    grid, rock, mixture, info, wells, base_model, state0, schedule = setup_separator_depletion_case(dims)
    _, single_separator, tree_separator = make_separator_tree()

    def clone_model(separator_group=None):
        return OverallCompositionModel(
            grid,
            rock,
            mixture,
            temperature=info.T,
            wells=copy.deepcopy(wells),
            rho_s=base_model.rho_s,
            max_iterations=base_model.max_iterations,
            separator_group=separator_group,
        )

    def packer(model, name):
        return pack_simulation_problem(
            state0,
            model,
            schedule,
            "depletion_spe5",
            name=name,
        )

    problem = packer(clone_model(None), "Overall-NoSep")
    problem_sep = packer(clone_model(single_separator), "Overall-SingleSep")
    problem_msep = packer(clone_model(tree_separator), "Overall-MultiSep")
    problems = [problem, problem_msep, problem_sep]
    if simulate:
        simulate_packed_problem(problems)
        _, states, reports, names = get_multiple_packed_simulator_outputs(problems)
    else:
        states, reports, names = [], [], [p.name for p in problems]
    separator_allocation_diagnostics = _packed_separator_allocation_diagnostics(problems, states)

    return {
        "grid": grid,
        "rock": rock,
        "mixture": mixture,
        "info": info,
        "wells": wells,
        "schedule": schedule,
        "stock_tank": single_separator.surface_separator,
        "single_separator": single_separator,
        "tree_separator": tree_separator,
        "problem": problem,
        "problem_sep": problem_sep,
        "problem_msep": problem_msep,
        "problems": problems,
        "states": states,
        "reports": reports,
        "names": names,
        "separator_allocation_diagnostics": separator_allocation_diagnostics,
    }


def _packed_separator_allocation_diagnostics(problems, states):
    """Return final separator-allocation diagnostics per packed problem.

    MRST's separator example is a facility comparison: the base problem has no
    separator tree, while the single- and multi-stage cases route production
    through separator groups. When the compact PyMRST problems are simulated,
    the returned final states now expose the separator allocation diagnostics in
    the same problem order.
    """
    diagnostics = []
    if states:
        for state_list in states:
            if state_list:
                diagnostics.append(state_list[-1].separator_group_diagnostics)
            else:
                diagnostics.append(None)
        return diagnostics
    for problem in problems:
        result = getattr(problem, "result", None)
        if result is None or not result.states:
            diagnostics.append(None)
        else:
            diagnostics.append(result.states[-1].separator_group_diagnostics)
    return diagnostics


def evaluate_separator_example_allocation_diagnostics(
    dims: tuple[int, int, int] = (3, 3, 2),
    *,
    q_o: float | None = None,
    q_g: float = 0.0,
):
    """Evaluate final separator allocation diagnostics for the example cases.

    This mirrors the separator comparison in MRST's example without forcing the
    optional packed Newton simulations to run. A producer state with signed
    surface rates is passed through the same final separator-group refresh used
    by `OverallCompositionModel.step`, returning diagnostics for the no
    separator, multi-stage separator, and single separator alternatives in MRST
    packed-problem order.
    """
    grid, rock, mixture, info, wells, base_model, state0, schedule = setup_separator_depletion_case(dims)
    _, single_separator, tree_separator = make_separator_tree()
    if q_o is None:
        q_o = float(wells[0].val)

    def clone(separator_group=None):
        return OverallCompositionModel(
            grid,
            rock,
            mixture,
            temperature=info.T,
            wells=copy.deepcopy(wells),
            rho_s=base_model.rho_s,
            max_iterations=base_model.max_iterations,
            separator_group=separator_group,
        )

    names = ["Overall-NoSep", "Overall-MultiSep", "Overall-SingleSep"]
    models = [clone(None), clone(tree_separator), clone(single_separator)]
    states = []
    diagnostics = []
    for model in models:
        state = state0.copy()
        state.qOs[0] = float(q_o)
        state.qGs[0] = float(q_g)
        out = model._apply_separator_group_to_state(state)
        states.append(out)
        diagnostics.append(out.separator_group_diagnostics)
    return {
        "grid": grid,
        "rock": rock,
        "mixture": mixture,
        "info": info,
        "wells": wells,
        "schedule": schedule,
        "single_separator": single_separator,
        "tree_separator": tree_separator,
        "names": names,
        "states": states,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    def test_single_separator_splits_spe5_stream():
        out = run_compositional_separator_example()
        single = out["single"]
        assert out["mixture"].n == 6
        assert len(single.surface_rates) == 2
        assert single.surface_rates[0].shape == (1,)
        assert single.surface_rates[1].shape == (1,)
        total_stream = single.phase_streams[0] + single.phase_streams[1]
        assert np.allclose(total_stream[0], out["stream_mole"], atol=1e-10)
        print("ok test_single_separator_splits_spe5_stream")

    def test_separator_tree_changes_surface_volumes_and_conserves_components():
        out = run_compositional_separator_example()
        single = np.array([q[0] for q in out["single"].surface_rates])
        tree = np.array([q[0] for q in out["tree"].surface_rates])
        assert not np.allclose(single, tree)
        final_streams = out["tree_surface_mole_streams"]
        total_final = final_streams[0] + final_streams[1]
        assert np.allclose(total_final[0], out["stream_mole"], atol=1e-10)
        assert out["tree_separator_streams"] is out["tree"].separator_streams
        assert len(out["tree_separator_streams"].molestreams) == len(out["tree_separator"].separators)
        assert np.array_equal(
            out["tree_separator"].phase_destination,
            np.array([[2, 3], [3, 4], [4, 0], [0, 0]]),
        )
        print("ok test_separator_tree_changes_surface_volumes_and_conserves_components")

    def test_schedule_can_switch_separator_group():
        case = make_boundary_conditions_case(nx=4, ny=2)
        model = case["overall_model"]
        state = case["initial_state"]
        stock_tank = EOSSeparator(pressure=ATM, T=300.0)
        sep = SeparatorGroup(stock_tank)
        schedule = simple_schedule(
            case["schedule"].step.val[:2],
            control=[
                ScheduleControl(bc=case["bc"], separator_group=None),
                ScheduleControl(bc=case["bc"], separator_group=sep),
            ],
            step_control=[1, 2],
        )
        result = run_schedule(model, state, schedule)
        assert len(result.states) == 2
        assert model.separator_group is None
        assert schedule.control[1].separator_group is sep
        print("ok test_schedule_can_switch_separator_group")

    def test_separator_example_builds_mrst_packed_problem_order():
        out = setup_compositional_separator_packed_problems()
        assert out["names"] == ["Overall-NoSep", "Overall-MultiSep", "Overall-SingleSep"]
        assert out["problem"].model.separator_group is None
        assert out["problem_msep"].model.separator_group is out["tree_separator"]
        assert out["problem_sep"].model.separator_group is out["single_separator"]
        assert out["schedule"].step.control.tolist() == [1]
        assert out["schedule"].control[0].wells[0].name == "Producer"
        assert out["wells"][0].val < 0.0
        assert out["tree_separator"].phase_destination.shape == (4, 2)
        print("ok test_separator_example_builds_mrst_packed_problem_order")

    def test_separator_example_exposes_allocation_diagnostics():
        out = evaluate_separator_example_allocation_diagnostics()
        diagnostics = out["diagnostics"]
        assert len(diagnostics) == 3
        assert diagnostics[0] is None
        assert diagnostics[1] is not None
        assert diagnostics[2] is not None
        for diag in diagnostics[1:]:
            assert diag["producer_indices"] == [0]
            assert diag["producer_names"] == ["Producer"]
            assert diag["producer_component_mass_streams"].shape == (1, out["mixture"].n)
            assert np.isclose(diag["allocation_shares"].sum(), 1.0)
            assert np.allclose(diag["positive_surface_rates_total"], -diag["allocated_surface_rates"].sum(axis=0))
        print("ok test_separator_example_exposes_allocation_diagnostics")

    test_single_separator_splits_spe5_stream()
    test_separator_tree_changes_surface_volumes_and_conserves_components()
    test_schedule_can_switch_separator_group()
    test_separator_example_builds_mrst_packed_problem_order()
    test_separator_example_exposes_allocation_diagnostics()
    print("\nAll tests passed!")


__all__ = [
    "make_separator_tree",
    "evaluate_separator_example_allocation_diagnostics",
    "run_compositional_separator_example",
    "setup_compositional_separator_packed_problems",
    "setup_separator_depletion_case",
]
