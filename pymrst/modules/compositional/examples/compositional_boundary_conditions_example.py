"""Port of MRST ``compositionalBoundaryConditionsExample.m``.

The MRST example injects a methane/CO2/n-decane stream through boundary
conditions: flux on ``xmin``, pressure on ``xmax``, and ``bc.components`` for
the injected composition.  This PyMRST port keeps that structure and runs the
same schedule with both natural variables and overall composition.  The default
grid is smaller so the file remains a fast smoke test; pass ``full_size=True``
to build the 50-by-50 MRST case.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.examples._shared import (
    SimpleSchedule,
    ScheduleControl,
    run_schedule,
    simple_schedule,
)
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import log_norm_layers, make_rock, pore_volume
from pymrst.params.wells_and_bc import fluxside, pside
from pymrst.utils.units import BARSA, DAY, DARCY, MILLI, YEAR


def boundary_condition_mixture() -> Mixture:
    """The exact three-component mixture from the MRST example."""
    return Mixture(
        names=("Methane", "CarbonDioxide", "n-Decane"),
        Tcrit=np.array([190.5640, 304.1282, 617.7000]),
        Pcrit=np.array([4599200.0, 7377300.0, 2103000.0]),
        Vcrit=np.array([9.8628e-05, 9.4118e-05, 6.0976e-04]),
        acentric=np.array([0.0114, 0.2239, 0.4884]),
        molarMass=np.array([0.0160, 0.0440, 0.1423]),
        name="MRST compositional boundary-condition mixture",
    )


def make_boundary_conditions_case(
    nx: int = 8,
    ny: int = 4,
    *,
    full_size: bool = False,
    permeability_md=None,
):
    """Build MRST's 2D compositional BC displacement case."""
    if full_size:
        nx, ny = 50, 50
    grid = compute_geometry(cart_grid([nx, ny, 1], [1000.0, 1000.0, 10.0]))
    if permeability_md is None:
        k_md = log_norm_layers((nx, ny, 1), rng=np.random.default_rng(0))
    else:
        k_md = np.asarray(permeability_md, dtype=float).ravel()
        if k_md.size != grid.cells.num:
            raise ValueError(f"permeability_md must have {grid.cells.num} entries")
    rock = make_rock(grid, k_md[:, None] * MILLI * DARCY, 0.2)
    mixture = boundary_condition_mixture()

    min_pressure = 75.0 * BARSA
    total_time = 20.0 * YEAR
    pv = pore_volume(grid, rock)
    bc = fluxside(None, grid, "xmin", pv.sum() / total_time, sat=np.array([1.0, 0.0]))
    bc = pside(bc, grid, "xmax", min_pressure, sat=np.array([1.0, 0.0]))
    injection = np.array([0.1, 0.9, 0.0])
    initial = np.array([0.3000, 0.1000, 0.6000])
    temperature = 423.15
    bc.components = np.tile(injection, (bc.face.size, 1))
    timesteps = rampup_timesteps(total_time, 75.0 * DAY, 12)
    schedule = simple_schedule(timesteps, bc=bc)

    overall_model = OverallCompositionModel(
        grid,
        rock,
        mixture,
        temperature=temperature,
        bc=bc,
        rho_s=(500.0, 500.0),
        n_kr=(2.0, 2.0),
    )
    state0 = overall_model.initial_state(1.5 * min_pressure, initial)
    return {
        "grid": grid,
        "rock": rock,
        "mixture": mixture,
        "bc": bc,
        "schedule": schedule,
        "overall_model": overall_model,
        "initial_state": state0,
        "injection": injection,
        "initial": initial,
        "temperature": temperature,
        "min_pressure": min_pressure,
        "total_time": total_time,
        "permeability_md": k_md,
    }


def run_compositional_boundary_conditions_example(
    nstep: int = 3,
    *,
    full_size: bool = False,
    permeability_md=None,
):
    """Run the BC displacement case with overall and natural variables."""
    case = make_boundary_conditions_case(full_size=full_size, permeability_md=permeability_md)
    grid = case["grid"]
    rock = case["rock"]
    mixture = case["mixture"]
    bc = case["bc"]
    schedule: SimpleSchedule = case["schedule"]
    if nstep is not None:
        schedule = simple_schedule(schedule.step.val[:int(nstep)], bc=bc)

    overall_model = case["overall_model"]
    state0 = case["initial_state"]
    result = run_schedule(overall_model, state0, schedule)
    natural_model = NaturalVariablesCompositionalModel(
        grid,
        rock,
        mixture,
        temperature=case["temperature"],
        bc=bc,
        rho_s=(500.0, 500.0),
        n_kr=(2.0, 2.0),
    )
    natural_state0 = natural_model.initial_state(state0.pressure, state0.z)
    natural_result = run_schedule(natural_model, natural_state0, schedule)
    case.update({
        "natural_model": natural_model,
        "natural_initial_state": natural_state0,
        "result": result,
        "natural_result": natural_result,
        "timesteps": schedule.step.val,
    })
    case["plot_fields"] = boundary_condition_plot_fields(case)
    case["iteration_plot"] = boundary_condition_iteration_plot(case)
    case["plot_sections"] = boundary_condition_plot_sections(case)
    return case


def boundary_condition_phase_saturations(model, state) -> tuple[np.ndarray, np.ndarray]:
    """Return oil/gas saturation columns corresponding to MRST ``state.s``."""
    props = backend_flow_props(model.mix, model.eos_model, state.pressure, model.T, state.z)
    s_o = np.asarray(props.sL, dtype=float)
    s_g = np.asarray(props.sV, dtype=float)
    if model.water:
        assert state.sW is not None
        s_o = (1.0 - state.sW) * s_o
        s_g = (1.0 - state.sW) * s_g
    if state.sT is not None:
        s_o = state.sT * s_o
        s_g = state.sT * s_g
    return s_o, s_g


def boundary_condition_plot_fields(case, *, component_index: int = 1):
    """Return MRST's pressure/gas-saturation/component plotting sections."""
    natural = case["natural_result"]
    overall = case["result"]
    natural_model = case["natural_model"]
    overall_model = case["overall_model"]
    n = min(len(natural.states), len(overall.states))
    frames = []
    for ix in range(n):
        natural_state = natural.states[ix]
        overall_state = overall.states[ix]
        frames.append({
            "step": int(ix + 1),
            "Natural variables": {
                "pressure": np.asarray(natural_state.pressure, dtype=float).copy(),
                "gas_saturation": boundary_condition_phase_saturations(natural_model, natural_state)[1].copy(),
                "component": np.asarray(natural_state.z[:, int(component_index)], dtype=float).copy(),
            },
            "Overall composition": {
                "pressure": np.asarray(overall_state.pressure, dtype=float).copy(),
                "gas_saturation": boundary_condition_phase_saturations(overall_model, overall_state)[1].copy(),
                "component": np.asarray(overall_state.z[:, int(component_index)], dtype=float).copy(),
            },
        })
    names = tuple(case["mixture"].names)
    return {
        "steps": np.arange(1, n + 1, dtype=int),
        "component_index": int(component_index),
        "component_name": names[int(component_index)],
        "fields": ("pressure", "gas_saturation", "component"),
        "formulations": ("Natural variables", "Overall composition"),
        "frames": frames,
    }


def boundary_condition_iteration_plot(case):
    """Return MRST's nonlinear-iteration comparison plot data."""
    natural = np.asarray(case["natural_result"].iterations, dtype=int).ravel()
    overall = np.asarray(case["result"].iterations, dtype=int).ravel()
    n = min(natural.size, overall.size)
    natural = natural[:n]
    overall = overall[:n]
    return {
        "names": ("Natural variables", "Overall composition"),
        "steps": np.arange(1, n + 1, dtype=int),
        "iterations": np.column_stack([natural, overall]),
        "cumulative": np.column_stack([np.cumsum(natural), np.cumsum(overall)]),
    }


def boundary_condition_plot_sections(case):
    """Return MRST's full plotting-section metadata for the BC example."""
    fields = case.get("plot_fields") or boundary_condition_plot_fields(case)
    iterations = case.get("iteration_plot") or boundary_condition_iteration_plot(case)
    component_name = fields["component_name"]
    return {
        "permeability": {
            "field": np.asarray(case["permeability_md"], dtype=float).copy(),
            "title": "Permeability [mD]",
            "colorbar": True,
            "grid_cells": int(case["grid"].cells.num),
        },
        "toolbars": (
            {
                "title": "Natural variables",
                "states": case["natural_result"].states,
                "grid_cells": int(case["grid"].cells.num),
            },
            {
                "title": "Overall composition",
                "states": case["result"].states,
                "grid_cells": int(case["grid"].cells.num),
            },
        ),
        "comparison_animation": {
            "subplot_layout": (3, 2),
            "formulations": fields["formulations"],
            "rows": (
                {"field": "pressure", "ylabel": "Pressure", "caxis": None},
                {"field": "gas_saturation", "ylabel": "Gas saturation", "caxis": (0.0, 1.0)},
                {"field": "component", "ylabel": component_name, "caxis": (0.0, 1.0)},
            ),
            "edge_color": "none",
            "axis": "tight",
            "hide_tick_labels": True,
            "right_bottom_axis": "tight off",
            "drawnow": True,
            "frames": fields["frames"],
        },
        "iteration_plot": {
            "data": iterations,
            "line_style": "-o",
            "legend": iterations["names"],
            "xlabel": "Step #",
            "ylabel": "Nonlinear iterations",
        },
    }


if __name__ == "__main__":
    def test_boundary_drive_enriches_left_side():
        out = run_compositional_boundary_conditions_example(2)
        initial = out["initial_state"]
        final = out["result"].final_state
        natural_final = out["natural_result"].final_state
        assert out["bc"].components.shape[1] == out["mixture"].n
        assert final.z[0, 1] > initial.z[0, 1]
        assert natural_final.z[0, 1] > out["natural_initial_state"].z[0, 1]
        assert final.pressure[-1] < final.pressure[0]
        assert natural_final.pressure[-1] < natural_final.pressure[0]
        assert np.all(out["timesteps"] == out["schedule"].step.val[:2])
        assert np.all(out["schedule"].step.control == 1)
        assert out["plot_fields"]["formulations"] == ("Natural variables", "Overall composition")
        assert out["plot_fields"]["frames"][0]["Natural variables"]["gas_saturation"].shape == (out["grid"].cells.num,)
        assert out["plot_fields"]["component_name"] == "CarbonDioxide"
        assert out["iteration_plot"]["iterations"].shape == (2, 2)
        assert out["plot_sections"]["comparison_animation"]["subplot_layout"] == (3, 2)
        assert out["plot_sections"]["iteration_plot"]["line_style"] == "-o"
        print("ok test_boundary_drive_enriches_left_side")

    def test_boundary_schedule_switches_components():
        case = make_boundary_conditions_case(nx=4, ny=2)
        bc_a = case["bc"]
        bc_b = fluxside(None, case["grid"], "xmin", case["bc"].value[:2].sum(), sat=np.array([1.0, 0.0]))
        bc_b = pside(bc_b, case["grid"], "xmax", case["min_pressure"], sat=np.array([1.0, 0.0]))
        bc_b.components = np.tile([0.8, 0.2, 0.0], (bc_b.face.size, 1))
        schedule = simple_schedule(
            case["schedule"].step.val[:2],
            control=[ScheduleControl(bc=bc_a), ScheduleControl(bc=bc_b)],
            step_control=[1, 2],
        )
        result = run_schedule(case["overall_model"], case["initial_state"], schedule)
        assert schedule.step.control.tolist() == [1, 2]
        assert np.allclose(schedule.control[1].bc.components[0], [0.8, 0.2, 0.0])
        assert result.final_state.z[0, 0] > case["initial_state"].z[0, 0]
        print("ok test_boundary_schedule_switches_components")

    test_boundary_drive_enriches_left_side()
    test_boundary_schedule_switches_components()
    print("\nAll tests passed!")


__all__ = [
    "boundary_condition_iteration_plot",
    "boundary_condition_mixture",
    "boundary_condition_phase_saturations",
    "boundary_condition_plot_fields",
    "boundary_condition_plot_sections",
    "make_boundary_conditions_case",
    "run_compositional_boundary_conditions_example",
]
