"""Port of MRST `book-ii/compositionalAccuracyExample.m`.

MRST compares three packed problems built from
`setupSimpleCompositionalExample`: explicit overall composition, natural
variables, and implicit overall composition. PyMRST now mirrors that structure.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.examples._shared import (
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    simple_schedule,
    simulate_packed_problem,
)
from pymrst.modules.compositional.examples.utils import setup_simple_compositional_example
from pymrst.modules.compositional.model import set_time_discretization
from pymrst.utils.units import CENTI, DAY, POISE


def run_compositional_accuracy_example(nx: int = 8, nstep: int = 3, dt: float = 10.0 * DAY):
    """Run MRST's explicit/natural/overall accuracy comparison structure."""
    overall_setup = setup_simple_compositional_example(False, nx=nx)
    natural_setup = setup_simple_compositional_example(True, nx=nx)
    timesteps = np.full(int(nstep), float(dt))
    schedule = simple_schedule(timesteps, wells=overall_setup.wells)

    explicit_model = set_time_discretization(overall_setup.model, "explicit")
    packer = lambda state, model, name: pack_simulation_problem(
        state,
        model,
        schedule,
        "compAccExample",
        name=name,
    )
    explicit_problem = packer(overall_setup.state, explicit_model, "Explicit")
    natural_problem = packer(natural_setup.state, natural_setup.model, "Natural")
    overall_problem = packer(overall_setup.state, overall_setup.model, "Overall")
    problems = [explicit_problem, natural_problem, overall_problem]
    simulate_packed_problem(problems)
    _, states, reports, names = get_multiple_packed_simulator_outputs(problems)

    explicit = explicit_problem.result
    natural = natural_problem.result
    overall = overall_problem.result
    assert explicit is not None and natural is not None and overall is not None

    final_explicit = explicit.final_state
    final_natural = natural.final_state
    final_overall = overall.final_state
    component_profiles = accuracy_component_profiles(states, names, overall_setup.model.mix.names)
    gas_profiles = accuracy_gas_saturation_profiles(overall_setup.model, states, names)
    property_profiles = accuracy_property_profiles(overall_setup.model, states[0])
    cumulative_iterations = accuracy_cumulative_iterations(reports, names)
    out = {
        "names": names,
        "packed_problems": problems,
        "states": states,
        "reports": reports,
        "time_discretization": {
            "Explicit": explicit_model.time_discretization,
            "Natural": "fully implicit natural variables",
            "Overall": overall_setup.model.time_discretization,
        },
        "timesteps": timesteps,
        "explicit": explicit,
        "explicit_states": explicit.states,
        "natural": natural,
        "overall": overall,
        "explicit_iterations": explicit.iterations,
        "natural_iterations": natural.iterations,
        "overall_iterations": overall.iterations,
        "natural_overall_pressure_error": float(np.linalg.norm(final_natural.pressure - final_overall.pressure)),
        "natural_overall_composition_error": float(np.linalg.norm(final_natural.z - final_overall.z)),
        "explicit_overall_pressure_error": float(np.linalg.norm(final_explicit.pressure - final_overall.pressure)),
        "explicit_overall_composition_error": float(np.linalg.norm(final_explicit.z - final_overall.z)),
        "component_profiles": component_profiles,
        "gas_saturation_profiles": gas_profiles,
        "property_profiles": property_profiles,
        "cumulative_iterations": cumulative_iterations,
    }
    out["plot_sections"] = accuracy_plot_sections(out)
    return out


def _mrst_plot_step_indices(states, step_stride: int = 10) -> np.ndarray:
    """Zero-based indices corresponding to MATLAB's ``1:step_stride:n``."""
    n = min(len(history) for history in states)
    if n <= 0:
        return np.array([], dtype=int)
    return np.arange(0, n, int(step_stride), dtype=int)


def accuracy_component_profiles(states, names, component_names=None, *, step_stride: int = 10):
    """Return the component-profile data plotted in MRST's first section."""
    indices = _mrst_plot_step_indices(states, step_stride)
    out = []
    for ix in indices:
        out.append({
            "step": int(ix + 1),
            "profiles": {
                str(name): np.asarray(history[ix].z, dtype=float).copy()
                for name, history in zip(names, states)
            },
        })
    ncomp = next(iter(out[0]["profiles"].values())).shape[1] if out else 0
    if component_names is None:
        component_names = tuple(f"Component {i + 1}" for i in range(ncomp))
    return {
        "steps": indices + 1,
        "names": tuple(str(name) for name in names),
        "component_names": tuple(component_names),
        "markers": ("-", "--", "--"),
        "line_widths": (1.0, 2.5, 5.0),
        "frames": out,
    }


def accuracy_phase_saturations(model, state) -> tuple[np.ndarray, np.ndarray]:
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


def accuracy_gas_saturation_profiles(model, states, names, *, step_stride: int = 10):
    """Return gas-saturation profiles from MRST's second plotting section."""
    indices = _mrst_plot_step_indices(states, step_stride)
    frames = []
    for ix in indices:
        frames.append({
            "step": int(ix + 1),
            "profiles": {
                str(name): accuracy_phase_saturations(model, history[ix])[1].copy()
                for name, history in zip(names, states)
            },
        })
    return {
        "steps": indices + 1,
        "names": tuple(str(name) for name in names),
        "line_styles": ("-", "-", "--"),
        "frames": frames,
    }


def accuracy_property_profiles(model, states, *, step_stride: int = 10, component_index: int = 1):
    """Return viscosity/density/gas-saturation/composition profiles.

    This mirrors the final property plot in MRST, which evaluates the first
    packed state history and plots liquid/vapour viscosity and density together
    with gas saturation and the second component's overall composition.
    """
    indices = _mrst_plot_step_indices([states], step_stride)
    frames = []
    for ix in indices:
        state = states[ix]
        props = backend_flow_props(model.mix, model.eos_model, state.pressure, model.T, state.z)
        liquid_absent = np.asarray(props.L, dtype=float) == 0.0
        vapour_absent = np.asarray(props.L, dtype=float) == 1.0
        mu_l = np.asarray(props.muL, dtype=float).copy() / (CENTI * POISE)
        mu_v = np.asarray(props.muV, dtype=float).copy() / (CENTI * POISE)
        rho_l = np.asarray(props.rhoL, dtype=float).copy()
        rho_v = np.asarray(props.rhoV, dtype=float).copy()
        mu_l[liquid_absent] = np.nan
        rho_l[liquid_absent] = np.nan
        mu_v[vapour_absent] = np.nan
        rho_v[vapour_absent] = np.nan
        frames.append({
            "step": int(ix + 1),
            "mu_liquid_cp": mu_l,
            "mu_vapour_cp": mu_v,
            "rho_liquid": rho_l,
            "rho_vapour": rho_v,
            "gas_saturation": accuracy_phase_saturations(model, state)[1].copy(),
            "component": np.asarray(state.z[:, int(component_index)], dtype=float).copy(),
            "component_index": int(component_index),
        })
    return {
        "steps": indices + 1,
        "frames": frames,
        "legend": ("mu_l", "mu_v", "rho_l", "rho_v", "S_g", "z (CO2)"),
    }


def accuracy_cumulative_iterations(reports, names):
    """Return MRST's cumulative-iteration plot for implicit formulations."""
    natural = np.array([int(r["Iterations"]) for r in reports[1]], dtype=int)
    overall = np.array([int(r["Iterations"]) for r in reports[2]], dtype=int)
    return {
        "names": tuple(names[1:]),
        "iterations": np.column_stack([natural, overall]),
        "cumulative": np.column_stack([np.cumsum(natural), np.cumsum(overall)]),
    }


def accuracy_plot_sections(output: dict) -> dict:
    """Return MRST's four plotting sections for the accuracy example."""
    component = output["component_profiles"]
    gas = output["gas_saturation_profiles"]
    properties = output["property_profiles"]
    cumulative = output["cumulative_iterations"]

    component_legend = tuple(
        f"{name} {component_name}"
        for name in component["names"]
        for component_name in component["component_names"]
    )
    return {
        "component_profiles": {
            "figure_position_offset": (0, 0, 350, 0),
            "steps": component["steps"],
            "frames": component["frames"],
            "legend": component_legend,
            "legend_location": "north",
            "legend_num_columns": 3,
            "ylim": (0.0, 1.0),
            "ylabel": "z",
            "font_size": 12,
            "markers": component["markers"],
            "line_widths": component["line_widths"],
        },
        "gas_saturation": {
            "steps": gas["steps"],
            "frames": gas["frames"],
            "legend": gas["names"],
            "legend_location": "north",
            "legend_num_columns": 3,
            "ylim": (0.0, 1.0),
            "ylabel": "z",
            "line_styles": gas["line_styles"],
        },
        "properties": {
            "steps": properties["steps"],
            "frames": properties["frames"],
            "legend": (r"\mu_l", r"\mu_v", r"\rho_l", r"\rho_v", "S_g", "z (CO_2)"),
            "legend_location": "east",
            "legend_num_columns": 3,
            "left_ylabel": "Viscosity [cP]",
            "right_ylabel": "Mass density [kg/m^3]",
            "overlays": ("S_g", "z (CO_2)"),
        },
        "cumulative_iterations": {
            "names": cumulative["names"],
            "iterations": cumulative["iterations"],
            "cumulative": cumulative["cumulative"],
            "legend": cumulative["names"],
            "title": "Cumulative number of iterations",
        },
    }


if __name__ == "__main__":
    def test_accuracy_example_has_three_mrst_cases():
        out = run_compositional_accuracy_example(nx=6, nstep=2, dt=5.0 * DAY)
        assert out["names"] == ["Explicit", "Natural", "Overall"]
        assert out["time_discretization"]["Explicit"] == "explicit"
        assert out["time_discretization"]["Overall"] == "implicit"
        assert out["explicit_iterations"].size == 2
        assert out["natural_iterations"].size == 2
        assert out["overall_iterations"].size == 2
        assert np.isfinite(out["natural_overall_pressure_error"])
        assert np.isfinite(out["explicit_overall_composition_error"])
        assert out["component_profiles"]["names"] == ("Explicit", "Natural", "Overall")
        assert out["gas_saturation_profiles"]["frames"][0]["profiles"]["Explicit"].shape == (6,)
        assert out["property_profiles"]["frames"][0]["mu_liquid_cp"].shape == (6,)
        assert out["cumulative_iterations"]["names"] == ("Natural", "Overall")
        assert out["plot_sections"]["component_profiles"]["legend_location"] == "north"
        assert out["plot_sections"]["gas_saturation"]["ylim"] == (0.0, 1.0)
        assert out["plot_sections"]["properties"]["legend_location"] == "east"
        assert out["plot_sections"]["cumulative_iterations"]["title"] == "Cumulative number of iterations"
        print("ok test_accuracy_example_has_three_mrst_cases")

    test_accuracy_example_has_three_mrst_cases()
    print("\nAll tests passed!")


__all__ = [
    "accuracy_component_profiles",
    "accuracy_cumulative_iterations",
    "accuracy_gas_saturation_profiles",
    "accuracy_phase_saturations",
    "accuracy_plot_sections",
    "accuracy_property_profiles",
    "run_compositional_accuracy_example",
]
