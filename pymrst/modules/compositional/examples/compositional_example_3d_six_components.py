"""Port of MRST `compositionalExample3DSixComponents.m`.

MRST builds a three-dimensional six-component `lumped_1` quarter five-spot
with a natural-variable compositional model. The default here is scaled down
for tests, while `full_size=True` selects the MATLAB dimensions
(`20 x 20 x 2`, `1000 x 1000 x 1`).
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.examples._shared import run_schedule, simple_schedule
from pymrst.modules.compositional.fluids import get_benchmark_mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import vertical_well
from pymrst.utils.units import BARSA, DAY, DARCY, MILLI, YEAR


def setup_3d_six_component_case(
    dims: tuple[int, int, int] = (4, 4, 2),
    *,
    full_size: bool = False,
    use_natural: bool = True,
    n_kr: float = 2.0,
):
    """Build MRST's 3D six-component quarter five-spot setup."""
    if full_size:
        dims = (20, 20, 2)
    dims = tuple(int(v) for v in dims)
    nx, ny, nz = dims
    physical_dims = (1000.0, 1000.0, 1.0)
    grid = compute_geometry(cart_grid(dims, physical_dims))
    rock = make_rock(grid, 50.0 * MILLI * DARCY, 0.25)

    case_name = "lumped_1"
    min_pressure = 100.0 * BARSA
    max_pressure = 200.0 * BARSA
    mixture, info = get_benchmark_mixture(case_name)
    pressure0 = min_pressure + 0.5 * (max_pressure - min_pressure)

    wells = vertical_well(
        None,
        grid,
        rock,
        0,
        0,
        None,
        well_type="rate",
        name="Inj",
        val=0.0015,
        sign=1,
        compi=np.array([1.0, 0.0]),
    )
    wells = vertical_well(
        wells,
        grid,
        rock,
        nx - 1,
        ny - 1,
        None,
        well_type="bhp",
        name="Prod",
        val=min_pressure,
        sign=-1,
        compi=np.array([0.5, 0.5]),
    )
    for well in wells:
        well.components = info.injection.copy()

    model_cls = NaturalVariablesCompositionalModel if use_natural else OverallCompositionModel
    model = model_cls(
        grid,
        rock,
        mixture,
        temperature=info.temp,
        wells=wells,
        water=False,
        n_kr=(float(n_kr), float(n_kr)),
        max_iterations=80,
    )
    state0 = model.initial_state(pressure0, info.initial)
    timesteps = rampup_timesteps(2.0 * YEAR, 7.0 * DAY, 5)
    schedule = simple_schedule(timesteps, wells=wells)
    return {
        "case_name": case_name,
        "dims": dims,
        "physical_dims": physical_dims,
        "grid": grid,
        "rock": rock,
        "mixture": mixture,
        "info": info,
        "min_pressure": min_pressure,
        "max_pressure": max_pressure,
        "pressure0": pressure0,
        "wells": wells,
        "model": model,
        "initial_state": state0,
        "schedule": schedule,
        "use_natural": bool(use_natural),
        "n_kr": float(n_kr),
    }


def run_compositional_example_3d_six_components(
    dims: tuple[int, int, int] = (4, 4, 2),
    *,
    full_size: bool = False,
    use_natural: bool = True,
    nstep: int | None = 2,
    max_cuts: int = 6,
):
    """Run the 3D six-component example schedule.

    `nstep=None` runs the full two-year ramped schedule. Tests normally pass a
    small value so the MRST setup is exercised without paying the full cost.
    """
    case = setup_3d_six_component_case(
        dims=dims,
        full_size=full_size,
        use_natural=use_natural,
    )
    schedule = case["schedule"]
    if nstep is not None:
        n = int(nstep)
        schedule = simple_schedule(schedule.step.val[:n], wells=case["wells"])
    result = run_schedule(
        case["model"],
        case["initial_state"],
        schedule,
        max_cuts=max_cuts,
    )
    case["result"] = result
    case["states"] = result.states
    case["iterations"] = result.iterations
    case["times"] = np.cumsum(schedule.step.val)
    case["plot_frames"] = six_component_plot_frames(case)
    case["plot_sections"] = six_component_plot_sections(case)
    return case


def extract_plot_fields(case_or_state):
    """Return the final cell fields MRST plots.

    This compatibility helper keeps the original final-state dictionary
    contract. Use :func:`six_component_plot_frames` for the full MRST plotting
    loop over every report state.
    """
    if isinstance(case_or_state, dict):
        state = case_or_state["states"][-1] if case_or_state.get("states") else case_or_state["initial_state"]
        grid = case_or_state["grid"]
        mixture = case_or_state["mixture"]
    else:
        state = case_or_state
        grid = None
        mixture = None
    fields = {
        "components": state.z.copy(),
        "pressure": state.pressure.copy(),
    }
    if hasattr(state, "sO"):
        fields["sO"] = state.sO.copy()
    elif hasattr(state, "s"):
        fields["sO"] = state.s[:, 0].copy()
    if hasattr(state, "sG"):
        fields["sG"] = state.sG.copy()
    elif hasattr(state, "s"):
        fields["sG"] = state.s[:, 1].copy()
    if grid is not None:
        fields["grid_cells"] = grid.cells.num
    if mixture is not None:
        fields["component_names"] = mixture.names
    return fields


def six_component_state_plot_fields(state) -> dict:
    """Return MRST's component/pressure/saturation fields for one state."""
    fields = {
        "components": np.asarray(state.z, dtype=float).copy(),
        "pressure": np.asarray(state.pressure, dtype=float).copy(),
    }
    if hasattr(state, "sO"):
        fields["sO"] = np.asarray(state.sO, dtype=float).copy()
    elif hasattr(state, "s"):
        fields["sO"] = np.asarray(state.s[:, 0], dtype=float).copy()
    if hasattr(state, "sG"):
        fields["sG"] = np.asarray(state.sG, dtype=float).copy()
    elif hasattr(state, "s"):
        fields["sG"] = np.asarray(state.s[:, 1], dtype=float).copy()
    return fields


def six_component_plot_frames(case_or_states, *, component_names=None):
    """Return the per-report plotting frames from the MRST 3D example."""
    if isinstance(case_or_states, dict):
        states = case_or_states.get("states") or []
        mixture = case_or_states.get("mixture")
        grid = case_or_states.get("grid")
        times = np.asarray(case_or_states.get("times", np.arange(1, len(states) + 1)), dtype=float)
        if component_names is None and mixture is not None:
            component_names = tuple(mixture.names)
        grid_cells = int(grid.cells.num) if grid is not None else None
    else:
        states = list(case_or_states)
        times = np.arange(1, len(states) + 1, dtype=float)
        grid_cells = None
    frames = []
    for ix, state in enumerate(states):
        fields = six_component_state_plot_fields(state)
        frames.append({
            "step": int(ix + 1),
            "time": float(times[ix]) if ix < times.size else np.nan,
            "fields": fields,
        })
    if component_names is None and frames:
        ncomp = frames[0]["fields"]["components"].shape[1]
        component_names = tuple(f"Component {i + 1}" for i in range(ncomp))
    return {
        "component_names": tuple(component_names or ()),
        "view": (-30, 60),
        "subplot_columns": 3,
        "subplot_rows": int(np.ceil(len(component_names or ()) / 2.0)) if component_names else 0,
        "extra_fields": ("pressure", "sO", "sG"),
        "grid_cells": grid_cells,
        "steps": np.arange(1, len(frames) + 1, dtype=int),
        "frames": frames,
    }


def six_component_plot_sections(case_or_frames):
    """Return MRST's plotting-section metadata for the 3D six-component example."""
    if isinstance(case_or_frames, dict) and "plot_frames" in case_or_frames:
        frames = case_or_frames["plot_frames"]
        states = case_or_frames.get("states", [])
        grid = case_or_frames.get("grid")
    else:
        frames = case_or_frames
        states = []
        grid = None
    component_names = tuple(frames["component_names"])
    ncomp = len(component_names)
    subplot_rows = int(np.ceil(ncomp / 2.0)) if ncomp else 0
    component_plots = tuple(
        {
            "field": "components",
            "component_index": int(i),
            "title": name,
            "edge_color": "none",
            "view": tuple(frames["view"]),
            "caxis": (0.0, 1.0),
        }
        for i, name in enumerate(component_names)
    )
    extra_plots = (
        {
            "field": "pressure",
            "title": "Pressure",
            "edge_color": "none",
            "view": tuple(frames["view"]),
            "caxis": None,
        },
        {
            "field": "sO",
            "title": "sO",
            "edge_color": "none",
            "view": tuple(frames["view"]),
            "caxis": None,
        },
        {
            "field": "sG",
            "title": "sG",
            "edge_color": "none",
            "view": tuple(frames["view"]),
            "caxis": None,
        },
    )
    return {
        "animation": {
            "figure_position_offset": (0, -200, 350, 200),
            "subplot_rows": subplot_rows,
            "subplot_columns": 3,
            "view": tuple(frames["view"]),
            "component_plots": component_plots,
            "extra_plots": extra_plots,
            "frames": frames["frames"],
            "drawnow": True,
        },
        "toolbar": {
            "figure": 1,
            "states": states,
            "grid_cells": None if grid is None else int(grid.cells.num),
            "view": tuple(frames["view"]),
            "axis": "tight",
        },
    }


def setup_3d_six_component_state():
    """Compatibility helper returning the initialized MRST-shaped case tuple."""
    case = setup_3d_six_component_case()
    return (
        case["grid"],
        case["rock"],
        case["mixture"],
        case["info"],
        case["model"],
        case["initial_state"],
    )


if __name__ == "__main__":
    def test_six_component_case_matches_mrst_setup():
        case = setup_3d_six_component_case((3, 3, 2))
        assert case["case_name"] == "lumped_1"
        assert case["dims"] == (3, 3, 2)
        assert case["mixture"].n == 6
        assert isinstance(case["model"], NaturalVariablesCompositionalModel)
        assert len(case["wells"]) == 2
        assert case["wells"][0].name == "Inj"
        assert case["wells"][1].name == "Prod"
        assert case["wells"][0].cells.size == 2
        assert case["wells"][1].cells.size == 2
        assert np.isclose(case["pressure0"], 150.0 * BARSA)
        assert np.allclose(case["wells"][0].components, case["info"].injection)
        print("ok test_six_component_case_matches_mrst_setup")

    def test_six_component_schedule_matches_mrst_rampup():
        case = setup_3d_six_component_case((3, 3, 2))
        dt = case["schedule"].step.val
        assert dt.size > 45
        assert np.isclose(dt.sum(), 2.0 * YEAR)
        assert np.isclose(dt[0], 7.0 * DAY / 2.0**5)
        assert np.all(case["schedule"].step.control == 1)
        print("ok test_six_component_schedule_matches_mrst_rampup")

    def test_six_component_run_returns_plot_fields():
        out = run_compositional_example_3d_six_components((2, 2, 2), nstep=1)
        fields = extract_plot_fields(out)
        assert out["iterations"].size == 1
        assert len(out["plot_frames"]["frames"]) == 1
        assert out["plot_frames"]["component_names"] == out["mixture"].names
        assert out["plot_frames"]["extra_fields"] == ("pressure", "sO", "sG")
        assert out["plot_sections"]["animation"]["subplot_columns"] == 3
        assert out["plot_sections"]["toolbar"]["axis"] == "tight"
        assert fields["components"].shape == (8, 6)
        assert fields["pressure"].shape == (8,)
        assert np.allclose(fields["components"].sum(axis=1), 1.0)
        assert np.all(np.isfinite(fields["pressure"]))
        print("ok test_six_component_run_returns_plot_fields")

    test_six_component_case_matches_mrst_setup()
    test_six_component_schedule_matches_mrst_rampup()
    test_six_component_run_returns_plot_fields()
    print("\nAll tests passed!")


__all__ = [
    "extract_plot_fields",
    "run_compositional_example_3d_six_components",
    "setup_3d_six_component_case",
    "setup_3d_six_component_state",
    "six_component_plot_frames",
    "six_component_plot_sections",
    "six_component_state_plot_fields",
]
