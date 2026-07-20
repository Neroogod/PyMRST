"""Port of MRST `book-ii/showNaturalOverall.m`.

MRST poses the example as a single-cell water-gas brine/CO2-style problem with
a flux boundary on `xmin` and a pressure boundary on `xmax`. PyMRST keeps the
standard W/O/G phase vector for water-active compositional models, so the
boundary saturation is represented in W/O/G order while the grid, rock,
boundary faces, component stream, and timestep follow the MATLAB script.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import flash
from pymrst.modules.compositional.fluids import get_compositional_fluid_case
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import fluxside, pside
from pymrst.utils.units import DARCY, DAY


def setup_show_natural_overall(max_change: float = 0.1):
    """Construct the MRST-style natural/overall comparison case."""
    grid = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    rock = make_rock(grid, 0.5 * DARCY, 0.5)
    mixture, info = get_compositional_fluid_case("simple")
    p = info.pressure
    sat_wog = np.array([0.0, 0.0, 1.0])
    bc = fluxside(None, grid, "xmin", 1.0 / DAY, sat=sat_wog)
    bc = pside(bc, grid, "xmax", p, sat=sat_wog)
    bc.components = np.tile(info.injection, (bc.face.size, 1))

    overall = OverallCompositionModel(grid, rock, mixture, temperature=info.T, water=True)
    natural = NaturalVariablesCompositionalModel(grid, rock, mixture, temperature=info.T, water=True)
    overall.dz_max = max_change
    natural.dz_max = max_change
    natural.ds_max = max_change
    state0_overall = overall.initial_state(p, info.initial, sW=0.5)
    state0_natural = natural.initial_state(p, info.initial, sW=0.5)
    return grid, rock, mixture, info, bc, overall, natural, state0_overall, state0_natural


def run_show_natural_overall(max_change: float = 0.1, dt: float = 100.0 * DAY):
    """Compute one timestep with natural-variable and overall formulations."""
    (
        grid,
        rock,
        mixture,
        info,
        bc,
        overall,
        natural,
        state0_overall,
        state0_natural,
    ) = setup_show_natural_overall(max_change=max_change)
    natural_state, natural_ok, natural_it = natural.step(state0_natural, dt, bc=bc)
    overall_state, overall_ok, overall_it = overall.step(state0_overall, dt, bc=bc)
    if not natural_ok or not overall_ok:
        raise RuntimeError("natural/overall boundary-condition comparison did not converge")
    flash_map = flash(
        mixture,
        state0_overall.pressure,
        np.full(state0_overall.pressure.shape, info.T),
        state0_overall.z,
    )
    return {
        "grid": grid,
        "rock": rock,
        "mixture": mixture,
        "info": info,
        "bc": bc,
        "initial_state_natural": state0_natural,
        "initial_state_overall": state0_overall,
        "natural_state": natural_state,
        "overall_state": overall_state,
        "iterations": np.array([natural_it, overall_it]),
        "vapour_fraction_initial": flash_map.V,
        "convergence_plots": show_natural_overall_convergence_plots(
            mixture,
            info,
            state0_natural,
            natural_state,
            state0_overall,
            overall_state,
            max_change=max_change,
        ),
    }


def _composition_line(info, ns: int, component_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return MRST-style composition-line samples for an n-component mixture."""
    component_values = np.linspace(0.01, 1.0, int(ns))
    z = np.zeros((component_values.size, np.asarray(info.initial).size), dtype=float)
    z[:, component_index] = component_values
    other = [i for i in range(z.shape[1]) if i != component_index]
    if other:
        weights = np.asarray(info.initial, dtype=float)[other]
        if np.sum(weights) <= 0.0:
            weights = np.ones(len(other), dtype=float) / len(other)
        else:
            weights = weights / np.sum(weights)
        z[:, other] = (1.0 - component_values)[:, None] * weights[None, :]
    return component_values, z


def show_natural_overall_convergence_plots(
    mixture,
    info,
    initial_natural,
    natural_state,
    initial_overall,
    overall_state,
    *,
    max_change: float = 0.1,
    ns: int = 50,
    component_index: int = 0,
) -> dict:
    """Return MRST's convergence-history plot payload.

    MRST plots nonlinear iteration histories over a flash-derived saturation
    contour map. PyMRST's current compact solver does not expose every
    nonlinear sub-iteration state, so this payload records the available
    initial-to-final paths while keeping the same two plot sections and flash
    contour construction.
    """
    p_ref = float(info.pressure)
    T = float(info.T)
    natural_dp = np.asarray([initial_natural.pressure[0], natural_state.pressure[0]], dtype=float) - p_ref
    overall_dp = np.asarray([initial_overall.pressure[0], overall_state.pressure[0]], dtype=float) - p_ref
    dp_min = float(min(np.min(natural_dp), np.min(overall_dp)))
    dp_max = float(1.1 * max(np.max(natural_dp), np.max(overall_dp)))
    if np.isclose(dp_min, dp_max):
        dp_min -= 1.0
        dp_max += 1.0
    pressure_offsets = np.linspace(dp_min, dp_max, int(ns))
    component_values, z_line = _composition_line(info, int(ns), component_index)
    dp_grid, comp_grid = np.meshgrid(pressure_offsets, component_values)
    z_grid = np.repeat(z_line, int(ns), axis=0)
    flash_map = flash(
        mixture,
        (dp_grid.reshape(-1) + p_ref),
        np.full(dp_grid.size, T),
        z_grid,
    )
    vapour = np.asarray(flash_map.V, dtype=float)
    liquid = 1.0 - vapour
    denom = liquid * flash_map.Z_L + vapour * flash_map.Z_V
    saturation = np.divide(
        liquid * flash_map.Z_L,
        denom,
        out=np.zeros_like(liquid, dtype=float),
        where=np.abs(denom) > 0.0,
    ).reshape(int(ns), int(ns))

    natural_z = np.asarray([initial_natural.z[0, component_index],
                            natural_state.z[0, component_index]], dtype=float)
    overall_z = np.asarray([initial_overall.z[0, component_index],
                            overall_state.z[0, component_index]], dtype=float)
    natural_s = np.asarray([initial_natural.sW[0], natural_state.sW[0]], dtype=float)
    overall_s = np.asarray([initial_overall.sW[0], overall_state.sW[0]], dtype=float)
    style = "-" if max_change < 0.1 else "-o"
    base = {
        "dp_grid": dp_grid,
        "saturation_contours": saturation,
        "natural_dp": natural_dp,
        "overall_dp": overall_dp,
        "line_style": style,
        "legend": ("Natural variables", "Overall composition", "Solution"),
        "legend_location": "southeast",
        "ylabel": r"\Delta p [Pa]",
    }
    return {
        "composition_pressure": {
            **base,
            "figure": 1,
            "x_grid": comp_grid,
            "natural_x": natural_z,
            "overall_x": overall_z,
            "solution_x": overall_z[-1],
            "solution_dp": overall_dp[-1],
            "xlabel": f"z_{component_index}",
        },
        "saturation_pressure": {
            **base,
            "figure": 2,
            "x_grid": saturation,
            "natural_x": natural_s,
            "overall_x": overall_s,
            "solution_x": overall_s[-1],
            "solution_dp": overall_dp[-1],
            "xlabel": "S_L",
        },
    }


if __name__ == "__main__":
    def test_show_natural_overall_has_mrst_boundary_setup():
        out = run_show_natural_overall()
        assert out["iterations"].shape == (2,)
        assert out["grid"].cells.num == 1
        assert out["bc"].face.size == 2
        assert np.all(out["grid"].faces.neighbors[out["bc"].face].min(axis=1) < 0)
        assert np.isclose(out["bc"].value[0], 1.0 / DAY)
        assert out["bc"].components.shape == (2, out["mixture"].n)
        assert out["convergence_plots"]["composition_pressure"]["figure"] == 1
        assert out["convergence_plots"]["saturation_pressure"]["figure"] == 2
        assert out["convergence_plots"]["composition_pressure"]["dp_grid"].shape == (50, 50)
        assert np.all(np.isfinite(out["natural_state"].pressure))
        assert np.all(np.isfinite(out["overall_state"].pressure))
        assert np.all(out["vapour_fraction_initial"] >= 0.0)
        print("ok test_show_natural_overall_has_mrst_boundary_setup")

    test_show_natural_overall_has_mrst_boundary_setup()
    print("\nAll tests passed!")


__all__ = [
    "run_show_natural_overall",
    "setup_show_natural_overall",
    "show_natural_overall_convergence_plots",
]
