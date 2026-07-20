"""Port of MRST `book-ii/showNaturalOverall2.m`."""

from __future__ import annotations

import time

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.fluids import get_benchmark_mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import fluxside, pside
from pymrst.utils.units import DARCY, DAY


def setup_show_natural_overall2(
    mixture_name: str = "lumped_1",
    T_offset: float = 0.0,
    p_offset: float = 0.0,
    max_change: float = 0.1,
):
    """Construct MRST's configurable natural/overall benchmark case."""
    grid = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    rock = make_rock(grid, 0.5 * DARCY, 0.5)
    mixture, info = get_benchmark_mixture(mixture_name)
    p = info.pressure + p_offset
    T = info.T + T_offset
    sat_wog = np.array([0.0, 0.0, 1.0])
    bc = fluxside(None, grid, "xmin", 1.0 / DAY, sat=sat_wog)
    bc = pside(bc, grid, "xmax", p, sat=sat_wog)
    bc.components = np.tile(info.injection, (bc.face.size, 1))

    overall = OverallCompositionModel(grid, rock, mixture, temperature=T, water=True)
    natural = NaturalVariablesCompositionalModel(grid, rock, mixture, temperature=T, water=True)
    overall.dz_max = max_change
    natural.dz_max = max_change
    natural.ds_max = max_change
    state0_overall = overall.initial_state(p, info.initial, sW=0.5)
    state0_natural = natural.initial_state(p, info.initial, sW=0.5)

    bad_z = np.full(mixture.n, 1.0e-8)
    bad_z[0] = 1.0
    bad_z = bad_z / bad_z.sum()
    init_overall = overall.initial_state(p, bad_z, sW=0.5)
    init_natural = natural.initial_state(p, bad_z, sW=0.5)
    return {
        "grid": grid,
        "rock": rock,
        "mixture": mixture,
        "info": info,
        "bc": bc,
        "overall": overall,
        "natural": natural,
        "state0_overall": state0_overall,
        "state0_natural": state0_natural,
        "init_guess_overall": init_overall,
        "init_guess_natural": init_natural,
    }


def run_show_natural_overall2(
    mixture_name: str = "lumped_1",
    T_offset: float = 0.0,
    p_offset: float = 0.0,
    max_change: float = 0.1,
    dt: float = 1.0 * DAY,
):
    """Run MRST's second natural/overall comparison setup."""
    case = setup_show_natural_overall2(
        mixture_name=mixture_name,
        T_offset=T_offset,
        p_offset=p_offset,
        max_change=max_change,
    )
    tic = time.perf_counter()
    natural_state, natural_ok, natural_it = case["natural"].step(
        case["init_guess_natural"], dt, bc=case["bc"]
    )
    natural_wall = time.perf_counter() - tic
    tic = time.perf_counter()
    overall_state, overall_ok, overall_it = case["overall"].step(
        case["init_guess_overall"], dt, bc=case["bc"]
    )
    overall_wall = time.perf_counter() - tic
    if not natural_ok or not overall_ok:
        raise RuntimeError("benchmark natural/overall boundary-condition comparison did not converge")
    iterations = np.array([natural_it, overall_it])
    wall_time = np.array([natural_wall, overall_wall], dtype=float)
    case.update(
        {
            "natural_state": natural_state,
            "overall_state": overall_state,
            "iterations": iterations,
            "wall_time": wall_time,
            "runtime_plot": show_natural_overall2_runtime_plot(iterations, wall_time),
        }
    )
    return case


def show_natural_overall2_runtime_plot(iterations, wall_time) -> dict:
    """Return MRST's final iteration/runtime bar-plot payload."""
    return {
        "layout": (1, 2),
        "x_tick_labels": ("Natural", "Overall"),
        "iterations": {
            "subplot": (1, 2, 1),
            "bar": np.asarray(iterations, dtype=float),
            "title": "Iterations",
        },
        "wall_time": {
            "subplot": (1, 2, 2),
            "bar": np.asarray(wall_time, dtype=float),
            "title": "Wall time",
        },
    }


if __name__ == "__main__":
    def test_show_natural_overall2_runs_from_bad_guess():
        out = run_show_natural_overall2()
        assert out["iterations"].shape == (2,)
        assert out["grid"].cells.num == 1
        assert out["bc"].face.size == 2
        assert np.all(out["grid"].faces.neighbors[out["bc"].face].min(axis=1) < 0)
        assert np.isclose(out["bc"].value[0], 1.0 / DAY)
        assert out["bc"].components.shape == (2, out["mixture"].n)
        assert np.isclose(out["init_guess_overall"].z[0, 0], 1.0, rtol=0, atol=1.0e-6)
        assert out["wall_time"].shape == (2,)
        assert np.all(out["wall_time"] >= 0.0)
        assert out["runtime_plot"]["x_tick_labels"] == ("Natural", "Overall")
        assert out["runtime_plot"]["iterations"]["title"] == "Iterations"
        assert out["runtime_plot"]["wall_time"]["title"] == "Wall time"
        assert np.all(np.isfinite(out["natural_state"].pressure))
        assert np.all(np.isfinite(out["overall_state"].pressure))
        print("ok test_show_natural_overall2_runs_from_bad_guess")

    test_show_natural_overall2_runs_from_bad_guess()
    print("\nAll tests passed!")


__all__ = [
    "run_show_natural_overall2",
    "setup_show_natural_overall2",
    "show_natural_overall2_runtime_plot",
]
