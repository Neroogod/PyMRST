"""Stochastic fracture network workflow (MRST ``stochasticFracs.m``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.shale.data import FracturePlane
from pymrst.modules.shale.examples.data_loader import load_shale_dataset
from pymrst.modules.shale.hfm_fns import edfm_shale_grid
from pymrst.modules.shale.other_mrst_fns import make_shale_rock
from pymrst.modules.shale.utils.plot_frac_system_nf import plot_frac_system_nf
from pymrst.modules.shale.utils.process_stochastic_fractures import process_stochastic_fractures
from pymrst.utils.units import DARCY, FT, MILLI


@dataclass
class StochasticFractureCase:
    grid: object
    hydraulic_set: tuple[np.ndarray, ...]
    natural_set: tuple[np.ndarray, ...]
    fracture_planes: tuple[FracturePlane, ...]
    hydraulic_count: int
    natural_count: int
    planar_count: int
    nonplanar_count: int
    areas: np.ndarray
    source: str
    global_grid: object | None = None
    processed_planes: tuple[FracturePlane, ...] = ()
    plot_result: object | None = None


def _polygons(values: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(value, dtype=float) for value in values)


def setup_stochastic_fractures(
    *,
    fracture_sets=None,
    generator: Callable | None = None,
    seed: int = 12345,
    tolerance: float = .01,
    build_edfm: bool = False,
    edfm_limit: int | None = None,
    plot: bool = False,
    plotter=None,
):
    """Load/generate, classify and optionally preprocess the MRST DFN.

    An external ADFNE-like generator receives ``(rng, physical_dimensions,
    tolerance)`` and must return ``(hydraulic_set, natural_set)``. Without it,
    the original 700-polygon MRST realization is loaded from package data.
    """
    physical = np.array([3300., 1300., 250.]) * FT
    dims = (98, 15, 3)
    grid = compute_geometry(cart_grid(dims, physical))
    if fracture_sets is not None and generator is not None:
        raise ValueError("provide fracture_sets or generator, not both")
    if generator is not None:
        fracture_sets = generator(np.random.default_rng(seed), physical.copy(), tolerance)
        source = "generator"
    elif fracture_sets is None:
        raw = load_shale_dataset("stochastic_fractures")
        fracture_sets = (raw["set1"], raw["set2"])
        source = "bundled-mrst"
    else:
        source = "provided"
    if len(fracture_sets) != 2:
        raise ValueError("fracture_sets must contain hydraulic and natural collections")
    hydraulic, natural = map(_polygons, fracture_sets)
    combined = hydraulic + natural
    planar, nonplanar, areas = process_stochastic_fractures(
        combined, area_tolerance=tolerance**2,
    )
    planes = []
    for index, polygon in enumerate(combined):
        vertices = polygon[:-1]
        if vertices.shape[0] < 3 or not np.all(np.isfinite(vertices)) or areas[index] <= 0:
            continue
        planes.append(FracturePlane(
            vertices,
            aperture=.003,
            porosity=.5,
            permeability=DARCY,
            metadata={
                "set": "hydraulic" if index < len(hydraulic) else "natural",
                "source_index": index if index < len(hydraulic) else index-len(hydraulic),
            },
        ))
    case = StochasticFractureCase(
        grid, hydraulic, natural, tuple(planes), len(hydraulic), len(natural),
        len(planar), len(nonplanar), areas, source,
    )
    if build_edfm:
        selected = planes if edfm_limit is None else planes[:int(edfm_limit)]
        rock = make_shale_rock(grid, 100*MILLI*DARCY, .1)
        case.global_grid, processed = edfm_shale_grid(grid, selected, rock)
        case.processed_planes = tuple(processed)
    if plot:
        case.plot_result = plot_frac_system_nf(
            grid, planes, min(len(hydraulic), len(planes)), (),
            label=False, plotter=plotter, show=plotter is None,
        )
    return case


stochasticFracs = setup_stochastic_fractures

if __name__ == "__main__":
    # `plot=True` with no plotter reaches `plot_frac_system_nf(show=True)`,
    # which opens a window and waits for it to be closed. Run as a captured
    # subprocess by `tests/test_all.py` that never returns, and the sweep sat
    # on this file for half an hour with no diagnostic. The geometry is what
    # the example is about, so build it and check it instead.
    def test_stochastic_fractures_builds_without_drawing():
        case = setup_stochastic_fractures(plot=False)
        assert case.plot_result is None, "plotting ran despite plot=False"
        print("ok test_stochastic_fractures_builds_without_drawing")

    test_stochastic_fractures_builds_without_drawing()
    print("\nAll tests passed!")


__all__ = ["StochasticFractureCase", "setup_stochastic_fractures", "stochasticFracs"]
