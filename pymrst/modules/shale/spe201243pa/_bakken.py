"""Shared native Bakken workflow for SPE-201243-PA Figures 21--23."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.examples._shared import run_schedule, simple_schedule
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.modules.compositional.utils.init_compositional_state import init_compositional_state
from pymrst.modules.shale.data import FracturePlane, ShaleNNC
from pymrst.modules.shale.hfm_fns import (
    edfm_shale_grid, fracture_fracture_shale_nncs_3d,
    setup_shale_edfm_operators_tpfa,
)
from pymrst.modules.shale.other_mrst_fns import (
    add_well_shale, make_shale_rock, well_fracture_nncs_3d,
)
from pymrst.modules.shale.pedfm_nnc import (
    projected_matrix_fracture_nncs_3d, setup_pedfm_operators_tpfa,
)
from pymrst.modules.shale.utils import get_shale_comp_fluid_case
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import DARCY, MICRO, MILLI, NANO, PSIA, YEAR

from pymrst.modules.shale.spe201243pa.multistage_planar_nonplanar_hf import multistage_planar_nonplanar_hf
from pymrst.modules.shale.spe201243pa.structured_grid_fracture_on_face import structured_grid_fracture_on_face


@dataclass
class BakkenCase:
    method: str
    matrix_grid: object
    hydraulic_fractures: tuple[FracturePlane, ...]
    natural_fractures: tuple[FracturePlane, ...]
    well_path: np.ndarray
    source: str
    global_grid: object | None = None
    well_connections: object | None = None
    operators: object | None = None
    wells: list | None = None
    model: object | None = None
    initial_state: object | None = None
    timesteps: np.ndarray | None = None
    result: object | None = None

    @property
    def fracture_planes(self):
        return self.hydraulic_fractures+self.natural_fractures

    @property
    def states(self):
        return [] if self.result is None else self.result.states


def _native_adfne(seed: int, physical: np.ndarray):
    """Generate the two seeded 500-plane statistical families from the paper."""
    rng = np.random.default_rng(seed)
    out = []
    margin = 1e-5
    for family, azimuth in enumerate((45., -45.)):
        azimuth = np.deg2rad(azimuth)
        strike = np.array([np.cos(azimuth), np.sin(azimuth), 0.])
        dip_horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
        dip_axis = (dip_horizontal+np.array([0., 0., 1.]))/np.sqrt(2.)
        for index in range(500):
            # ADFNE input uses minl=10, mu=10, maxl=20. A bounded lognormal
            # preserves that exact support and characteristic scale.
            length = float(np.clip(rng.lognormal(np.log(10.), .28), 10., 20.))
            height = float(np.clip(rng.lognormal(np.log(10.), .28), 10., 20.))
            half_extent = .5*(np.abs(strike)*length+np.abs(dip_axis)*height)
            low, high = half_extent+margin, physical-half_extent-margin
            center = rng.uniform(low, high)
            points = np.array([
                center-strike*length/2-dip_axis*height/2,
                center+strike*length/2-dip_axis*height/2,
                center+strike*length/2+dip_axis*height/2,
                center-strike*length/2+dip_axis*height/2,
            ])
            out.append((family, index, points))
    return out


def _merge_nnc(a: ShaleNNC, b: ShaleNNC):
    return ShaleNNC(
        np.vstack([a.cells, b.cells]), np.r_[a.T, b.T], a.type+b.type,
        np.r_[a.area, b.area], np.vstack([a.normal, b.normal]),
        np.vstack([a.plane_point, b.plane_point]),
    )


def _completion_sides(global_grid, completions):
    offsets = np.cumsum(
        [global_grid.n_matrix]
        + [projection.matrix_cells.size for projection in global_grid.projections]
    )
    sides = []
    for cell, plane_index in completions:
        plane_index = int(plane_index)
        local = int(cell-offsets[plane_index])
        polygon = np.asarray(global_grid.projections[plane_index].polygons[local])
        _, axis1, axis2 = global_grid.fractures[plane_index].frame()
        rel = polygon-polygon.mean(axis=0)
        sides.append([np.ptp(rel@axis1), np.ptp(rel@axis2)])
    return np.asarray(sides)


def setup_bakken_case(
    *, method="edfm", seed=12345, fracture_generator: Callable | None = None,
    natural_fracture_limit: int | None = None, build_edfm=False,
):
    method = str(method).lower()
    if method not in {"edfm", "pedfm"}:
        raise ValueError("method must be 'edfm' or 'pedfm'")
    physical = np.array([1200., 300., 90.])
    hydraulic, centers = multistage_planar_nonplanar_hf(
        num_stages=8, fracture_spacing=100., fractures_per_stage=3,
        fracture_half_length=70., fracture_height=60., theta=np.pi/2,
        cluster_spacing=10., heel_coordinate=[125., 150., 45.],
    )
    for index, plane in enumerate(hydraulic):
        plane.aperture, plane.porosity, plane.permeability = 3*MILLI, .5, DARCY
        plane.metadata.update(kind="hydraulic", source_index=index)
    grid = compute_geometry(structured_grid_fracture_on_face(
        physical, centers, num_stages=8, fractures_per_stage=3,
        ny=12, nz=5, nx_refine_small=4, fracture_spacing=100., nx_refine=5,
    ))
    rock = make_shale_rock(grid, 100*NANO*DARCY, .063)
    generated = (
        _native_adfne(seed, physical)
        if fracture_generator is None else fracture_generator(
            np.random.default_rng(seed), physical.copy(), 1e-5,
        )
    )
    source = "native-seeded-adfne" if fracture_generator is None else "provided-generator"
    generated = list(generated)
    if natural_fracture_limit is not None:
        if int(natural_fracture_limit) < 0:
            raise ValueError("natural_fracture_limit must be non-negative")
        generated = generated[:int(natural_fracture_limit)]
    natural = []
    for ordinal, item in enumerate(generated):
        if len(item) == 3:
            family, source_index, points = item
        else:
            family, source_index, points = ordinal//500, ordinal%500, item
        natural.append(FracturePlane(
            np.asarray(points, float), aperture=100*NANO, porosity=.5,
            permeability=(NANO if ordinal % 2 == 0 else 100*MICRO)*DARCY,
            metadata={"kind": "natural", "family": int(family),
                      "source_index": int(source_index)},
        ))
    case = BakkenCase(
        method, grid, tuple(hydraulic), tuple(natural),
        np.array([[90.,150.,45.], [1000.,150.,45.]]), source,
        timesteps=rampup_timesteps(30*YEAR, YEAR, 45),
    )
    if not build_edfm:
        return case
    planes = case.fracture_planes
    global_grid, _ = edfm_shale_grid(grid, planes, rock, tolerance=1e-5)
    ff = fracture_fracture_shale_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        aperture=[plane.aperture for plane in planes], tolerance=1e-5,
    )
    global_grid.nnc = _merge_nnc(global_grid.nnc, ff)
    connections = well_fracture_nncs_3d(
        [case.well_path], global_grid.fractures, global_grid.projections,
        matrix_cell_count=grid.cells.num, tolerance=1e-5,
    )
    completions = np.column_stack([
        connections.fracture_cells, connections.fracture_index,
    ])
    mixture, info = get_shale_comp_fluid_case("bakken")
    wells = add_well_shale(
        [], global_grid, completions,
        fracture_aperture=[plane.aperture for plane in planes],
        fracture_cell_side_lengths=_completion_sides(global_grid, completions),
        radius=.1, well_type="bhp", val=2000*PSIA,
        compi=[.23,.76,.01], sign=-1, name="Prod1",
        hydraulic_plane_count=len(hydraulic), components=info.initial,
    )
    half_t = get_face_transmissibility(global_grid.grid, global_grid.rock)
    volumes = np.asarray(global_grid.grid.cells.volumes)
    if method == "pedfm":
        projected = projected_matrix_fracture_nncs_3d(
            global_grid.edfm, global_grid.fractures, global_grid.projections,
            tolerance=1e-5,
        )
        operators = setup_pedfm_operators_tpfa(
            global_grid.grid, global_grid.rock, half_t, global_grid.nnc,
            projected=projected, tolerance=1e-5, cell_volumes=volumes,
        )
    else:
        operators = setup_shale_edfm_operators_tpfa(
            global_grid.grid, global_grid.rock, half_t, global_grid.nnc,
            cell_volumes=volumes,
        )
    model = NaturalVariablesCompositionalModel(
        global_grid.grid, global_grid.rock, mixture, temperature=info.T,
        wells=wells, water=True, n_kr=(2,2,2), gravity=0,
        max_iterations=40,
    )
    model.ops, model.pv = operators, operators.pv
    model.gdz = np.zeros(operators.N.shape[0])
    state = init_compositional_state(
        model, info.pressure, info.T,
        np.tile([.23,.70,.07], (global_grid.n_cells, 1)), info.initial,
    )
    case.global_grid, case.well_connections = global_grid, connections
    case.operators, case.wells, case.model, case.initial_state = operators, wells, model, state
    return case


def run_bakken_case(*, nstep=None, timesteps=None, max_cuts=8, **kwargs):
    kwargs["build_edfm"] = True
    case = setup_bakken_case(**kwargs)
    steps = case.timesteps if timesteps is None else np.asarray(timesteps, float)
    steps = steps if nstep is None else steps[:int(nstep)]
    case.timesteps = steps
    case.result = run_schedule(
        case.model, case.initial_state, simple_schedule(steps, wells=case.wells),
        max_cuts=max_cuts,
    )
    return case


__all__ = ["BakkenCase", "setup_bakken_case", "run_bakken_case"]
