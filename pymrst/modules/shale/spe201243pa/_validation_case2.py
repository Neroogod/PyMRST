"""Shared native implementation for SPE-201243-PA validation Case 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import compute_geometry
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
from pymrst.params.wells_and_bc import add_well
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import DARCY, DAY, FT, MICRO, NANO, PSIA

from pymrst.modules.shale.spe201243pa.data_loader import load_cmg_timesteps
from pymrst.modules.shale.spe201243pa.explicit_stencil import explicit_stencil
from pymrst.modules.shale.spe201243pa.explicit_stencil_3_fracs import explicit_stencil_3_fracs


@dataclass
class ValidationVariant:
    name: str
    grid: object
    rock: object
    model: object
    wells: list
    initial_state: object
    result: object | None = None

    @property
    def states(self):
        return [] if self.result is None else self.result.states


@dataclass
class ValidationCase2:
    structured: bool
    matrix_grid: object
    fracture_planes: tuple[FracturePlane, ...]
    global_grid: object
    well_path: np.ndarray
    well_connections: object
    pedfm: ValidationVariant
    edfm: ValidationVariant
    explicit: ValidationVariant
    explicit_fracture_cells: np.ndarray
    explicit_well_cells: np.ndarray
    timesteps: np.ndarray


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


def _make_variant(name, global_grid, operators, wells, mixture, info):
    model = NaturalVariablesCompositionalModel(
        global_grid.grid, global_grid.rock, mixture, temperature=info.T,
        wells=wells, water=False, n_kr=(2, 2, 2), gravity=0,
        max_iterations=40,
    )
    model.ops = operators
    model.pv = operators.pv
    model.gdz = np.zeros(operators.N.shape[0])
    saturation = np.tile([1., 0.], (global_grid.n_cells, 1))
    state = init_compositional_state(
        model, 5700*PSIA, info.T, saturation, info.initial,
    )
    return ValidationVariant(name, global_grid.grid, global_grid.rock, model, wells, state)


def setup_validation_case2(*, structured=False):
    physical = np.array([1990., 1990., 150.])*FT
    aperture, half_length = .01*FT, 350*FT
    if structured:
        matrix, _, _ = explicit_stencil(
            physical, aperture=aperture, fracture_spacing=physical[0],
            fracture_half_length=half_length, nx_refine=32, tip_ny=8,
            ny=8, nz=6, grid_type="cartesian",
        )
    else:
        matrix, _, _ = explicit_stencil_3_fracs(
            physical, aperture=aperture, fracture_spacing=physical[0],
            fracture_half_length=half_length, nx_refine=12,
            nx_refine_small=10, tip_ny=8, ny=8, nz=6,
            grid_type="cartesian",
        )
    matrix = compute_geometry(matrix)
    matrix_rock = make_shale_rock(matrix, 100*MICRO*DARCY, .16)
    xfeet = (930., 995., 1060.)
    permeabilities = (NANO*DARCY, 50*DARCY, NANO*DARCY)
    planes = tuple(FracturePlane(
        np.array([[x,645,0], [x,1345,0], [x,1345,150], [x,645,150]])*FT,
        aperture=aperture, porosity=.16, permeability=perm,
        metadata={"kind": "barrier" if index != 1 else "hydraulic"},
    ) for index, (x, perm) in enumerate(zip(xfeet, permeabilities)))
    global_grid, processed = edfm_shale_grid(matrix, planes, matrix_rock, tolerance=1e-3)
    ff = fracture_fracture_shale_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        aperture=[aperture]*3, tolerance=1e-3,
    )
    global_grid.nnc = _merge_nnc(global_grid.nnc, ff)
    midpoint = physical/2
    well_path = np.array([
        midpoint+[-5,0,0], midpoint+[5,0,0],
    ])
    connections = well_fracture_nncs_3d(
        [well_path], global_grid.fractures, global_grid.projections,
        matrix_cell_count=matrix.cells.num, tolerance=1e-3,
    )
    completions = np.column_stack([
        connections.fracture_cells, connections.fracture_index,
    ])
    mixture, info = get_shale_comp_fluid_case("c1c2c3")
    wells = add_well_shale(
        [], global_grid, completions,
        fracture_aperture=[aperture]*3,
        fracture_cell_side_lengths=_completion_sides(global_grid, completions),
        radius=.25*FT, well_type="bhp", val=1000*PSIA,
        compi=[1,0], sign=-1, name="Prod1", components=info.initial,
    )
    half_t = get_face_transmissibility(global_grid.grid, global_grid.rock)
    edfm_ops = setup_shale_edfm_operators_tpfa(
        global_grid.grid, global_grid.rock, half_t, global_grid.nnc,
        cell_volumes=global_grid.grid.cells.volumes,
    )
    projected = projected_matrix_fracture_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        tolerance=1e-3,
    )
    pedfm_ops = setup_pedfm_operators_tpfa(
        global_grid.grid, global_grid.rock, half_t, global_grid.nnc,
        projected=projected, tolerance=1e-3,
        cell_volumes=global_grid.grid.cells.volumes,
    )
    pedfm = _make_variant("pedfm", global_grid, pedfm_ops, wells, mixture, info)
    edfm = _make_variant("edfm", global_grid, edfm_ops, wells, mixture, info)

    explicit_grid, frac_ids, well_ids = explicit_stencil_3_fracs(
        physical, aperture=aperture, fracture_spacing=physical[0],
        fracture_half_length=half_length, nx_refine=12,
        nx_refine_small=10, tip_ny=8, ny=8, nz=6,
        grid_type="geomspacing",
    )
    explicit_grid = compute_geometry(explicit_grid)
    explicit_rock = make_shale_rock(explicit_grid, 100*MICRO*DARCY, .16)
    for ids, perm in zip(frac_ids, permeabilities):
        explicit_rock.perm[ids, :] = perm
    explicit_wells = add_well(
        [], explicit_grid, explicit_rock, well_ids, well_type="bhp",
        val=1000*PSIA, radius=.25*FT, compi=[1,0], sign=-1,
        name="Prod1", components=info.initial,
    )
    explicit_model = NaturalVariablesCompositionalModel(
        explicit_grid, explicit_rock, mixture, temperature=info.T,
        wells=explicit_wells, water=False, n_kr=(2,2,2), gravity=0,
        max_iterations=40,
    )
    explicit_state = init_compositional_state(
        explicit_model, 5700*PSIA, info.T,
        np.tile([1.,0.], (explicit_grid.cells.num, 1)), info.initial,
    )
    explicit = ValidationVariant(
        "explicit", explicit_grid, explicit_rock, explicit_model,
        explicit_wells, explicit_state,
    )
    return ValidationCase2(
        bool(structured), matrix, tuple(processed), global_grid, well_path,
        connections, pedfm, edfm, explicit, frac_ids, well_ids,
        load_cmg_timesteps()*DAY,
    )


def run_validation_case2(*, nstep=None, timesteps=None, max_cuts=8, **kwargs):
    case = setup_validation_case2(**kwargs)
    steps = case.timesteps if timesteps is None else np.asarray(timesteps, float)
    steps = steps if nstep is None else steps[:int(nstep)]
    case.timesteps = steps
    for variant in (case.pedfm, case.edfm, case.explicit):
        variant.result = run_schedule(
            variant.model, variant.initial_state,
            simple_schedule(steps, wells=variant.wells), max_cuts=max_cuts,
        )
    return case


__all__ = ["ValidationCase2", "ValidationVariant", "setup_validation_case2", "run_validation_case2"]
