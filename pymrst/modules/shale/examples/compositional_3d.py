"""3-D fractured compositional waterflood (MRST ``compositional3D.m``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.examples._shared import run_schedule, simple_schedule
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.modules.compositional.utils.init_compositional_state import init_compositional_state
from pymrst.modules.shale.data import FracturePlane, ShaleNNC
from pymrst.modules.shale.hfm_fns import (
    edfm_shale_grid,
    fracture_fracture_shale_nncs_3d,
    setup_shale_edfm_operators_tpfa,
)
from pymrst.modules.shale.other_mrst_fns import make_shale_rock
from pymrst.modules.shale.utils import get_shale_comp_fluid_case
from pymrst.params.wells_and_bc import add_well
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI, YEAR


@dataclass
class Compositional3DCase:
    grid: object
    global_grid: object
    fracture_planes: tuple
    wells: list
    mixture: object
    model: object
    initial_state: object
    timesteps: np.ndarray
    result: object | None = None

    @property
    def states(self):
        return [] if self.result is None else self.result.states


def _merge_nnc(first: ShaleNNC, second: ShaleNNC) -> ShaleNNC:
    return ShaleNNC(
        np.vstack([first.cells, second.cells]),
        np.r_[first.T, second.T],
        first.type + second.type,
        np.r_[first.area, second.area],
        np.vstack([first.normal, second.normal]),
        np.vstack([first.plane_point, second.plane_point]),
    )


def _reference_fractures(scale) -> list[FracturePlane]:
    p1 = np.array([[90,100,0], [140,160,0], [140,160,100], [90,100,100]], float)
    p2 = np.array([[130,160,0], [340,40,0], [340,40,100], [130,160,100]], float)
    normal = np.cross(p2[1]-p2[0], p2[2]-p2[0])
    normal /= np.linalg.norm(normal)
    p2[:2] += normal*5
    p2[2:] -= normal*5
    p3 = np.array([[250,70,0], [330,160,0], [330,160,100], [250,70,100]], float)
    return [
        FracturePlane(points*np.asarray(scale), aperture=1/25, porosity=.8,
                      permeability=10000*DARCY)
        for points in (p1, p2, p3)
    ]


def setup_compositional_3d_case(*, full_size: bool = False):
    """Build the complete MRST fractured compositional waterflood setup."""
    dims = (25, 10, 5) if full_size else (5, 3, 2)
    physical = (500., 200., 100.) if full_size else (100., 60., 20.)
    scale = np.asarray(physical) / np.array([500., 200., 100.])
    grid = compute_geometry(cart_grid(dims, physical))
    rock = make_shale_rock(grid, 100*MILLI*DARCY, .3)
    planes = _reference_fractures(scale)
    global_grid, processed = edfm_shale_grid(grid, planes, rock)
    fracfrac = fracture_fracture_shale_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        aperture=[plane.aperture for plane in planes], tolerance=1e-5,
    )
    global_grid.nnc = _merge_nnc(global_grid.nnc, fracfrac)
    face_t = get_face_transmissibility(global_grid.grid, global_grid.rock)
    operators = setup_shale_edfm_operators_tpfa(
        global_grid.grid, global_grid.rock, face_t, global_grid.nnc,
        cell_volumes=global_grid.grid.cells.volumes,
    )

    nx, ny, nz = dims
    total_time = 5*YEAR if full_size else 20*DAY
    injector_cells = np.array([nx*ny*k for k in range(nz)])
    producer_cells = np.array([nx*ny*(k+1)-1 for k in range(nz)])
    injection_rate = float(np.sum(operators.pv)/total_time*.7)
    wells = add_well(
        [], grid, rock, injector_cells, well_type="rate", val=injection_rate,
        radius=.1, compi=[1,0,0], name="Injector", sign=1,
        components=[1,0],
    )
    wells = add_well(
        wells, grid, rock, producer_cells, well_type="bhp", val=50*BARSA,
        radius=.1, compi=[.5,.5,0], name="Producer", sign=-1,
        components=[.1,.9],
    )
    mixture, info = get_shale_comp_fluid_case("verysimple")
    model = NaturalVariablesCompositionalModel(
        global_grid.grid, global_grid.rock, mixture, temperature=info.T,
        wells=wells, water=True, n_kr=(2,2,2), gravity=0,
        max_iterations=40,
    )
    model.ops = operators
    model.pv = operators.pv
    model.gdz = np.zeros(operators.N.shape[0])
    saturation = np.tile([0., 1., 0.], (global_grid.n_cells, 1))
    state0 = init_compositional_state(
        model, 100*BARSA, info.T, saturation, info.initial,
    )
    timesteps = rampup_timesteps(
        total_time, 30*DAY if full_size else 5*DAY, 8 if full_size else 2,
    )
    if full_size:
        timesteps = timesteps[:51]
    return Compositional3DCase(
        grid, global_grid, tuple(processed), wells, mixture, model, state0, timesteps,
    )


def run_compositional_3d_case(*, full_size=False, nstep=None, max_cuts=8,
                              verbose=False):
    case = setup_compositional_3d_case(full_size=full_size)
    steps = case.timesteps if nstep is None else case.timesteps[:int(nstep)]
    case.timesteps = steps
    case.result = run_schedule(
        case.model, case.initial_state, simple_schedule(steps, wells=case.wells),
        max_cuts=max_cuts, verbose=verbose,
    )
    return case


__all__ = ["Compositional3DCase", "setup_compositional_3d_case", "run_compositional_3d_case"]
