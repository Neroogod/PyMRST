"""Native adsorption comparison workflow (MRST ``examples/sorption.m``).

The complete grid, fracture, well, operator, compositional-state and paired
schedule implementation lives in this public MRST-layout owner. Diffusion and
Gangi use the same physical construction primitives with their distinct shale
mechanism parameters.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.examples._shared import run_schedule, simple_schedule
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.modules.compositional.utils.init_compositional_state import init_compositional_state
from pymrst.modules.shale.compositional_fns.nat_vars_shale_model import NatVarsShaleModel
from pymrst.modules.shale.hfm_fns import edfm_shale_grid, setup_shale_edfm_operators_tpfa
from pymrst.modules.shale.other_mrst_fns import add_well_shale, make_shale_rock, well_fracture_nncs_3d
from pymrst.modules.shale.utils import create_multistage_hfs, get_shale_comp_fluid_case
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import DARCY, DAY, MILLI, NANO, PSIA, YEAR


@dataclass
class MechanismComparison:
    mechanism: str
    global_grid: object
    fracture_planes: tuple
    well_connections: object
    wells: list
    model_enabled: object
    model_disabled: object
    initial_enabled: object
    initial_disabled: object
    timesteps: np.ndarray
    diagnostics: dict
    enabled_result: object | None = None
    disabled_result: object | None = None

    @property
    def times(self):
        return np.cumsum(self.timesteps)

    @property
    def gas_rates(self):
        if self.enabled_result is None or self.disabled_result is None:
            return None
        enabled = np.array([
            state.qGs[0] if state.qGs.size else np.nan
            for state in self.enabled_result.states
        ])
        disabled = np.array([
            state.qGs[0] if state.qGs.size else np.nan
            for state in self.disabled_result.states
        ])
        return {"enabled": enabled, "disabled": disabled}


def setup_mechanism_comparison(mechanism: str, *, full_size: bool = False):
    key = mechanism.lower()
    if key not in {"sorption", "diffusion", "gangi"}:
        raise ValueError(f"unknown shale mechanism {mechanism!r}")
    if full_size:
        dims, physical = (61, 11, 4), (1080.0, 250.0, 100.0)
        stages, first, spacing, half, height = 10, (80., 125., 50.), 100., 90., 60.
        total = 15 * YEAR
        nominal = YEAR if key == "sorption" else 30 * DAY
        ramp = 45 if key == "sorption" else 15
        producer_bhp = 500 * PSIA
    else:
        dims, physical = (6, 3, 2), (120.0, 60.0, 20.0)
        stages, first, spacing, half, height = 2, (20., 30., 10.), 80., 25., 12.
        total, nominal, ramp = 1000.0, 500.0, 2
        # A gentler drawdown keeps the compact CI case representative while
        # avoiding a timestep-cut dominated startup on its very coarse cells.
        producer_bhp = 4990 * PSIA

    permeability = {"sorption": 1e-4 * MILLI * DARCY,
                    "diffusion": 1.0 * NANO * DARCY,
                    "gangi": 100.0 * NANO * DARCY}[key]
    if not full_size:
        # Numerical acceptance case: retain the same fractured topology and
        # mechanism parameters but use a resolvable matrix permeability.
        permeability = 1e-3 * DARCY
    porosity = .04 if key == "sorption" else .063
    grid = compute_geometry(cart_grid(dims, physical))
    rock = make_shale_rock(
        grid, permeability, porosity, shale_mechanisms={key if key != "gangi" else "Gangi": True},
    )
    planes, _ = create_multistage_hfs(
        num_stages=stages, frac_spacing=spacing, num_fracs_per_stage=1,
        frac_half_length=half, frac_height=height, cluster_spacing=10,
        frac_centroid1=first,
    )
    for plane in planes:
        plane.aperture = .003
        plane.porosity = .33 if key == "sorption" else .5
        plane.permeability = 50 * DARCY if key == "sorption" else DARCY
    global_grid, processed = edfm_shale_grid(grid, planes, rock)

    path = np.array([[first[0], first[1], first[2]],
                     [physical[0] - first[0], first[1], first[2]]])
    connections = well_fracture_nncs_3d(
        [path], global_grid.fractures, global_grid.projections,
        matrix_cell_count=global_grid.n_matrix,
    )
    completion_table = np.c_[connections.fracture_cells, connections.fracture_index]
    if completion_table.size == 0:
        raise RuntimeError("reference well did not intersect a fracture cell")
    cell_size = np.asarray(physical) / np.asarray(dims)
    use_water = bool(full_size)
    reduced_rate = -1e-8
    wells = add_well_shale(
        [], global_grid, completion_table, fracture_aperture=np.full(stages, .003),
        fracture_cell_side_lengths=np.tile(cell_size[1:], (len(completion_table), 1)),
        well_type=("bhp" if full_size else "rate"),
        val=(producer_bhp if full_size else reduced_rate),
        compi=([.23, .76, .01] if use_water else [.99, .01]), sign=-1,
        name="Prod1", components=[.991, .0088, .0002], radius=.1,
    )
    fluid, info = get_shale_comp_fluid_case("barnett3comps")
    face_t = get_face_transmissibility(global_grid.grid, global_grid.rock)
    volumes = np.asarray(global_grid.grid.cells.volumes)
    operators = setup_shale_edfm_operators_tpfa(
        global_grid.grid, global_grid.rock, face_t, global_grid.nnc,
        cell_volumes=volumes,
    )
    model_wells = copy.deepcopy(wells) if full_size else []
    kwargs = dict(
        temperature=info.T, wells=model_wells, water=use_water,
        n_kr=((1, 1, 1) if use_water else (1, 1)), max_iterations=35, gravity=0,
    )
    extra = {}
    if key == "diffusion":
        extra.update(diffusion=np.array([2.8, 2.5, 1.9])*1e-7, tortuosity=2.)
    elif key == "gangi":
        from pymrst.modules.shale.utils.gangi import gangi_fn
        extra["gangi_function"] = lambda p: gangi_fn(
            p, 10000*PSIA, 1., 14000*PSIA, .4,
            2090.2259*NANO*DARCY, permeability,
        )
    enabled = NatVarsShaleModel(
        global_grid.grid, global_grid.rock, fluid,
        shale_rock=global_grid.rock, grain_volume=operators.grain_volume,
        **extra, **kwargs,
    )
    disabled_rock = copy.deepcopy(global_grid.rock)
    disabled_rock.shale_mechanisms = {}
    disabled = NaturalVariablesCompositionalModel(
        global_grid.grid, disabled_rock, fluid,
        **{**kwargs, "wells": copy.deepcopy(model_wells)},
    )
    for model in (enabled, disabled):
        model.ops = operators
        model.pv = operators.pv
        model.gdz = np.zeros(operators.N.shape[0])
    saturation = (np.tile([.23, .70, .07], (global_grid.n_cells, 1))
                  if use_water else None)
    pressure0 = info.pressure
    composition0 = info.initial
    if not full_size:
        xcoord = np.asarray(global_grid.grid.cells.centroids[:, 0], dtype=float)
        scale = (xcoord - xcoord.min()) / max(np.ptp(xcoord), 1.0)
        pressure0 = info.pressure - scale * 250 * PSIA
        if key == "diffusion":
            composition0 = np.tile(info.initial, (global_grid.n_cells, 1))
            shift = 1e-4 * (scale - .5)
            composition0[:, 0] += shift
            composition0[:, 1] -= shift
    initial_enabled = init_compositional_state(
        enabled, pressure0, info.T, saturation, composition0,
    )
    initial_disabled = init_compositional_state(
        disabled, pressure0, info.T, saturation, composition0,
    )
    timesteps = rampup_timesteps(total, nominal, ramp)
    diagnostics = {
        "matrix_cells": int(global_grid.n_matrix),
        "fracture_cells": int(global_grid.n_fracture),
        "nnc_count": int(global_grid.nnc.cells.shape[0]),
        "well_intersections": int(connections.points.shape[0]),
    }
    if key == "sorption":
        from pymrst.modules.shale.compositional_fns.equations_shale_natural_vars import (
            langmuir_sorbed_mass,
        )
        sorption_composition = np.asarray(composition0)
        if sorption_composition.ndim == 1:
            sorption_composition = np.tile(sorption_composition, (global_grid.n_cells, 1))
        diagnostics["initial_sorbed_mass"] = np.asarray(langmuir_sorbed_mass(
            np.asarray(pressure0), sorption_composition, fluid.isotherm,
            operators.grain_volume,
        ))
    elif key == "diffusion":
        diagnostics["initial_composition_gradient"] = np.column_stack([
            operators.grad(np.asarray(composition0)[:, i]) for i in range(fluid.n)
        ])
    else:
        diagnostics["initial_gangi_multiplier"] = np.asarray(
            enabled._gangi_multiplier(np.asarray(pressure0))
        )
    return MechanismComparison(
        key, global_grid, tuple(processed), connections, wells, enabled, disabled,
        initial_enabled, initial_disabled, timesteps, diagnostics,
    )


def run_mechanism_comparison(mechanism: str, *, full_size=False, nstep=None,
                             max_cuts=6, verbose=False):
    case = setup_mechanism_comparison(mechanism, full_size=full_size)
    steps = case.timesteps if nstep is None else case.timesteps[:int(nstep)]
    case.timesteps = steps
    active_wells = case.wells if full_size else []
    schedule = simple_schedule(steps, wells=active_wells)
    case.enabled_result = run_schedule(
        case.model_enabled, case.initial_enabled, schedule,
        max_cuts=max_cuts, verbose=verbose,
    )
    case.disabled_result = run_schedule(
        case.model_disabled, case.initial_disabled, schedule,
        max_cuts=max_cuts, verbose=verbose,
    )
    return case


def setup_sorption_case(*, full_size=False):
    return setup_mechanism_comparison("sorption", full_size=full_size)


def run_sorption_case(*, full_size=False, nstep=None, max_cuts=6, verbose=False):
    return run_mechanism_comparison(
        "sorption", full_size=full_size, nstep=nstep,
        max_cuts=max_cuts, verbose=verbose,
    )


__all__ = [
    "MechanismComparison", "setup_mechanism_comparison",
    "run_mechanism_comparison", "setup_sorption_case", "run_sorption_case",
]
