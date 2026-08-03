"""Foundation analytic benchmarks using PyMRST numerical owners."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import j0
import warnings

from pymrst.examples.flow_solver_tutorial1 import run_flow_solver_tutorial1
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp import explicit_transport, incomp_tpfa
from pymrst.modules.incomp.fluid import init_simple_fluid
from pymrst.modules.ad_mechanics.examples.mandel import run_mandel
from pymrst.modules.ad_mechanics.examples.terzaghi import run_terzaghi
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.compositional.eos import flash, simple_mixture
from pymrst.modules.compositional.properties.fugacity_lv import fugacity_lv
from pymrst.modules.vemmech.utils.linear_displacement import linear_displacement
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c, enu2c_2d
from pymrst.modules.vemmech.vem_assemble import vem_assemble
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast
from pymrst.params.rock import make_rock, pore_volume
from pymrst.params.wells_and_bc import add_source
from pymrst.solvers.state import init_res_sol
from pymrst.solvers.trans import compute_trans
from pymrst.utils.units import DAY, DARCY, MILLI
from pymrst.utils.adi import init_variables_adi
from ..analytic import (
    AnalyticCaseResult, AnalyticCaseSpec, AnalyticRegistry, RefinementLevel,
    observed_order,
)


HYDROSTATIC_SPEC = AnalyticCaseSpec(
    case_id="hydrostatic_tpfa_column",
    name="Single-phase hydrostatic TPFA column",
    assumptions=("incompressible fluid", "constant density", "constant gravity",
                 "top Dirichlet pressure", "no-flow remaining boundaries"),
    domain="1 m x 1 m x 30 m vertical Cartesian column",
    variables=("cell_pressure", "face_flux"),
    units={"cell_pressure": "Pa", "face_flux": "m^3/s"},
    expected_order=None,
)

DIFFUSION_SPEC = AnalyticCaseSpec(
    case_id="pressure_diffusion_1d",
    name="One-dimensional constant-coefficient pressure diffusion",
    assumptions=("constant diffusivity", "zero initial excess pressure",
                 "unit left Dirichlet step", "zero right Dirichlet pressure",
                 "cell-centred TPFA", "backward Euler with dt proportional to dx^2"),
    domain="0 <= x <= 100 m",
    variables=("normalized_pressure",),
    units={"normalized_pressure": "1", "x": "m", "time": "s"},
    expected_order=2.0,
    minimum_order=1.9,
)

RADIAL_SPEC = AnalyticCaseSpec(
    case_id="radial_pressure_bessel_transient",
    name="Axisymmetric radial pressure transient on an annulus",
    assumptions=("constant diffusivity", "axisymmetric flow", "unit thickness",
                 "exact time-dependent Dirichlet data at both radii",
                 "cell-centred radial finite volumes", "backward Euler"),
    domain="1 m <= r <= 10 m",
    variables=("normalized_pressure",),
    units={"normalized_pressure": "1", "r": "m", "time": "s"},
    expected_order=2.0, minimum_order=1.85,
)

BUCKLEY_LEVERETT_SPEC = AnalyticCaseSpec(
    case_id="buckley_leverett_equal_mobility",
    name="Buckley--Leverett displacement with quadratic relative permeability",
    assumptions=("incompressible two-phase flow", "no gravity or capillarity",
                 "equal viscosities", "kr_w=Sw^2", "kr_o=(1-Sw)^2",
                 "one-dimensional homogeneous rock", "0.3 pore volumes injected"),
    domain="0 <= x_D <= 1",
    variables=("water_saturation", "shock_position"),
    units={"water_saturation": "1", "shock_position": "pore-volume coordinate"},
    expected_order=1.0, minimum_order=0.7,
)

TERZAGHI_SPEC = AnalyticCaseSpec(
    case_id="terzaghi_consolidation",
    name="Terzaghi one-dimensional consolidation",
    assumptions=("linear poroelasticity", "top drained", "bottom sealed",
                 "instantaneous constant vertical load", "one-dimensional strain",
                 "backward Euler pressure diffusion"),
    domain="20 m poroelastic column",
    variables=("excess_pressure", "top_settlement"),
    units={"excess_pressure": "Pa", "top_settlement": "m"},
    expected_order=1.0, minimum_order=0.85,
)

MANDEL_SPEC = AnalyticCaseSpec(
    case_id="mandel_cryer_coupling",
    name="Mandel constant-load poroelastic slab",
    assumptions=("linear poroelasticity", "plane strain", "constant plate load",
                 "far side drained", "remaining boundaries sealed",
                 "moving rigid plate"),
    domain="10 m x 1 m poroelastic half-slab",
    variables=("normalized_pressure", "plate_displacement"),
    units={"normalized_pressure": "1", "plate_displacement": "m"},
    expected_order=1.0, minimum_order=0.8,
)

ELASTICITY_PATCH_SPEC = AnalyticCaseSpec(
    case_id="elasticity_linear_patch",
    name="VEM linear-elasticity patch tests",
    assumptions=("homogeneous isotropic linear elasticity",
                 "analytic affine displacement prescribed on the boundary",
                 "zero body force", "regular and distorted polyhedral cells"),
    domain="2D distorted quadrilateral and 3D Cartesian patches",
    variables=("nodal_displacement", "elastic_energy"),
    units={"nodal_displacement": "m", "elastic_energy": "J"},
    expected_order=None,
)

FLASH_EQUILIBRIUM_SPEC = AnalyticCaseSpec(
    case_id="peng_robinson_flash_identities",
    name="Peng--Robinson two-phase flash identities",
    assumptions=("isothermal-isobaric equilibrium", "three-component mixture",
                 "positive normalized feed", "two-phase states selected explicitly"),
    domain="pressure-temperature-composition state ensemble",
    variables=("vapour_fraction", "liquid_composition", "vapour_composition",
               "component_fugacity"),
    units={"vapour_fraction": "1", "liquid_composition": "1",
           "vapour_composition": "1", "component_fugacity": "Pa"},
    expected_order=None,
)

ACCUMULATION_FLUX_SPEC = AnalyticCaseSpec(
    case_id="manufactured_accumulation_flux_balance",
    name="Manufactured AD accumulation and TPFA flux balance",
    assumptions=("one-dimensional Cartesian grid", "constant compressibility",
                 "constant permeability", "closed external boundary",
                 "source manufactured from the prescribed pressure field"),
    domain="1D cell-centred TPFA chains",
    variables=("cell_residual", "face_flux", "residual_jacobian"),
    units={"cell_residual": "m^3/s", "face_flux": "m^3/s",
           "residual_jacobian": "m^3/(Pa s)"},
    expected_order=None,
)


def run_hydrostatic_benchmark(resolutions=(10, 20, 40, 80)) -> AnalyticCaseResult:
    levels = []
    for nz in resolutions:
        result = run_flow_solver_tutorial1(nz=int(nz))
        error = result.pressure - result.analytic_pressure()
        levels.append(RefinementLevel(
            int(nz), 30.0 / int(nz), None, None,
            errors={
                "l1_pressure_Pa": np.mean(np.abs(error)),
                "l2_pressure_Pa": np.sqrt(np.mean(error * error)),
                "linf_pressure_Pa": np.max(np.abs(error)),
            },
            invariants={"maximum_absolute_face_flux_m3_s":
                        np.max(np.abs(np.asarray(result.state.flux, float)))},
        ))
    levels = tuple(levels)
    maximum_pressure = max(level.errors["linf_pressure_Pa"] for level in levels)
    maximum_flux = max(level.invariants["maximum_absolute_face_flux_m3_s"]
                       for level in levels)
    acceptance = {"pressure_absolute_tolerance_Pa": 1e-5,
                  "flux_absolute_tolerance_m3_s": 1e-15}
    return AnalyticCaseResult(
        HYDROSTATIC_SPEC, levels, {},
        bool(maximum_pressure <= acceptance["pressure_absolute_tolerance_Pa"]
             and maximum_flux <= acceptance["flux_absolute_tolerance_m3_s"]),
        acceptance,
    )


def diffusion_exact(x, time, *, length=100.0, diffusivity=0.02,
                    terms=300) -> np.ndarray:
    """Exact normalized solution for a left-boundary unit pressure step."""
    x = np.asarray(x, float).reshape(-1)
    n = np.arange(1, int(terms) + 1, dtype=float)[:, None]
    scaled_x = x[None, :] / float(length)
    tau = float(diffusivity) * float(time) / float(length)**2
    transient = (2.0 / np.pi) * np.sum(
        np.sin(n * np.pi * scaled_x) / n * np.exp(-(n * np.pi)**2 * tau), axis=0,
    )
    return 1.0 - x / float(length) - transient


def _diffusion_level(ncells: int, *, length=100.0, diffusivity=0.02,
                     final_tau=0.05, step_factor=0.2) -> RefinementLevel:
    n = int(ncells)
    dx = float(length) / n
    final_time = float(final_tau) * float(length)**2 / float(diffusivity)
    target_dt = float(step_factor) * dx * dx / float(diffusivity)
    steps = int(np.ceil(final_time / target_dt))
    dt = final_time / steps

    main = -2.0 * np.ones(n)
    main[[0, -1]] = -3.0  # half-cell distance to Dirichlet faces
    operator = (float(diffusivity) / dx**2) * sp.diags(
        (np.ones(n - 1), main, np.ones(n - 1)), (-1, 0, 1), format="csc",
    )
    forcing = np.zeros(n)
    forcing[0] = 2.0 * float(diffusivity) / dx**2
    solve = spla.factorized(sp.eye(n, format="csc") - dt * operator)
    pressure = np.zeros(n)
    cumulative_net_boundary_volume = 0.0
    for _ in range(steps):
        pressure = solve(pressure + dt * forcing)
        inflow = 2.0 * float(diffusivity) * (1.0 - pressure[0]) / dx
        outflow = 2.0 * float(diffusivity) * pressure[-1] / dx
        cumulative_net_boundary_volume += dt * (inflow - outflow)

    x = (np.arange(n) + 0.5) * dx
    exact = diffusion_exact(x, final_time, length=length, diffusivity=diffusivity)
    error = pressure - exact
    stored = float(np.sum(pressure) * dx)
    return RefinementLevel(
        n, dx, dt, steps,
        errors={"l1": np.mean(np.abs(error)),
                "l2": np.sqrt(np.mean(error * error)),
                "linf": np.max(np.abs(error))},
        invariants={
            "storage_boundary_balance": abs(stored - cumulative_net_boundary_volume),
            "minimum_pressure": np.min(pressure),
            "maximum_pressure": np.max(pressure),
        },
    )


def run_diffusion_benchmark(resolutions=(10, 20, 40, 80)) -> AnalyticCaseResult:
    levels = tuple(_diffusion_level(n) for n in resolutions)
    orders = {norm: observed_order(levels, norm) for norm in ("l1", "l2", "linf")}
    max_balance = max(level.invariants["storage_boundary_balance"] for level in levels)
    bounded = all(level.invariants["minimum_pressure"] >= -1e-14 and
                  level.invariants["maximum_pressure"] <= 1.0 + 1e-14
                  for level in levels)
    acceptance = {"minimum_l2_order": 1.9, "maximum_finest_l2_error": 2e-4,
                  "maximum_storage_balance": 1e-10}
    passed = (
        orders["l2"] >= acceptance["minimum_l2_order"]
        and levels[-1].errors["l2"] <= acceptance["maximum_finest_l2_error"]
        and max_balance <= acceptance["maximum_storage_balance"] and bounded
    )
    return AnalyticCaseResult(DIFFUSION_SPEC, levels, orders, bool(passed), acceptance)


def radial_bessel_exact(radius, time, *, diffusivity=0.5, wave_number=0.3):
    """Exact radial diffusion mode ``J0(alpha*r) exp(-D alpha^2 t)``."""
    radius = np.asarray(radius, float)
    return j0(float(wave_number) * radius) * np.exp(
        -float(diffusivity) * float(wave_number)**2 * float(time)
    )


def _radial_level(ncells: int, *, inner=1.0, outer=10.0, diffusivity=0.5,
                  wave_number=0.3, final_time=0.4, step_factor=0.2):
    n = int(ncells)
    faces = np.linspace(inner, outer, n + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    dr = (outer - inner) / n
    volumes = np.pi * (faces[1:]**2 - faces[:-1]**2)
    stiffness = sp.lil_matrix((n, n))
    for i in range(n - 1):
        trans = 2 * np.pi * faces[i + 1] * diffusivity / (centers[i + 1] - centers[i])
        stiffness[i, i] -= trans
        stiffness[i, i + 1] += trans
        stiffness[i + 1, i] += trans
        stiffness[i + 1, i + 1] -= trans
    west = 2 * np.pi * inner * diffusivity / (centers[0] - inner)
    east = 2 * np.pi * outer * diffusivity / (outer - centers[-1])
    stiffness[0, 0] -= west
    stiffness[-1, -1] -= east
    operator = sp.diags(1.0 / volumes) @ stiffness.tocsr()
    target_dt = step_factor * dr * dr / diffusivity
    steps = int(np.ceil(final_time / target_dt))
    dt = final_time / steps
    solve = spla.factorized(sp.eye(n, format="csc") - dt * operator)
    pressure = radial_bessel_exact(centers, 0.0, diffusivity=diffusivity,
                                   wave_number=wave_number)
    initial_storage = float(volumes @ pressure)
    integrated_boundary_flux = 0.0
    for step in range(steps):
        time = (step + 1) * dt
        left_value = float(radial_bessel_exact(inner, time, diffusivity=diffusivity,
                                               wave_number=wave_number))
        right_value = float(radial_bessel_exact(outer, time, diffusivity=diffusivity,
                                                wave_number=wave_number))
        rhs = pressure.copy()
        rhs[0] += dt * west / volumes[0] * left_value
        rhs[-1] += dt * east / volumes[-1] * right_value
        pressure = solve(rhs)
        inward = west * (left_value - pressure[0])
        outward = east * (pressure[-1] - right_value)
        integrated_boundary_flux += dt * (inward - outward)
    exact = radial_bessel_exact(centers, final_time, diffusivity=diffusivity,
                                wave_number=wave_number)
    error = pressure - exact
    weight = volumes / np.sum(volumes)
    return RefinementLevel(
        n, dr, dt, steps,
        errors={"l1": np.sum(weight * np.abs(error)),
                "l2": np.sqrt(np.sum(weight * error * error)),
                "linf": np.max(np.abs(error))},
        invariants={
            "storage_boundary_balance": abs(
                float(volumes @ pressure) - initial_storage - integrated_boundary_flux),
        },
    )


def run_radial_pressure_benchmark(resolutions=(20, 40, 80, 160)):
    levels = tuple(_radial_level(n) for n in resolutions)
    orders = {norm: observed_order(levels, norm) for norm in ("l1", "l2", "linf")}
    max_balance = max(level.invariants["storage_boundary_balance"] for level in levels)
    acceptance = {"minimum_l2_order": 1.85, "maximum_finest_l2_error": 5e-6,
                  "maximum_storage_balance": 1e-10}
    passed = (orders["l2"] >= acceptance["minimum_l2_order"]
              and levels[-1].errors["l2"] <= acceptance["maximum_finest_l2_error"]
              and max_balance <= acceptance["maximum_storage_balance"])
    return AnalyticCaseResult(RADIAL_SPEC, levels, orders, bool(passed), acceptance)


def buckley_leverett_exact(x_d, injected_pore_volumes):
    """Welge solution for equal viscosity and quadratic relative permeability."""
    x = np.asarray(x_d, float)
    time = float(injected_pore_volumes)
    saturation = np.linspace(0.0, 1.0, 200001)
    denominator = saturation**2 + (1.0 - saturation)**2
    fractional_flow = saturation**2 / denominator
    derivative = 2.0 * saturation * (1.0 - saturation) / denominator**2
    shock_saturation = np.sqrt(0.5)
    shock_flow = shock_saturation**2 / (
        shock_saturation**2 + (1.0 - shock_saturation)**2)
    shock_speed = shock_flow / shock_saturation
    profile = np.zeros_like(x)
    behind = x <= shock_speed * time
    branch = saturation >= shock_saturation
    profile[behind] = np.interp(
        x[behind] / time, derivative[branch][::-1], saturation[branch][::-1]
    )
    return profile, shock_speed * time, shock_saturation


def _buckley_leverett_level(nx: int, *, injected_pore_volumes=0.3):
    n = int(nx)
    grid = compute_geometry(cart_grid([n, 1, 1], [100.0, 1.0, 1.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.2)
    fluid = init_simple_fluid(mu=[1e-3, 1e-3], rho=[1000.0, 1000.0], n=[2, 2])
    half_trans = compute_trans(grid, rock)
    rate = grid.cells.volumes[0] * 0.2 * n / (100 * DAY)
    source = add_source(None, np.array([0]), rate, sat=np.array([[1.0, 0.0]]))
    source = add_source(source, np.array([n - 1]), -rate,
                        sat=np.array([[0.0, 1.0]]))
    state = init_res_sol(grid, 0.0, s0=np.array([[0.0, 1.0]]))
    state = incomp_tpfa(state, grid, half_trans, fluid, src=source)
    final_time = injected_pore_volumes * 100 * DAY
    state = explicit_transport(state, grid, final_time, rock, fluid, src=source)
    x_d = grid.cells.centroids[:, 0] / 100.0
    exact, shock_position, shock_saturation = buckley_leverett_exact(
        x_d, injected_pore_volumes)
    error = state.s[:, 0] - exact
    above = np.flatnonzero(state.s[:, 0] > 0.5 * shock_saturation)
    numerical_front = float(x_d[above[-1]]) if above.size else 0.0
    water_volume = float(pore_volume(grid, rock) @ state.s[:, 0])
    injected_volume = rate * final_time
    return RefinementLevel(
        n, 1.0 / n, None, None,
        errors={"l1": np.mean(np.abs(error)),
                "l2": np.sqrt(np.mean(error * error)),
                "linf": np.max(np.abs(error)),
                "shock_position": abs(numerical_front - shock_position)},
        invariants={"water_volume_balance_m3": abs(water_volume - injected_volume),
                    "minimum_saturation": np.min(state.s[:, 0]),
                    "maximum_saturation": np.max(state.s[:, 0])},
    )


def run_buckley_leverett_benchmark(resolutions=(50, 100, 200, 400)):
    levels = tuple(_buckley_leverett_level(n) for n in resolutions)
    orders = {norm: observed_order(levels, norm) for norm in ("l1", "l2")}
    max_balance = max(level.invariants["water_volume_balance_m3"] for level in levels)
    bounded = all(level.invariants["minimum_saturation"] >= -1e-12 and
                  level.invariants["maximum_saturation"] <= 1 + 1e-12
                  for level in levels)
    acceptance = {"minimum_l1_order": 0.7, "maximum_finest_l1_error": 6e-3,
                  "maximum_finest_shock_error": 5e-3,
                  "maximum_water_balance_m3": 1e-12}
    passed = (orders["l1"] >= acceptance["minimum_l1_order"]
              and levels[-1].errors["l1"] <= acceptance["maximum_finest_l1_error"]
              and levels[-1].errors["shock_position"] <= acceptance["maximum_finest_shock_error"]
              and max_balance <= acceptance["maximum_water_balance_m3"] and bounded)
    return AnalyticCaseResult(BUCKLEY_LEVERETT_SPEC, levels, orders,
                              bool(passed), acceptance)


def _terzaghi_level(layers: int):
    n = int(layers)
    times = np.array([0.02, 0.1, 0.4, 1.0])
    substeps = max(20, n // 2)
    result = run_terzaghi(times, layers=n, substeps=substeps)
    scale = float(result["maximum_pressure_pa"])
    pressure_error = ((result["pressure_profiles_pa"]
                       - result["analytic_pressure_profiles_pa"]) / scale)
    pressure_ratio = result["pressure_profiles_pa"] / scale
    settlement = np.asarray(result["top_displacement_m"], float)
    return RefinementLevel(
        n, 1.0 / n, float(np.max(np.diff(np.r_[0.0, times])) / substeps),
        int(substeps * times.size),
        errors={"pressure_l2": np.sqrt(np.mean(pressure_error**2)),
                "pressure_linf": np.max(np.abs(pressure_error)),
                "settlement_linf_relative": np.max(result["settlement_relative_errors"])},
        invariants={"minimum_pressure_ratio": np.min(pressure_ratio),
                    "maximum_pressure_ratio": np.max(pressure_ratio),
                    "minimum_settlement_increment": np.min(np.diff(settlement))},
    )


def run_terzaghi_benchmark(resolutions=(40, 80, 160, 320)):
    levels = tuple(_terzaghi_level(n) for n in resolutions)
    orders = {name: observed_order(levels, name) for name in
              ("pressure_l2", "pressure_linf", "settlement_linf_relative")}
    bounded = all(level.invariants["minimum_pressure_ratio"] >= -1e-12 and
                  level.invariants["maximum_pressure_ratio"] <= 1 + 1e-12
                  for level in levels)
    monotone = all(level.invariants["minimum_settlement_increment"] > 0
                   for level in levels)
    acceptance = {"minimum_pressure_order": 0.85,
                  "maximum_finest_pressure_linf": 1e-3,
                  "maximum_finest_settlement_relative_error": 5e-4}
    passed = (orders["pressure_linf"] >= acceptance["minimum_pressure_order"]
              and levels[-1].errors["pressure_linf"]
              <= acceptance["maximum_finest_pressure_linf"]
              and levels[-1].errors["settlement_linf_relative"]
              <= acceptance["maximum_finest_settlement_relative_error"]
              and bounded and monotone)
    return AnalyticCaseResult(TERZAGHI_SPEC, levels, orders, bool(passed), acceptance)


def _mandel_level(nx: int):
    n = int(nx)
    times = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    substeps = max(8, n // 2)
    result = run_mandel(times, nx=n, nz=4, substeps=substeps)
    profiles = np.asarray(result["pressure_profiles"], float)
    exact = np.asarray(result["analytic_profiles"], float)
    error = profiles - exact
    centre = int(np.argmin(result["x_over_a"]))
    centre_peak = float(np.max(profiles[:, centre]))
    case = result["case"]
    coords = np.asarray(case.G.cells.centroids[:, 0], float)
    z_spread = 0.0
    for x in np.unique(np.round(coords, 10)):
        selected = np.isclose(coords, x)
        values = profiles[:, selected]
        z_spread = max(z_spread, float(np.max(np.ptp(values, axis=1))))
    final_force_error = abs(case.plate_force(result["states"][-1]) - case.load) / case.load
    return RefinementLevel(
        n, 1.0 / n, float(np.max(np.diff(np.r_[0.0, times])) / substeps),
        int(substeps * times.size),
        errors={"pressure_l2": np.sqrt(np.mean(error**2)),
                "pressure_linf": np.max(np.abs(error))},
        invariants={"centre_peak_pressure_ratio": centre_peak,
                    "maximum_column_pressure_spread": z_spread,
                    "final_plate_force_relative_error": final_force_error},
    )


def run_mandel_benchmark(resolutions=(12, 20, 30, 40)):
    levels = tuple(_mandel_level(n) for n in resolutions)
    orders = {name: observed_order(levels, name)
              for name in ("pressure_l2", "pressure_linf")}
    peak = max(level.invariants["centre_peak_pressure_ratio"] for level in levels)
    max_z_spread = max(level.invariants["maximum_column_pressure_spread"]
                       for level in levels)
    max_force_error = max(level.invariants["final_plate_force_relative_error"]
                          for level in levels)
    acceptance = {"minimum_pressure_order": 0.8,
                  "maximum_finest_pressure_linf": 0.01,
                  "minimum_overshoot_ratio": 1.05,
                  "maximum_column_pressure_spread": 1e-6,
                  "maximum_plate_force_relative_error": 1e-6}
    passed = (orders["pressure_linf"] >= acceptance["minimum_pressure_order"]
              and levels[-1].errors["pressure_linf"]
              <= acceptance["maximum_finest_pressure_linf"]
              and peak >= acceptance["minimum_overshoot_ratio"]
              and max_z_spread <= acceptance["maximum_column_pressure_spread"]
              and max_force_error <= acceptance["maximum_plate_force_relative_error"])
    return AnalyticCaseResult(MANDEL_SPEC, levels, orders, bool(passed), acceptance)


def _elasticity_patch_level(resolution: int) -> RefinementLevel:
    """Exercise the actual VEM solve against independent affine fields."""
    n = int(resolution)
    dims = [n, n] if n != 3 else [3, 3, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = create_augmented_grid(compute_geometry(cart_grid(dims)))
    dim = grid.griddim
    if dim == 2:
        rng = np.random.default_rng(731)
        coords = grid.nodes.coords
        interior = np.all((coords > 1e-12) & (coords < n - 1e-12), axis=1)
        coords[interior] += 0.2 * (rng.random((int(interior.sum()), 2)) - 0.5)
        grid = create_augmented_grid(compute_geometry(grid))
    C = (enu2c_2d if dim == 2 else enu2c)(
        np.full(grid.cells.num, 1.0e6), np.full(grid.cells.num, 0.25))
    boundary_faces = np.flatnonzero(np.any(grid.faces.neighbors < 0, axis=1))
    boundary_nodes = np.unique(np.concatenate(
        [grid.face_nodes(face) for face in boundary_faces]))
    fields, names = linear_displacement(dim)
    maximum_error = 0.0
    maximum_rigid_energy = 0.0
    stiffness = vem_assemble(grid, C)
    stiffness_scale = max(float(abs(stiffness).max()), 1.0)
    for field, name in zip(fields, names):
        values = field(grid.nodes.coords[boundary_nodes, :dim])
        bc = ElasticBC(
            disp_nodes=boundary_nodes,
            disp_mask=np.ones((boundary_nodes.size, dim), dtype=bool),
            disp_vals=values,
            force_faces=np.zeros(0, dtype=np.int64),
            force=np.zeros((0, dim)),
        )
        displacement, _ = vem_lin_elast(grid, C, bc, lambda x: np.zeros_like(x))
        exact = field(grid.nodes.coords[:, :dim])
        scale = max(float(np.max(np.abs(exact))), 1.0)
        maximum_error = max(maximum_error,
                            float(np.max(np.abs(displacement - exact))) / scale)
        if name.startswith("cons_disp") or name.startswith("rot_"):
            vector = exact.ravel()
            normalized_energy = abs(float(vector @ (stiffness @ vector))) / (
                stiffness_scale * max(float(vector @ vector), 1.0))
            maximum_rigid_energy = max(maximum_rigid_energy, normalized_energy)
    asymmetry = float(abs(stiffness - stiffness.T).max()) / stiffness_scale
    return RefinementLevel(
        grid.cells.num, 1.0 / n, None, None,
        errors={"maximum_relative_displacement": maximum_error},
        invariants={"maximum_normalized_rigid_energy": maximum_rigid_energy,
                    "relative_stiffness_asymmetry": asymmetry,
                    "fields_exercised": len(fields)},
    )


def run_elasticity_patch_benchmark(resolutions=(2, 4, 3)) -> AnalyticCaseResult:
    levels = tuple(_elasticity_patch_level(n) for n in resolutions)
    acceptance = {"maximum_relative_displacement": 1e-9,
                  "maximum_normalized_rigid_energy": 1e-8,
                  "maximum_relative_stiffness_asymmetry": 1e-12}
    passed = all(
        level.errors["maximum_relative_displacement"]
        <= acceptance["maximum_relative_displacement"]
        and level.invariants["maximum_normalized_rigid_energy"]
        <= acceptance["maximum_normalized_rigid_energy"]
        and level.invariants["relative_stiffness_asymmetry"]
        <= acceptance["maximum_relative_stiffness_asymmetry"]
        for level in levels)
    return AnalyticCaseResult(ELASTICITY_PATCH_SPEC, levels, {}, bool(passed), acceptance)


def _flash_equilibrium_level(state_count: int) -> RefinementLevel:
    count = int(state_count)
    mixture = simple_mixture()
    pressure = np.linspace(1.0e6, 2.0e6, count)
    temperature = np.linspace(250.0, 360.0, count)
    phase = np.linspace(0.0, 1.0, count)
    feed = np.column_stack((0.15 + 0.10 * phase,
                            0.25 + 0.10 * (1.0 - phase),
                            0.60 - 0.10 * phase))
    feed /= feed.sum(axis=1, keepdims=True)
    result = flash(mixture, pressure, temperature, feed)
    liquid_fugacity, vapour_fugacity = fugacity_lv(
        mixture, pressure, temperature, result.x, result.y)
    active = np.asarray(result.two_phase, bool)
    reconstructed = ((1.0 - result.V[:, None]) * result.x
                     + result.V[:, None] * result.y)
    log_fugacity_error = np.max(np.abs(np.log(
        np.maximum(liquid_fugacity[active], 1e-300)
        / np.maximum(vapour_fugacity[active], 1e-300))))
    return RefinementLevel(
        count, 1.0 / count, None, int(result.iterations),
        errors={"material_balance": np.max(np.abs(reconstructed - feed)),
                "fugacity_log_ratio": log_fugacity_error,
                "liquid_closure": np.max(np.abs(result.x.sum(axis=1) - 1.0)),
                "vapour_closure": np.max(np.abs(result.y.sum(axis=1) - 1.0))},
        invariants={"minimum_vapour_fraction": np.min(result.V),
                    "maximum_vapour_fraction": np.max(result.V),
                    "two_phase_fraction": np.mean(active),
                    "minimum_composition": min(np.min(result.x), np.min(result.y)),
                    "finite_outputs": float(np.all(np.isfinite(np.column_stack(
                        (result.V, result.x, result.y, result.Z_L, result.Z_V)))))},
    )


def run_flash_equilibrium_benchmark(state_counts=(8, 32, 128)) -> AnalyticCaseResult:
    levels = tuple(_flash_equilibrium_level(n) for n in state_counts)
    acceptance = {"maximum_material_balance": 1e-9,
                  "maximum_log_fugacity_ratio": 1e-8,
                  "maximum_composition_closure": 1e-12,
                  "minimum_two_phase_fraction": 1.0}
    passed = all(
        level.errors["material_balance"] <= acceptance["maximum_material_balance"]
        and level.errors["fugacity_log_ratio"] <= acceptance["maximum_log_fugacity_ratio"]
        and max(level.errors["liquid_closure"], level.errors["vapour_closure"])
        <= acceptance["maximum_composition_closure"]
        and level.invariants["two_phase_fraction"]
        >= acceptance["minimum_two_phase_fraction"]
        and level.invariants["minimum_composition"] >= 0.0
        and level.invariants["finite_outputs"] == 1.0
        for level in levels)
    return AnalyticCaseResult(FLASH_EQUILIBRIUM_SPEC, levels, {}, bool(passed), acceptance)


def _accumulation_flux_level(ncells: int) -> RefinementLevel:
    n = int(ncells)
    grid = compute_geometry(cart_grid([n, 1, 1], [float(n), 1.0, 1.0]))
    rock = make_rock(grid, 2.0e-13, 0.2)
    operators = setup_operators_tpfa(grid, rock)
    p0 = 1.0e7 + 2.0e5 * np.sin(np.linspace(0.0, np.pi, n))
    pressure = p0 + 1.0e4 * np.cos(np.linspace(0.0, 2.0 * np.pi, n))
    compressibility, dt = 1.5e-8, 7200.0
    storage = operators.pv * compressibility / dt
    numerical_flux = -operators.T * operators.grad(pressure)
    manufactured_source = storage * (pressure - p0) + operators.div(numerical_flux)
    (pressure_ad,) = init_variables_adi(pressure)
    flux_ad = -operators.T * operators.grad(pressure_ad)
    residual_ad = storage * (pressure_ad - p0) + operators.div(flux_ad) \
        - manufactured_source
    analytic_jacobian = (sp.diags(storage, format="csr")
                         + operators.C.T @ sp.diags(operators.T) @ operators.C)
    jacobian_error = residual_ad.jac[0] - analytic_jacobian
    maximum_jacobian_error = (float(abs(jacobian_error).max())
                              if jacobian_error.nnz else 0.0)
    global_balance = abs(float(np.sum(manufactured_source)
                               - np.sum(storage * (pressure - p0))))
    return RefinementLevel(
        n, 1.0 / n, dt, 1,
        errors={"maximum_residual": np.max(np.abs(residual_ad.val)),
                "maximum_jacobian_error": maximum_jacobian_error},
        invariants={"global_source_storage_balance": global_balance,
                    "jacobian_nnz": residual_ad.jac[0].nnz,
                    "expected_jacobian_nnz": analytic_jacobian.nnz,
                    "maximum_face_flux": np.max(np.abs(numerical_flux))},
    )


def run_accumulation_flux_benchmark(resolutions=(3, 9, 33)) -> AnalyticCaseResult:
    levels = tuple(_accumulation_flux_level(n) for n in resolutions)
    acceptance = {"maximum_residual": 1e-20,
                  "maximum_jacobian_error": 1e-24,
                  "maximum_global_balance": 1e-20}
    passed = all(
        level.errors["maximum_residual"] <= acceptance["maximum_residual"]
        and level.errors["maximum_jacobian_error"]
        <= acceptance["maximum_jacobian_error"]
        and level.invariants["global_source_storage_balance"]
        <= acceptance["maximum_global_balance"]
        and level.invariants["jacobian_nnz"] == level.invariants["expected_jacobian_nnz"]
        for level in levels)
    return AnalyticCaseResult(ACCUMULATION_FLUX_SPEC, levels, {}, bool(passed), acceptance)


def foundation_analytic_registry() -> AnalyticRegistry:
    registry = AnalyticRegistry()
    registry.register(HYDROSTATIC_SPEC, run_hydrostatic_benchmark)
    registry.register(DIFFUSION_SPEC, run_diffusion_benchmark)
    registry.register(RADIAL_SPEC, run_radial_pressure_benchmark)
    registry.register(BUCKLEY_LEVERETT_SPEC, run_buckley_leverett_benchmark)
    registry.register(TERZAGHI_SPEC, run_terzaghi_benchmark)
    registry.register(MANDEL_SPEC, run_mandel_benchmark)
    registry.register(ELASTICITY_PATCH_SPEC, run_elasticity_patch_benchmark)
    registry.register(FLASH_EQUILIBRIUM_SPEC, run_flash_equilibrium_benchmark)
    registry.register(ACCUMULATION_FLUX_SPEC, run_accumulation_flux_benchmark)
    return registry


__all__ = ["ACCUMULATION_FLUX_SPEC", "BUCKLEY_LEVERETT_SPEC", "DIFFUSION_SPEC",
           "ELASTICITY_PATCH_SPEC", "FLASH_EQUILIBRIUM_SPEC", "HYDROSTATIC_SPEC",
           "MANDEL_SPEC", "RADIAL_SPEC", "TERZAGHI_SPEC",
           "buckley_leverett_exact", "diffusion_exact",
           "foundation_analytic_registry", "run_accumulation_flux_benchmark",
           "run_diffusion_benchmark", "run_elasticity_patch_benchmark",
           "run_flash_equilibrium_benchmark",
           "radial_bessel_exact", "run_buckley_leverett_benchmark",
           "run_hydrostatic_benchmark", "run_mandel_benchmark",
           "run_radial_pressure_benchmark", "run_terzaghi_benchmark"]
