"""MRST ``run2DCase.m`` compatibility surface for 2D VEM mechanics.

The original MRST example is a family of 2D poromechanics cases. This module
provides the same landing point for PyMRST's implemented subset: plane-strain
water mechanics and MRST-shaped fully coupled oil-water mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import blackoil_status
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_mechanics.mech_blackoil import MechBlackOilModel
from pymrst.modules.ad_mechanics.mechanics import MechanicModel
from pymrst.modules.ad_mechanics.poromech import (
    FixedStressBlackOilWellModel,
    FixedStressOilWaterWellModel,
    FixedStressSplitModel,
    FixedStressWaterWellModel,
    MechWaterModel,
    MechWaterWellModel,
)
from pymrst.modules.ad_props.blackoil_fluid import init_deck_blackoil_fluid
from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c_2d
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, GIGA, POISE


class TwoDVEMNotPorted(NotImplementedError):
    """Backward-compatible name for older callers.

    The 2D VEM path and fully coupled oil-water wrapper are implemented.
    Fixed-stress multiphase and black-oil variants still use this exception.
    """


@dataclass
class TwoDCaseResult:
    """Result bundle returned by :func:`run_2d_case`."""

    G: object
    rock: object
    mech: MechanicModel
    model: object
    states: list[State]
    reports: list[dict]
    diagnostics: list[dict]
    dts: np.ndarray


def _build_case(
    nx: int,
    ny: int,
    physdims: tuple[float, float],
    E: float,
    nu: float,
    alpha: float,
    perm: float,
    poro: float,
    p_ref: float,
):
    g = create_augmented_grid(compute_geometry(cart_grid([nx, ny], physdims)))
    rock = make_rock(g, perm, poro)
    C = enu2c_2d(np.full(g.cells.num, E), np.full(g.cells.num, nu))

    pts = g.nodes.coords[:, :2]
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    bottom = np.flatnonzero(pts[:, 1] < ymin + 1e-9)
    disp_nodes = bottom
    disp_mask = np.ones((bottom.size, 2), dtype=bool)

    bd = np.any(g.faces.neighbors == -1, axis=1)
    fc = g.faces.centroids[:, :2]
    side_faces = np.flatnonzero(
        bd & (
            (fc[:, 0] < xmin + 1e-9)
            | (fc[:, 0] > xmax - 1e-9)
            | (fc[:, 1] > ymax - 1e-9)
        )
    )
    signcoef = (
        (g.faces.neighbors[side_faces, 0] == -1).astype(float)
        - (g.faces.neighbors[side_faces, 1] == -1).astype(float)
    )
    normals = g.faces.normals[side_faces, :2] \
        * (signcoef / g.faces.areas[side_faces])[:, None]
    force = normals * p_ref

    el_bc = ElasticBC(
        disp_nodes=disp_nodes,
        disp_mask=disp_mask,
        disp_vals=np.zeros((disp_nodes.size, 2)),
        force_faces=side_faces,
        force=force,
    )
    mech = MechanicModel(g, C, alpha=alpha, el_bc=el_bc)
    return g, rock, mech


def _center_cell(nx: int, ny: int) -> int:
    """MRST ``sub2ind(cartDims, ceil(cartDims/2))``, converted to 0-based."""
    ix, iy = int(np.ceil(nx / 2.0)) - 1, int(np.ceil(ny / 2.0)) - 1
    return ix + nx * iy


def _oil_water_state(g, model, p_ref: float, rate: float) -> State:
    nw = len(model.wells)
    state = State(
        pressure=np.full(g.cells.num, p_ref),
        sW=np.zeros(g.cells.num),
        sO=np.ones(g.cells.num),
        qWs=np.zeros(nw),
        qOs=np.zeros(nw),
        bhp=np.full(nw, p_ref),
    )
    state["qWs"][0] = rate
    return model.seed_state(state)


def _spe1_deck_path() -> Path:
    return Path(__file__).resolve().parents[5] / (
        "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
    )


def _blackoil_fluid_from_spe1():
    deck = read_spe1_deck(_spe1_deck_path())
    fluid = init_deck_blackoil_fluid(deck)
    # MRST's run2DCase replaces SPE1's tabulated relperms by smoother
    # quadratic curves for this mechanics example.
    fluid.kr_w = lambda s: s**2
    fluid.kr_g = lambda s: s**2
    fluid.kr_ow = lambda s: s**2
    fluid.kr_og = lambda s: s**2
    return deck, fluid


def _blackoil_state(g, model, fluid, p_ref: float, rate: float) -> State:
    nw = len(model.wells)
    sw = np.zeros(g.cells.num)
    so = np.ones(g.cells.num)
    sg = np.zeros(g.cells.num)
    state = State(
        pressure=np.full(g.cells.num, p_ref),
        sW=sw,
        sO=so,
        sG=sg,
        rs=0.5 * fluid.rs_sat(np.full(g.cells.num, p_ref)),
        status=blackoil_status(sw, so, sg, disgas=True),
        qWs=np.zeros(nw),
        qOs=np.zeros(nw),
        qGs=np.zeros(nw),
        bhp=np.full(nw, p_ref),
    )
    state["qGs"][0] = rate
    return model.seed_state(state)


def run_2d_case(
    fluid_model: str = "water",
    method: str = "fully_coupled",
    nx: int = 100,
    ny: int = 10,
    physdims: tuple[float, float] = (100.0, 10.0),
    nsteps: int = 3,
    dt: float | np.ndarray = 5.0 * DAY,
    E: float = 1.0 * GIGA,
    nu: float = 0.3,
    alpha: float = 1.0,
    perm: float = 1.0 * DARCY,
    poro: float = 0.3,
    mu: float = 1.0 * CENTI * POISE,
    rho: float = 1000.0,
    compressibility: float = 1e-10,
    rock_compressibility: float = 4e-10,
    p_ref: float = 270.0 * BARSA,
    mrst_oilwater_bug: bool = True,
) -> TwoDCaseResult:
    """Run a 2D plane-strain poromechanics case.

    Parameters mirror MRST's example surface where practical. The MRST-shaped
    well-driven branch is ``fluid_model="oil water"`` with fully coupled
    mechanics. The water branch remains a boundary-driven smoke test.
    """
    fluid_key = fluid_model.lower().replace("-", " ")
    deck = None
    if fluid_key in {"blackoil", "black oil"}:
        deck, blackoil_fluid = _blackoil_fluid_from_spe1()
        p_ref = float(deck["pvtw"]["p_ref"])

    g, rock, mech = _build_case(
        nx, ny, physdims, E, nu, alpha, perm, poro, p_ref
    )

    key = method.lower().replace("-", "_").replace(" ", "_")
    if fluid_key in {"oil water", "oil_water"} and key in {
        "fully_coupled",
        "full",
        "monolithic",
    }:
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 10.0]) * CENTI * POISE,
            rho=np.array([1000.0, 700.0]),
            n=np.array([1.0, 1.0]),
            c=np.array([1e-10, 1e-10]),
            p_ref=p_ref,
        )
        rate = 1e4 / DAY
        wells = add_well(
            None,
            g,
            rock,
            np.array([_center_cell(nx, ny)]),
            well_type="rate",
            val=rate,
            compi=np.array([1.0, 0.0]),
            sign=1,
            ref_depth=float(np.min(g.cells.centroids[:, g.griddim - 1])),
        )
        model = MechBlackOilModel(
            g, rock, fluid, wells, mech,
            mrst_oilwater_bug=mrst_oilwater_bug,
        )
        states = [_oil_water_state(g, model, p_ref, rate)]
        solver = NonLinearSolver(max_iterations=100)
    elif fluid_key in {"water", "single_phase", "single phase"}:
        def b(p):
            from pymrst.utils.adi import ad_exp
            return ad_exp(compressibility * (p - p_ref))

        def pv_mult(p):
            return 1.0 + rock_compressibility * (p - p_ref)

        if key in {"fully_coupled", "full", "monolithic"}:
            rate = 1e-3 / DAY
            wells = add_well(
                None,
                g,
                rock,
                np.array([_center_cell(nx, ny)]),
                well_type="rate",
                val=rate,
                compi=np.array([1.0]),
                sign=1,
                ref_depth=float(np.min(g.cells.centroids[:, g.griddim - 1])),
            )
            model = MechWaterWellModel(
                g, rock, mech, wells, b, mu, rho,
                pv_mult=pv_mult,
                tol_well=1e-8,
            )
            states = [model.initial_state(np.full(g.cells.num, p_ref))]
            solver = NonLinearSolver(max_iterations=100)
        elif key in {
            "fixed_stress",
            "fixed_stress_splitting",
            "split",
            "splitting",
        }:
            rate = 1e-3 / DAY
            wells = add_well(
                None,
                g,
                rock,
                np.array([_center_cell(nx, ny)]),
                well_type="rate",
                val=rate,
                compi=np.array([1.0]),
                sign=1,
                ref_depth=float(np.min(g.cells.centroids[:, g.griddim - 1])),
            )
            model = FixedStressWaterWellModel(
                g, rock, mech, wells, b, mu, rho, pv_mult=pv_mult
            )
            states = [model.initial_state(np.full(g.cells.num, p_ref))]
            solver = None
        else:
            raise ValueError(f"unknown 2D poromechanics method: {method!r}")
    elif fluid_key in {"blackoil", "black oil"} and key in {
        "fully_coupled",
        "full",
        "monolithic",
    }:
        rate = 1e2 / DAY
        wells = add_well(
            None,
            g,
            rock,
            np.array([_center_cell(nx, ny)]),
            well_type="rate",
            val=rate,
            compi=np.array([0.0, 0.0, 1.0]),
            sign=1,
            ref_depth=float(np.min(g.cells.centroids[:, g.griddim - 1])),
        )
        model = MechBlackOilModel(
            g, rock, blackoil_fluid, wells, mech, disgas=True,
        )
        states = [_blackoil_state(g, model, blackoil_fluid, p_ref, rate)]
        solver = NonLinearSolver(max_iterations=100)
    elif fluid_key in {"blackoil", "black oil"} and key in {
        "fixed_stress",
        "fixed_stress_splitting",
        "split",
        "splitting",
    }:
        rate = 1e2 / DAY
        wells = add_well(
            None,
            g,
            rock,
            np.array([_center_cell(nx, ny)]),
            well_type="rate",
            val=rate,
            compi=np.array([0.0, 0.0, 1.0]),
            sign=1,
            ref_depth=float(np.min(g.cells.centroids[:, g.griddim - 1])),
        )
        model = FixedStressBlackOilWellModel(
            g, rock, blackoil_fluid, wells, mech,
        )
        states = [
            model.initial_state(
                np.full(g.cells.num, p_ref),
                sw0=0.0,
                sg0=0.0,
                rs=0.5 * blackoil_fluid.rs_sat(np.full(g.cells.num, p_ref)),
            )
        ]
        solver = None
    elif fluid_key in {"oil water", "oil_water"} and key in {
        "fixed_stress",
        "fixed_stress_splitting",
        "split",
        "splitting",
    }:
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 10.0]) * CENTI * POISE,
            rho=np.array([1000.0, 700.0]),
            n=np.array([1.0, 1.0]),
            c=np.array([1e-10, 1e-10]),
            p_ref=p_ref,
        )
        rate = 1e4 / DAY
        wells = add_well(
            None,
            g,
            rock,
            np.array([_center_cell(nx, ny)]),
            well_type="rate",
            val=rate,
            compi=np.array([1.0, 0.0]),
            sign=1,
            ref_depth=float(np.min(g.cells.centroids[:, g.griddim - 1])),
        )
        model = FixedStressOilWaterWellModel(
            g, rock, fluid, wells, mech,
            mrst_oilwater_bug=mrst_oilwater_bug,
        )
        states = [model.initial_state(np.full(g.cells.num, p_ref), sw0=0.0)]
        solver = None
    else:
        raise ValueError(f"unknown fluid_model: {fluid_model!r}")

    reports: list[dict] = []
    dts = np.asarray(dt, dtype=float).ravel()
    if dts.size == 1:
        dts = np.concatenate([[1.0 * DAY], np.full(nsteps - 1, float(dts[0]))])
    for step, dti in enumerate(dts, start=1):
        if solver is None:
            state, ok, its = model.step(states[-1], float(dti))
        else:
            state, rep = solver.solve_ministep(
                model, states[-1], states[-1].copy(), float(dti)
            )
            ok, its = rep.converged, rep.iterations
        states.append(state)
        reports.append({"step": step, "converged": ok, "iterations": its})
        if not ok:
            break

    diagnostics = [
        mech.diagnostics(st, "pressure") if "u" in st
        else mech.diagnostics(
            State(u=mech.full_displacement(st["xd"]), pressure=st["pressure"]),
            "pressure",
        )
        for st in states
    ]
    return TwoDCaseResult(
        G=g,
        rock=rock,
        mech=mech,
        model=model,
        states=states,
        reports=reports,
        diagnostics=diagnostics,
        dts=dts[:len(reports)],
    )


__all__ = ["run_2d_case", "TwoDCaseResult", "TwoDVEMNotPorted"]


if __name__ == "__main__":
    result = run_2d_case()
    last = result.states[-1]
    diag = result.diagnostics[-1]
    print(f"ok run_2d_case: {len(result.states) - 1} steps")
    print(f"  pressure range: {last['pressure'].min():.3e} .. "
          f"{last['pressure'].max():.3e} Pa")
    print(f"  max displacement: {diag['displacement_magnitude'].max():.3e} m")
