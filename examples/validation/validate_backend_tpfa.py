"""Validate the experimental backend TPFA solve against PyMRST's CPU solver."""

from __future__ import annotations

import numpy as np

from pymrst.backends import get_backend
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp import incomp_tpfa_ad as incomp_tpfa, incomp_tpfa_backend_pressure
from pymrst.modules.incomp.fluid import init_single_fluid
from pymrst.params.gravity_const import gravity_vector
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc, add_well
from pymrst.solvers.state import init_res_sol
from pymrst.solvers.trans import compute_trans
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _bc_case():
    g = compute_geometry(cart_grid([10, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
    fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
    t_half = compute_trans(g, rock)
    west = np.flatnonzero(
        np.any(g.faces.neighbors == -1, axis=1) & (g.faces.centroids[:, 0] < 1e-10)
    )
    east = np.flatnonzero(
        np.any(g.faces.neighbors == -1, axis=1)
        & (g.faces.centroids[:, 0] > 100.0 - 1e-10)
    )
    bc = add_bc(None, west, "pressure", 200 * BARSA)
    bc = add_bc(bc, east, "pressure", 100 * BARSA)
    return g, t_half, fluid, bc


def _well_case():
    g = compute_geometry(cart_grid([8, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
    fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
    t_half = compute_trans(g, rock)
    wells = add_well(
        None,
        g,
        rock,
        np.array([0]),
        well_type="rate",
        val=1e-3,
        compi=np.array([1.0]),
    )
    wells = add_well(
        wells,
        g,
        rock,
        np.array([g.cells.num - 1]),
        well_type="bhp",
        val=100 * BARSA,
        compi=np.array([1.0]),
    )
    return g, t_half, fluid, wells


def _gravity_bc_case():
    g = compute_geometry(cart_grid([1, 1, 8], [10.0, 10.0, 80.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
    fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
    t_half = compute_trans(g, rock)
    top = np.flatnonzero(
        np.any(g.faces.neighbors == -1, axis=1) & (g.faces.centroids[:, 2] < 1e-10)
    )
    bottom = np.flatnonzero(
        np.any(g.faces.neighbors == -1, axis=1)
        & (g.faces.centroids[:, 2] > 80.0 - 1e-10)
    )
    bc = add_bc(None, top, "pressure", 200 * BARSA)
    bc = add_bc(bc, bottom, "pressure", 250 * BARSA)
    return g, t_half, fluid, bc, gravity_vector(g.griddim)


def _gravity_well_case():
    g = compute_geometry(cart_grid([1, 1, 8], [10.0, 10.0, 80.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
    fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
    t_half = compute_trans(g, rock)
    wells = add_well(
        None,
        g,
        rock,
        np.array([0, 1, 2]),
        well_type="rate",
        val=5e-4,
        compi=np.array([1.0]),
        sign=1,
        gravity=gravity_vector(g.griddim),
    )
    wells = add_well(
        wells,
        g,
        rock,
        np.array([5, 6, 7]),
        well_type="bhp",
        val=180 * BARSA,
        compi=np.array([1.0]),
        sign=-1,
        gravity=gravity_vector(g.griddim),
    )
    return g, t_half, fluid, wells, gravity_vector(g.griddim)


def _compare_backend(
    name: str,
    backend: str,
    device: str | None = None,
    linear_solver: str = "auto",
) -> int:
    g, t_half, fluid, bc = _bc_case()
    ref = incomp_tpfa(init_res_sol(g, 0.0), g, t_half, fluid, bc=bc)
    p, flux = incomp_tpfa_backend_pressure(
        g,
        t_half,
        totmob=1.0 / fluid.mu[0],
        bc=bc,
        backend=backend,
        device=device,
        linear_solver=linear_solver,
    )
    be = get_backend(backend, device=device, linear_solver=linear_solver)
    p = be.to_numpy(p)
    flux = be.to_numpy(flux)
    pdiff = float(np.max(np.abs(p - ref.pressure)))
    fdiff = float(np.max(np.abs(flux - ref.flux)))
    ok_bc = pdiff < 1e-7 and fdiff < 1e-12
    print(f"{'ok  ' if ok_bc else 'FAIL'} {name} BC: pressure {pdiff:.3e}, flux {fdiff:.3e}")

    g, t_half, fluid, wells = _well_case()
    ref = incomp_tpfa(init_res_sol(g, 0.0), g, t_half, fluid, wells=wells)
    p, flux, well_data = incomp_tpfa_backend_pressure(
        g,
        t_half,
        totmob=1.0 / fluid.mu[0],
        wells=wells,
        backend=backend,
        device=device,
        linear_solver=linear_solver,
        return_wells=True,
    )
    p = be.to_numpy(p)
    flux = be.to_numpy(flux)
    bhp = be.to_numpy(well_data["bhp"])
    wflux = [be.to_numpy(v) for v in well_data["flux"]]
    pdiff = float(np.max(np.abs(p - ref.pressure)))
    fdiff = float(np.max(np.abs(flux - ref.flux)))
    bdiff = float(np.max(np.abs(bhp - np.array([ws.pressure for ws in ref.wellSol]))))
    wdiff = float(
        max(np.max(np.abs(a - b.flux)) for a, b in zip(wflux, ref.wellSol))
    )
    ok_well = pdiff < 1e-7 and fdiff < 1e-12 and bdiff < 1e-7 and wdiff < 1e-12
    print(
        f"{'ok  ' if ok_well else 'FAIL'} {name} wells: pressure {pdiff:.3e}, "
        f"flux {fdiff:.3e}, bhp {bdiff:.3e}, perf {wdiff:.3e}"
    )
    g, t_half, fluid, bc, gravity = _gravity_bc_case()
    ref = incomp_tpfa(init_res_sol(g, 0.0), g, t_half, fluid, bc=bc, gravity=gravity)
    p, flux = incomp_tpfa_backend_pressure(
        g,
        t_half,
        totmob=1.0 / fluid.mu[0],
        omega=fluid.rho[0],
        bc=bc,
        gravity=gravity,
        backend=backend,
        device=device,
        linear_solver=linear_solver,
    )
    p = be.to_numpy(p)
    flux = be.to_numpy(flux)
    pdiff = float(np.max(np.abs(p - ref.pressure)))
    fdiff = float(np.max(np.abs(flux - ref.flux)))
    ok_gbc = pdiff < 1e-7 and fdiff < 1e-12
    print(
        f"{'ok  ' if ok_gbc else 'FAIL'} {name} gravity BC: "
        f"pressure {pdiff:.3e}, flux {fdiff:.3e}"
    )

    g, t_half, fluid, wells, gravity = _gravity_well_case()
    ref = incomp_tpfa(init_res_sol(g, 0.0), g, t_half, fluid, wells=wells, gravity=gravity)
    p, flux, well_data = incomp_tpfa_backend_pressure(
        g,
        t_half,
        totmob=1.0 / fluid.mu[0],
        omega=fluid.rho[0],
        wells=wells,
        gravity=gravity,
        backend=backend,
        device=device,
        linear_solver=linear_solver,
        return_wells=True,
    )
    p = be.to_numpy(p)
    flux = be.to_numpy(flux)
    bhp = be.to_numpy(well_data["bhp"])
    wflux = [be.to_numpy(v) for v in well_data["flux"]]
    pdiff = float(np.max(np.abs(p - ref.pressure)))
    fdiff = float(np.max(np.abs(flux - ref.flux)))
    bdiff = float(np.max(np.abs(bhp - np.array([ws.pressure for ws in ref.wellSol]))))
    wdiff = float(max(np.max(np.abs(a - b.flux)) for a, b in zip(wflux, ref.wellSol)))
    ok_gwell = pdiff < 1e-7 and fdiff < 1e-12 and bdiff < 1e-7 and wdiff < 1e-12
    print(
        f"{'ok  ' if ok_gwell else 'FAIL'} {name} gravity wells: "
        f"pressure {pdiff:.3e}, flux {fdiff:.3e}, bhp {bdiff:.3e}, perf {wdiff:.3e}"
    )

    return 0 if ok_bc and ok_well and ok_gbc and ok_gwell else 1


def main() -> int:
    failed = _compare_backend("numpy backend", "numpy")
    failed += _compare_backend("numpy backend/scipy solver", "numpy", linear_solver="scipy")
    try:
        import cupy  # noqa: F401
    except ImportError:
        print("skip cupy sparse solver: cupy is not installed")
    else:
        failed += _compare_backend("numpy backend/cupy sparse solver", "numpy", linear_solver="cupy")
    try:
        import torch
    except ImportError:
        print("skip torch backend: torch is not installed")
    else:
        failed += _compare_backend("torch CPU backend", "torch", device="cpu")
        if torch.cuda.is_available():
            failed += _compare_backend("torch CUDA backend", "torch", device="cuda")
        else:
            print("skip torch CUDA backend: CUDA is not available")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
