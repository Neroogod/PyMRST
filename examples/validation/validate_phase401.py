"""Phase 401: mimetic gravity and rate/BHP well branches."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.mimetic import computeMimeticIP, incomp_mimetic
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc, add_well


def _grid():
    g = cart_grid([7, 6], [7.0, 6.0])
    coordinates = np.asarray(g.nodes.coords, dtype=float)
    x, y = coordinates.T.copy()
    coordinates[:, 0] += 0.18*np.sin(np.pi*x/7.0)*np.sin(np.pi*y/6.0)
    coordinates[:, 1] += 0.11*np.sin(2*np.pi*x/7.0)*np.sin(np.pi*y/6.0)
    g.nodes.coords = coordinates
    return compute_geometry(g)


def _divergence(g, half_flux):
    cellno = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    return np.bincount(cellno, weights=half_flux, minlength=g.cells.num)


def main():
    g = _grid()
    rock = make_rock(
        g, np.tile([2.0, 0.25, 0.8], (g.cells.num, 1))*1e-12, 0.2
    )
    system = computeMimeticIP(g, rock)
    neighbors = np.asarray(g.faces.neighbors)
    boundary = np.flatnonzero(np.any(neighbors < 0, axis=1))

    rho = 1000.0
    gravity = np.array([0.0, -9.81])
    reference = 2.0e5
    boundary_pressure = (
        reference + rho*(np.asarray(g.faces.centroids)[boundary] @ gravity)
    )
    bc = add_bc(None, boundary, "pressure", boundary_pressure)
    hydro = incomp_mimetic(
        g, system, bc=bc, density=rho, gravity=gravity
    )
    exact_pressure = (
        reference + rho*(np.asarray(g.cells.centroids) @ gravity)
    )
    hydro_pressure_error = float(np.max(np.abs(
        hydro.pressure - exact_pressure
    )))
    hydro_flux_error = float(np.max(np.abs(hydro.flux)))
    hydro_conservation = float(np.max(np.abs(
        _divergence(g, hydro.half_flux)
    )))

    wells = add_well(
        [], g, rock, np.array([0]), well_type="rate", val=5e-5,
        radius=0.1, well_dir="z", name="I",
    )
    wells = add_well(
        wells, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
        val=1.5e5, radius=0.1, well_dir="z", name="P",
    )
    injector, producer = wells
    well_result = incomp_mimetic(
        g, system, wells=wells, mobility=2.5
    )
    injection = float(np.sum(well_result.well_sol[0].flux))
    production = float(np.sum(well_result.well_sol[1].flux))
    material_balance = abs(injection + production)
    bhp_error = abs(well_result.well_sol[1].pressure - producer.val)
    cell_well_balance = _divergence(g, well_result.half_flux)
    for well, solution in zip(wells, well_result.well_sol):
        np.add.at(cell_well_balance, well.cells, -solution.flux)
    well_conservation = float(np.max(np.abs(cell_well_balance)))
    ordering_ok = bool(
        well_result.pressure[0] > well_result.pressure[-1]
    )

    print("Phase 401: mimetic gravity and wells")
    print(f"  hydrostatic pressure patch: {hydro_pressure_error:.3e}")
    print(f"  hydrostatic maximum flux: {hydro_flux_error:.3e}")
    print(f"  hydrostatic conservation: {hydro_conservation:.3e}")
    print(f"  rate-well injection: {injection:.6e}")
    print(f"  BHP-well production: {production:.6e}")
    print(f"  well material balance: {material_balance:.3e}")
    print(f"  producer BHP error: {bhp_error:.3e}")
    print(f"  cell/well conservation: {well_conservation:.3e}")
    print(f"  pressure ordering: {ordering_ok}")

    failed = (
        hydro_pressure_error > 1e-8
        or hydro_flux_error > 1e-16
        or hydro_conservation > 1e-16
        or abs(injection - injector.val) > 1e-12
        or material_balance > 1e-10
        or bhp_error > 1e-12
        or well_conservation > 1e-10
        or not ordering_ok
    )
    print("\nPhase 401 passed." if not failed else "\nPhase 401 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
