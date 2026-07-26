"""Phase 388: MPFA sources, mixed boundary conditions, mobility and fluxes."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.mpfa import compute_multipoint_trans, incomp_mpfa
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc, add_source


def _grid():
    g = cart_grid([6, 4], [6.0, 4.0])
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y = c[:, 0].copy(), c[:, 1].copy()
    c[:, 0] += 0.2*np.sin(np.pi*x/6.0)*np.sin(np.pi*y/4.0)
    g.nodes.coords = c
    return compute_geometry(g)


def main():
    g = _grid()
    rock = make_rock(g, np.tile([2.0, 0.35, 1.0], (g.cells.num, 1))*1e-13, 0.2)
    T = compute_multipoint_trans(g, rock)
    neigh = np.asarray(g.faces.neighbors)
    boundary = np.any(neigh < 0, axis=1)
    left = np.flatnonzero(boundary & (g.faces.centroids[:, 0] < 1e-10))
    right = np.flatnonzero(boundary & (g.faces.centroids[:, 0] > 6.0 - 1e-10))

    # A pressure outlet plus a distributed flux inlet exercises both kinds in
    # one solve; all remaining boundary faces are implicitly no-flow.
    bc = add_bc(None, left, "pressure", 0.0)
    inlet_rate = 2.5e-4
    bc = add_bc(bc, right, "flux", inlet_rate/right.size)
    mixed = incomp_mpfa(g, T, bc=bc, return_result=True)
    residual = np.asarray(T["e_div"] @ mixed.subface_flux).ravel()
    cell_residual = float(np.max(np.abs(residual[:g.cells.num])))
    face_of_subface = np.asarray(T["d1"].argmax(axis=1)).ravel()
    boundary_face_of_row = face_of_subface[np.flatnonzero(T["sb"])]
    right_rows = np.isin(boundary_face_of_row, right)
    imposed_flux = float(residual[g.cells.num:][right_rows].sum())

    # Balanced sources with no pressure boundary exercise the pure-Neumann
    # null-space treatment while retaining every local conservation equation.
    src = add_source(None, np.array([0]), 1e-4)
    src = add_source(src, np.array([g.cells.num - 1]), -1e-4)
    source_result = incomp_mpfa(g, T, src=src, return_result=True)
    source_residual = np.asarray(T["e_div"] @ source_result.subface_flux).ravel()
    expected = np.zeros(g.cells.num)
    expected[0], expected[-1] = 1e-4, -1e-4
    source_error = float(np.max(np.abs(source_residual[:g.cells.num] - expected)))
    noflow_error = float(np.max(np.abs(source_residual[g.cells.num:])))
    mean_pressure = float(abs(source_result.pressure.mean()))

    # For a prescribed pressure field, homogeneous mobility scales flux
    # exactly while leaving pressure unchanged.
    pressure_bc = {
        int(f): float(10.0 - g.faces.centroids[f, 0])
        for f in np.flatnonzero(boundary)
    }
    unit = incomp_mpfa(g, T, pressure_bc, return_result=True)
    triple = incomp_mpfa(g, T, pressure_bc, mobility=3.0, return_result=True)
    pressure_scale_error = float(np.max(np.abs(unit.pressure - triple.pressure)))
    flux_scale_error = float(np.max(np.abs(triple.flux - 3.0*unit.flux)))

    # Hydrostatic pressure p = rho*g dot x has exactly zero Darcy flux.
    rho = 1000.0
    gravity = np.array([0.0, -9.81])
    hydro_bc = {
        int(f): float(2.0e5 + rho*(g.faces.centroids[f] @ gravity))
        for f in np.flatnonzero(boundary)
    }
    hydro = incomp_mpfa(
        g, T, hydro_bc, density=rho, gravity=gravity, return_result=True
    )
    hydro_exact = 2.0e5 + rho*(np.asarray(g.cells.centroids) @ gravity)
    hydro_pressure_error = float(np.max(np.abs(hydro.pressure - hydro_exact)))
    hydro_flux_error = float(np.max(np.abs(hydro.flux)))

    print("Phase 388: extended MPFA pressure solve")
    print(f"  mixed-BC cell conservation: {cell_residual:.3e}")
    print(f"  imposed boundary-row flux: {imposed_flux:.3e}")
    print(f"  balanced-source conservation: {source_error:.3e}")
    print(f"  unspecified-boundary no-flow: {noflow_error:.3e}")
    print(f"  pure-Neumann mean pressure: {mean_pressure:.3e}")
    print(f"  mobility pressure invariance: {pressure_scale_error:.3e}")
    print(f"  mobility flux scaling: {flux_scale_error:.3e}")
    print(f"  hydrostatic pressure patch: {hydro_pressure_error:.3e}")
    print(f"  hydrostatic zero flux: {hydro_flux_error:.3e}")

    failed = (
        cell_residual > 1e-12
        or abs(imposed_flux - inlet_rate) > 1e-12
        or source_error > 1e-12
        or noflow_error > 1e-12
        or mean_pressure > 1e-8
        or pressure_scale_error > 1e-10
        or flux_scale_error > 1e-18
        or hydro_pressure_error > 1e-8
        or hydro_flux_error > 1e-18
    )
    print("\nPhase 388 passed." if not failed else "\nPhase 388 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
