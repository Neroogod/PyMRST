"""Phase 400: conservative mimetic pressure solve with mixed forcing."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.mimetic import (
    computeMimeticIP,
    incompMimetic,
    incomp_mimetic,
)
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc, add_source


def _grid():
    g = cart_grid([7, 6], [7.0, 6.0])
    coordinates = np.asarray(g.nodes.coords, dtype=float)
    x, y = coordinates.T.copy()
    coordinates[:, 0] += 0.2*np.sin(np.pi*x/7)*np.sin(np.pi*y/6)
    g.nodes.coords = coordinates
    return compute_geometry(g)


def _cell_divergence(g, half_flux):
    cellno = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    return np.bincount(cellno, weights=half_flux, minlength=g.cells.num)


def main():
    g = _grid()
    rock = make_rock(
        g, np.tile([2.0, 0.3, 0.9], (g.cells.num, 1))*1e-12, 0.2
    )
    system = computeMimeticIP(g, rock, Type="hybrid")
    neighbors = np.asarray(g.faces.neighbors)
    boundary = np.any(neighbors < 0, axis=1)
    left = np.flatnonzero(boundary & (g.faces.centroids[:, 0] < 1e-12))
    right = np.flatnonzero(boundary & (g.faces.centroids[:, 0] > 7 - 1e-12))

    pressure_values = 10.0 - g.faces.centroids[left, 0]
    bc = add_bc(None, left, "pressure", pressure_values)
    total_rate = 2.5e-5
    bc = add_bc(bc, right, "flux", total_rate/right.size)
    result = incomp_mimetic(g, system, bc=bc, matrix_output=True)
    divergence = _cell_divergence(g, result.half_flux)
    cell_error = float(np.max(np.abs(divergence)))
    imposed_rate = float(np.sum(result.flux[right]))
    continuity = []
    faces = np.asarray(g.cells.faces)
    cellno = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    for face in np.flatnonzero(np.all(neighbors >= 0, axis=1)):
        rows = np.flatnonzero(faces == face)
        continuity.append(abs(np.sum(result.half_flux[rows])))
    continuity_error = float(max(continuity))

    source = add_source(None, np.array([0]), 1e-6)
    source = add_source(source, np.array([g.cells.num - 1]), -1e-6)
    neumann = incomp_mimetic(g, system, src=source)
    expected = np.zeros(g.cells.num)
    expected[0], expected[-1] = 1e-6, -1e-6
    source_error = float(np.max(np.abs(
        _cell_divergence(g, neumann.half_flux) - expected
    )))
    mean_pressure = float(abs(np.mean(neumann.pressure)))

    # MRST state-mutating entry point must use the same numerical kernel.
    class State:
        pass
    state = State()
    state = incompMimetic(state, g, system, None, bc=bc)
    entry_error = max(
        float(np.max(np.abs(state.pressure - result.pressure))),
        float(np.max(np.abs(state.flux - result.flux))),
    )

    print("Phase 400: mimetic incompressible solve")
    print(f"  cell conservation: {cell_error:.3e}")
    print(f"  internal flux continuity: {continuity_error:.3e}")
    print(f"  imposed boundary rate: {imposed_rate:.6e}")
    print(f"  balanced-source conservation: {source_error:.3e}")
    print(f"  pure-Neumann mean pressure: {mean_pressure:.3e}")
    print(f"  public/state entry-point parity: {entry_error:.3e}")
    print(f"  matrix output present: {result.A is not None}")

    failed = (
        cell_error > 1e-12
        or continuity_error > 1e-12
        or abs(imposed_rate - total_rate) > 1e-12
        or source_error > 1e-12
        or mean_pressure > 1e-8
        or entry_error > 1e-12
        or result.A is None
    )
    print("\nPhase 400 passed." if not failed else "\nPhase 400 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
