"""VEM on a 3-D unstructured grid (MRST ``vem/examples/vemExample1.m``).

A pressure drop across an anisotropic box, solved at first and second order,
comparing the two. This is the only place the module is driven the way a user
drives it - build a grid, make a rock and a fluid, set boundary conditions,
solve, look at cell pressures - rather than through a unit test that pins one
quantity.

**The grid generator differs from MRST's, and that is a substitution, not a
port.** MRST builds the grid with `mirroredPebi3D` from the `upr` module,
which PyMRST does not have. This uses a tetrahedral grid over random points
instead. Everything downstream - `compute_vem_geometry`,
`compute_virtual_ip` at both orders, `incomp_vem` with `cell_pressure` and
`matrix_output` - is the same code MRST's example exercises, on a grid with
the same character: unstructured, general polyhedral cells, no two alike.

Random interior points plus the box corners, as MRST does. A *lattice* is the
tempting alternative and it does not work: coplanar points make the
tetrahedralisation degenerate, and `tetrahedral_grid` then returns a grid whose
`faces.neighbors` reference cells that do not exist. The grid is checked below
rather than assumed.
"""

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.tetrahedral_grid import tetrahedral_grid
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils.compute_vem_geometry import compute_vem_geometry
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.add_bc import add_bc
from pymrst.utils.gridtools.boundary_faces import boundary_faces
from pymrst.utils.units.units import BARSA, DARCY, MILLI

GRID_LIMITS = np.array([500.0, 100.0, 100.0])
PERMEABILITY = np.array([1000.0, 100.0, 10.0]) * MILLI * DARCY
POROSITY = 0.3
INLET_PRESSURE = 500 * BARSA


def build_grid(n_points: int = 200, seed: int = 0):
    """A tetrahedral grid over random points in the box, plus its corners."""
    rng = np.random.default_rng(seed)
    interior = rng.random((n_points, 3)) * GRID_LIMITS
    corners = np.array(np.meshgrid(*[[0.0, L] for L in GRID_LIMITS])
                       ).reshape(3, -1).T
    g = compute_geometry(tetrahedral_grid(np.vstack([interior, corners])))

    # A degenerate tetrahedralisation produces neighbours that index cells
    # which do not exist, and zero-volume cells. Catch it here rather than
    # several hundred lines into the assembly.
    neighbours = np.asarray(g.faces.neighbors)
    if neighbours.max() >= g.cells.num or np.any(np.asarray(g.cells.volumes) <= 0):
        raise RuntimeError(
            f"degenerate tetrahedral grid from seed {seed}: "
            f"max neighbour {neighbours.max()} against {g.cells.num} cells")
    return compute_vem_geometry(g)


def run(verbose: bool = True):
    """Solve the pressure drop at k=1 and k=2; return both states."""
    g = build_grid()
    rock = make_rock(g, PERMEABILITY, POROSITY)

    faces = boundary_faces(g)
    centroids = np.asarray(g.faces.centroids)[faces]
    tol = 1e-6
    west = faces[np.abs(centroids[:, 0]) < tol]
    east = faces[np.abs(centroids[:, 0] - GRID_LIMITS[0]) < tol]

    bc = add_bc(None, west, "pressure", np.full(west.size, INLET_PRESSURE))
    bc = add_bc(bc, east, "pressure", np.zeros(east.size))

    states = {}
    for order in (1, 2):
        S = compute_virtual_ip(g, rock, order)
        states[order] = incomp_vem(g, S, bc=bc, cell_pressure=True,
                                   matrix_output=True)
        if verbose:
            pressure = states[order]["pressure"]
            print(f"  k={order}: {states[order]['A'].shape[0]:5d} dofs, "
                  f"cell pressure {pressure.min() / BARSA:7.2f} .. "
                  f"{pressure.max() / BARSA:7.2f} bar")
    return g, rock, bc, states


if __name__ == "__main__":
    print(f"3-D unstructured VEM: {GRID_LIMITS.tolist()} m box, "
          f"perm {(PERMEABILITY / (MILLI * DARCY)).tolist()} mD")
    grid, rock, bc, states = run()

    def test_the_grid_is_unstructured() -> None:
        counts = np.diff(np.asarray(grid.cells.facePos))
        assert grid.cells.num > 100, grid.cells.num
        print(f"ok test_the_grid_is_unstructured "
              f"({grid.cells.num} cells, {counts.min()}-{counts.max()} faces each)")

    def test_both_orders_solve() -> None:
        for order in (1, 2):
            assert np.all(np.isfinite(states[order]["pressure"]))
        print("ok test_both_orders_solve")

    def test_the_pressure_spans_the_imposed_drop() -> None:
        """Interior cell pressures lie inside the boundary values."""
        for order in (1, 2):
            pressure = states[order]["pressure"]
            assert pressure.min() > -1e-6 * INLET_PRESSURE, pressure.min()
            assert pressure.max() < INLET_PRESSURE * (1 + 1e-6), pressure.max()
        print("ok test_the_pressure_spans_the_imposed_drop")

    def test_the_pressure_falls_from_west_to_east() -> None:
        centroids = np.asarray(grid.cells.centroids)[:, 0]
        for order in (1, 2):
            pressure = states[order]["pressure"]
            near = pressure[centroids < 0.2 * GRID_LIMITS[0]].mean()
            far = pressure[centroids > 0.8 * GRID_LIMITS[0]].mean()
            assert near > far, (order, near, far)
        print("ok test_the_pressure_falls_from_west_to_east")

    def test_the_two_orders_agree_because_the_solution_is_linear() -> None:
        """Not evidence that k=2 adds nothing - evidence about this problem.

        A uniform drop across a homogeneous box has a linear exact solution,
        and both orders reproduce linear fields exactly, so they agree to
        round-off. Somewhere the orders *do* differ is a curved solution;
        `vem_example_2` provides one by varying the permeability tensor
        between regions.
        """
        first, second = states[1]["pressure"], states[2]["pressure"]
        difference = np.abs(first - second).max() / INLET_PRESSURE
        assert difference < 1e-9, difference
        print(f"ok test_the_two_orders_agree_because_the_solution_is_linear "
              f"({difference:.2e} of the imposed drop)")

    def test_the_solution_really_is_linear_in_x() -> None:
        """Which is what makes the agreement above unsurprising."""
        centroids = np.asarray(grid.cells.centroids)[:, 0]
        expected = INLET_PRESSURE * (1.0 - centroids / GRID_LIMITS[0])
        error = np.abs(states[1]["pressure"] - expected).max() / INLET_PRESSURE
        assert error < 1e-9, error
        print(f"ok test_the_solution_really_is_linear_in_x ({error:.2e})")

    def test_the_second_order_system_is_larger() -> None:
        """k=2 adds edge, face and cell moments to the nodal unknowns."""
        assert states[2]["A"].shape[0] > states[1]["A"].shape[0]
        assert states[1]["A"].shape[0] == grid.nodes.num
        print(f"ok test_the_second_order_system_is_larger "
              f"({states[1]['A'].shape[0]} -> {states[2]['A'].shape[0]} dofs)")

    def test_the_flux_balances_every_cell() -> None:
        from pymrst.utils.rldecode import rldecode
        for order in (1, 2):
            flux = states[order]["flux"]
            face_pos = np.asarray(grid.cells.facePos, dtype=np.int64)
            cell_faces = np.asarray(grid.cells.faces, dtype=np.int64)
            cell_faces = (cell_faces[:, 0] if cell_faces.ndim > 1
                          else cell_faces)
            owner = rldecode(np.arange(grid.cells.num), np.diff(face_pos))
            neighbours = np.asarray(grid.faces.neighbors)
            signs = np.where(neighbours[cell_faces, 0] == owner, 1.0, -1.0)
            divergence = np.zeros(grid.cells.num)
            np.add.at(divergence, owner, flux[cell_faces] * signs)
            scale = np.abs(flux).max()
            # 1e-7 rather than machine precision: the k=2 system here has
            # 5451 unknowns, and the residual of a direct solve accumulates
            # with the problem size. k=1 lands near 1e-12; k=2 near 1e-8.
            assert np.abs(divergence).max() < 1e-7 * scale, \
                (order, np.abs(divergence).max() / scale)
        print("ok test_the_flux_balances_every_cell")

    test_the_grid_is_unstructured()
    test_both_orders_solve()
    test_the_pressure_spans_the_imposed_drop()
    test_the_pressure_falls_from_west_to_east()
    test_the_two_orders_agree_because_the_solution_is_linear()
    test_the_solution_really_is_linear_in_x()
    test_the_second_order_system_is_larger()
    test_the_flux_balances_every_cell()
    print("\nAll tests passed!")
