"""VEM on a 2-D PEBI grid with rotated anisotropy (MRST ``vem/examples/vemExample2.m``).

An injector and a producer in a square, across four regions whose permeability
tensors are the same anisotropy rotated by different angles. The rotation is
the point: flow refracts at each region boundary, so the solution is genuinely
curved and the two orders do not agree - unlike `vem_example_1`, whose linear
solution both orders reproduce exactly.

**The grid generator differs from MRST's.** MRST uses `pebiGrid2D` from the
`upr` module with face and cell constraints, so its region boundaries follow
grid faces exactly and its wells sit in dedicated cells. PyMRST has no `upr`,
so this builds a PEBI grid from a Delaunay triangulation of a jittered lattice
and assigns regions by centroid. Two consequences worth stating: region
boundaries cut through cells rather than following faces, so the tensor is
piecewise constant per *cell* rather than exactly per region; and the
generating points must stay on a lattice, for the reason `build_grid`
documents.

Everything else - `compute_vem_geometry`, `compute_virtual_ip` at both orders,
`incomp_vem` with sources, `conserve_flux` - is the code MRST's example
exercises.
"""

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.pebi import pebi
from pymrst.gridprocessing.triangle_grid import triangle_grid
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils.compute_vem_geometry import compute_vem_geometry
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.add_bc import add_bc
from pymrst.params.wells_and_bc.add_source import add_source
from pymrst.utils.units.units import DARCY, MILLI, YEAR

LENGTH = 1000.0
# MRST scales the rate to the pore volume: one pore volume in ten years. A
# rate in m^3/s instead gives pressures around 1e14 Pa, which are not wrong so
# much as meaningless, and they make the linear system harder to read.
INJECTION_YEARS = 10.0
BASE_PERM = np.diag([1000.0, 10.0]) * MILLI * DARCY
ANGLES = (np.pi / 4, 0.0, np.pi / 2, 0.0)


def _rotate(K, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ K @ R.T


def build_grid(n: int = 22, jitter: float = 0.0, seed: int = 0):
    """A PEBI grid over a lattice of generating points.

    **`jitter` defaults to zero, and perturbing it is not an improvement.**
    Jittering the generators is the obvious way to make the grid look less
    regular, and with PyMRST's `pebi` it produces sliver cells: at any jitter
    from 0.02 to 0.35 the cell volumes span five orders of magnitude and the
    local VEM blocks pick up negative eigenvalues, which leaves the assembled
    system indefinite and the solve inaccurate to ~1e-6. An unjittered lattice
    gives 4- to 7-sided cells with a volume ratio of 4 - genuinely polygonal,
    and well conditioned.

    That is a property of the PEBI construction from near-cocircular points,
    not of the VEM code, and it is recorded here because a reader's first
    instinct is to turn the jitter up.
    """
    rng = np.random.default_rng(seed)
    step = LENGTH / n
    axis = (np.arange(n) + 0.5) * step
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"),
                      axis=-1).reshape(-1, 2)
    if jitter:
        points += (rng.random(points.shape) - 0.5) * jitter * step
    g = compute_vem_geometry(compute_geometry(pebi(triangle_grid(points))))

    volumes = np.asarray(g.cells.volumes)
    ratio = volumes.max() / volumes.min()
    if ratio > 100.0:
        raise RuntimeError(
            f"grid has sliver cells (volume ratio {ratio:.3g}); the VEM system "
            f"will be indefinite. See this function's docstring on `jitter`.")
    return g


def region_of(centroids: np.ndarray) -> np.ndarray:
    """Four diagonal bands, as MRST's face constraints produce."""
    diagonal = (centroids[:, 0] + centroids[:, 1]) / (2 * LENGTH)
    return np.clip((diagonal * 4).astype(int), 0, 3)


def build_rock(g):
    """One rotated tensor per region, as (kxx, kxy, kyy) per cell."""
    regions = region_of(np.asarray(g.cells.centroids)[:, :2])
    perm = np.zeros((g.cells.num, 3))
    for index, angle in enumerate(ANGLES):
        K = _rotate(BASE_PERM, angle)
        perm[regions == index] = [K[0, 0], K[0, 1], K[1, 1]]
    return make_rock(g, perm, 0.3), regions


def run(verbose: bool = True):
    """Inject at one corner, produce at the other; solve at both orders."""
    g = build_grid()
    rock, regions = build_rock(g)

    centroids = np.asarray(g.cells.centroids)[:, :2]
    injector = int(np.argmin(np.sum((centroids - 0.1 * LENGTH) ** 2, axis=1)))
    producer = int(np.argmin(np.sum((centroids - 0.9 * LENGTH) ** 2, axis=1)))
    rate = float(np.sum(np.asarray(g.cells.volumes) * np.asarray(rock.poro))
                 / (INJECTION_YEARS * YEAR))
    src = add_source(None, [injector, producer], [rate, -rate])

    # Seal the boundary explicitly. Without a bc the reconstruction leaves
    # boundary faces at their computed flux, so the domain leaks and the
    # injected volume is not the produced volume.
    outer = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
    bc = add_bc(None, outer, "flux", np.zeros(outer.size))

    states = {}
    for order in (1, 2):
        S = compute_virtual_ip(g, rock, order)
        states[order] = incomp_vem(g, S, src=src, bc=bc,
                                   cell_pressure=True)
        if verbose:
            pressure = states[order]["pressure"]
            print(f"  k={order}: pressure range "
                  f"{pressure.min():.4g} .. {pressure.max():.4g}")
    return g, rock, regions, src, states, rate


if __name__ == "__main__":
    print(f"2-D PEBI VEM: {LENGTH:.0f} m square, four rotated anisotropy "
          f"regions {[round(np.degrees(a)) for a in ANGLES]} deg")
    grid, rock, regions, src, states, rate = run()

    def test_the_grid_is_a_polygonal_pebi_grid() -> None:
        counts = np.diff(np.asarray(grid.cells.facePos))
        assert counts.max() > 4, "a PEBI grid has cells with more than 4 sides"
        print(f"ok test_the_grid_is_a_polygonal_pebi_grid "
              f"({grid.cells.num} cells, {counts.min()}-{counts.max()} sides)")

    def test_all_four_regions_are_populated() -> None:
        present = np.unique(regions)
        assert present.size == 4, present
        print(f"ok test_all_four_regions_are_populated "
              f"({np.bincount(regions).tolist()} cells)")

    def test_the_regions_carry_different_tensors() -> None:
        perm = np.asarray(rock.perm)
        first = perm[regions == 0][0]
        second = perm[regions == 1][0]
        # atol=0: permeabilities are ~1e-13 in SI, so the default absolute
        # tolerance of 1e-8 would call any two of them equal.
        assert not np.allclose(first, second, atol=0.0), (first, second)
        # Likewise here: the 45 deg off-diagonal is (1000 - 10)/2 mD, which
        # is 4.9e-13 in SI and "close to zero" under the default tolerance.
        assert abs(first[1]) > 1e-3 * abs(first[0]), \
            "the 45 deg tensor should not be diagonal"
        assert second[1] == 0.0, "the unrotated tensor should be diagonal"
        print("ok test_the_regions_carry_different_tensors")

    def test_both_orders_solve() -> None:
        for order in (1, 2):
            assert np.all(np.isfinite(states[order]["pressure"]))
        print("ok test_both_orders_solve")

    def test_the_injector_is_the_high_pressure_cell() -> None:
        centroids = np.asarray(grid.cells.centroids)[:, :2]
        injector = int(np.argmin(np.sum((centroids - 0.1 * LENGTH) ** 2, axis=1)))
        producer = int(np.argmin(np.sum((centroids - 0.9 * LENGTH) ** 2, axis=1)))
        for order in (1, 2):
            pressure = states[order]["pressure"]
            assert pressure[injector] > pressure[producer], order
        print("ok test_the_injector_is_the_high_pressure_cell")

    def test_the_source_is_balanced_by_the_sink() -> None:
        """No boundary flow, so what is injected must be produced."""
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
            assert abs(divergence.sum()) < 1e-8 * rate, divergence.sum()
        print("ok test_the_source_is_balanced_by_the_sink")

    def test_the_two_orders_differ_on_this_curved_solution() -> None:
        """The contrast with `vem_example_1`, whose solution is linear.

        Rotated tensors refract the flow at every region boundary, so the
        solution is not a polynomial either order reproduces exactly and the
        two disagree by a discretisation error rather than by round-off.
        """
        first, second = states[1]["pressure"], states[2]["pressure"]
        scale = first.max() - first.min()
        difference = np.abs(first - second).max() / scale
        assert difference > 1e-6, difference
        print(f"ok test_the_two_orders_differ_on_this_curved_solution "
              f"({difference:.2e} of the pressure range)")

    def test_anisotropy_steers_the_flow_off_the_diagonal() -> None:
        """A rotated tensor must move the answer, or the rock is not being read."""
        isotropic = make_rock(grid, np.tile([1000.0 * MILLI * DARCY, 0.0,
                                             1000.0 * MILLI * DARCY],
                                            (grid.cells.num, 1)), 0.3)
        S = compute_virtual_ip(grid, isotropic, 1)
        outer = np.flatnonzero(
            np.any(np.asarray(grid.faces.neighbors) < 0, axis=1))
        sealed = add_bc(None, outer, "flux", np.zeros(outer.size))
        plain = incomp_vem(grid, S, src=src, bc=sealed,
                           cell_pressure=True)["pressure"]
        anisotropic = states[1]["pressure"]
        scale = anisotropic.max() - anisotropic.min()
        assert np.abs(plain - anisotropic).max() > 0.01 * scale
        print("ok test_anisotropy_steers_the_flow_off_the_diagonal")

    test_the_grid_is_a_polygonal_pebi_grid()
    test_all_four_regions_are_populated()
    test_the_regions_carry_different_tensors()
    test_both_orders_solve()
    test_the_injector_is_the_high_pressure_cell()
    test_the_source_is_balanced_by_the_sink()
    test_the_two_orders_differ_on_this_curved_solution()
    test_anisotropy_steers_the_flow_off_the_diagonal()
    print("\nAll tests passed!")
