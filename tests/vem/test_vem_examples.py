"""The worked examples still run, and still say what they claim.

An example that stops working is a regression nobody notices, because nothing
imports it. These run both, on reduced grids so the suite stays quick, and
check the properties each example exists to demonstrate rather than merely
that they did not raise.

`vem_example_1` and `vem_example_2` are substitutions rather than ports -
MRST builds their grids with the `upr` module, which PyMRST does not have - so
what is pinned here is the physics and the code path, not agreement with
MRST's exact grid.
"""

import numpy as np
import pytest

from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.examples import vem_example_1, vem_example_2
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.params.wells_and_bc.add_bc import add_bc
from pymrst.params.wells_and_bc.add_source import add_source
from pymrst.utils.gridtools.boundary_faces import boundary_faces
from pymrst.utils.units.units import BARSA


@pytest.fixture(scope="module")
def three_dimensional():
    """Example 1 on a smaller point set than the script uses."""
    g = vem_example_1.build_grid(n_points=60, seed=0)
    from pymrst.params.rock import make_rock
    rock = make_rock(g, vem_example_1.PERMEABILITY, vem_example_1.POROSITY)

    faces = boundary_faces(g)
    centroids = np.asarray(g.faces.centroids)[faces]
    west = faces[np.abs(centroids[:, 0]) < 1e-6]
    east = faces[np.abs(centroids[:, 0] - vem_example_1.GRID_LIMITS[0]) < 1e-6]
    bc = add_bc(None, west, "pressure",
                np.full(west.size, vem_example_1.INLET_PRESSURE))
    bc = add_bc(bc, east, "pressure", np.zeros(east.size))

    states = {}
    for order in (1, 2):
        S = compute_virtual_ip(g, rock, order)
        states[order] = incomp_vem(g, S, bc=bc, cell_pressure=True,
                                   matrix_output=True)
    return g, states


@pytest.fixture(scope="module")
def two_dimensional():
    """Example 2 on a coarser grid than the script uses."""
    g = vem_example_2.build_grid(n=12)
    rock, regions = vem_example_2.build_rock(g)

    centroids = np.asarray(g.cells.centroids)[:, :2]
    injector = int(np.argmin(
        np.sum((centroids - 0.1 * vem_example_2.LENGTH) ** 2, axis=1)))
    producer = int(np.argmin(
        np.sum((centroids - 0.9 * vem_example_2.LENGTH) ** 2, axis=1)))
    rate = 1.0
    src = add_source(None, [injector, producer], [rate, -rate])
    outer = boundary_faces(g)
    bc = add_bc(None, outer, "flux", np.zeros(outer.size))

    S = compute_virtual_ip(g, rock, 1)
    state = incomp_vem(g, S, src=src, bc=bc, cell_pressure=True)
    return g, rock, regions, state, injector, producer


# ---------------------------------------------------------------- example 1

def test_the_three_dimensional_grid_is_valid(three_dimensional):
    """`build_grid` rejects the degenerate tetrahedralisations a lattice gives."""
    g, _ = three_dimensional
    assert np.asarray(g.faces.neighbors).max() < g.cells.num
    assert np.all(np.asarray(g.cells.volumes) > 0)


def test_a_lattice_is_rejected_as_degenerate():
    """The guard exists because this is the tempting thing to try."""
    limits = vem_example_1.GRID_LIMITS
    axes = [np.linspace(0.0, limits[i], 5) for i in range(3)]
    lattice = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    from pymrst.gridprocessing import compute_geometry
    from pymrst.gridprocessing.tetrahedral_grid import tetrahedral_grid
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = compute_geometry(tetrahedral_grid(lattice))
    bad = (np.asarray(g.faces.neighbors).max() >= g.cells.num
           or np.any(np.asarray(g.cells.volumes) <= 0))
    assert bad, "a lattice was expected to give a degenerate tetrahedralisation"


def test_the_pressure_drop_is_reproduced_at_both_orders(three_dimensional):
    g, states = three_dimensional
    inlet = vem_example_1.INLET_PRESSURE
    for order in (1, 2):
        pressure = states[order]["pressure"]
        assert pressure.min() > -1e-6 * inlet
        assert pressure.max() < inlet * (1 + 1e-6)


def test_the_three_dimensional_solution_is_linear(three_dimensional):
    """Which is why the two orders agree there - see the next test."""
    g, states = three_dimensional
    x = np.asarray(g.cells.centroids)[:, 0]
    inlet = vem_example_1.INLET_PRESSURE
    expected = inlet * (1.0 - x / vem_example_1.GRID_LIMITS[0])
    assert np.abs(states[1]["pressure"] - expected).max() < 1e-9 * inlet


def test_the_orders_agree_on_the_linear_solution(three_dimensional):
    _, states = three_dimensional
    difference = np.abs(states[1]["pressure"] - states[2]["pressure"]).max()
    assert difference < 1e-9 * vem_example_1.INLET_PRESSURE


def test_the_second_order_system_is_larger(three_dimensional):
    g, states = three_dimensional
    assert states[1]["A"].shape[0] == g.nodes.num
    assert states[2]["A"].shape[0] > states[1]["A"].shape[0]


# ---------------------------------------------------------------- example 2

def test_the_two_dimensional_grid_is_polygonal(two_dimensional):
    g = two_dimensional[0]
    sides = np.diff(np.asarray(g.cells.facePos))
    assert sides.max() > 4, sides.max()


def test_a_jittered_pebi_grid_is_rejected_as_slivered():
    """The other trap the examples record: jitter makes sliver cells."""
    with pytest.raises(RuntimeError, match="sliver"):
        vem_example_2.build_grid(n=12, jitter=0.3)


def test_all_four_anisotropy_regions_are_present(two_dimensional):
    regions = two_dimensional[2]
    assert np.unique(regions).size == 4


def test_the_rotated_region_has_an_off_diagonal_tensor(two_dimensional):
    _, rock, regions, _, _, _ = two_dimensional
    perm = np.asarray(rock.perm)
    rotated = perm[regions == 0][0]
    upright = perm[regions == 1][0]
    # atol=0: permeabilities are ~1e-13, so the default tolerance calls
    # every one of them zero.
    assert abs(rotated[1]) > 1e-3 * abs(rotated[0])
    assert upright[1] == 0.0


def test_the_injector_is_above_the_producer(two_dimensional):
    _, _, _, state, injector, producer = two_dimensional
    assert state["pressure"][injector] > state["pressure"][producer]


def test_the_sealed_domain_balances(two_dimensional):
    g, _, _, state, _, _ = two_dimensional
    from pymrst.utils.rldecode import rldecode
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    owner = rldecode(np.arange(g.cells.num), np.diff(face_pos))
    neighbours = np.asarray(g.faces.neighbors)
    signs = np.where(neighbours[cell_faces, 0] == owner, 1.0, -1.0)
    divergence = np.zeros(g.cells.num)
    np.add.at(divergence, owner, state["flux"][cell_faces] * signs)
    assert abs(divergence.sum()) < 1e-9
    assert np.abs(state["flux"][boundary_faces(g)]).max() < 1e-12
