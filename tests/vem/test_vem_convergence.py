"""The VEM converges, and at the rate its order promises.

Patch tests show a method reproduces polynomials it ought to reproduce
exactly. They say nothing about what happens on a solution it cannot
represent, which is every real problem. A method can pass every patch test and
still not converge - the tests elsewhere in this module would not notice.

The solution used here is `exp(x) sin(y)`, which is harmonic (so no source
term is needed) and is not a polynomial of any degree (so neither order
reproduces it exactly, and the error is genuine discretisation error rather
than round-off).

Expected rates, measured: **2 for k=1 and about 3.8 for k=2** in the nodal
maximum norm. The thresholds below are set well under those so a small change
in the grid or the solver does not trip them, while a method that has dropped
to first order, or stopped converging, fails immediately.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
from pymrst.modules.vem.utils.compute_vem_geometry import compute_vem_geometry
from pymrst.params.rock import make_rock

RESOLUTIONS = (4, 8, 16, 32)


def harmonic(points: np.ndarray) -> np.ndarray:
    """`exp(x) sin(y)`: harmonic, smooth, and not a polynomial."""
    return np.exp(points[:, 0]) * np.sin(points[:, 1])


def boundary_faces_of(g):
    return np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))


def nodal_error(n: int, order: int) -> float:
    """Max nodal error of the VEM solution on an n-by-n unit square."""
    g = compute_vem_geometry(compute_geometry(cart_grid([n, n], [1.0, 1.0])))
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), order)
    bc = add_bc_vem(None, boundary_faces_of(g), "pressure", harmonic)
    state = incomp_vem(g, S, bc=bc, reconstruct_flux=False)
    exact = harmonic(np.asarray(g.nodes.coords)[:, :2])
    return float(np.abs(state["node_pressure"] - exact).max())


@pytest.fixture(scope="module")
def errors():
    return {order: [nodal_error(n, order) for n in RESOLUTIONS]
            for order in (1, 2)}


def rates(sequence):
    return [np.log2(sequence[i] / sequence[i + 1])
            for i in range(len(sequence) - 1)]


@pytest.mark.parametrize("order", [1, 2])
def test_the_error_falls_with_refinement(errors, order):
    sequence = errors[order]
    assert all(sequence[i] > sequence[i + 1] for i in range(len(sequence) - 1)), \
        sequence


def test_first_order_converges_quadratically(errors):
    """k=1 is superconvergent at the nodes: 2, not 1."""
    observed = rates(errors[1])
    assert min(observed) > 1.8, observed


def test_second_order_converges_faster_than_first(errors):
    assert min(rates(errors[2])) > max(rates(errors[1])) + 0.5, \
        (rates(errors[1]), rates(errors[2]))


def test_second_order_converges_at_least_cubically(errors):
    observed = rates(errors[2])
    assert min(observed) > 3.0, observed


def test_the_second_order_solution_is_more_accurate_everywhere(errors):
    """At every resolution, not only asymptotically."""
    for coarse, fine in zip(errors[1], errors[2]):
        assert fine < coarse, (coarse, fine)


def test_the_rates_are_steady_rather_than_a_single_lucky_pair(errors):
    """A method that stalls shows it here, where a two-grid check would not."""
    for order in (1, 2):
        observed = rates(errors[order])
        assert max(observed) - min(observed) < 1.0, (order, observed)


def test_the_solution_is_actually_hard_for_both_orders(errors):
    """Guard on the fixture: a polynomial would be reproduced exactly and
    every rate above would be meaningless."""
    assert errors[1][0] > 1e-6, errors[1][0]
    assert errors[2][0] > 1e-9, errors[2][0]
