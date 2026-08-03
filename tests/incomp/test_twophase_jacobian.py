"""Two-phase transport residual and Jacobian (MRST ``twophaseJacobian``).

On F0.4's risk list with no focused test. A hand-written Jacobian is the
classic place for a silent defect: Newton still converges from a wrong one, it
just takes more iterations, so nothing fails and the cost hides in the solver.
The check that actually settles it is the analytic Jacobian against central
differences of the residual it claims to differentiate.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp.fluid import init_simple_fluid
from pymrst.modules.incomp.transport.private.twophase_jacobian import (
    two_phase_residual_jacobian,
)
from pymrst.params.rock import make_rock
from pymrst.solvers.state import State


def _case(dims=(4, 1), size=(4.0, 1.0), flux=1.0e-6, saturation=0.4):
    g = compute_geometry(cart_grid(list(dims), list(size)))
    rock = make_rock(g, perm=np.full(g.cells.num, 1.0e-13),
                     poro=np.full(g.cells.num, 0.3))
    fluid = init_simple_fluid(mu=[1.0e-3, 5.0e-3], rho=[1000.0, 800.0], n=[2, 2])
    face_flux = np.zeros(g.faces.num)
    interior = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
    face_flux[interior] = flux
    state = State(
        np.zeros(g.cells.num), face_flux,
        np.column_stack([np.full(g.cells.num, saturation),
                         np.full(g.cells.num, 1.0 - saturation)]),
    )
    return g, state, rock, fluid


def _finite_difference(residual, s, s0, dt, eps=1.0e-7):
    columns = []
    for index in range(s.size):
        step = np.zeros_like(s)
        step[index] = eps
        columns.append(
            (residual(s + step, s0, dt) - residual(s - step, s0, dt))/(2.0*eps)
        )
    return np.column_stack(columns)


def _assert_jacobian_matches(g, state, rock, fluid, *, gravity=None, dt=1000.0):
    residual, jacobian, _, _ = two_phase_residual_jacobian(
        g, state, rock, fluid, gravity=gravity,
    )
    s = np.linspace(0.3, 0.7, g.cells.num)
    s0 = np.full(g.cells.num, 0.5)
    analytic = jacobian(s, s0, dt).toarray()
    numerical = _finite_difference(residual, s, s0, dt)
    scale = max(np.max(np.abs(numerical)), 1.0e-300)
    assert np.max(np.abs(analytic - numerical))/scale < 1.0e-6, (
        f"Jacobian disagrees with central differences by "
        f"{np.max(np.abs(analytic - numerical))/scale:.3e} relative"
    )
    return analytic, numerical


def test_jacobian_matches_central_differences_of_its_own_residual():
    g, state, rock, fluid = _case()
    _assert_jacobian_matches(g, state, rock, fluid)


def test_jacobian_still_matches_with_gravity_active():
    """Gravity adds a second upwind direction, chosen per phase.

    The gravity flux can oppose the Darcy flux, and then water and oil are
    upwinded from opposite cells. That branch is invisible without a non-zero
    density difference and a gravity vector.
    """
    g, state, rock, fluid = _case(dims=(1, 4), size=(1.0, 4.0))
    analytic, _ = _assert_jacobian_matches(
        g, state, rock, fluid, gravity=np.array([0.0, -9.80665]),
    )
    assert np.any(analytic != 0.0)


def test_jacobian_matches_on_a_two_dimensional_grid():
    g, state, rock, fluid = _case(dims=(3, 3), size=(3.0, 3.0))
    _assert_jacobian_matches(g, state, rock, fluid)


def test_the_residual_is_zero_when_nothing_moves():
    """No flux, no gravity, no sources: saturation cannot change.

    A residual that is not identically zero here would drive Newton away from
    a correct steady state.
    """
    g, state, rock, fluid = _case(flux=0.0)
    residual, _, _, _ = two_phase_residual_jacobian(g, state, rock, fluid)
    s0 = np.full(g.cells.num, 0.5)
    np.testing.assert_allclose(residual(s0, s0, 1000.0), 0.0, atol=1e-15)


def test_the_accumulation_term_is_the_identity_at_zero_timestep():
    """At dt = 0 the residual is s - s0 and the Jacobian is I.

    This isolates the accumulation from every flux term, so a sign error in
    accumulation cannot hide behind a compensating one in the divergence.
    """
    g, state, rock, fluid = _case()
    residual, jacobian, _, _ = two_phase_residual_jacobian(g, state, rock, fluid)
    s = np.linspace(0.2, 0.8, g.cells.num)
    s0 = np.full(g.cells.num, 0.5)
    np.testing.assert_allclose(residual(s, s0, 0.0), s - s0, atol=1e-15)
    np.testing.assert_allclose(
        jacobian(s, s0, 0.0).toarray(), np.eye(g.cells.num), atol=1e-15,
    )


def test_water_flows_downstream_and_the_divergence_conserves():
    """Interior transport moves saturation without creating it.

    With no sources the divergence must sum to zero over the cells, so the
    total pore-volume-weighted change equals nothing. A one-sided divergence -
    adding to the downstream cell without subtracting upstream - passes every
    single-cell check and fails this one.
    """
    g, state, rock, fluid = _case()
    residual, _, _, _ = two_phase_residual_jacobian(g, state, rock, fluid)
    s0 = np.full(g.cells.num, 0.5)
    s = s0.copy()
    from pymrst.params.rock import pore_volume
    pv = pore_volume(g, rock)
    change = residual(s, s0, 1000.0)*pv          # dt*(div - q_w), q_w = 0
    np.testing.assert_allclose(change.sum(), 0.0, atol=1e-12*np.abs(change).max()
                               if np.abs(change).max() else 1e-15)


def test_water_moves_downstream_not_upstream():
    """The physics the finite-difference check cannot see.

    Comparing the Jacobian against differences of the residual establishes
    that they agree, not that either is right: swapping the upwind side
    changes both consistently and the comparison still passes. It did — that
    mutation survived every FD test here.

    Direction has to be asserted against the flow. With water only in the
    inlet cell and a positive flux, the cell downstream of it must gain water
    and the far cell must be untouched. Upwinding from the wrong side reverses
    both.
    """
    g, state, rock, fluid = _case()
    residual, _, _, _ = two_phase_residual_jacobian(g, state, rock, fluid)
    s0 = np.array([1.0, 0.0, 0.0, 0.0])          # water only in the inlet cell
    r = residual(s0, s0, 1000.0)

    # residual = s - s0 + dt/pv*(div - q); at s = s0 it is dt/pv*div, so a
    # negative entry is net inflow and the saturation there has to rise.
    assert r[0] > 0.0, "the inlet cell should be losing water downstream"
    assert r[1] < 0.0, "the cell downstream of the water should be gaining it"
    np.testing.assert_allclose(r[2:], 0.0, atol=1e-15)


def test_an_injection_source_adds_water_and_a_producer_removes_the_flowing_mix():
    """`q_w` treats the two signs differently, and the asymmetry is the point.

    Injection enters as pure water, so it is `max(q, 0)` unscaled; production
    leaves at the cell's own fractional flow, so it is `min(q, 0)*f`. Dropping
    the term entirely changes nothing in a case without sources, which is why
    every other test here missed it.
    """
    from pymrst.params.wells_and_bc import Source

    g, state, rock, fluid = _case(flux=0.0)
    rate = 1.0e-6
    src = Source(cell=np.array([0, 3]), rate=np.array([rate, -rate]),
                 sat=np.column_stack([[1.0, 1.0], [0.0, 0.0]]))
    residual, _, _, q = two_phase_residual_jacobian(
        g, state, rock, fluid, src=src,
    )
    np.testing.assert_allclose(q[[0, 3]], [rate, -rate])
    np.testing.assert_allclose(q[[1, 2]], 0.0)

    s0 = np.full(g.cells.num, 0.5)
    dt = 1000.0
    r = residual(s0, s0, dt)
    np.testing.assert_allclose(r[[1, 2]], 0.0, atol=1e-15)

    # Signs alone do not separate the two treatments - both send the injector
    # negative and the producer positive - so the magnitudes are asserted.
    # At s = 0.5 this fluid's fractional flow is 5/6, so the producer removes
    # five sixths of what it would if it drew pure water.
    from pymrst.params.rock import pore_volume
    pv = pore_volume(g, rock)
    mobility = fluid.mobility(np.column_stack([s0, 1.0 - s0]))
    fractional_flow = mobility[:, 0]/mobility.sum(axis=1)
    np.testing.assert_allclose(fractional_flow[3], 5.0/6.0)

    np.testing.assert_allclose(r[0], -dt/pv[0]*rate, rtol=1e-12)
    np.testing.assert_allclose(r[3], dt/pv[3]*rate*fractional_flow[3], rtol=1e-12)
