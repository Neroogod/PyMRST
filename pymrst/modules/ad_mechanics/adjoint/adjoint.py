"""Adjoint gradients for coupled poromechanics (MRST's runAdjointExample).

The question this answers is the one geomechanics is usually asked in
practice: *how should the wells be operated to keep the ground still?*
Injection lifts the surface; the objective is the **uplift** - the vertical
displacement of a chosen surface node, integrated over the schedule
(MRST's `objUplift` / `computeUpliftForState`) - and what is wanted is its
gradient with respect to every well control at every timestep.

Computed by one backward sweep rather than one simulation per control. For
a schedule of `n` steps and `w` wells that is one extra linear solve per
step instead of `n*w` forward runs.

The discrete adjoint, for residuals `R_n(x_n, x_{n-1}, u_n) = 0` and
`J = sum_n J_n(x_n)`:

    (dR_n/dx_n)^T lam_n = -(dJ_n/dx_n)^T - (dR_{n+1}/dx_n)^T lam_{n+1}
    dJ/du_n             = (dR_n/du_n)^T lam_n

which needs three Jacobians. `dR_n/dx_n` is the one the forward Newton
already builds. The other two are assembled here by seeding the *previous*
state and the controls as AD variables instead of the current state.

**How it is verified.** As with the flow adjoint (Phase 18), there is no
MATLAB array to compare against and none is needed: a gradient is either the
derivative of the forward model or it is not, and finite differences of that
model settle the question. The forward model it differentiates is the coupled
poroelastic one validated in Phase 27 against MATLAB, Terzaghi and Mandel, so
the chain is closed. The two blocks the adjoint adds - `dR/du` and
`dR/dx_{n-1}` - are each finite-differenced separately as well, because a
single end-to-end check can hide compensating errors in them.

**One caveat.** `dR_n/dx_{n-1}` differentiates the previous state's pore
volume through `div(u_{n-1})`, which is rebuilt from `xd_{n-1}` against the
*current* prescribed displacements. That is exact whenever the displacement
boundary conditions do not move in time, which is the case for wells-driven
poroelasticity. A moving boundary (Mandel's plate) would need the stored
`u` differentiated instead, and this adjoint does not support it.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_mechanics.mech_blackoil import MechBlackOilModel
from pymrst.utils.adi import ADI, combine_equations, value


@dataclass
class AdjointResult:
    """Objective value, its gradient, and the states it was evaluated at."""

    value: float
    gradient: np.ndarray            # (n_steps, n_wells)
    states: List[State] = field(default_factory=list)


def _stack(eq) -> sp.csr_matrix:
    return sp.hstack([j for j in eq.jac], format="csr")


def _set_controls(model: MechBlackOilModel, u: np.ndarray) -> None:
    for w, val in zip(model.wells, np.asarray(u, float).ravel()):
        w.val = float(val)


def _layout(model: MechBlackOilModel):
    """(n_columns, column offset of xd, row offset of the control equations)."""
    nc, nw, nph = model.G.cells.num, model.facility.n_wells, model.n_phases
    n_cellvars = 3 if model.disgas else 2
    n_free = model.mech.n_free

    xd_col = n_cellvars * nc + (nph + 1) * nw
    n_cols = xd_col + n_free
    ctrl_row = nph * nc + nph * nw
    n_rows = ctrl_row + nw + n_free
    return n_cols, xd_col, ctrl_row, n_rows


def _jac_wrt_current(model, st0: State, st: State, dt: float) -> sp.csc_matrix:
    """dR_n/dx_n - the matrix the forward Newton already solves with."""
    eqs = model.get_equations(st0, st, dt)
    return combine_equations(*eqs).jac[0].tocsc()


def _jac_wrt_previous(model, st0: State, st: State,
                      dt: float) -> sp.csc_matrix:
    """dR_n/dx_{n-1}.

    Only the accumulation term sees the previous state - through the
    previous mass, and hence the previous *pore volume*, and hence the
    previous displacement. The well, control and mechanics rows carry no
    dependence on it at all and are padded with zeros.
    """
    st0_ad = model.init_state_ad(st0)      # AD in every primary variable
    st_plain = model._state0_props(st)     # current state, plain values

    eqs = model.conservation_equations(st0_ad, st_plain, dt)

    n_cols, _, _, n_rows = _layout(model)
    blocks = [_stack(e) for e in eqs]
    n_cell_rows = sum(b.shape[0] for b in blocks)
    blocks.append(sp.csr_matrix((n_rows - n_cell_rows, n_cols)))
    return sp.vstack(blocks, format="csc")


def _jac_wrt_controls(model) -> sp.csc_matrix:
    """dR_n/du_n.

    The control target enters exactly one equation, linearly: the control
    residual is `bhp - target` or `sum(q) - target`, so the derivative is -1
    in that row. `test_control_derivative_is_minus_one` checks this against
    the model rather than trusting the comment.
    """
    nw = model.facility.n_wells
    _, _, ctrl_row, n_rows = _layout(model)
    rows = ctrl_row + np.arange(nw)
    return sp.csc_matrix((-np.ones(nw), (rows, np.arange(nw))),
                         shape=(n_rows, nw))


def uplift_objective(model: MechBlackOilModel, node: int, direction: int = 2):
    """dt * u_node[direction]: MRST's `objUplift`.

    The displacement of one surface node, weighted by the timestep, summed
    over the schedule. Returns (J_n, dJ_n/dx_n) for a given state.
    """
    is_dir = model.mech.operators.is_dir
    dof = 3 * int(node) + int(direction)
    if is_dir[dof]:
        raise ValueError(
            f"node {node} has its component {direction} prescribed - its "
            "displacement cannot be an objective"
        )
    # index of that dof among the free ones (MRST does the same search)
    free_index = int(np.count_nonzero(~is_dir[:dof]))

    n_cols, xd_col, _, _ = _layout(model)
    col = xd_col + free_index

    def evaluate(st: State, dt: float) -> Tuple[float, np.ndarray]:
        j = float(st["xd"][free_index]) * dt
        djdx = np.zeros(n_cols)
        djdx[col] = dt
        return j, djdx

    return evaluate


def simulate_forward(model: MechBlackOilModel, state0: State,
                     dts: Sequence[float], controls: np.ndarray,
                     solver=None) -> List[State]:
    """Run the schedule, keeping every state (the adjoint needs them all)."""
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver

    solver = solver or NonLinearSolver()
    states = [state0]
    for n, dt in enumerate(dts):
        _set_controls(model, controls[n])
        st, rep = solver.solve_ministep(model, states[-1], states[-1].copy(),
                                        float(dt))
        if not rep.converged:
            raise RuntimeError(f"forward step {n} did not converge")
        states.append(st)
    return states


def evaluate_objective(objective, states: Sequence[State],
                       dts: Sequence[float]) -> float:
    """J = sum_n J_n, on an already-simulated schedule."""
    return sum(objective(states[n + 1], float(dt))[0]
               for n, dt in enumerate(dts))


def compute_gradient_adjoint(
    model: MechBlackOilModel,
    state0: State,
    dts: Sequence[float],
    controls: np.ndarray,
    objective,
    states: Optional[Sequence[State]] = None,
) -> AdjointResult:
    """dJ/du for every control of every step, in one backward sweep."""
    controls = np.atleast_2d(np.asarray(controls, float))
    nsteps, nw = controls.shape
    assert len(dts) == nsteps, "one control row per timestep"
    assert nw == model.facility.n_wells, "one control per well"

    if states is None:
        states = simulate_forward(model, state0, dts, controls)

    dRdu = _jac_wrt_controls(model)
    grad = np.zeros((nsteps, nw))
    jtotal = 0.0
    lam = None

    for n in reversed(range(nsteps)):
        _set_controls(model, controls[n])
        st_prev, st = states[n], states[n + 1]
        dt = float(dts[n])

        jn, djdx = objective(st, dt)
        jtotal += jn

        rhs = -djdx
        if lam is not None:
            # dR_{n+1}/dx_n couples this step to the one after it
            _set_controls(model, controls[n + 1])
            b = _jac_wrt_previous(model, st, states[n + 2], float(dts[n + 1]))
            rhs = rhs - b.T @ lam
            _set_controls(model, controls[n])

        a = _jac_wrt_current(model, st_prev, st, dt)
        lam = spla.spsolve(a.T.tocsc(), rhs)
        grad[n] = dRdu.T @ lam

    return AdjointResult(value=jtotal, gradient=grad, states=list(states))


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_mechanics.mechanics import MechanicModel
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.modules.vemmech.vem import (
        ElasticBC,
        create_augmented_grid,
        enu2c,
    )
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def build(nx=4, ny=4, nz=2):
        """A shallow poroelastic block, free at the surface, on rollers below.

        Injection at one corner lifts the surface above it - the uplift the
        objective measures.
        """
        phys = [40.0 * nx / 4, 40.0 * ny / 4, 10.0 * nz / 2]
        g = create_augmented_grid(
            compute_geometry(cart_grid([nx, ny, nz], phys))
        )
        rock = make_rock(g, 500 * MILLI * DARCY, 0.3)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=100 * BARSA,
        )

        pts, tol = g.nodes.coords, 1e-9
        xf = np.flatnonzero((pts[:, 0] < tol) | (pts[:, 0] > phys[0] - tol))
        yf = np.flatnonzero((pts[:, 1] < tol) | (pts[:, 1] > phys[1] - tol))
        zf = np.flatnonzero(pts[:, 2] > phys[2] - tol)     # base
        nodes = np.concatenate([xf, yf, zf])
        mask = np.vstack([
            np.tile([True, False, False], (xf.size, 1)),
            np.tile([False, True, False], (yf.size, 1)),
            np.tile([False, False, True], (zf.size, 1)),
        ])
        el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask,
                          disp_vals=np.zeros((nodes.size, 3)))
        C = enu2c(np.full(g.cells.num, 1e7), np.full(g.cells.num, 0.3))
        mech = MechanicModel(g, C, alpha=1.0, el_bc=el_bc)

        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=2e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=90 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)

        model = MechBlackOilModel(g, rock, fluid, w, mech)
        st = State(
            pressure=np.full(g.cells.num, 100.0 * BARSA),
            sW=np.full(g.cells.num, 0.2),
            qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -2e-3]),
            bhp=np.array([100.0 * BARSA, 90.0 * BARSA]),
        )
        st = model.seed_state(st)

        # a free surface node above the injector
        surface = np.flatnonzero(
            (pts[:, 2] < tol) & (pts[:, 0] < tol) & (pts[:, 1] < tol)
        )
        # that corner is on the roller planes; take an interior surface node
        interior_surface = np.flatnonzero(
            (pts[:, 2] < tol) & (pts[:, 0] > tol) & (pts[:, 0] < phys[0] - tol)
            & (pts[:, 1] > tol) & (pts[:, 1] < phys[1] - tol)
        )
        node = int(interior_surface[0])
        return model, st, node

    DTS = [10 * DAY, 10 * DAY]

    def test_control_derivative_is_minus_one():
        """dR/du claimed analytically, checked against the model."""
        model, st, _ = build()
        u = np.array([2e-3, 90 * BARSA])
        _set_controls(model, u)
        st1, rep = NonLinearSolver().solve_ministep(model, st, st.copy(),
                                                    DTS[0])
        assert rep.converged

        n_cols, _, ctrl_row, n_rows = _layout(model)
        analytic = _jac_wrt_controls(model).toarray()

        # finite-difference the residual with respect to each control
        base = np.concatenate(
            [value(e) for e in model.get_equations(st, st1, DTS[0])])
        fd = np.zeros((n_rows, u.size))
        for k in range(u.size):
            up = u.copy()
            h = 1e-6 * max(abs(u[k]), 1.0)
            up[k] += h
            _set_controls(model, up)
            pert = np.concatenate(
                [value(e) for e in model.get_equations(st, st1, DTS[0])])
            fd[:, k] = (pert - base) / h
            _set_controls(model, u)

        assert np.allclose(analytic, fd, atol=1e-6), (
            f"dR/du wrong: max diff {np.abs(analytic - fd).max():.2e}"
        )
        print("ok test_control_derivative_is_minus_one")

    def test_jac_wrt_previous_matches_finite_difference():
        """dR/dx_{n-1} - including the previous displacement, which enters
        through the previous pore volume and is the block a flow-only adjoint
        would not have."""
        model, st, _ = build()
        st1, rep = NonLinearSolver().solve_ministep(model, st, st.copy(),
                                                    DTS[0])
        assert rep.converged

        analytic = _jac_wrt_previous(model, st, st1, DTS[0]).toarray()

        def residual(prev: State) -> np.ndarray:
            return np.concatenate(
                [value(e) for e in model.get_equations(prev, st1, DTS[0])])

        base = residual(st)
        n_cols = analytic.shape[1]
        nc, nw = model.G.cells.num, model.facility.n_wells
        fd = np.zeros_like(analytic)

        fields = [("pressure", nc), ("sW", nc), ("qWs", nw), ("qOs", nw),
                  ("bhp", nw), ("xd", model.mech.n_free)]
        col = 0
        for name, n in fields:
            for i in range(n):
                pert = st.copy()
                scale = max(abs(float(pert[name][i])), 1.0)
                h = 1e-7 * scale
                pert[name] = pert[name].astype(float)
                pert[name][i] += h
                if name == "xd":
                    pert["u"] = model.mech.full_displacement(pert["xd"])
                pert.clear_cache()
                fd[:, col] = (residual(pert) - base) / h
                col += 1
        assert col == n_cols

        scale = max(np.abs(analytic).max(), 1e-30)
        err = np.abs(analytic - fd).max() / scale
        assert err < 1e-5, f"dR/dx_prev off by {err:.2e} relative"
        print(f"ok test_jac_wrt_previous_matches_finite_difference "
              f"(rel {err:.1e})")

    def test_uplift_gradient_matches_finite_difference():
        """THE test: the adjoint gradient of the uplift is the derivative of
        the coupled forward model."""
        model, st, node = build()
        obj = uplift_objective(model, node)
        u0 = np.array([[2e-3, 90 * BARSA], [2e-3, 90 * BARSA]])

        res = compute_gradient_adjoint(model, st, DTS, u0, obj)
        assert res.value != 0.0, "the surface must move at all"

        fd = np.zeros_like(res.gradient)
        for n in range(u0.shape[0]):
            for k in range(u0.shape[1]):
                h = 1e-6 * abs(u0[n, k])
                for sgn in (+1, -1):
                    up = u0.copy()
                    up[n, k] += sgn * h
                    states = simulate_forward(model, st, DTS, up)
                    j = evaluate_objective(obj, states, DTS)
                    fd[n, k] += sgn * j
                fd[n, k] /= 2 * h

        scale = max(np.abs(fd).max(), 1e-30)
        err = np.abs(res.gradient - fd).max() / scale
        assert err < 1e-5, (
            f"adjoint gradient off finite differences by {err:.2e}\n"
            f"adjoint:\n{res.gradient}\nfd:\n{fd}"
        )
        print(f"ok test_uplift_gradient_matches_finite_difference "
              f"(rel {err:.1e})")

    def test_injection_lifts_the_surface():
        """The sign the whole exercise depends on: injecting more raises the
        ground, so the gradient of uplift w.r.t. the injection rate is > 0.

        A flipped sign still passes a gradient check - it agrees with the
        finite differences of a wrong objective - so it needs its own test.
        """
        model, st, node = build()
        obj = uplift_objective(model, node)
        u0 = np.array([[2e-3, 90 * BARSA], [2e-3, 90 * BARSA]])
        res = compute_gradient_adjoint(model, st, DTS, u0, obj)

        # uplift is the -z displacement (z points down), so more injection
        # must make xd at that node *more negative*: gradient < 0.
        assert np.all(res.gradient[:, 0] < 0), (
            f"injecting must lift the surface: {res.gradient[:, 0]}"
        )
        # and raising the producer's bhp (producing less) also lifts it
        assert np.all(res.gradient[:, 1] < 0), res.gradient[:, 1]
        print("ok test_injection_lifts_the_surface")

    test_control_derivative_is_minus_one()
    test_jac_wrt_previous_matches_finite_difference()
    test_uplift_gradient_matches_finite_difference()
    test_injection_lifts_the_surface()
    print("\nAll tests passed!")
