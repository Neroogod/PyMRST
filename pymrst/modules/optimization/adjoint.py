"""Adjoint gradients and rate optimization (MRST optimization/, ad-core).

Ports computeGradientAdjointAD. A reservoir simulation is a chain of
implicit steps

    R_n(x_n, x_{n-1}, u_n) = 0,     n = 1..N

and the thing worth optimizing is a sum over the schedule, J = sum_n
J_n(x_n, u_n) - typically net present value. Differentiating J by every
control with finite differences costs one extra simulation per control.
The adjoint costs one *linear* solve per step regardless of how many
controls there are, which is what makes optimizing a real schedule
affordable.

Sweeping backwards, the adjoint variable solves

    (dR_n/dx_n)^T lam_n = -(dJ_n/dx_n)^T - (dR_{n+1}/dx_n)^T lam_{n+1}

with lam_{N+1} = 0, and each step then contributes

    dJ/du_n = dJ_n/du_n + (dR_n/du_n)^T lam_n.

dR_n/dx_n is the Newton Jacobian the forward solve already builds, so the
backward sweep reuses the same assembly with a different variable seeded.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)
from pymrst.utils.adi import ADI, init_variables_adi, value

# An objective term: given the model, the AD state variables of step n, the
# controls, dt and the step index, return an ADI scalar.
Objective = Callable[
    [TwoPhaseOilWaterModelAD, ADI, ADI, ADI, ADI, ADI, np.ndarray, float, int],
    ADI,
]


@dataclass
class AdjointResult:
    """Objective value, its gradient, and the states it was evaluated at."""

    value: float
    gradient: np.ndarray          # (n_steps, n_wells)
    states: List[OWState]         # states[0] is the initial state


def _set_controls(model: TwoPhaseOilWaterModelAD, u: np.ndarray) -> None:
    for w, val in zip(model.wells, np.asarray(u, float).ravel()):
        w.val = float(val)


def _stack(eq: ADI) -> sp.csr_matrix:
    """The Jacobian of one equation block as a single sparse matrix."""
    return sp.hstack(eq.jac, format="csr")


def _jac_wrt_current(model, st_prev: OWState, st: OWState,
                     dt: float) -> sp.csc_matrix:
    """dR_n / dx_n - the same matrix the forward Newton solves with."""
    p, sw, qws, qos, bhp = init_variables_adi(
        st.pressure, st.sW, st.qWs, st.qOs, st.bhp)
    eqs = model.assemble(st_prev, p, sw, qws, qos, bhp, dt)
    return sp.vstack([_stack(e) for e in eqs], format="csc")


def _jac_wrt_previous(model, st_prev: OWState, st: OWState,
                      dt: float) -> sp.csc_matrix:
    """dR_n / dx_{n-1}.

    Seeds the *previous* state as the AD variable and leaves the current
    state as plain numbers. Only the accumulation term sees x_{n-1}, so the
    well rows come out empty; they are padded with zeros to keep the block
    structure of the full residual.
    """
    p0, sw0, qws0, qos0, bhp0 = init_variables_adi(
        st_prev.pressure, st_prev.sW, st_prev.qWs, st_prev.qOs, st_prev.bhp)
    st0_ad = OWState(pressure=p0, sW=sw0, qWs=qws0, qOs=qos0, bhp=bhp0)

    water, oil, eq_w, eq_o, ctrl = model.assemble(
        st0_ad, st.pressure, st.sW, st.qWs, st.qOs, st.bhp, dt)

    n = 2 * model.g.cells.num + 3 * len(model.wells)
    blocks = [_stack(water), _stack(oil)]
    # the three well equations carry no dependence on the previous state
    n_well_rows = sum(np.asarray(value(e)).size for e in (eq_w, eq_o, ctrl))
    blocks.append(sp.csr_matrix((n_well_rows, n)))
    return sp.vstack(blocks, format="csc")


def _jac_wrt_controls(model: TwoPhaseOilWaterModelAD) -> sp.csc_matrix:
    """dR_n / du_n.

    The control target enters exactly one equation, linearly: the well's
    control residual is `bhp - target` or `qWs + qOs - target`, so the
    derivative is -1 in that row and zero everywhere else. `test_control_
    derivative_is_minus_one` checks this against the model rather than
    trusting the comment.
    """
    nc, nw = model.g.cells.num, len(model.wells)
    n = 2 * nc + 3 * nw
    rows = 2 * nc + 2 * nw + np.arange(nw)      # the control-equation rows
    return sp.csc_matrix(
        (-np.ones(nw), (rows, np.arange(nw))), shape=(n, nw))


def _objective_derivatives(model, objective: Objective, st: OWState,
                           u: np.ndarray, dt: float, n: int
                           ) -> Tuple[float, np.ndarray]:
    """J_n and dJ_n/dx_n (dJ_n/du_n is zero for the objectives here)."""
    p, sw, qws, qos, bhp = init_variables_adi(
        st.pressure, st.sW, st.qWs, st.qOs, st.bhp)
    j = objective(model, p, sw, qws, qos, bhp, u, dt, n)
    jv = float(np.asarray(value(j)).ravel()[0])
    djdx = np.concatenate(
        [np.asarray(b.todense()).ravel() if sp.issparse(b)
         else np.asarray(b).ravel() for b in j.jac])
    return jv, djdx


def simulate_forward(model: TwoPhaseOilWaterModelAD, state0: OWState,
                     dts: Sequence[float], controls: np.ndarray
                     ) -> List[OWState]:
    """Run the schedule, keeping every state (the adjoint needs them all)."""
    states = [state0]
    for n, dt in enumerate(dts):
        _set_controls(model, controls[n])
        st, ok, _ = model.step(states[-1], float(dt))
        if not ok:
            raise RuntimeError(f"forward step {n} did not converge")
        states.append(st)
    return states


def evaluate_objective(model: TwoPhaseOilWaterModelAD, objective: Objective,
                       states: Sequence[OWState], dts: Sequence[float],
                       controls: np.ndarray) -> float:
    """J = sum_n J_n, on an already-simulated schedule."""
    total = 0.0
    for n, dt in enumerate(dts):
        j = objective(model, states[n + 1].pressure, states[n + 1].sW,
                      states[n + 1].qWs, states[n + 1].qOs,
                      states[n + 1].bhp, controls[n], float(dt), n)
        total += float(np.asarray(value(j)).ravel()[0])
    return total


def compute_gradient_adjoint(
    model: TwoPhaseOilWaterModelAD,
    state0: OWState,
    dts: Sequence[float],
    controls: np.ndarray,
    objective: Objective,
    states: Optional[Sequence[OWState]] = None,
) -> AdjointResult:
    """dJ/du for every control of every step, in one backward sweep."""
    controls = np.atleast_2d(np.asarray(controls, float))
    nsteps, nw = controls.shape
    assert len(dts) == nsteps, "one control row per timestep"
    assert nw == len(model.wells), "one control per well"

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

        jn, djdx = _objective_derivatives(
            model, objective, st, controls[n], dt, n)
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


def npv_objective(oil_price: float = 60.0, water_cost: float = 6.0,
                  water_inj_cost: float = 6.0,
                  discount_rate: float = 0.0,
                  dts: Optional[Sequence[float]] = None,
                  time_unit: float = 365.0 * 86400.0) -> Objective:
    """Net present value (MRST's NPVOW).

    Produced oil earns `oil_price` per surface m3, produced water costs
    `water_cost` to handle, and injected water costs `water_inj_cost`.

    Rates are signed: production is negative, injection positive. So a
    producer earns -oil_price * qOs (positive, since qOs < 0) and pays
    water_cost * qWs (negative, a cost), while an injector pays
    -water_inj_cost * qWs. That is MRST's NPVOW term for term.

    Discounting needs elapsed time, so pass `dts` to use `discount_rate`.
    """
    if discount_rate and dts is None:
        raise ValueError("discounting needs the timestep sizes; pass dts=")
    elapsed = (np.cumsum(np.asarray(dts, float)) if dts is not None
               else None)

    def obj(model, p, sw, qws, qos, bhp, u, dt, n):
        j = None
        for k in range(len(model.wells)):
            qw, qo = qws[k:k + 1], qos[k:k + 1]
            producing = float(value(qo)[0]) + float(value(qw)[0]) < 0.0
            if producing:
                term = (-oil_price) * qo + water_cost * qw
            else:
                term = (-water_inj_cost) * qw
            j = term if j is None else j + term

        disc = ((1.0 + discount_rate) ** (-(elapsed[n] / time_unit))
                if discount_rate else 1.0)
        return j * (dt * disc)

    return obj


def optimize_rates(
    model: TwoPhaseOilWaterModelAD,
    state0: OWState,
    dts: Sequence[float],
    controls: np.ndarray,
    objective: Objective,
    tunable: np.ndarray,
    bounds: Tuple[float, float],
    iterations: int = 10,
    step: float = 0.5,
) -> Tuple[np.ndarray, List[float]]:
    """Projected gradient ascent on the tunable controls.

    `tunable` is a boolean mask the shape of `controls`. The step is scaled
    by the control range and backtracked when the objective fails to
    improve, which is enough to show the gradient points uphill without
    dragging in a full line-search library.
    """
    lo, hi = bounds
    u = np.array(controls, float, copy=True)
    res = compute_gradient_adjoint(model, state0, dts, u, objective)
    history = [res.value]
    span = hi - lo

    for _ in range(iterations):
        g = res.gradient * tunable
        gmax = np.abs(g).max()
        if gmax == 0.0:
            break

        trial_step = step
        for _ in range(8):                       # backtrack until it improves
            cand = np.clip(u + trial_step * span * g / gmax, lo, hi)
            cand = np.where(tunable, cand, u)
            try:
                cres = compute_gradient_adjoint(model, state0, dts, cand,
                                                objective)
            except RuntimeError:                 # a step that will not converge
                trial_step *= 0.5
                continue
            if cres.value > res.value:
                u, res = cand, cres
                break
            trial_step *= 0.5
        else:
            break
        history.append(res.value)

    return u, history


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model(nx=6, ny=6):
        g = compute_geometry(cart_grid([nx, ny, 1], [300.0, 300.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_simple_adi_fluid(
            mu=[1 * CENTI * POISE, 5 * CENTI * POISE],
            rho=[1000.0, 800.0], n=[2.0, 2.0],
            c=[1e-10, 1e-9], p_ref=200 * BARSA,
        )
        prod = g.cells.num - 1
        W = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=60.0 / DAY, compi=np.array([1.0, 0.0]), name="I")
        W = add_well(W, g, rock, np.array([prod]), well_type="bhp",
                     val=200 * BARSA, compi=np.array([1.0, 0.0]), name="P")
        model = TwoPhaseOilWaterModelAD(g, rock, fluid, W)
        return model, model.initial_state(200 * BARSA, 0.2)

    def test_control_derivative_is_minus_one():
        """dR/du really is -1 in the control row, as _jac_wrt_controls says."""
        model, st0 = _model(4, 4)
        dt = 30.0 * DAY
        st, ok, _ = model.step(st0, dt)
        assert ok

        def residual(u):
            _set_controls(model, u)
            p, sw, qws, qos, bhp = init_variables_adi(
                st.pressure, st.sW, st.qWs, st.qOs, st.bhp)
            eqs = model.assemble(st0, p, sw, qws, qos, bhp, dt)
            return np.concatenate([np.asarray(value(e)).ravel() for e in eqs])

        u0 = np.array([w.val for w in model.wells], float)
        analytic = np.asarray(_jac_wrt_controls(model).todense())
        for k in range(len(model.wells)):
            h = 1e-6 * max(abs(u0[k]), 1.0)
            up, um = u0.copy(), u0.copy()
            up[k] += h
            um[k] -= h
            fd = (residual(up) - residual(um)) / (2 * h)
            assert np.allclose(fd, analytic[:, k], atol=1e-6), (
                f"well {k}: dR/du does not match the -1 in the control row")
        _set_controls(model, u0)
        print("ok test_control_derivative_is_minus_one")

    def test_jac_wrt_previous_matches_finite_difference():
        """dR/dx_{n-1} from AD equals a finite difference of the residual."""
        model, st0 = _model(4, 4)
        dt = 20.0 * DAY
        st, ok, _ = model.step(st0, dt)
        assert ok

        b = np.asarray(_jac_wrt_previous(model, st0, st, dt).todense())

        def residual(prev: OWState):
            p, sw, qws, qos, bhp = (st.pressure, st.sW, st.qWs, st.qOs, st.bhp)
            eqs = model.assemble(prev, p, sw, qws, qos, bhp, dt)
            return np.concatenate([np.asarray(value(e)).ravel() for e in eqs])

        nc = model.g.cells.num
        rng = np.random.default_rng(0)
        for col in rng.choice(2 * nc, size=6, replace=False):
            h = 1e-6 if col >= nc else 1.0        # saturation vs pressure scale
            up, dn = st0.copy(), st0.copy()
            if col < nc:
                up.pressure[col] += h
                dn.pressure[col] -= h
            else:
                up.sW[col - nc] += h
                dn.sW[col - nc] -= h
            fd = (residual(up) - residual(dn)) / (2 * h)
            err = np.abs(fd - b[:, col]).max() / max(np.abs(b[:, col]).max(), 1e-30)
            assert err < 1e-5, f"column {col}: relative error {err:.2e}"
        print("ok test_jac_wrt_previous_matches_finite_difference")

    def test_npv_sign_convention():
        """Producing oil earns money; injecting and handling water costs it.

        The whole optimization is meaningless if this sign is wrong, and a
        flipped sign still passes a gradient check, so pin it down here.
        """
        model, st0 = _model(5, 5)
        dts = [30.0 * DAY] * 2
        u = np.array([[40.0 / DAY, 200 * BARSA]] * len(dts))
        states = simulate_forward(model, st0, dts, u)

        # the producer really is producing (negative rates)
        assert states[-1].qOs[1] < 0.0 and states[-1].qWs[0] > 0.0

        free = evaluate_objective(
            model, npv_objective(oil_price=60.0, water_cost=0.0,
                                 water_inj_cost=0.0), states, dts, u)
        assert free > 0.0, "producing oil must earn money"

        # charging for the water can only reduce NPV
        charged = evaluate_objective(
            model, npv_objective(oil_price=60.0, water_cost=6.0,
                                 water_inj_cost=12.0), states, dts, u)
        assert charged < free, "water costs must reduce NPV"

        # raising the oil price raises NPV proportionally
        dearer = evaluate_objective(
            model, npv_objective(oil_price=120.0, water_cost=0.0,
                                 water_inj_cost=0.0), states, dts, u)
        assert np.isclose(dearer, 2.0 * free, rtol=1e-12)
        print(f"ok test_npv_sign_convention (NPV {charged:.3e} with costs, "
              f"{free:.3e} without)")

    def test_gradient_matches_finite_difference():
        """The adjoint gradient equals the finite-difference gradient.

        This is the test that matters: the adjoint is exact or it is not.
        """
        model, st0 = _model(5, 5)
        dts = [25.0 * DAY] * 3
        nw = len(model.wells)
        u0 = np.array([[60.0 / DAY, 200 * BARSA]] * len(dts))
        obj = npv_objective()

        res = compute_gradient_adjoint(model, st0, dts, u0, obj)

        fd = np.zeros_like(u0)
        for n in range(len(dts)):
            for k in range(nw):
                h = 1e-6 * max(abs(u0[n, k]), 1.0)
                up, dn = u0.copy(), u0.copy()
                up[n, k] += h
                dn[n, k] -= h
                ju = evaluate_objective(
                    model, obj, simulate_forward(model, st0, dts, up), dts, up)
                jd = evaluate_objective(
                    model, obj, simulate_forward(model, st0, dts, dn), dts, dn)
                fd[n, k] = (ju - jd) / (2 * h)

        scale = np.abs(fd).max()
        err = np.abs(res.gradient - fd).max() / scale
        assert err < 1e-5, (
            f"adjoint gradient differs from finite differences by {err:.2e}\n"
            f"adjoint:\n{res.gradient}\nfinite diff:\n{fd}")
        print(f"ok test_gradient_matches_finite_difference "
              f"(NPV = {res.value:.3e}, max rel err = {err:.2e})")

    def test_adjoint_is_cheaper_than_finite_differences():
        """One backward sweep, not one simulation per control."""
        model, st0 = _model(4, 4)
        dts = [20.0 * DAY] * 4
        u0 = np.array([[50.0 / DAY, 200 * BARSA]] * len(dts))

        calls = {"n": 0}
        orig_step = model.step

        def counting_step(state, dt):
            calls["n"] += 1
            return orig_step(state, dt)

        model.step = counting_step
        compute_gradient_adjoint(model, st0, dts, u0, npv_objective())
        model.step = orig_step

        # exactly one forward simulation; finite differences would need
        # 2 * n_steps * n_wells = 16 more
        assert calls["n"] == len(dts), calls["n"]
        print(f"ok test_adjoint_is_cheaper_than_finite_differences "
              f"({calls['n']} steps vs {2 * len(dts) * 2 * len(dts)} for FD)")

    def test_optimize_rates_increases_npv():
        """Gradient ascent on the injection rate improves NPV."""
        model, st0 = _model(5, 5)
        dts = [30.0 * DAY] * 3
        obj = npv_objective(oil_price=60.0, water_cost=6.0,
                            water_inj_cost=12.0)

        u0 = np.array([[20.0 / DAY, 200 * BARSA]] * len(dts))
        tunable = np.zeros_like(u0, dtype=bool)
        tunable[:, 0] = True                       # only the injection rate

        u, hist = optimize_rates(model, st0, dts, u0, obj, tunable,
                                 bounds=(5.0 / DAY, 200.0 / DAY),
                                 iterations=6, step=0.4)
        assert hist[-1] > hist[0], hist
        assert np.all(np.diff(hist) > 0), "ascent must be monotone"
        # the producer's bhp was not tunable, so it must be untouched
        assert np.allclose(u[:, 1], u0[:, 1])
        print(f"ok test_optimize_rates_increases_npv "
              f"(NPV {hist[0]:.3e} -> {hist[-1]:.3e}, "
              f"rate {u0[0, 0] * DAY:.0f} -> {u[0, 0] * DAY:.0f} m3/day)")

    test_control_derivative_is_minus_one()
    test_jac_wrt_previous_matches_finite_difference()
    test_npv_sign_convention()
    test_gradient_matches_finite_difference()
    test_adjoint_is_cheaper_than_finite_differences()
    test_optimize_rates_increases_npv()
    print("\nAll tests passed!")
