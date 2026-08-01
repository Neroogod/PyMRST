"""Fetkovich analytic aquifers (MRST ad-core `AquiferModel`).

An analytic aquifer stands in for the water body around a reservoir without
gridding it. Each aquifer is one lumped tank of known pressure and volume,
connected to a set of boundary cells; it pushes water in as the reservoir is
produced and its own pressure falls in response. This is the ECLIPSE Fetkovich
model (`AQUFETP`/`AQUANCON`), and MRST implements it in `AquiferModel.m`.

The influx through one connection is

    q = alpha * J * (p_aq - p_w + rho_w * g * (z_conn - z_aq)) * coef

with the Fetkovich time-constant relaxation

    Tc   = C * V_aq / J
    coef = (1 - exp(-dt/Tc)) / (dt/Tc)      (coef -> 1 as dt -> 0)

`J` is the connection productivity index, `alpha` its share of the aquifer's
influx area (the shares sum to one per aquifer), `C` the aquifer
compressibility and `V_aq` its volume. After each converged step the aquifer
pressure and volume are updated explicitly from the total flux it delivered:

    Q      = dt * sum_conn q          (per aquifer)
    p_aq  -= Q / (C * V_aq)
    V_aq  -= Q

**Units.** `q` is a surface-volume water rate. MRST's conservation equations
are mass-based, so it adds `-q * rhoWS` to the water equation; PyMRST's
equations are on a surface-volume basis (see `ad_core/components.py`), so the
contribution is simply `-q` - the two differ by the one constant that separates
the two conventions, and give the same solution.
"""

from typing import Any, List, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value


class AquiferConnections:
    """The static geometry of one or more Fetkovich aquifers.

    All connection arrays have length `n_conn`; `aquid` (0-based) says which
    aquifer each connection belongs to. Per-aquifer compressibility `c_aq` has
    length `n_aq` and is used by the after-convergence update, while the
    per-connection `c` (its aquifer's value broadcast to each connection) is
    used by the flux relaxation.
    """

    def __init__(
        self,
        aquid: np.ndarray,
        cells: np.ndarray,
        j: np.ndarray,
        c: np.ndarray,
        alpha: np.ndarray,
        depth_conn: np.ndarray,
        depth_aq: np.ndarray,
        c_aq: np.ndarray,
    ):
        self.aquid = np.asarray(aquid, dtype=np.int64).ravel()
        self.cells = np.asarray(cells, dtype=np.int64).ravel()
        self.j = np.asarray(j, dtype=float).ravel()
        self.c = np.asarray(c, dtype=float).ravel()
        self.depth_conn = np.asarray(depth_conn, dtype=float).ravel()
        self.depth_aq = np.asarray(depth_aq, dtype=float).ravel()
        self.c_aq = np.asarray(c_aq, dtype=float).ravel()

        self.n_conn = self.aquid.size
        self.n_aq = int(self.aquid.max()) + 1 if self.n_conn else 0

        # aquid2conn[c, a] = 1 when connection c belongs to aquifer a. It maps
        # a per-aquifer value to each connection (matrix @ vec) and sums
        # per-connection fluxes back to each aquifer (matrix.T @ vec).
        self.aquid2conn = sp.csr_matrix(
            (np.ones(self.n_conn), (np.arange(self.n_conn), self.aquid)),
            shape=(self.n_conn, self.n_aq),
        )

        # Normalise the influx shares to sum to one per aquifer, as MRST's
        # processAquifer does (alpha = alpha / sum_over_its_connections).
        alpha = np.asarray(alpha, dtype=float).ravel()
        share_sum = self.aquid2conn @ (self.aquid2conn.T @ alpha)
        self.alpha = alpha / share_sum

    def map_to_conn(self, per_aquifer: np.ndarray) -> np.ndarray:
        """Broadcast a per-aquifer array to its connections."""
        return self.aquid2conn @ np.asarray(per_aquifer, dtype=float)

    def sum_to_aquifer(self, per_conn: np.ndarray) -> np.ndarray:
        """Sum a per-connection array back to each aquifer."""
        return self.aquid2conn.T @ np.asarray(per_conn, dtype=float)


def fetkovich_coefficient(c: np.ndarray, v_aq_conn: np.ndarray,
                          j: np.ndarray, dt: float) -> np.ndarray:
    """The Fetkovich relaxation factor `coef`, per connection.

    Over a finite step the influx is not the instantaneous rate but its
    integral, which for a first-order tank gives the `(1-exp(-dt/Tc))/(dt/Tc)`
    factor. It tends to one as `dt -> 0` and decays as the step grows past the
    aquifer's time constant `Tc = C V / J`.
    """
    if dt == 0.0:
        return np.ones_like(j)
    tc = c * v_aq_conn / j
    x = dt / tc
    return (1.0 - np.exp(-x)) / x


def compute_aquifer_fluxes(conn: AquiferConnections, p_aq: np.ndarray,
                           v_aq: np.ndarray, p_w: Any, b_w: Any,
                           rho_ws: float, g: float, dt: float) -> Any:
    """Water influx through every connection (MRST `computeAquiferFluxes`).

    `p_w` and `b_w` are the water phase pressure and shrinkage factor at the
    connection cells; they may be AD, in which case the returned flux carries
    the derivatives that couple the aquifer into the reservoir Jacobian. `p_aq`
    and `v_aq` are the (explicit, per-aquifer) aquifer pressure and volume.
    """
    p_aq_c = conn.map_to_conn(p_aq)
    v_aq_c = conn.map_to_conn(v_aq)
    coef = fetkovich_coefficient(conn.c, v_aq_c, conn.j, dt)
    rho_w = b_w * rho_ws
    head = rho_w * (g * (conn.depth_conn - conn.depth_aq))
    return conn.alpha * conn.j * (p_aq_c - p_w + head) * coef


def update_aquifer_state(conn: AquiferConnections, p_aq: np.ndarray,
                         v_aq: np.ndarray, q: np.ndarray,
                         dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Advance aquifer pressure and volume after a converged step.

    `q` is the converged connection flux (values, not AD). The aquifer loses
    the water it delivered, and its pressure falls in proportion through its
    compressibility (MRST `updateAfterConvergence`).
    """
    q_total = dt * conn.sum_to_aquifer(np.asarray(value(q), dtype=float))
    p_new = np.asarray(p_aq, dtype=float) - q_total / (conn.c_aq * v_aq)
    v_new = np.asarray(v_aq, dtype=float) - q_total
    return p_new, v_new


def equilibrium_aquifer_pressure(conn: AquiferConnections, p_w: np.ndarray,
                                 b_w: np.ndarray, rho_ws: float, g: float,
                                 ) -> np.ndarray:
    """The aquifer pressures that make the net influx vanish (MRST
    `computeInitAquifer`).

    At `dt = 0` the flux is affine in `p_aq`: `q = M p_aq + r`. The pressures
    that zero it in the least-squares sense are `p_aq = -(M'M)^{-1} M'r`, which
    for connections that do not share cells is just each aquifer's
    area-weighted mean connection potential.
    """
    p_w = np.asarray(p_w, dtype=float)
    b_w = np.asarray(b_w, dtype=float)
    w = conn.alpha * conn.j
    head = b_w * rho_ws * (g * (conn.depth_conn - conn.depth_aq))
    r = w * (head - p_w)
    m = sp.diags(w) @ conn.aquid2conn
    mtm = (m.T @ m).toarray()
    return np.linalg.solve(mtm, -(m.T @ r))


class AquiferModel:
    """A set of Fetkovich aquifers attached to a reservoir model.

    Mirrors MRST's `AquiferModel`: it holds the connection geometry and the
    initial aquifer state, contributes an influx to the water equation, and
    updates its own pressure and volume once a step has converged. The reservoir
    model calls it; the aquifer pressure/volume live on the reservoir `State`
    under `aquifer_p`/`aquifer_V` so they survive Newton iterations and
    timestep cuts.
    """

    water_index: int = 0

    def __init__(self, connections: AquiferConnections,
                 pressures: np.ndarray, volumes: np.ndarray):
        self.connections = connections
        self.pressures = np.asarray(pressures, dtype=float).ravel()
        self.volumes = np.asarray(volumes, dtype=float).ravel()

    def init_state(self, state: State) -> State:
        """Seed a reservoir state with this aquifer's pressure and volume."""
        state["aquifer_p"] = self.pressures.copy()
        state["aquifer_V"] = self.volumes.copy()
        return state

    def _water_props(self, model, state: State) -> Tuple[Any, Any]:
        """Water phase pressure and shrinkage factor at the connection cells."""
        cells = self.connections.cells
        p_phase = model.get_props(state, "PhasePressures")
        b = model.get_props(state, "ShrinkageFactors")
        return p_phase[self.water_index][cells], b[self.water_index][cells]

    def fluxes(self, model, state: State, p_aq: np.ndarray,
               v_aq: np.ndarray, dt: float) -> Any:
        """Connection fluxes at `state` (AD if `state` carries AD variables)."""
        p_w, b_w = self._water_props(model, state)
        rho_ws = float(model.fluid.rho_s[self.water_index])
        return compute_aquifer_fluxes(self.connections, p_aq, v_aq, p_w, b_w,
                                      rho_ws, model.gravity_magnitude, dt)

    def add_contribution(self, model, eqs: List[Any], ad_state: State,
                         p_aq: np.ndarray, v_aq: np.ndarray,
                         dt: float) -> List[Any]:
        """Subtract the aquifer influx from the water equation.

        A single connection cell must appear once per aquifer set, matching
        MRST, which indexes the equation directly rather than accumulating.
        """
        cells = self.connections.cells
        assert cells.size == np.unique(cells).size, (
            "aquifer connections must be distinct cells"
        )
        q = self.fluxes(model, ad_state, p_aq, v_aq, dt)
        # The influx is a surface-volume water rate. In the mass basis it has to
        # be weighed by the water's surface density before it can enter a mass
        # equation - which is exactly the `rhoWS` MRST multiplies by here, and
        # exactly what the surface-volume basis does not need.
        q = model._basis_scale()[self.water_index] * q
        eq = eqs[self.water_index]
        eq[cells] = eq[cells] - q
        return eqs

    def after_convergence(self, model, state: State, p0: np.ndarray,
                          v0: np.ndarray, dt: float,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """New aquifer pressure and volume from the converged flux."""
        q = value(self.fluxes(model, state, p0, v0, dt))
        return update_aquifer_state(self.connections, p0, v0, q, dt)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _single_aquifer(cells, j=1e-10, c=1e-9, alpha=None, depth_conn=0.0,
                        depth_aq=0.0, p=250 * BARSA, v=1e7):
        cells = np.atleast_1d(cells)
        n = cells.size
        alpha = np.ones(n) if alpha is None else np.asarray(alpha, float)
        conn = AquiferConnections(
            aquid=np.zeros(n), cells=cells, j=np.full(n, j),
            c=np.full(n, c), alpha=alpha,
            depth_conn=np.full(n, depth_conn), depth_aq=np.full(n, depth_aq),
            c_aq=np.array([c]),
        )
        return AquiferModel(conn, np.array([p]), np.array([v]))

    def test_flux_matches_hand_computation():
        """q = alpha J (p_aq - p_w) with coef = 1 at dt = 0, no gravity."""
        aq = _single_aquifer(np.array([0]), j=2e-10, p=250 * BARSA, v=1e7)
        p_w = np.array([200.0 * BARSA])
        b_w = np.array([1.0])
        q = compute_aquifer_fluxes(aq.connections, aq.pressures, aq.volumes,
                                   p_w, b_w, rho_ws=1000.0, g=0.0, dt=0.0)
        want = 1.0 * 2e-10 * (250e5 - 200e5)
        assert np.isclose(q[0], want, rtol=1e-14), (q[0], want)
        print("ok test_flux_matches_hand_computation")

    def test_coefficient_limits():
        """coef -> 1 as dt -> 0, and decays below 1 for a finite step."""
        c = np.array([1e-9])
        v = np.array([1e7])
        j = np.array([1e-10])
        assert np.isclose(fetkovich_coefficient(c, v, j, 0.0)[0], 1.0)
        small = fetkovich_coefficient(c, v, j, 1.0 * DAY)[0]
        big = fetkovich_coefficient(c, v, j, 1000.0 * DAY)[0]
        assert 0.0 < big < small < 1.0, (small, big)
        print("ok test_coefficient_limits")

    def test_gravity_head_shifts_the_drive():
        """A connection below the aquifer datum feels a heavier water column
        and so a larger driving potential."""
        deep = _single_aquifer(np.array([0]), depth_conn=50.0, depth_aq=0.0)
        p_w = np.array([250.0 * BARSA])           # equal to aquifer pressure
        b_w = np.array([1.0])
        q0 = compute_aquifer_fluxes(deep.connections, deep.pressures,
                                    deep.volumes, p_w, b_w, 1000.0, 0.0, 0.0)
        qg = compute_aquifer_fluxes(deep.connections, deep.pressures,
                                    deep.volumes, p_w, b_w, 1000.0, 9.80665, 0.0)
        assert np.isclose(q0[0], 0.0) and qg[0] > 0.0, (q0[0], qg[0])
        print("ok test_gravity_head_shifts_the_drive")

    def test_equilibrium_pressure_zeros_a_single_connection():
        """With one connection the equilibrium pressure zeros the flux exactly."""
        conn = AquiferConnections(
            aquid=np.array([0]), cells=np.array([0]), j=np.array([2e-10]),
            c=np.array([1e-9]), alpha=np.array([1.0]),
            depth_conn=np.array([30.0]), depth_aq=np.array([20.0]),
            c_aq=np.array([1e-9]),
        )
        p_w = np.array([210.0 * BARSA])
        b_w = np.array([1.0])
        p_eq = equilibrium_aquifer_pressure(conn, p_w, b_w, 1000.0, 9.80665)
        q = compute_aquifer_fluxes(conn, p_eq, np.array([1e7]), p_w, b_w,
                                   1000.0, 9.80665, 0.0)
        assert abs(float(q[0])) < 1e-12, q
        print("ok test_equilibrium_pressure_zeros_a_single_connection")

    def test_equilibrium_pressure_solves_normal_equations():
        """With several connections the flux cannot be zeroed everywhere; the
        equilibrium solve minimises ||q||^2, so the weighted residual M'q (the
        normal equations MRST's computeInitAquifer solves) vanishes instead."""
        conn = AquiferConnections(
            aquid=np.array([0, 0]), cells=np.array([0, 3]),
            j=np.array([1e-10, 3e-10]), c=np.full(2, 1e-9),
            alpha=np.array([1.0, 1.0]),
            depth_conn=np.array([10.0, 30.0]), depth_aq=np.full(2, 20.0),
            c_aq=np.array([1e-9]),
        )
        p_w = np.array([210.0 * BARSA, 205.0 * BARSA])
        b_w = np.array([1.0, 1.0])
        p_eq = equilibrium_aquifer_pressure(conn, p_w, b_w, 1000.0, 9.80665)
        q = compute_aquifer_fluxes(conn, p_eq, np.array([1e7]), p_w, b_w,
                                   1000.0, 9.80665, 0.0)
        weighted = conn.aquid2conn.T @ (conn.alpha * conn.j * q)
        assert abs(float(weighted[0])) < 1e-18, weighted
        print("ok test_equilibrium_pressure_solves_normal_equations")

    def test_update_drains_the_aquifer():
        """Delivering water lowers the aquifer's volume and pressure; the
        volume drop equals the integrated flux."""
        aq = _single_aquifer(np.array([0]), c=1e-9, v=1e7, p=250 * BARSA)
        q = np.array([2.0e-3])
        dt = 30.0 * DAY
        p_new, v_new = update_aquifer_state(aq.connections, aq.pressures,
                                            aq.volumes, q, dt)
        assert np.isclose(aq.volumes[0] - v_new[0], dt * q[0], rtol=1e-14)
        assert p_new[0] < aq.pressures[0]
        print("ok test_update_drains_the_aquifer")

    def test_two_aquifers_map_independently():
        """Connections route to the right aquifer through aquid2conn."""
        conn = AquiferConnections(
            aquid=np.array([0, 1, 1]), cells=np.array([0, 5, 9]),
            j=np.full(3, 1e-10), c=np.full(3, 1e-9),
            alpha=np.array([1.0, 1.0, 1.0]), depth_conn=np.zeros(3),
            depth_aq=np.zeros(3), c_aq=np.array([1e-9, 1e-9]),
        )
        assert conn.n_aq == 2
        # aquifer 1's two connections each get half the influx share
        assert np.allclose(conn.alpha, [1.0, 0.5, 0.5])
        totals = conn.sum_to_aquifer(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(totals, [1.0, 5.0])
        print("ok test_two_aquifers_map_independently")

    def _sealed_ow_model(aq, gravity=0.0):
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
        g = compute_geometry(cart_grid([3, 1, 1], [30.0, 10.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.3)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        model = GenericBlackOilModel(g, rock, fluid, wells=[],
                                     gravity=gravity, aquifer=aq)
        return g, model

    def _sealed_state(g, model, p0=200 * BARSA, sw0=0.5):
        st = State(pressure=np.full(g.cells.num, float(p0)),
                   sW=np.full(g.cells.num, float(sw0)),
                   qWs=np.zeros(0), qOs=np.zeros(0), bhp=np.zeros(0))
        return model.aquifer.init_state(st)

    def test_aquifer_drives_a_sealed_reservoir():
        """A higher-pressure aquifer pushes water into a sealed box: reservoir
        pressure rises, aquifer pressure falls, and the influx decays as they
        approach each other."""
        from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
        aq = _single_aquifer(np.array([0]), j=3e-14, c=1e-9, v=1e8,
                             p=260 * BARSA)
        g, model = _sealed_ow_model(aq)
        st = _sealed_state(g, model, p0=200 * BARSA)

        solver = NonLinearSolver()
        p_res = [float(st["pressure"].mean())]
        p_aq = [float(st["aquifer_p"][0])]
        q_drive = []
        for _ in range(8):
            # The flux that drives *this* step, at the start-of-step state.
            q_drive.append(float(value(model.aquifer.fluxes(
                model, st, st["aquifer_p"], st["aquifer_V"], 0.0))[0]))
            st, rep = solver.solve_ministep(model, st, st.copy(), 30.0 * DAY)
            assert rep.converged, rep.residuals[-1]
            p_res.append(float(st["pressure"].mean()))
            p_aq.append(float(st["aquifer_p"][0]))

        # The reservoir pressurises towards the aquifer and never falls back;
        # the aquifer depletes and never recovers; the driving flux tapers off
        # monotonically as the two pressures close.
        assert np.all(np.diff(p_res) > 0), p_res
        assert p_res[-1] > p_res[0] + 10 * BARSA, p_res
        assert np.all(np.diff(p_aq) < 0), p_aq
        assert p_res[-1] < p_aq[-1], (p_res[-1], p_aq[-1])
        assert np.all(np.diff(q_drive) < 0), q_drive
        print(f"ok test_aquifer_drives_a_sealed_reservoir "
              f"(p_res {p_res[0]/BARSA:.1f} -> {p_res[-1]/BARSA:.1f} bar)")

    def test_a_balanced_aquifer_does_not_move():
        """The check no reference can make: an aquifer at the reservoir's own
        pressure, no gravity, moves nothing at all."""
        from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
        aq = _single_aquifer(np.array([0]), j=5e-11, c=1e-9, v=1e8,
                             p=200 * BARSA)
        g, model = _sealed_ow_model(aq)
        st = _sealed_state(g, model, p0=200 * BARSA)

        solver = NonLinearSolver()
        for _ in range(3):
            st, rep = solver.solve_ministep(model, st, st.copy(), 100.0 * DAY)
            assert rep.converged
        assert abs(float(st["pressure"].mean()) - 200 * BARSA) < 1e-3
        assert abs(float(st["aquifer_V"][0]) - 1e8) < 1e-6
        print("ok test_a_balanced_aquifer_does_not_move")

    test_flux_matches_hand_computation()
    test_coefficient_limits()
    test_gravity_head_shifts_the_drive()
    test_equilibrium_pressure_zeros_a_single_connection()
    test_equilibrium_pressure_solves_normal_equations()
    test_update_drains_the_aquifer()
    test_two_aquifers_map_independently()
    test_aquifer_drives_a_sealed_reservoir()
    test_a_balanced_aquifer_does_not_move()
    print("\nAll tests passed!")
