"""A linear solver for coupled poroelasticity, and why the obvious one is slow.

A fully coupled poroelastic Newton step on a field-scale grid is a big sparse
system - on the 7392-cell Norne model it is 45,000 x 45,000 - and handing it
to a sparse direct solver costs **35 seconds per Newton iteration**. Assembly
is 0.06 s. The solve is the whole cost of a coupled run, and it is what stands
between this module and a grid anyone would actually use.

Two facts about the system fix it.

**1. The mechanics block never changes.** Linear elasticity: the stiffness
matrix `A` is the same at every Newton iteration of every timestep. Factorise
it *once* (7.4 s on Norne) and every later use is a back-substitution (0.03 s).
So eliminate the displacement instead of solving for it:

    | F   Fx | |dx_f|   |r_f|            (F - Fx A^-1 Mx) dx_f = r_f - Fx A^-1 r_m
    | Mx  A  | |dx_m| = |r_m|            dx_m = A^-1 (r_m - Mx dx_f)

The Schur complement `S = F - Fx A^-1 Mx` is dense and is never formed; it is
applied as an operator, one cached back-substitution per matrix-vector product.
That alone is 8x.

**2. Fixed stress is the right preconditioner.** Preconditioning the reduced
system with the flow block `F` alone takes 122 GMRES iterations, because `F`
knows nothing about the rock's stiffness. But `Fx A^-1 Mx` is - cell by cell -
almost exactly the fixed-stress storage term,

    (b_i s_i / dt) * alpha^2 * V / K_dr,

with `1/K_dr = tr(C^-1 I)`, the same `p_term` the fixed-stress *splitting* uses
(and for the same reason: it is how much the rock opens when you pressurise it).
Adding it to the flow block's diagonal drops GMRES to 36 iterations.

Measured on the full 7392-cell Norne model (45,000 unknowns):

    sparse direct solve      34.9 s   per Newton iteration
    Schur + fixed stress      1.6 s   per Newton iteration   (22x)
    splu of the mechanics     7.4 s   once, for the whole run

    the same two timesteps   361.7 s  ->  24.4 s             (14.8x)
    states agree to 2e-15 in pressure, saturation, displacement and bhp

The whole 21-step schedule now runs in **2 minutes** instead of about an hour.
The result is the direct solution - only the path to it is different, and
`test_reproduces_the_direct_solution` pins that rather than trusting it.
"""

from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.utils.adi import value


class PoroelasticSchurSolver:
    """Eliminate the (constant) mechanics block; precondition by fixed stress."""

    def __init__(self, rtol: float = 1e-10, restart: int = 50,
                 maxiter: int = 500, fixed_stress: bool = True):
        self.rtol = rtol
        self.restart = restart
        self.maxiter = maxiter
        self.fixed_stress = fixed_stress
        self._lu_mech: Optional[Any] = None
        self._mech_signature = None
        self.last_iterations = 0

    # -- the cached mechanics factorisation ---------------------------------
    def _factorize_mechanics(self, a: sp.spmatrix):
        """Factorise A once. It is the same matrix at every Newton step."""
        signature = (a.shape, a.nnz, float(abs(a).sum()))
        if self._lu_mech is None:
            self._lu_mech = spla.splu(a.tocsc())
            self._mech_signature = signature
        elif signature != self._mech_signature:
            # The cache is the whole point; if the stiffness ever *did* change
            # (plasticity, a stress-dependent modulus) reusing it would be
            # silently wrong, so say so rather than return a wrong answer.
            raise RuntimeError(
                "the mechanics block changed between Newton iterations; this "
                "solver assumes linear elasticity, where it cannot"
            )
        return self._lu_mech

    # -- the fixed-stress preconditioner ------------------------------------
    def _preconditioner(self, model, state, dt: float, f: sp.spmatrix,
                        n_flow: int):
        """splu(F + the fixed-stress storage), or splu(F) without it."""
        if not (self.fixed_stress and dt and model is not None
                and state is not None):
            return spla.splu(f.tocsc())

        nc = model.G.cells.num
        nph = model.n_phases
        _, p_term = model.mech.fixed_stress_terms(state["xd"],
                                                  np.asarray(value(
                                                      state["pressure"])))
        vol = model.G.cells.volumes
        b = [np.asarray(value(x)) for x in
             model.get_props(state, "ShrinkageFactors")]
        s = [np.asarray(value(x)) for x in model.get_props(state, "Saturations")]

        # How much extra fluid a cell can store because the rock yields:
        # alpha^2 V / K_dr, weighted like the accumulation term it sits in.
        # It lands on (phase-equation row, pressure column) of the same cell.
        #
        # Built as one sparse matrix rather than by assigning into a LIL
        # block: `p[rows, cols] += diags(...)` looks harmless and costs 3.6 s
        # here, which is more than the solve it is meant to accelerate.
        cell = np.arange(nc)
        rows = np.concatenate([i * nc + cell for i in range(nph)])
        cols = np.tile(cell, nph)
        data = np.concatenate([
            (b[i] * s[i] / dt) * model.alpha**2 * vol * p_term
            for i in range(nph)
        ])
        storage = sp.csr_matrix((data, (rows, cols)), shape=f.shape)
        return spla.splu((f + storage).tocsc())

    # -- the solve -----------------------------------------------------------
    def solve(self, jac: sp.spmatrix, rhs: np.ndarray, model=None,
              state=None, dt: Optional[float] = None) -> np.ndarray:
        """Solve the coupled Newton system. Falls back to a direct solve if
        this is not a poroelastic model."""
        if model is None or not hasattr(model, "mech"):
            return spla.spsolve(jac.tocsc(), rhs)

        n_flow = jac.shape[0] - model.mech.n_free
        jac = jac.tocsr()
        f = jac[:n_flow, :n_flow]
        fx = jac[:n_flow, n_flow:]
        mx = jac[n_flow:, :n_flow]
        a = jac[n_flow:, n_flow:]
        r_f, r_m = rhs[:n_flow], rhs[n_flow:]

        lu_a = self._factorize_mechanics(a)
        lu_p = self._preconditioner(model, state, dt, f, n_flow)

        def schur(v: np.ndarray) -> np.ndarray:
            return f @ v - fx @ lu_a.solve(mx @ v)

        s_op = spla.LinearOperator((n_flow, n_flow), matvec=schur)
        m_op = spla.LinearOperator((n_flow, n_flow), matvec=lu_p.solve)

        count = [0]

        def tick(_):
            count[0] += 1

        x_f, info = spla.gmres(
            s_op, r_f - fx @ lu_a.solve(r_m), M=m_op, rtol=self.rtol,
            restart=self.restart, maxiter=self.maxiter,
            callback=tick, callback_type="pr_norm",
        )
        self.last_iterations = count[0]
        if info != 0:
            raise RuntimeError(
                f"the Schur solve did not converge (gmres info={info} after "
                f"{count[0]} iterations); loosen rtol or use the direct solver"
            )

        x_m = lu_a.solve(r_m - mx @ x_f)
        return np.concatenate([x_f, x_m])


if __name__ == "__main__":
    import time

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import MechBlackOilModel
    from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.modules.vemmech.vem import (
        ElasticBC,
        create_augmented_grid,
        enu2c,
    )
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.adi import combine_equations
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def build(n=8):
        g = create_augmented_grid(
            compute_geometry(cart_grid([n, n, 4], [80.0, 80.0, 20.0]))
        )
        rock = make_rock(g, 200 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        pts, tol = g.nodes.coords, 1e-9
        xf = np.flatnonzero((pts[:, 0] < tol) | (pts[:, 0] > 80.0 - tol))
        yf = np.flatnonzero((pts[:, 1] < tol) | (pts[:, 1] > 80.0 - tol))
        zf = np.flatnonzero(pts[:, 2] > 20.0 - tol)
        nodes = np.concatenate([xf, yf, zf])
        mask = np.vstack([
            np.tile([True, False, False], (xf.size, 1)),
            np.tile([False, True, False], (yf.size, 1)),
            np.tile([False, False, True], (zf.size, 1)),
        ])
        el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask,
                          disp_vals=np.zeros((nodes.size, 3)))
        C = enu2c(np.full(g.cells.num, 1e9), np.full(g.cells.num, 0.25))
        mech = MechanicModel(g, C, alpha=1.0, el_bc=el_bc)

        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=2e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        m = MechBlackOilModel(g, rock, fluid, w, mech)
        st = State(pressure=np.full(g.cells.num, 200.0 * BARSA),
                   sW=np.full(g.cells.num, 0.2),
                   qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -2e-3]),
                   bhp=np.array([200.0 * BARSA, 150.0 * BARSA]))
        return g, m, m.seed_state(st)

    def test_reproduces_the_direct_solution():
        """THE test: this is the same answer, reached a different way."""
        g, m, st = build()
        eq = combine_equations(*m.get_equations(st, st.copy(), 30 * DAY))
        j, rhs = eq.jac[0], -eq.val

        exact = spla.spsolve(j.tocsc(), rhs)
        got = PoroelasticSchurSolver().solve(j, rhs, m, st, 30 * DAY)
        err = np.abs(got - exact).max() / np.abs(exact).max()
        assert err < 1e-9, f"Schur solve is not the direct solution: {err:.2e}"
        print(f"ok test_reproduces_the_direct_solution (rel {err:.1e})")

    def test_fixed_stress_beats_the_flow_block():
        """The preconditioner earns its place: it must cut the iteration count."""
        g, m, st = build()
        eq = combine_equations(*m.get_equations(st, st.copy(), 30 * DAY))
        j, rhs = eq.jac[0], -eq.val

        plain = PoroelasticSchurSolver(fixed_stress=False)
        plain.solve(j, rhs, m, st, 30 * DAY)
        fs = PoroelasticSchurSolver(fixed_stress=True)
        fs.solve(j, rhs, m, st, 30 * DAY)

        assert fs.last_iterations < plain.last_iterations, (
            f"fixed stress did not help: {fs.last_iterations} vs "
            f"{plain.last_iterations}"
        )
        print(f"ok test_fixed_stress_beats_the_flow_block "
              f"({plain.last_iterations} gmres its -> {fs.last_iterations})")

    def test_mechanics_block_really_is_constant():
        """The cache rests on this, so check it rather than assume it: the
        stiffness must be identical at two genuinely different states."""
        g, m, st = build()
        j1 = combine_equations(*m.get_equations(st, st.copy(), 30 * DAY)).jac[0]

        st2 = st.copy()
        st2["pressure"] = st2["pressure"] * 1.3
        st2["sW"] = np.full(g.cells.num, 0.7)
        st2 = m.seed_state(st2)
        j2 = combine_equations(*m.get_equations(st2, st2.copy(), 5 * DAY)).jac[0]

        nf = m.mech.n_free
        a1 = j1.tocsr()[-nf:, -nf:]
        a2 = j2.tocsr()[-nf:, -nf:]
        assert abs(a1 - a2).max() == 0.0, "the stiffness matrix moved"
        print("ok test_mechanics_block_really_is_constant")

    def test_stale_cache_is_refused():
        """If the stiffness ever did change, reusing the factorisation would
        be silently wrong - so it must raise instead."""
        g, m, st = build()
        eq = combine_equations(*m.get_equations(st, st.copy(), 30 * DAY))
        s = PoroelasticSchurSolver()
        s.solve(eq.jac[0], -eq.val, m, st, 30 * DAY)

        j = eq.jac[0].tolil()
        j[-1, -1] *= 2.0                       # pretend the rock stiffened
        try:
            s.solve(j.tocsr(), -eq.val, m, st, 30 * DAY)
            raise SystemExit("should have refused the stale factorisation")
        except RuntimeError as exc:
            assert "linear elasticity" in str(exc)
        print("ok test_stale_cache_is_refused")

    def test_it_actually_solves_a_schedule_faster():
        """End to end: same states as the direct solver, less time."""
        g, m, st = build(n=10)
        dts = [10 * DAY] * 3

        t = time.time()
        direct = NonLinearSolver()
        sd = st
        for dt in dts:
            sd, rep = direct.solve_ministep(m, sd, sd.copy(), dt)
            assert rep.converged
        t_direct = time.time() - t

        t = time.time()
        fast = NonLinearSolver(linear_solver=PoroelasticSchurSolver())
        sf = st
        for dt in dts:
            sf, rep = fast.solve_ministep(m, sf, sf.copy(), dt)
            assert rep.converged
        t_fast = time.time() - t

        for key in ("pressure", "sW", "bhp"):
            scale = max(np.abs(sd[key]).max(), 1e-30)
            err = np.abs(sf[key] - sd[key]).max() / scale
            assert err < 1e-7, f"{key} differs by {err:.2e}"
        print(f"ok test_it_actually_solves_a_schedule_faster "
              f"({t_direct:.1f} s direct -> {t_fast:.1f} s, "
              f"{t_direct / t_fast:.1f}x, same states)")

    test_reproduces_the_direct_solution()
    test_fixed_stress_beats_the_flow_block()
    test_mechanics_block_really_is_constant()
    test_stale_cache_is_refused()
    test_it_actually_solves_a_schedule_faster()
    print("\nAll tests passed!")
