"""Single-phase Compressible AD Solver (MRST ``flowSolverTutorialAD.m``).

A complete simulator written from first principles rather than assembled from
model classes. That is the point of the tutorial: every term is visible, so the
reader sees what a reservoir simulator actually solves.

The physics is single-phase flow of a compressible fluid in a compressible rock:

    pore volume   pv(p)  = pv_r * exp(cr * (p - p_r))
    density       rho(p) = rho_r * exp(c  * (p - p_r))
    Darcy flux    v      = -(T/mu) * (grad(p) - g*avg(rho)*grad(z))
    conservation  d/dt (pv*rho) + div(avg(rho)*v) = q

closed by a well with eight perforations under bottom-hole pressure control.
Three unknown groups are solved together, so the Jacobian is block structured:
cell pressures, the bottom-hole pressure, and the surface rate.

Two details are worth flagging against MRST's version, because both are places
a port can go quietly wrong:

* MRST marks exterior faces with a neighbour of 0 and writes `all(N ~= 0, 2)`.
  PyMRST marks them with -1, so the interior test is `all(N != -1)`. Getting
  this wrong keeps the boundary faces and leaks flow out of the domain.
* The initial condition solves `dp/dz = g*rho(p)` as an ODE, because the fluid
  is compressible and the profile is therefore not linear in depth. MRST uses
  `ode23` with `deval`; the port uses `solve_ivp` with dense output, which is
  the same method of solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import NORM_GRAVITY
from pymrst.params.rock import make_rock
from pymrst.params.rock.pore_volume import pore_volume
from pymrst.params.wells_and_bc import add_well
from pymrst.solvers.compute_trans import compute_trans
from pymrst.utils.adi import (
    ad_exp,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
)
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


@dataclass
class CompressibleSetup:
    """Grid, rock, fluid model, well and discrete operators."""

    G: object
    rock: object
    well: object
    p_ref: float
    rho_ref: float
    rho_surface: float
    mu: float
    c_fluid: float
    c_rock: float
    trans: np.ndarray
    neighbors: np.ndarray
    C: sp.csr_matrix
    pv_ref: np.ndarray

    def pore_volume_at(self, p):
        return self.pv_ref * ad_exp(self.c_rock * (p - self.p_ref))

    def density_at(self, p):
        return self.rho_ref * ad_exp(self.c_fluid * (p - self.p_ref))

    def grad(self, x):
        return self.C @ x if not hasattr(x, "jac") else _spmat_mul(self.C, x)

    def div(self, x):
        ct = (-self.C.T).tocsr()
        return ct @ x if not hasattr(x, "jac") else _spmat_mul(ct, x)

    def avg(self, x):
        n0, n1 = self.neighbors[:, 0], self.neighbors[:, 1]
        return 0.5 * (x[n0] + x[n1])


def _spmat_mul(mat, x):
    """`mat @ x` for an ADI vector, applied to value and every Jacobian block."""
    from pymrst.utils.adi import ad_matmul
    return ad_matmul(mat, x)


@dataclass
class CompressibleResult:
    """History from the tutorial's 52-step run."""

    time: np.ndarray
    pressure: np.ndarray
    bhp: np.ndarray
    qS: np.ndarray
    iterations: list[int] = field(default_factory=list)
    setup: CompressibleSetup | None = None


def build_setup(*, dims=(10, 10, 10), physdims=(200.0, 200.0, 50.0),
                perm: float = 30 * MILLI * DARCY, poro: float = 0.3,
                p_ref: float = 200 * BARSA,
                c_rock: float = 1e-6 / BARSA,
                c_fluid: float = 1e-3 / BARSA,
                mu: float = 5 * CENTI * POISE,
                rho_ref: float = 850.0,
                rho_surface: float = 750.0,
                nperf: int = 8) -> CompressibleSetup:
    """Assemble every ingredient MRST's tutorial builds before the time loop."""
    g = compute_geometry(cart_grid(list(dims), list(physdims)))
    rock = make_rock(g, perm, poro)

    nx, ny, _ = dims
    # MRST: I = 2 (1-based) repeated, J = 2..nperf+1, K = 5. Converted to
    # 0-based: i = 1, j = 1..nperf, k = 4.
    i = np.full(nperf, 1, dtype=np.int64)
    j = np.arange(1, nperf + 1, dtype=np.int64)
    k = np.full(nperf, 4, dtype=np.int64)
    cell_inx = i + nx * (j + ny * k)
    wells = add_well(None, g, rock, cell_inx, well_type="bhp",
                     val=100 * BARSA, name="P1", well_dir="y", sign=-1)

    trans_half = compute_trans(g, rock)
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    # PyMRST marks an exterior face with -1 where MRST uses 0.
    interior = np.all(neighbors != -1, axis=1)
    cf = np.asarray(g.cells.faces, dtype=np.int64)
    if cf.ndim > 1:
        cf = cf[:, 0]
    # Harmonic average of the two half-transmissibilities on each face.
    inv_t = np.bincount(cf, weights=1.0 / trans_half, minlength=g.faces.num)
    trans = (1.0 / inv_t)[interior]
    n_int = neighbors[interior]

    n = n_int.shape[0]
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([n_int[:, 0], n_int[:, 1]])
    vals = np.concatenate([-np.ones(n), np.ones(n)])
    c_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, g.cells.num))

    return CompressibleSetup(
        G=g, rock=rock, well=wells[0], p_ref=p_ref, rho_ref=rho_ref,
        rho_surface=rho_surface, mu=mu, c_fluid=c_fluid, c_rock=c_rock,
        trans=trans, neighbors=n_int, C=c_mat, pv_ref=pore_volume(g, rock))


def hydrostatic_initial_pressure(setup: CompressibleSetup) -> np.ndarray:
    """Solve `dp/dz = g*rho(p)` for the equilibrium profile.

    A compressible fluid does not give a linear profile, so this is an ODE
    rather than a formula. MRST integrates it with `ode23` and evaluates the
    dense output at the cell centroids.
    """
    z = np.asarray(setup.G.cells.centroids[:, 2], dtype=float)
    g_acc = NORM_GRAVITY

    def rhs(_z, p):
        return g_acc * setup.rho_ref * np.exp(setup.c_fluid * (p - setup.p_ref))

    sol = solve_ivp(rhs, (0.0, float(z.max())), [setup.p_ref],
                    dense_output=True, rtol=1e-10, atol=1e-8)
    return np.asarray(sol.sol(z)).ravel()


def run_flow_solver_tutorial_ad(*, num_steps: int = 52,
                                total_time: float = 365 * DAY,
                                tol: float = 1e-5, max_its: int = 10,
                                bhp_target: float = 100 * BARSA,
                                setup: CompressibleSetup | None = None,
                                verbose: bool = False) -> CompressibleResult:
    """Run the tutorial's fully implicit single-phase compressible simulation."""
    setup = build_setup() if setup is None else setup
    g = setup.G
    nc = g.cells.num
    p_init = hydrostatic_initial_pressure(setup)

    wc = np.asarray(setup.well.cells, dtype=np.int64)
    wi = np.asarray(setup.well.WI, dtype=float)
    dz = np.asarray(setup.well.dZ, dtype=float)
    g_acc = NORM_GRAVITY
    gradz = setup.C @ np.asarray(g.cells.centroids[:, 2], dtype=float)

    dt = float(total_time) / int(num_steps)
    p_hist = [p_init.copy()]
    bhp_hist = [float(p_init[wc[0]])]
    qs_hist = [0.0]
    times = [0.0]
    iterations: list[int] = []

    p_prev = p_init.copy()
    bhp_prev = float(p_init[wc[0]])
    qs_prev = 0.0

    for step in range(int(num_steps)):
        p_ad, bhp_ad, qs_ad = init_variables_adi(
            p_prev.copy(), np.array([bhp_prev]), np.array([qs_prev]))
        p0 = p_prev.copy()
        converged = False
        nit = 0
        while not converged and nit < int(max_its):
            rho = setup.density_at(p_ad)
            rho_avg = setup.avg(rho)
            flux = -(setup.trans / setup.mu) * (
                _spmat_mul(setup.C, p_ad) - g_acc * rho_avg * gradz)
            acc = (1.0 / dt) * (setup.pore_volume_at(p_ad) * rho
                                - setup.pv_ref * np.exp(
                                    setup.c_rock * (p0 - setup.p_ref))
                                * setup.rho_ref * np.exp(
                                    setup.c_fluid * (p0 - setup.p_ref)))
            eq1 = acc + setup.div(rho_avg * flux)

            rho_bhp = setup.rho_ref * ad_exp(setup.c_fluid * (bhp_ad - setup.p_ref))
            p_conn = bhp_ad + g_acc * dz * rho_bhp
            q_conn = wi * (setup.density_at(p_ad[wc]) / setup.mu) * (p_conn - p_ad[wc])
            eq1[wc] = eq1[wc] - q_conn

            rate_eq = qs_ad - ad_sum(q_conn) / setup.rho_surface
            ctrl_eq = bhp_ad - bhp_target

            # `combine_equations` is the counterpart of MRST's `cat` here: it
            # stacks the three residuals and assembles the per-variable
            # Jacobian blocks into one square matrix. `vertcat_adi` stacks the
            # values but leaves the blocks separate, which gives a 1002x1000
            # system and no solve.
            eq = combine_equations(eq1, rate_eq, ctrl_eq)
            res = np.asarray(eq.val, dtype=float)
            upd = spla.spsolve(eq.jac[0].tocsc(), -res)

            p_ad = p_ad + upd[:nc]
            bhp_ad = bhp_ad + upd[nc]
            qs_ad = qs_ad + upd[nc + 1]
            resid = float(np.max(np.abs(res)))
            converged = resid < tol
            nit += 1

        if not converged:
            raise RuntimeError(
                f"step {step + 1} did not converge in {max_its} iterations "
                f"(residual {resid:.3e})")

        p_prev = np.asarray(value(p_ad), dtype=float).ravel()
        bhp_prev = float(np.asarray(value(bhp_ad)).ravel()[0])
        qs_prev = float(np.asarray(value(qs_ad)).ravel()[0])
        iterations.append(nit)
        times.append(times[-1] + dt)
        p_hist.append(p_prev.copy())
        bhp_hist.append(bhp_prev)
        qs_hist.append(qs_prev)
        if verbose:
            print(f"  step {step + 1:3d}: {times[-1] / DAY:7.2f} d, "
                  f"{nit} iterations, qS = {qs_prev:.6e}")

    return CompressibleResult(
        time=np.asarray(times), pressure=np.vstack(p_hist),
        bhp=np.asarray(bhp_hist), qS=np.asarray(qs_hist),
        iterations=iterations, setup=setup)


if __name__ == "__main__":
    def test_setup_matches_the_tutorial_geometry():
        s = build_setup()
        assert s.G.cells.num == 1000, s.G.cells.num
        assert np.asarray(s.well.cells).size == 8, s.well.cells
        # 10x10x10 has 3*10*10*9 = 2700 interior faces.
        assert s.C.shape == (2700, 1000), s.C.shape
        assert s.trans.shape == (2700,), s.trans.shape
        print("ok test_setup_matches_the_tutorial_geometry")

    def test_exterior_faces_are_excluded():
        """PyMRST marks exterior neighbours -1 where MRST uses 0.

        Testing `!= 0` instead of `!= -1` would keep the boundary faces and let
        flow leave the domain, which is the kind of error that still produces a
        plausible picture.
        """
        s = build_setup()
        assert np.all(s.neighbors >= 0), "an exterior face survived"
        # Each interior face contributes -1 and +1, so rows sum to zero.
        assert np.allclose(np.asarray(s.C.sum(axis=1)).ravel(), 0.0)
        print("ok test_exterior_faces_are_excluded")

    def test_initial_pressure_is_in_equilibrium():
        """The ODE profile must satisfy dp/dz = g*rho(p) at the cell centres."""
        s = build_setup()
        p = hydrostatic_initial_pressure(s)
        z = np.asarray(s.G.cells.centroids[:, 2])
        order = np.argsort(z)
        zs, ps = z[order], p[order]
        uniq = np.unique(np.round(zs, 9))
        pz = np.array([ps[np.isclose(zs, u)].mean() for u in uniq])
        dpdz = np.gradient(pz, uniq)
        rho = s.rho_ref * np.exp(s.c_fluid * (pz - s.p_ref))
        assert np.allclose(dpdz, NORM_GRAVITY * rho, rtol=2e-3), (
            np.max(np.abs(dpdz - NORM_GRAVITY * rho)))
        assert np.isclose(p.min(), s.p_ref, rtol=5e-3), p.min()
        print("ok test_initial_pressure_is_in_equilibrium")

    def test_short_run_depletes_and_produces():
        """Draw the well down to 100 bar and the reservoir must deplete.

        The well is below the initial pressure everywhere, so pressure falls
        monotonically and the surface rate is negative throughout.
        """
        out = run_flow_solver_tutorial_ad(num_steps=6, total_time=60 * DAY)
        assert out.pressure.shape == (7, 1000), out.pressure.shape
        mean_p = out.pressure.mean(axis=1)
        assert np.all(np.diff(mean_p) < 0.0), "reservoir did not deplete"
        assert np.allclose(out.bhp[1:], 100 * BARSA, rtol=1e-9), out.bhp[:3]
        assert np.all(out.qS[1:] < 0.0), out.qS[:3]
        assert max(out.iterations) <= 10, out.iterations
        print(f"ok test_short_run_depletes_and_produces "
              f"(max {max(out.iterations)} Newton iterations)")

    def test_mass_is_conserved_against_the_well():
        """Change in reservoir mass must equal what the well removed.

        This is the closure check the tutorial's plots imply but never make.
        Accumulated surface rate times surface density is the mass produced,
        and it should match the drop in `sum(pv(p)*rho(p))` to solver
        tolerance.
        """
        out = run_flow_solver_tutorial_ad(num_steps=6, total_time=60 * DAY)
        s = out.setup
        def mass(p):
            pv = s.pv_ref * np.exp(s.c_rock * (p - s.p_ref))
            rho = s.rho_ref * np.exp(s.c_fluid * (p - s.p_ref))
            return float(np.sum(pv * rho))
        dm = mass(out.pressure[0]) - mass(out.pressure[-1])
        dt = out.time[1] - out.time[0]
        produced = -float(np.sum(out.qS[1:])) * dt * s.rho_surface
        assert np.isclose(dm, produced, rtol=1e-3), (dm, produced)
        print(f"ok test_mass_is_conserved_against_the_well "
              f"(balance {abs(dm - produced) / dm:.2e})")

    def test_incompressible_limit_gives_a_linear_profile():
        """With c -> 0 the equilibrium profile must become linear in depth.

        A useful check on the ODE initial condition: the nonlinear solve has to
        reduce to the textbook answer when the nonlinearity is removed.
        """
        s = build_setup(c_fluid=1e-14 / BARSA)
        p = hydrostatic_initial_pressure(s)
        z = np.asarray(s.G.cells.centroids[:, 2])
        expect = s.p_ref + s.rho_ref * NORM_GRAVITY * z
        assert np.allclose(p, expect, rtol=1e-8), np.max(np.abs(p - expect))
        print("ok test_incompressible_limit_gives_a_linear_profile")

    test_setup_matches_the_tutorial_geometry()
    test_exterior_faces_are_excluded()
    test_initial_pressure_is_in_equilibrium()
    test_short_run_depletes_and_produces()
    test_mass_is_conserved_against_the_well()
    test_incompressible_limit_gives_a_linear_profile()
    print("\nAll tests passed!")
