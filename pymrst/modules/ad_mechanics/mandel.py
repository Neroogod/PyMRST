"""Mandel's problem: the benchmark that catches wrong poroelastic coupling.

A saturated poroelastic slab is squeezed between two rigid, impermeable,
frictionless plates under a constant force, and drains through its side
faces. The famous result (Mandel 1953; Cryer 1963) is that the pressure in
the middle of the slab does **not** simply decay: it first *rises above*
its initial value, then falls. Fluid squeezed out near the drained edge
lets that region soften, which transfers load onto the still-undrained
core and pressurises it further.

That overshoot is why this is the benchmark worth having. A model with
one-way coupling, a lagged coupling term, or a wrong Biot modulus gives a
monotonic decay and can still look plausible. Only a correctly coupled
model reproduces the non-monotonic curve, and it reproduces it to the
analytic series solution.

The plates apply constant *force*, not constant displacement, so they
settle as the slab consolidates: each step iterates the prescribed plate
displacement until the reaction force matches the load, as MRST's
`mandel.m` does. Run as plane strain (a 3D slab with the out-of-plane
displacement fixed), which is exactly the 2D problem MRST solves.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import brentq

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
from pymrst.modules.ad_mechanics.models.poromechanics import MechWaterModel
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c
from pymrst.params.rock import make_rock
from pymrst.utils.units import DARCY


def mandel_roots(nu: float, nu_u: float, n: int = 60) -> np.ndarray:
    """Roots of tan(a) = (1-nu)/(nu_u-nu) * a, one per rising branch of tan.

    The branches are (m*pi, m*pi + pi/2) for m = 0, 1, 2, ..., where tan runs
    from 0 to +infinity. The m = 0 branch matters: it holds the slowest-
    decaying mode and therefore most of the solution. (Bracketing from
    pi/2 instead - an easy mistake, since that is where tan's *poles* are -
    silently drops it, and the series then fails to reduce to 1 at t = 0.)
    """
    k = (1.0 - nu) / (nu_u - nu)
    f = lambda a: np.tan(a) - k * a
    eps = 1e-10
    roots = []
    for m in range(n):
        lo = m * np.pi + eps
        hi = m * np.pi + 0.5 * np.pi - eps
        roots.append(brentq(f, lo, hi, xtol=1e-14))
    return np.array(roots)


def mandel_pressure(x_over_a: np.ndarray, tau: np.ndarray, nu: float,
                    nu_u: float, n: int = 60) -> np.ndarray:
    """p(x, t) / p(x, 0) for Mandel's problem; tau = c t / a^2.

    Normalised by the initial (uniform) pressure, so the applied force
    cancels and the comparison needs no poroelastic prefactor.
    """
    a_n = mandel_roots(nu, nu_u, n)[:, None]
    x = np.atleast_1d(x_over_a)[None, :]
    tau = np.atleast_1d(tau)

    amp = np.sin(a_n) / (a_n - np.sin(a_n) * np.cos(a_n))
    shape = np.cos(a_n * x) - np.cos(a_n)
    out = np.empty((tau.size, x.size))
    for i, t in enumerate(tau):
        out[i] = 2.0 * np.sum(amp * shape * np.exp(-a_n**2 * t), axis=0)
    return out


def poro_params(E: float, nu: float, alpha: float, M: float) -> Dict[str, float]:
    """The poroelastic constants Mandel's solution is written in."""
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    K_u = K + alpha**2 * M                       # undrained bulk modulus
    B = alpha * M / K_u                          # Skempton coefficient
    nu_u = (3.0 * K_u - 2.0 * G) / (2.0 * (3.0 * K_u + G))
    return {"K": K, "G": G, "K_u": K_u, "B": B, "nu_u": nu_u}


def consolidation_coefficient(perm: float, mu_f: float, E: float, nu: float,
                              alpha: float, M: float) -> float:
    """Mandel's c (Cheng & Detournay)."""
    p = poro_params(E, nu, alpha, M)
    G, B, nu_u = p["G"], p["B"], p["nu_u"]
    return (2.0 * perm * B**2 * G * (1.0 - nu) * (1.0 + nu_u) ** 2) / (
        9.0 * mu_f * (1.0 - nu_u) * (nu_u - nu)
    )


class MandelCase:
    """The slab, its plates, and the force iteration that drives them."""

    def __init__(self, nx: int = 40, nz: int = 6, a: float = 10.0,
                 h: float = 1.0, E: float = 1e8, nu: float = 0.2,
                 alpha: float = 1.0, poro: float = 0.1,
                 perm: float = 200e-3 * DARCY, mu_f: float = 0.89e-3,
                 c_f: float = 1e-9, load: float = 1e6):
        self.a, self.h, self.load = a, h, load
        self.E, self.nu, self.alpha = E, nu, alpha

        # Plane strain: one cell thick in y, with u_y = 0 on both sides.
        g = create_augmented_grid(
            compute_geometry(cart_grid([nx, 1, nz], [a, h / nx, h]))
        )
        self.G = g
        self.rock = make_rock(g, perm, poro)
        self.mu_f, self.perm, self.poro, self.c_f = mu_f, perm, poro, c_f

        # alpha = 1 (incompressible grains), so 1/M = phi c_f.
        self.M = 1.0 / (poro * c_f)
        self.params = poro_params(E, nu, alpha, self.M)
        self.cv = consolidation_coefficient(perm, mu_f, E, nu, alpha, self.M)

        pts, fc = g.nodes.coords, g.faces.centroids
        bd = np.any(g.faces.neighbors == -1, axis=1)
        ymax = pts[:, 1].max()

        self.top_nodes = np.flatnonzero(pts[:, 2] < 1e-9)
        bot = np.flatnonzero(pts[:, 2] > h - 1e-9)
        left = np.flatnonzero(pts[:, 0] < 1e-9)              # symmetry plane
        yfix = np.flatnonzero((pts[:, 1] < 1e-9) | (pts[:, 1] > ymax - 1e-9))

        dn = np.concatenate([self.top_nodes, bot, left, yfix])
        dm = np.vstack([
            np.tile([False, False, True], (self.top_nodes.size, 1)),
            np.tile([False, False, True], (bot.size, 1)),
            np.tile([True, False, False], (left.size, 1)),
            np.tile([False, True, False], (yfix.size, 1)),   # plane strain
        ])
        el_bc = ElasticBC(disp_nodes=dn, disp_mask=dm,
                          disp_vals=np.zeros((dn.size, 3)))

        C = enu2c(np.full(g.cells.num, E), np.full(g.cells.num, nu))
        self.mech = MechanicModel(g, C, alpha=alpha, el_bc=el_bc)

        # Drained at the far side (x = a); everything else sealed.
        self.drain_faces = np.flatnonzero(bd & (fc[:, 0] > a - 1e-9))
        self.top_cells = np.flatnonzero(
            g.cells.centroids[:, 2] < g.cells.centroids[:, 2].min() + 1e-9
        )

        self.model = MechWaterModel(
            g, self.rock, self.mech, lambda p: 1.0 + c_f * p, mu_f, 1000.0,
            self.drain_faces, np.zeros(self.drain_faces.size),
        )

    def plate_force(self, st) -> float:
        """Mean total vertical stress under the plate (MRST's `tforce_sim`)."""
        stress = self.mech.effective_stress(st["xd"])[self.top_cells, 2]
        p = st["pressure"][self.top_cells]
        return float(np.mean(-stress + self.alpha * p))

    def _step_at(self, st0, dt: float, plate: float):
        self.mech.set_dirichlet(self.top_nodes, 2, plate)
        st, ok, _ = self.model.step(st0, dt)
        assert ok, "coupled step failed"
        return st, self.plate_force(st)

    def solve_step(self, st0, dt: float, plate: float,
                   tol: float = 1e-6, max_iter: int = 10):
        """One step, moving the plate until its reaction force matches the load.

        The plate carries a *constant force* and settles as the slab drains -
        that is the whole point of Mandel - so its displacement is an unknown
        of each step.

        The reaction force is affine in the plate displacement (the pressure
        supplies the offset), so a secant iteration lands on it in one step.
        MRST rescales instead (`dy = dy * load / force`), which implicitly
        assumes the force is *proportional* to the displacement; here the
        slope is so flat that the rescaling crawls and never arrives.
        """
        st, f0 = self._step_at(st0, dt, plate)
        if abs((self.load - f0) / self.load) < tol:
            return st, plate

        d0 = plate
        d1 = plate * 1.05 if plate != 0.0 else 1e-6
        st, f1 = self._step_at(st0, dt, d1)

        for _ in range(max_iter):
            if abs((self.load - f1) / self.load) < tol:
                return st, d1
            slope = (f1 - f0) / (d1 - d0)
            d0, f0 = d1, f1
            d1 = d1 + (self.load - f1) / slope
            st, f1 = self._step_at(st0, dt, d1)
        raise RuntimeError("plate force iteration did not converge")

    def initial_state(self, plate: float = -1e-4):
        """The undrained response at t = 0+.

        Not imposed by hand: the slab starts unloaded and undeformed, the
        plate is applied, and one very short step is taken. With no time to
        drain, the mass balance alone forces the pressure up - which is the
        undrained response, and it comes out *non-uniform near the drained
        edge*, as it physically must. (Imposing a uniform Skempton pressure
        instead quietly destroys the Mandel-Cryer overshoot, because the
        overshoot is driven by exactly that edge.)
        """
        from pymrst.modules.ad_core.statefunctions.state_function import State

        self.mech.set_dirichlet(self.top_nodes, 2, 0.0)
        nc = self.G.cells.num
        # `u` must be given explicitly, not left to be rebuilt from xd: the
        # plate is about to move, and without a stored `u` this state's
        # div(u) would be re-evaluated against the *new* plate position - so
        # the slab would appear to have been deformed before it was loaded.
        unloaded = State(pressure=np.zeros(nc),
                         xd=np.zeros(self.mech.n_free),
                         u=np.zeros(3 * self.G.nodes.num))

        dt0 = 1e-6 * self.a**2 / self.cv
        return self.solve_step(unloaded, dt0, plate)


if __name__ == "__main__":
    def test_mandel_roots_are_the_right_branches():
        """One root per branch of tan, each satisfying the transcendental eq."""
        nu, nu_u = 0.2, 0.44
        a = mandel_roots(nu, nu_u, 8)
        k = (1 - nu) / (nu_u - nu)
        assert np.allclose(np.tan(a), k * a, atol=1e-8)
        assert np.all(np.diff(a) > 2.0)          # one per pi-branch
        print("ok test_mandel_roots_are_the_right_branches")

    def test_analytic_shows_the_overshoot():
        """The analytic solution itself is non-monotonic at the centre - if
        it were not, the benchmark would be testing nothing."""
        nu, nu_u = 0.2, 0.44
        tau = np.array([1e-8, 0.01, 0.05, 0.2, 2.0])
        p_centre = mandel_pressure(np.array([0.0]), tau, nu, nu_u)[:, 0]
        assert abs(p_centre[0] - 1.0) < 5e-3, "series must start at p = p0"
        assert p_centre.max() > 1.05, "no Mandel-Cryer overshoot in the analytic"
        assert p_centre.argmax() > 0, "the peak must not be at t = 0"
        assert p_centre[-1] < 0.1, "pressure must eventually drain away"
        print(f"ok test_analytic_shows_the_overshoot "
              f"(peak p/p0 = {p_centre.max():.3f})")

    def test_undrained_state_is_uniform_in_z():
        """Mandel's pressure depends on x alone. Any z-variation is an
        artifact - and it was one: it is what a stale div(u) at the moving
        plate produces, and it kills the overshoot."""
        case = MandelCase(nx=20, nz=4)
        st, _ = case.initial_state()
        cc = case.G.cells.centroids
        for xi in np.unique(np.round(cc[:, 0], 9)):
            col = st["pressure"][np.isclose(cc[:, 0], xi)]
            spread = (col.max() - col.min()) / abs(col.mean())
            assert spread < 1e-6, f"pressure varies with z at x={xi}: {spread}"
        assert np.all(st["pressure"] > 0), "undrained loading must pressurise"
        print("ok test_undrained_state_is_uniform_in_z")

    def test_mandel_simulation_matches_analytic():
        """THE test: the coupled model reproduces Mandel, overshoot and all.

        A one-way-coupled, lagged, or wrongly-scaled model decays
        monotonically here and would pass a Terzaghi test regardless. Only
        the correct coupling turns the pressure back *up* after loading.
        """
        case = MandelCase(nx=40, nz=4)
        nu, nu_u = case.nu, case.params["nu_u"]
        t_char = case.a**2 / case.cv

        st, plate = case.initial_state()
        x = case.G.cells.centroids[:, 0] / case.a
        centre = int(np.argmin(x))
        p0 = st["pressure"][centre]
        assert p0 > 0, "undrained loading must pressurise the slab"

        taus = [0.01, 0.05, 0.1, 0.2, 0.5]   # the overshoot peaks near 0.05
        n_sub = 20
        t_prev, worst, centre_hist = 0.0, 0.0, [1.0]
        for tau in taus:
            dt = (tau - t_prev) * t_char / n_sub
            for _ in range(n_sub):
                st, plate = case.solve_step(st, dt, plate)
            t_prev = tau

            sim = st["pressure"] / p0
            exact = mandel_pressure(x, np.array([tau]), nu, nu_u)[0]
            err = np.max(np.abs(sim - exact))
            worst = max(worst, err)
            centre_hist.append(sim[centre])
            print(f"  tau={tau}: max |p/p0 - analytic| = {err:.4f}"
                  f"  (centre p/p0 = {sim[centre]:.3f}, "
                  f"analytic {exact[centre]:.3f})")

        peak = max(centre_hist)
        assert peak > 1.05, f"no Mandel-Cryer overshoot: {centre_hist}"
        assert centre_hist.index(peak) > 0, "the peak must not be at t = 0"
        assert worst < 0.02, f"profile off the analytic by {worst:.3f}"
        print(f"ok test_mandel_simulation_matches_analytic "
              f"(peak p/p0 = {peak:.3f}, analytic 1.098; "
              f"worst deviation {worst:.4f})")

    test_mandel_roots_are_the_right_branches()
    test_analytic_shows_the_overshoot()
    test_undrained_state_is_uniform_in_z()
    test_mandel_simulation_matches_analytic()
    print("\nAll tests passed!")
