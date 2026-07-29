"""Discontinuous-Galerkin transport of a scalar (MRST ``TransportModelDG`` core).

The DG weak form of a scalar conservation law `d(u)/dt + div(f(u) v) = 0` on the
grid: per cell and test function `psi_i`,

    integral psi_i du/dt  -  integral f(u) v . grad psi_i  +  boundary  f(uhat) (v.n) psi_i = 0,

with an upwind numerical flux `f(uhat)` on faces. This module assembles that
residual from the `DGDiscretization` operators (mass, stiffness) and a face
cubature, and time-steps it implicitly. Linear advection (`f(u)=u`) gives a
linear system solved directly; nonlinear fractional flow uses Newton.
"""

from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.modules.dg.cubatures.element_cubature import LineCubature
from pymrst.modules.dg.discretizations.dg_discretization import DGDiscretization


class DGScalarTransport:
    """Implicit DG transport of a scalar on a 2-D grid with a total velocity.

    `flux_fn(u)` is the scalar flux function `f(u)` (identity for linear
    advection, fractional flow for two-phase); `velocity` is the constant total
    velocity vector. Faces carry the total flux `q = velocity . area-normal`.
    """

    def __init__(self, disc: DGDiscretization, velocity, flux_fn: Callable,
                 dflux_fn: Callable = None):
        self.disc = disc
        self.g = disc.g
        self.vel = np.asarray(velocity, dtype=float)
        self.flux_fn = flux_fn
        self.dflux_fn = dflux_fn
        self.nDof = disc.basis.n_dof
        self._precompute()

    def _precompute(self):
        g, disc = self.g, self.disc
        self.M = [disc.mass_matrix(c) for c in range(g.cells.num)]
        self.S = [disc.stiffness_matrix(c) for c in range(g.cells.num)]
        # Face cubature: points, weights and per-face slice.
        fc = LineCubature(g, 2 * disc.degree + 1)
        self.fpts, self.fw, self.fpos = fc.points, fc.weights, fc.pos
        self.areas = np.asarray(g.faces.areas, dtype=float)
        normals = np.asarray(g.faces.normals, dtype=float)[:, :g.griddim]
        self.qn = normals @ self.vel                   # total flux sign per face
        self.neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
        # Basis values at each face's cubature points, from each adjacent cell.
        self.psiL, self.psiR = {}, {}
        for f in range(g.faces.num):
            i0, i1 = self.fpos[f], self.fpos[f + 1]
            pts = self.fpts[i0:i1]
            for side, cell in ((self.psiL, self.neigh[f, 0]),
                               (self.psiR, self.neigh[f, 1])):
                if cell >= 0:
                    xhat, _ = disc.transform_coords(pts, np.full(pts.shape[0], cell))
                    side[f] = disc.basis.evaluate(xhat)   # (nq, nDof)

    def _face_u(self, dofs, f, side):                  # DG value at face points
        cell = self.neigh[f, side]
        psi = (self.psiL if side == 0 else self.psiR)[f]
        return psi @ dofs[cell]

    def residual(self, dofs, dofs_old, dt, inflow_u: float = 0.0) -> np.ndarray:
        """DG residual (n_cells, nDof) for one implicit time step."""
        R = np.zeros_like(dofs)
        for c in range(self.g.cells.num):
            R[c] = self.M[c] @ (dofs[c] - dofs_old[c]) / dt
            # Volume term  -integral f(u) v . grad psi_i  (via cubature).
            pts, w = self.disc._cell_points(c)
            xhat, _ = self.disc.transform_coords(pts, np.full(pts.shape[0], c))
            V = self.disc.basis.evaluate(xhat)
            G = self.disc.basis.evaluate_gradient(xhat)
            _, scaling = self.disc.transform_coords(pts[:1], np.array([c]))
            Gphys = G * scaling[0][None, None, :]
            fu = self.flux_fn(V @ dofs[c])             # f(u) at cell cub points
            vol = float(np.asarray(self.g.cells.volumes)[c])
            for d in range(self.g.griddim):
                R[c] -= self.vel[d] * vol * (Gphys[:, :, d].T * w) @ fu
        # Face upwind flux.
        for f in range(self.g.faces.num):
            L, Rn = self.neigh[f]
            i0, i1 = self.fpos[f], self.fpos[f + 1]
            wq = self.fw[i0:i1]
            qn = self.qn[f]
            if L >= 0 and Rn >= 0:                      # interior face
                up = L if qn >= 0 else Rn
                fu = self.flux_fn(self._face_u(dofs, f, 0 if up == L else 1))
                contrib = self.areas[f] * qn * (wq * fu)
                R[L] += self.psiL[f].T @ contrib
                R[Rn] -= self.psiR[f].T @ contrib
            else:                                      # boundary face
                cell = int(max(L, Rn))
                psi = self.psiL[f] if L >= 0 else self.psiR[f]
                if qn * (1 if L >= 0 else -1) >= 0:    # outflow: use cell value
                    fu = self.flux_fn(psi @ dofs[cell])
                else:                                  # inflow: prescribed u
                    fu = self.flux_fn(np.full(i1 - i0, inflow_u))
                sgn = 1.0 if L >= 0 else -1.0
                R[cell] += sgn * self.areas[f] * qn * (psi.T @ (wq * fu))
        return R

    def step(self, dofs_old, dt, inflow_u: float = 0.0,
             tol: float = 1e-10, maxit: int = 30) -> np.ndarray:
        """One implicit time step (damped Newton for nonlinear flux).

        Fractional-flow derivatives vanish at residual saturation.  A full
        Newton step can consequently jump between the two ends of the physical
        interval without reducing the residual.  Backtracking makes the solve
        robust in that regime while retaining full Newton steps near the root.
        """
        dofs = dofs_old.copy()
        converged = False
        for _ in range(maxit):
            R = self.residual(dofs, dofs_old, dt, inflow_u).ravel()
            norm_r = np.linalg.norm(R)
            if norm_r < tol:
                converged = True
                break
            J = self._jacobian(dofs, dofs_old, dt, inflow_u)
            delta = spsolve(J, R).reshape(dofs.shape)
            alpha = 1.0
            while alpha > 1e-8:
                candidate = dofs - alpha * delta
                trial = self.residual(candidate, dofs_old, dt, inflow_u).ravel()
                if np.linalg.norm(trial) < norm_r * (1.0 - 1e-4 * alpha):
                    break
                alpha *= 0.5
            dofs = candidate
        if not converged:
            R = self.residual(dofs, dofs_old, dt, inflow_u).ravel()
            if np.linalg.norm(R) >= tol:
                raise RuntimeError(
                    f"DG transport Newton solve did not converge in {maxit} "
                    f"iterations (residual {np.linalg.norm(R):.3e})")
        return dofs

    def solve_linear(self, dofs0, dt, nsteps, inflow_u: float = 0.0):
        """Time-step a linear-flux problem: the operator is affine, so assemble
        the Jacobian once and take one linear solve per step."""
        zero = np.zeros_like(dofs0)
        J = self._jacobian(zero, zero, dt, inflow_u)
        lu = sp.linalg.splu(J.tocsc())
        dofs = dofs0.copy()
        for _ in range(nsteps):
            R = self.residual(dofs, dofs, dt, inflow_u).ravel()
            dofs = (dofs.ravel() - lu.solve(R)).reshape(dofs.shape)
        return dofs

    def _jacobian(self, dofs, dofs_old, dt, inflow_u):
        """Analytical Jacobian d(residual)/d(dofs) (sparse). Uses `dflux_fn`; if
        it is not supplied, falls back to finite differences."""
        if self.dflux_fn is None:
            return self._jacobian_fd(dofs, dofs_old, dt, inflow_u)
        nd = self.nDof
        rows, cols, vals = [], [], []

        def block(ci, cj, B):
            ii, jj = np.meshgrid(np.arange(nd), np.arange(nd), indexing="ij")
            rows.append((ci * nd + ii).ravel())
            cols.append((cj * nd + jj).ravel())
            vals.append(B.ravel())

        for c in range(self.g.cells.num):
            pts, w = self.disc._cell_points(c)
            xhat, scaling = self.disc.transform_coords(pts, np.full(pts.shape[0], c))
            V = self.disc.basis.evaluate(xhat)
            G = self.disc.basis.evaluate_gradient(xhat) * scaling[:, None, :]
            fp = self.dflux_fn(V @ dofs[c])
            vol = float(np.asarray(self.g.cells.volumes)[c])
            B = self.M[c] / dt
            for d in range(self.g.griddim):
                B -= self.vel[d] * vol * (G[:, :, d].T * (w * fp)) @ V
            block(c, c, B)
        for f in range(self.g.faces.num):
            L, Rn = self.neigh[f]
            i0, i1 = self.fpos[f], self.fpos[f + 1]
            wq = self.fw[i0:i1]
            qn = self.qn[f]
            if L >= 0 and Rn >= 0:
                up, psi_up = (L, self.psiL[f]) if qn >= 0 else (Rn, self.psiR[f])
                fp = self.dflux_fn(psi_up @ dofs[up])
                D = self.areas[f] * qn * (psi_up * (wq * fp)[:, None])   # (nq,nDof)
                block(L, up, self.psiL[f].T @ D)
                block(Rn, up, -self.psiR[f].T @ D)
            else:
                cell = int(max(L, Rn))
                psi = self.psiL[f] if L >= 0 else self.psiR[f]
                sgn = 1.0 if L >= 0 else -1.0
                if qn * sgn >= 0:                      # outflow depends on cell u
                    fp = self.dflux_fn(psi @ dofs[cell])
                    D = sgn * self.areas[f] * qn * (psi * (wq * fp)[:, None])
                    block(cell, cell, psi.T @ D)
        n = self.g.cells.num * nd
        return sp.csc_array((np.concatenate(vals),
                             (np.concatenate(rows), np.concatenate(cols))),
                            shape=(n, n))

    def _jacobian_fd(self, dofs, dofs_old, dt, inflow_u):
        n = dofs.size
        base = self.residual(dofs, dofs_old, dt, inflow_u).ravel()
        rows, cols, vals = [], [], []
        h = 1e-7
        flat = dofs.ravel()
        for j in range(n):
            pert = flat.copy()
            pert[j] += h
            Rp = self.residual(pert.reshape(dofs.shape), dofs_old, dt, inflow_u).ravel()
            col = (Rp - base) / h
            nz = np.flatnonzero(np.abs(col) > 1e-14)
            rows.extend(nz)
            cols.extend([j] * nz.size)
            vals.extend(col[nz])
        return sp.csc_array((vals, (rows, cols)), shape=(n, n))


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_linear_advection_converges_to_exact_translate():
        # 1-D linear advection on a thin 2-D strip: an interior Gaussian pulse is
        # translated by the velocity. Backward-Euler time stepping is dissipative,
        # so the DG solution converges to the exact shifted pulse as dt -> 0.
        g = compute_geometry(cart_grid([40, 1], [10.0, 1.0]))
        disc = DGDiscretization.build(g, 2)
        tr = DGScalarTransport(disc, np.array([1.0, 0.0]), flux_fn=lambda u: u)
        pulse = lambda P: np.exp(-((P[:, 0] - 2.0) ** 2) / 1.0)
        cc = np.asarray(g.cells.centroids)
        exact = np.exp(-((cc[:, 0] - 5.0) ** 2) / 1.0)  # pulse at x = 5 at t = 3
        errs = []
        for dt in (0.02, 0.005):
            dofs = tr.solve_linear(disc.project(pulse), dt, int(3.0 / dt))
            errs.append(np.max(np.abs(disc.cell_mean(dofs) - exact)))
        assert errs[1] < errs[0] * 0.5, errs           # ~first-order in dt
        assert errs[1] < 0.02, errs
        print(f"ok test_linear_advection_converges_to_exact_translate "
              f"(err {errs[0]:.2e} -> {errs[1]:.2e})")

    def test_advection_conserves_mass_interior():
        # With the pulse away from boundaries, total mass is conserved.
        g = compute_geometry(cart_grid([30, 1], [10.0, 1.0]))
        disc = DGDiscretization.build(g, 1)
        tr = DGScalarTransport(disc, np.array([1.0, 0.0]), flux_fn=lambda u: u)
        pulse = lambda P: np.exp(-((P[:, 0] - 2.0) ** 2) / 1.0)
        dofs = disc.project(pulse)
        vols = np.asarray(g.cells.volumes)
        m0 = vols @ disc.cell_mean(dofs)
        dofs = tr.solve_linear(dofs, 0.02, 100)         # to t = 2, still interior
        m1 = vols @ disc.cell_mean(dofs)
        assert abs(m1 - m0) < 1e-9 * abs(m0), (m0, m1)
        print("ok test_advection_conserves_mass_interior")

    def test_buckley_leverett_matches_welge_front():
        # Nonlinear DG transport of a Buckley-Leverett displacement: with
        # quadratic relperm and equal viscosity, f_w = s^2/(s^2+(1-s)^2). The DG
        # shock front sits at the analytic Welge front position.
        from scipy.optimize import brentq
        fw = lambda s: s ** 2 / (s ** 2 + (1 - s) ** 2)
        dfw = lambda s: (2 * s * (s ** 2 + (1 - s) ** 2)
                         - s ** 2 * (2 * s - 2 * (1 - s))) / (s ** 2 + (1 - s) ** 2) ** 2
        g = compute_geometry(cart_grid([50, 1], [1.0, 1.0]))   # unit PV, vT=1
        disc = DGDiscretization.build(g, 1)
        tr = DGScalarTransport(disc, np.array([1.0, 0.0]), flux_fn=fw, dflux_fn=dfw)
        dofs = disc.assign_from_cell_values(np.full(g.cells.num, 1e-6))
        T, dt = 0.3, 0.004
        for _ in range(int(T / dt)):
            dofs = tr.step(dofs, dt, inflow_u=1.0)
        m = disc.cell_mean(dofs)
        cc = np.asarray(g.cells.centroids)
        s_star = brentq(lambda s: fw(s) / s - dfw(s), 0.3, 0.99)
        x_front = dfw(s_star) * T
        dg_front = cc[m > s_star * 0.5, 0].max()
        assert abs(dg_front - x_front) < 0.03, (dg_front, x_front)
        assert 0.0 <= m.min() and m.max() <= 1.001            # bounded
        print(f"ok test_buckley_leverett_matches_welge_front "
              f"(DG front {dg_front:.3f} vs Welge {x_front:.3f})")

    test_linear_advection_converges_to_exact_translate()
    test_advection_conserves_mass_interior()
    test_buckley_leverett_matches_welge_front()
    print("\nAll tests passed!")
