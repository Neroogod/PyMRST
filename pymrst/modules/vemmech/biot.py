"""Linear Biot poroelasticity: VEM mechanics + TPFA flow, fully coupled.

Backward-Euler monolithic solver for

    div(sigma'(u) - alpha p I) = f          (mechanics, VEM)
    (1/M) dp/dt + alpha d(div u)/dt         (flow, TPFA)
        - div(k/mu grad p) = q

following the coupling used by MRST's ad-mechanics module (the pressure
enters mechanics through the VEM divergence operator's transpose). The
system is linear, so each time step is one saddle-point solve:

    [ A          -alpha Q']  [u]   [f]
    [ alpha Q/dt  F + V/dt]  [p] = [alpha Q u0/dt + V p0/dt + q]
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.vem import ElasticBC, vem3d_div, vem_assemble
from pymrst.params.rock import Rock
from pymrst.solvers.trans import compute_trans


@dataclass
class BiotBC:
    """Flow boundary conditions: Dirichlet pressure faces (rest no-flow)."""

    pressure_faces: np.ndarray
    pressure_vals: np.ndarray


class BiotModel:
    """Fully coupled linear poroelasticity on a 3D grid."""

    def __init__(
        self,
        g: Grid,
        C: np.ndarray,
        rock: Rock,
        el_bc: ElasticBC,
        flow_bc: BiotBC,
        mu_fluid: float,
        alpha: float = 1.0,
        M_biot: float = 1e30,
    ):
        self.g = g
        self.alpha = alpha

        # Mechanics: stiffness, Dirichlet reduction, coupling operator.
        S = vem_assemble(g, C)
        ndof = 3 * g.nodes.num
        bcnodes, first_ix = np.unique(el_bc.disp_nodes, return_index=True)
        mask = np.zeros((bcnodes.size, 3), dtype=bool)
        inv = np.searchsorted(bcnodes, el_bc.disp_nodes)
        np.logical_or.at(mask, inv, el_bc.disp_mask.astype(bool))
        dirdofs = (3 * bcnodes[:, None] + np.arange(3))[mask]
        self.isdir = np.zeros(ndof, dtype=bool)
        self.isdir[dirdofs] = True
        self.free = ~self.isdir

        self.A = S[self.free][:, self.free]
        self.A = (self.A + self.A.T) / 2.0
        self.Q = vem3d_div(g)[:, self.free]

        # Constant mechanics load from boundary tractions.
        self.f_mech = np.zeros(ndof)
        if el_bc.force_faces is not None and el_bc.force_faces.size > 0:
            from pymrst.modules.vemmech.vem import calculate_qc
            from pymrst.utils.mcolon import mcolon
            _, qf_all, _ = calculate_qc(g)
            faces = el_bc.force_faces
            lnn = np.diff(g.faces.nodePos)[faces]
            inodes = mcolon(g.faces.nodePos[faces],
                            g.faces.nodePos[faces + 1])
            nodes = g.faces.nodes[inodes]
            force = np.repeat(el_bc.force, lnn, axis=0) \
                * qf_all[inodes][:, None]
            fdofs = (3 * nodes[:, None] + np.arange(3)).ravel()
            self.f_mech += np.bincount(fdofs, weights=force.ravel(),
                                       minlength=ndof)
        self.f_mech = self.f_mech[self.free]

        # Flow: TPFA matrix with Dirichlet pressure faces.
        t_half = compute_trans(g, rock)
        t_face = 1.0 / np.bincount(g.cells.faces, weights=1.0 / t_half,
                                   minlength=g.faces.num) / mu_fluid
        n = g.faces.neighbors
        internal = np.all(n >= 0, axis=1)
        ni = n[internal]
        nc = g.cells.num
        d = np.bincount(ni.ravel(order="F"),
                        weights=np.tile(t_face[internal], 2), minlength=nc)
        self.q_bc = np.zeros(nc)
        pf = flow_bc.pressure_faces
        if pf.size > 0:
            bc_cell = n[pf].sum(axis=1) + 1
            np.add.at(d, bc_cell, t_face[pf])
            np.add.at(self.q_bc, bc_cell, t_face[pf] * flow_bc.pressure_vals)
        rows = np.concatenate([ni[:, 0], ni[:, 1], np.arange(nc)])
        cols = np.concatenate([ni[:, 1], ni[:, 0], np.arange(nc)])
        vals = np.concatenate([-t_face[internal], -t_face[internal], d])
        self.F = sp.csr_array((vals, (rows, cols)), shape=(nc, nc))

        self.Vm = sp.diags_array(g.cells.volumes / M_biot, format="csr")

    def initial_undrained(self, p0: np.ndarray) -> np.ndarray:
        """Mechanics equilibrium at fixed pressure (undrained response)."""
        rhs = self.f_mech + self.alpha * (self.Q.T @ p0)
        return spla.spsolve(self.A.tocsc(), rhs)

    def step(self, u_free: np.ndarray, p: np.ndarray,
             dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """One backward-Euler step of the coupled system."""
        a = self.alpha
        top = sp.hstack([self.A, -a * self.Q.T], format="csr")
        bot = sp.hstack([(a / dt) * self.Q, self.F + self.Vm / dt],
                        format="csr")
        sys = sp.vstack([top, bot], format="csc")
        rhs = np.concatenate([
            self.f_mech,
            (a / dt) * (self.Q @ u_free) + (self.Vm / dt) @ p + self.q_bc,
        ])
        x = spla.spsolve(sys, rhs)
        nf = self.A.shape[0]
        return x[:nf], x[nf:]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c
    from pymrst.params.rock import make_rock
    from pymrst.utils.mcolon import mcolon
    from pymrst.utils.units import DARCY

    def terzaghi_analytic(z_over_L, tau, nterms=200):
        """p/p0 for consolidation with drainage at z=0, sealed at z=L."""
        out = np.zeros_like(z_over_L)
        for m in range(nterms):
            n = 2 * m + 1
            out += (4.0 / (np.pi * n)) * np.sin(n * np.pi * z_over_L / 2.0) \
                * np.exp(-(n * np.pi / 2.0) ** 2 * tau)
        return out

    def test_terzaghi():
        """1D consolidation matches the analytic series solution."""
        nz, L = 40, 10.0
        g = compute_geometry(cart_grid([1, 1, nz], [1.0, 1.0, L]))
        g = create_augmented_grid(g)

        E, nu = 1e9, 0.25
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu_s = E / (2 * (1 + nu))
        kv = lam + 2 * mu_s
        alpha, M = 1.0, 1e9
        perm, mu_f = 1.0 * DARCY, 1e-3
        sigma0 = 1e6

        C = enu2c(np.full(nz, E), np.full(nz, nu))
        rock = make_rock(g, perm, 0.2)

        pts = g.nodes.coords
        bd = np.any(g.faces.neighbors == -1, axis=1)
        fc = g.faces.centroids
        top_f = np.flatnonzero(bd & (fc[:, 2] < 1e-9))
        bot_nodes = np.flatnonzero(pts[:, 2] > L - 1e-9)
        nx_nodes = np.flatnonzero((pts[:, 0] < 1e-9) | (pts[:, 0] > 1 - 1e-9))
        ny_nodes = np.flatnonzero((pts[:, 1] < 1e-9) | (pts[:, 1] > 1 - 1e-9))
        dn = np.concatenate([bot_nodes, nx_nodes, ny_nodes])
        dm = np.vstack([
            np.tile([False, False, True], (bot_nodes.size, 1)),
            np.tile([True, False, False], (nx_nodes.size, 1)),
            np.tile([False, True, False], (ny_nodes.size, 1)),
        ])
        el_bc = ElasticBC(
            disp_nodes=dn, disp_mask=dm, disp_vals=np.zeros((dn.size, 3)),
            force_faces=top_f,
            force=np.tile([0.0, 0.0, sigma0], (top_f.size, 1)),
        )
        flow_bc = BiotBC(pressure_faces=top_f, pressure_vals=np.zeros(1))

        model = BiotModel(g, C, rock, el_bc, flow_bc,
                          mu_fluid=mu_f, alpha=alpha, M_biot=M)

        # Undrained initial condition.
        p0_val = alpha * M * sigma0 / (kv + alpha**2 * M)
        p = np.full(nz, p0_val)
        u = model.initial_undrained(p)

        cv = (perm / mu_f) * kv * M / (kv + alpha**2 * M)
        z = g.cells.centroids[:, 2] / L
        for tau_target in (0.1, 0.5):
            t_target = tau_target * L**2 / cv
            nsteps = 100
            dt = t_target / nsteps if tau_target == 0.1 else \
                (tau_target - 0.1) * L**2 / cv / nsteps
            for _ in range(nsteps):
                u, p = model.step(u, p, dt)
            exact = p0_val * terzaghi_analytic(z, tau_target)
            err = np.max(np.abs(p - exact)) / p0_val
            assert err < 0.02, f"tau={tau_target}: rel err {err:.4f}"
            print(f"  tau={tau_target}: max rel err {err:.4f}")

        # Long-time limit: full drainage and drained settlement.
        for _ in range(60):
            u, p = model.step(u, p, 0.5 * L**2 / cv / 10)
        assert np.max(np.abs(p)) < 1e-3 * p0_val
        top_nodes = np.flatnonzero(pts[:, 2] < 1e-9)
        ndof_map = np.flatnonzero(model.free)
        u_full = np.zeros(3 * g.nodes.num)
        u_full[ndof_map] = u
        settle = u_full.reshape(-1, 3)[top_nodes, 2].mean()
        assert np.isclose(settle, sigma0 * L / kv, rtol=1e-3), (
            f"settlement {settle:.5e} vs drained {sigma0 * L / kv:.5e}"
        )
        print("ok test_terzaghi")

    test_terzaghi()
    print("\nAll tests passed!")
