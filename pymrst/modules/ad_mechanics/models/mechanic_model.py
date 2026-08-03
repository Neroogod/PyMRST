"""Poroelastic mechanics on a VEM discretization (MRST ad-mechanics).

The mechanical half of the coupled problem: given a fluid pressure field,
find the displacement that balances the effective stress,

    div(sigma'(u) - alpha p I) = load,    sigma'(u) = C : eps(u)

which discretizes (MRST `equationsPoroMechanics`) to the linear residual

    A xd - gradP (alpha p) - rhs = 0

on the free (non-Dirichlet) displacement degrees of freedom `xd`. `A` is
the VEM stiffness matrix, `gradP` is the transpose of the VEM divergence
operator restricted to those dofs - the pressure enters mechanics through
exactly the operator by which displacement enters flow, which is what
makes the coupled system symmetric.

The residual is linear in both `xd` and `p`, so the fully coupled model can
assemble it with AD and the splitting scheme can solve it directly.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.vemmech.vem import (
    ElasticBC,
    calculate_qf,
    calculate_qc,
    inv_ci,
    vem_div,
    vem_assemble,
    vem_strain_operator,
    vem_stress_operator,
)
from pymrst.utils.mcolon import mcolon

VOIGT_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
VOIGT_TRACE_2D = np.array([1.0, 1.0, 0.0])


def _nlin(g: Grid) -> int:
    return 6 if g.griddim == 3 else 3


def _trace_vector(g: Grid) -> np.ndarray:
    return VOIGT_TRACE if g.griddim == 3 else VOIGT_TRACE_2D


@dataclass
class MechanicOperators:
    """The discrete mechanics operators (MRST `setupOperatorsVEM`)."""

    A: sp.csr_array                 # stiffness on free dofs
    gradP: sp.csr_array             # (n_free, nc): pressure -> mechanics force
    div: sp.csr_array               # (nc, n_free): xd -> volume-integrated div
    div_rhs: np.ndarray             # (nc,) divergence of the Dirichlet part
    strain: sp.csr_array            # (nlin*nc, ndof): u -> strain
    stress: sp.csr_array            # (nlin*nc, ndof): u -> effective stress
    rhs: np.ndarray                 # (n_free,) load + traction + Dirichlet
    is_dir: np.ndarray              # (ndof,) bool
    u_dir: np.ndarray               # (ndof,) prescribed displacement
    inv_ci: np.ndarray              # (nc, nlin) = C^-1 : I


class MechanicModel:
    """Linear poroelastic mechanics: displacement given a pressure field."""

    def __init__(self, g: Grid, C: np.ndarray, alpha: np.ndarray,
                 el_bc: ElasticBC, load=None, alpha_scaling: float = 1.0,
                 stabilization=None, S=None, initial_stress=None):
        assert g.cells.nodes is not None, "run create_augmented_grid first"
        self.G = g
        assert g.griddim in (2, 3), "ad_mechanics supports 2D and 3D VEM grids"
        self.C = np.asarray(C, dtype=np.float64)
        self.alpha = np.broadcast_to(np.asarray(alpha, dtype=float).ravel(),
                                     (g.cells.num,)).copy()
        if stabilization is not None and S is not None:
            raise ValueError("provide only one of stabilization or MRST alias S")
        self.operators = self._setup_operators(
            el_bc, load, alpha_scaling, stabilization if S is None else S,
            initial_stress,
        )
        # VEM mechanics is linear: pressure and moving displacement data only
        # change the right-hand side.  Fixed-stress may solve this same system
        # hundreds of times, so factor the immutable free-DOF stiffness once.
        # scipy's factorized callable is equivalent to repeated spsolve calls
        # but avoids repeated symbolic and numeric LU work.
        # Lazy because operator-only inspection may intentionally construct an
        # under-constrained (singular) mechanics problem without solving it.
        self._solve_stiffness = None
        # MRST scales the mechanics residual so it is comparable with the
        # flow residual in the coupled system (equationsPoroMechanics).
        self.res_scale = 1.0 / (1e6 * float(np.mean(g.cells.volumes)))

    def _setup_operators(self, el_bc: ElasticBC, load,
                         alpha_scaling: float, stabilization=None,
                         initial_stress=None) -> MechanicOperators:
        g = self.G
        dim = g.griddim
        ndof = dim * g.nodes.num

        S = vem_assemble(
            g, self.C, alpha_scaling=alpha_scaling,
            stabilization=stabilization,
        )

        # Dirichlet dofs, from the per-node component mask.
        bcnodes = np.unique(el_bc.disp_nodes)
        mask = np.zeros((bcnodes.size, dim), dtype=bool)
        vals = np.zeros((bcnodes.size, dim))
        inv = np.searchsorted(bcnodes, el_bc.disp_nodes)
        np.logical_or.at(mask, inv, el_bc.disp_mask[:, :dim].astype(bool))
        np.add.at(vals, inv,
                  el_bc.disp_vals[:, :dim] * el_bc.disp_mask[:, :dim].astype(float))
        counts = np.bincount(inv, minlength=bcnodes.size)[:, None]
        vals = np.divide(vals, np.maximum(counts, 1))

        dofs = dim * bcnodes[:, None] + np.arange(dim)
        is_dir = np.zeros(ndof, dtype=bool)
        is_dir[dofs[mask]] = True
        u_dir = np.zeros(ndof)
        u_dir[dofs[mask]] = vals[mask]
        free = ~is_dir

        A = S[free][:, free]
        A = ((A + A.T) / 2.0).tocsr()

        div_full = vem_div(g)
        div = div_full[:, free].tocsr()
        div_rhs = div_full @ u_dir

        # Load: body force, boundary tractions, and the Dirichlet reaction.
        f = np.zeros(ndof)
        if load is not None:
            q_cell = calculate_qc(g)[0] if dim == 3 else calculate_qf(g)[0]
            cells_nodes = g.cells.nodes
            nlc = np.diff(g.cells.nodePos)
            xc = np.repeat(g.cells.centroids[:, :dim], nlc, axis=0)
            rel_vec = -(xc - g.nodes.coords[cells_nodes, :dim])
            nodal_load = np.asarray(load(xc), dtype=float)[:, :dim] \
                * (q_cell[:, :dim] * rel_vec)
            fdofs = dim * cells_nodes[:, None] + np.arange(dim)
            np.add.at(f, fdofs.ravel(), nodal_load.ravel())

        strain_operator = vem_strain_operator(g)
        if initial_stress is not None:
            sigma0 = np.asarray(initial_stress, dtype=float)
            nlin = _nlin(g)
            if sigma0.shape == (nlin,):
                sigma0 = np.broadcast_to(sigma0, (g.cells.num, nlin))
            if sigma0.shape != (g.cells.num, nlin):
                raise ValueError(
                    f"initial_stress must have shape ({g.cells.num}, {nlin})"
                )
            weighted = (sigma0 * g.cells.volumes[:, None]).reshape(-1)
            f -= np.asarray(strain_operator.T @ weighted).reshape(-1)

        if el_bc.force_faces is not None and el_bc.force_faces.size > 0:
            faces = el_bc.force_faces
            lnn = np.diff(g.faces.nodePos)[faces]
            inodes = mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
            nodes = g.faces.nodes[inodes]
            if dim == 3:
                qf = calculate_qc(g)[1]
            else:
                qf = np.repeat(g.faces.areas / 2.0, np.diff(g.faces.nodePos))
            force = np.repeat(el_bc.force[:, :dim], lnn, axis=0) \
                * qf[inodes][:, None]
            fdofs = dim * nodes[:, None] + np.arange(dim)
            np.add.at(f, fdofs.ravel(), force.ravel())

        rhs = f[free] - S[free][:, is_dir] @ u_dir[is_dir]

        # Kept so prescribed displacements can be changed without
        # re-assembling the stiffness matrix - Mandel's rigid plate moves
        # every timestep, and reassembling VEM each time is unaffordable.
        self._S_free_dir = S[free][:, is_dir].tocsr()
        self._load = f
        self._div_full = div_full

        return MechanicOperators(
            A=A,
            gradP=div.T.tocsr(),
            div=div,
            div_rhs=div_rhs,
            strain=strain_operator,
            stress=vem_stress_operator(g, self.C),
            rhs=rhs,
            is_dir=is_dir,
            u_dir=u_dir,
            inv_ci=inv_ci(self.C),
        )

    def set_dirichlet(self, nodes: np.ndarray, component: int,
                      value: float) -> None:
        """Change a prescribed displacement in place (the moving rigid plate).

        Only the right-hand side depends on it, so the stiffness matrix -
        the expensive part - is untouched.
        """
        op = self.operators
        dim = self.G.griddim
        dofs = dim * np.asarray(nodes, dtype=np.int64) + component
        assert np.all(op.is_dir[dofs]), "those dofs are not Dirichlet"
        op.u_dir[dofs] = value
        op.rhs = (self._load[~op.is_dir]
                  - self._S_free_dir @ op.u_dir[op.is_dir])
        op.div_rhs = self._div_full @ op.u_dir

    @property
    def n_free(self) -> int:
        return int(np.count_nonzero(~self.operators.is_dir))

    # -- the residual and its solution ---------------------------------------
    def equations(self, xd, p):
        """MRST's `equationsPoroMechanics`: A xd - gradP (alpha p) - rhs."""
        op = self.operators
        from pymrst.utils.adi import ad_matmul
        eq = (ad_matmul(op.A, xd) - ad_matmul(op.gradP, self.alpha * p)
              - op.rhs)
        return eq * self.res_scale

    variable_names = ("xd", "uu", "u", "stress", "strain", "vdiv")
    primary_variable_names = ("xd",)

    def get_equations(self, state0: Any, state: Any, dt: float,
                      driving_forces: Any, **kwargs: Any):
        pressure = (driving_forces.get("fluidp")
                    if hasattr(driving_forces, "get")
                    else driving_forces.fluidp)
        if pressure is None:
            raise ValueError("mechanics equations require driving force 'fluidp'")
        return self.equations(state["xd"], pressure)

    getEquations = get_equations

    def get_valid_driving_forces(self) -> dict[str, Any]:
        return {"fluidp": None}

    getValidDrivingForces = get_valid_driving_forces

    def get_variable_field(self, name: str):
        aliases = {"uu": "u", "vdiv": "volumetric_strain"}
        if name not in self.variable_names:
            raise KeyError(f"Unknown mechanics variable {name!r}")
        return aliases.get(name, name), None if name == "xd" else slice(None)

    getVariableField = get_variable_field

    def get_all_vars_names(self) -> list[str]:
        return list(self.variable_names)

    getAllVarsNames = get_all_vars_names

    def get_all_primary_variables(self) -> list[str]:
        return list(self.primary_variable_names)

    getAllPrimaryVariables = get_all_primary_variables

    def update_state(self, state: Any, increment: Any):
        out = state.copy()
        dx = np.asarray(increment, dtype=float).ravel()
        if dx.size != self.n_free:
            raise ValueError(
                f"mechanics increment has {dx.size} entries, expected {self.n_free}"
            )
        out["xd"] = np.asarray(
            out.get("xd", np.zeros(self.n_free)), dtype=float
        ) + dx
        out["u"] = self.full_displacement(out["xd"])
        pressure = np.asarray(
            out.get("pressure", np.zeros(self.G.cells.num)), dtype=float
        )
        derived = self.diagnostics({"u": out["u"], "pressure": pressure})
        out.update(
            strain=derived["strain"],
            stress=derived["effective_stress"],
            vdiv=derived["volumetric_strain"],
        )
        return out

    updateState = update_state

    def solve(self, p: np.ndarray) -> np.ndarray:
        """The displacement in equilibrium with pressure p (one linear solve)."""
        op = self.operators
        rhs = op.rhs + op.gradP @ (self.alpha * np.asarray(p, dtype=float))
        if self._solve_stiffness is None:
            self._solve_stiffness = spla.factorized(op.A.tocsc())
        return np.asarray(self._solve_stiffness(rhs), dtype=float)

    # -- derived quantities --------------------------------------------------
    def full_displacement(self, xd) -> np.ndarray:
        """Free dofs plus the prescribed Dirichlet values (MRST's `u`)."""
        op = self.operators
        from pymrst.utils.adi import ADI
        if isinstance(xd, ADI):
            u = ADI(op.u_dir.copy(),
                    [sp.csr_array((op.u_dir.size, j.shape[1]))
                     for j in xd.jac])
            free = np.flatnonzero(~op.is_dir)
            u.val[free] = xd.val
            for k, j in enumerate(xd.jac):
                rowmap = sp.csr_array(
                    (np.ones(free.size), (free, np.arange(free.size))),
                    shape=(op.u_dir.size, free.size),
                )
                u.jac[k] = (rowmap @ j).tocsr()
            return u
        u = op.u_dir.copy()
        u[~op.is_dir] = xd
        return u

    def volumetric_strain(self, xd):
        """div(u) per cell from the free dofs, using the *current* Dirichlet
        values. Correct for the state being solved for; see
        `volumetric_strain_full` for why an old state needs the other one."""
        from pymrst.utils.adi import ad_matmul
        op = self.operators
        return (ad_matmul(op.div, xd) + op.div_rhs) / self.G.cells.volumes

    def volumetric_strain_full(self, u):
        """div(u) per cell from the *full* displacement vector.

        This is what a stored state must use. `div_rhs` is built from the
        current prescribed displacements, so re-deriving an old state's
        div(u) from its free dofs would silently evaluate it against the
        *new* boundary values - and where those move in time (Mandel's plate
        settles every step), that cancels the very term the flow equation is
        there to see. MRST stores `state.u` for the same reason.
        """
        from pymrst.utils.adi import ad_matmul
        return ad_matmul(self._div_full, u) / self.G.cells.volumes

    def strain_full(self, u) -> np.ndarray:
        """Cell strain in Voigt form, (nc, nlin), from full node displacement.

        The component order is MRST/VEM Voigt order:
        3D ``[exx, eyy, ezz, eyz, exz, exy]``; 2D ``[exx, eyy, exy]``.
        """
        return (self.operators.strain @ np.asarray(u, dtype=float)).reshape(
            -1, _nlin(self.G)
        )

    def effective_stress_full(self, u) -> np.ndarray:
        """Cell effective stress in Voigt form, (nc, nlin), from full displacement."""
        return (self.operators.stress @ np.asarray(u, dtype=float)).reshape(
            -1, _nlin(self.G)
        )

    def effective_stress(self, xd) -> np.ndarray:
        """Cell effective stress in Voigt form, (nc, nlin)."""
        u = self.full_displacement(np.asarray(xd, dtype=float))
        return self.effective_stress_full(u)

    def total_stress_full(self, u, p: np.ndarray) -> np.ndarray:
        """Cell total stress ``sigma = sigma' - alpha*p*I`` in Voigt form."""
        p = np.asarray(p, dtype=float).ravel()
        sigma = self.effective_stress_full(u).copy()
        sigma -= (self.alpha * p)[:, None] * _trace_vector(self.G)
        return sigma

    def diagnostics(self, state, pressure_key: str = "pressure") -> dict:
        """Derived mechanics fields for a saved state.

        Returns full-cell strain, effective stress, total stress and scalar
        summaries without mutating the simulation state.
        """
        u = np.asarray(state["u"], dtype=float)
        p = np.asarray(state[pressure_key], dtype=float)
        strain = self.strain_full(u)
        eff = self.effective_stress_full(u)
        total = eff - (self.alpha * p)[:, None] * _trace_vector(self.G)
        divu = self.volumetric_strain_full(u)
        dim = self.G.griddim
        return {
            "strain": strain,
            "volumetric_strain": np.asarray(divu, dtype=float),
            "effective_stress": eff,
            "total_stress": total,
            "displacement": u.reshape(-1, dim),
            "displacement_magnitude": np.linalg.norm(u.reshape(-1, dim), axis=1),
        }

    def fixed_stress_terms(self, xd, p: np.ndarray):
        """(sTerm, pTerm) with div(u) = sTerm + pTerm * p (MRST computeMechTerm).

        Exact, not an approximation: total stress is sigma_T = C eps - p I, so
        tr(eps) = tr(C^-1 sigma_T) + p tr(C^-1 I). The *splitting* is what
        lags sTerm; pTerm - the drained compressibility 1/K - is kept implicit
        in the flow solve, and that is what makes fixed-stress stable where
        the naive (drained) split is not.
        """
        inv_c_i = self.operators.inv_ci
        trace = _trace_vector(self.G)
        total = self.effective_stress(xd) - np.asarray(p)[:, None] * trace
        s_term = np.sum(inv_c_i * total, axis=1)
        p_term = inv_c_i @ trace
        return s_term, p_term


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c, enu2c_2d

    def _column(nz=6, L=10.0, E=1e9, nu=0.25):
        """A confined column: sides on rollers, base fixed, free top."""
        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, nz], [1.0, 1.0, L]))
        )
        pts = g.nodes.coords
        bot = np.flatnonzero(pts[:, 2] > L - 1e-9)
        nx = np.flatnonzero((pts[:, 0] < 1e-9) | (pts[:, 0] > 1 - 1e-9))
        ny = np.flatnonzero((pts[:, 1] < 1e-9) | (pts[:, 1] > 1 - 1e-9))
        dn = np.concatenate([bot, nx, ny])
        dm = np.vstack([
            np.tile([False, False, True], (bot.size, 1)),
            np.tile([True, False, False], (nx.size, 1)),
            np.tile([False, True, False], (ny.size, 1)),
        ])
        el_bc = ElasticBC(disp_nodes=dn, disp_mask=dm,
                          disp_vals=np.zeros((dn.size, 3)))
        C = enu2c(np.full(nz, E), np.full(nz, nu))
        return g, C, el_bc, E, nu, L

    def test_inv_ci_is_the_drained_compressibility():
        """tr(C^-1 I) = 1/K, the constant fixed-stress splitting relies on."""
        E, nu = 1e9, 0.25
        C = enu2c(np.array([E]), np.array([nu]))
        ici = inv_ci(C)
        K = E / (3.0 * (1.0 - 2.0 * nu))
        assert np.isclose(ici @ VOIGT_TRACE, 1.0 / K, rtol=1e-12)
        assert np.allclose(ici[0, 3:], 0.0)          # no shear from unit p
        print("ok test_inv_ci_is_the_drained_compressibility")

    def test_uniform_pressure_gives_uniaxial_strain():
        """A confined column under uniform pressure compacts by p/K_v.

        The closed form for uniaxial strain: div(u) = alpha p / (lambda + 2 mu).
        """
        g, C, el_bc, E, nu, L = _column()
        m = MechanicModel(g, C, alpha=1.0, el_bc=el_bc)
        p = np.full(g.cells.num, 1e6)
        xd = m.solve(p)
        divu = m.volumetric_strain(xd)
        kv = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))     # oedometer modulus
        assert np.allclose(divu, 1e6 / kv, rtol=1e-6), divu
        print("ok test_uniform_pressure_gives_uniaxial_strain")

    def test_residual_vanishes_at_the_solution():
        """equations() is zero where solve() puts it, and linear in p."""
        g, C, el_bc, *_ = _column()
        m = MechanicModel(g, C, alpha=1.0, el_bc=el_bc)
        p = np.linspace(1e5, 1e6, g.cells.num)
        xd = m.solve(p)
        r = m.equations(xd, p)
        assert np.max(np.abs(r)) < 1e-8 * max(1.0, np.max(np.abs(xd))), r.max()
        # linearity: doubling p doubles the displacement (zero load here)
        assert np.allclose(m.solve(2 * p), 2 * xd, rtol=1e-8)
        print("ok test_residual_vanishes_at_the_solution")

    def test_fixed_stress_terms_reproduce_div_u():
        """sTerm + pTerm * p must equal div(u) exactly - it is an identity,
        not an approximation, and if it drifts the splitting is wrong."""
        g, C, el_bc, *_ = _column()
        m = MechanicModel(g, C, alpha=1.0, el_bc=el_bc)
        p = np.linspace(1e5, 1e6, g.cells.num)
        xd = m.solve(p)
        s_term, p_term = m.fixed_stress_terms(xd, p)
        divu = m.volumetric_strain(xd)
        assert np.allclose(s_term + p_term * p, divu, rtol=1e-8), (
            np.max(np.abs(s_term + p_term * p - divu))
        )
        print("ok test_fixed_stress_terms_reproduce_div_u")

    def test_2d_uniform_pressure_gives_plane_strain_diagnostics():
        """2D VEM mechanics returns MRST-shaped strain/stress fields."""
        L = 6.0
        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 6], [1.0, L]))
        )
        pts = g.nodes.coords[:, :2]
        bot = np.flatnonzero(pts[:, 1] < 1e-9)
        left = np.flatnonzero(pts[:, 0] < 1e-9)
        right = np.flatnonzero(pts[:, 0] > 1.0 - 1e-9)
        dn = np.concatenate([bot, left, right])
        dm = np.vstack([
            np.tile([False, True], (bot.size, 1)),
            np.tile([True, False], (left.size, 1)),
            np.tile([True, False], (right.size, 1)),
        ])
        el_bc = ElasticBC(disp_nodes=dn, disp_mask=dm,
                          disp_vals=np.zeros((dn.size, 2)))
        E, nu = 1e9, 0.25
        C = enu2c_2d(np.full(g.cells.num, E), np.full(g.cells.num, nu))
        m = MechanicModel(g, C, alpha=1.0, el_bc=el_bc)
        p = np.full(g.cells.num, 1e6)
        xd = m.solve(p)
        divu = m.volumetric_strain(xd)
        kv = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        assert np.allclose(divu, 1e6 / kv, rtol=1e-6), divu

        u = m.full_displacement(xd)
        diag = m.diagnostics({"u": u, "pressure": p})
        assert diag["strain"].shape == (g.cells.num, 3)
        assert diag["effective_stress"].shape == (g.cells.num, 3)
        assert diag["total_stress"].shape == (g.cells.num, 3)
        assert diag["displacement"].shape == (g.nodes.num, 2)
        s_term, p_term = m.fixed_stress_terms(xd, p)
        assert np.allclose(s_term + p_term * p, divu, rtol=1e-8)
        print("ok test_2d_uniform_pressure_gives_plane_strain_diagnostics")

    test_inv_ci_is_the_drained_compressibility()
    test_uniform_pressure_gives_uniaxial_strain()
    test_residual_vanishes_at_the_solution()
    test_fixed_stress_terms_reproduce_div_u()
    test_2d_uniform_pressure_gives_plane_strain_diagnostics()
    print("\nAll tests passed!")
