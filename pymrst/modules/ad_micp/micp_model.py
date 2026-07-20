"""MICP reactive transport model (MRST ad-micp: MICPModel/equationsMICP).

Single-phase water flow (the 'oil' equation of the MATLAB model just
enforces sW = 1, so it is dropped here) with five components:
microbes, oxygen, urea, biofilm, calcite.

    pv/dt (poro - poro0) + div(vW)                                  = q_w
    pv/dt (poro m - poro0 m0) + div(vM - d_m grad m) - pv R_m       = q_m
    pv/dt (poro o - poro0 o0) + div(vO - d_o grad o) + pv R_o       = q_o
    pv/dt (poro u - poro0 u0) + div(vU - d_u grad u) + pv R_u       = q_u
    (b - b0)/dt - R_b = 0,   (c - c0)/dt - R_c = 0

Faithful to MRST, porosity (poro = poro_0 - b - c), the permeability it
induces, the dispersion tensors, and the detachment pressure gradient
dpW are all evaluated at the *current Newton iterate's values* (MATLAB
strips the AD from poro with `poro.val`), so those couplings are lagged
inside a Newton step rather than differentiated.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_micp.micp_fluid import MICPFluid
from pymrst.params.rock import Rock, make_rock
from pymrst.params.wells_and_bc import BoundaryCondition, Well
from pymrst.solvers.trans import compute_trans
from pymrst.utils.adi import (
    ADI,
    ad_matmul,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
)

NCOMP = 5  # m, o, u, b, c


@dataclass
class MICPState:
    """Reservoir state of the MICP model."""

    pressure: np.ndarray
    m: np.ndarray
    o: np.ndarray
    u: np.ndarray
    b: np.ndarray
    c: np.ndarray
    bhp: np.ndarray
    flux: Optional[np.ndarray] = None    # face fluxes, for dispersion

    def copy(self) -> "MICPState":
        return MICPState(
            self.pressure.copy(), self.m.copy(), self.o.copy(),
            self.u.copy(), self.b.copy(), self.c.copy(), self.bhp.copy(),
            None if self.flux is None else self.flux.copy(),
        )


@dataclass
class MICPWell:
    """Injection well with component concentrations (W.m/W.o/W.u)."""

    well: Well
    m: float = 0.0
    o: float = 0.0
    u: float = 0.0


@dataclass
class MICPBC:
    """Pressure boundary conditions with inflow component concentrations."""

    bc: BoundaryCondition
    m: Optional[np.ndarray] = None
    o: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None


class MICPModelAD:
    """Fully-implicit MICP model on a 3D grid."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: MICPFluid,
        wells: Sequence[MICPWell] = (),
        bc: Optional[MICPBC] = None,
        gravity: float = 0.0,
        tol: float = 1e-14,
        max_iterations: int = 25,
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.wells = list(wells)
        self.bc = bc
        self.tol = tol
        self.max_iterations = max_iterations

        self.poro0 = rock.poro.copy()
        self.pv = g.cells.volumes.copy()      # MRST uses raw cell volumes

        neighbors = g.faces.neighbors
        self.internal = np.all(neighbors >= 0, axis=1)
        self.N = neighbors[self.internal]
        nif = self.N.shape[0]
        nc = g.cells.num
        self.C = sp.csr_array(
            (np.tile([-1.0, 1.0], nif),
             (np.repeat(np.arange(nif), 2), self.N.ravel())),
            shape=(nif, nc),
        )
        self.gdz = gravity * (
            g.cells.centroids[self.N[:, 1], 2]
            - g.cells.centroids[self.N[:, 0], 2]
        )

    # -- helpers -----------------------------------------------------------
    def grad(self, x):
        return ad_matmul(self.C, x)

    def div(self, x):
        return ad_matmul(-self.C.T, x)

    def _face_trans(self, poro: np.ndarray) -> np.ndarray:
        """Interior face transmissibilities for the clogged permeability."""
        k = self.fluid.permeability(poro)
        rock = make_rock(self.g, np.atleast_2d(k).T, poro[:, None])
        t_half = compute_trans(self.g, rock)
        t_face = 1.0 / np.bincount(self.g.cells.faces, weights=1.0 / t_half,
                                   minlength=self.g.faces.num)
        return t_face[self.internal]

    def _full_flux(self, v_internal: np.ndarray) -> np.ndarray:
        """Internal fluxes padded with zeros on boundary faces (storeFluxes)."""
        out = np.zeros(self.g.faces.num)
        out[self.internal] = v_internal
        return out

    def _cell_velocity(self, flux: np.ndarray) -> np.ndarray:
        """faceFlux2cellVelocity: reconstruct a cell velocity vector."""
        g = self.g
        cell_no = g.cell_no()
        cf = g.cells.faces
        sgn = np.where(g.faces.neighbors[cf, 0] == cell_no, 1.0, -1.0)
        cflux = sgn * flux[cf]
        cvec = g.faces.centroids[cf] - g.cells.centroids[cell_no]
        v = np.zeros((g.cells.num, 3))
        np.add.at(v, cell_no, cflux[:, None] * cvec)
        return v / g.cells.volumes[:, None]

    def _dispersion(self, poro: np.ndarray, flux: np.ndarray,
                    well_cells: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray]:
        """Dispersion face coefficients and the detachment |grad p|."""
        f = self.fluid
        v = self._cell_velocity(flux)
        if well_cells.size:
            # MRST doubles the velocity in the boundary injection cell.
            v[well_cells] *= 2.0
        v_norm = np.sqrt(np.sum(v**2, axis=1))
        k = f.permeability(poro)
        dpw = f.muw * v_norm / k

        eps = np.finfo(float).eps
        dd = (f.alphaL - f.alphaT) * np.column_stack([
            poro * v[:, 0] * v[:, 0], poro * v[:, 0] * v[:, 1],
            poro * v[:, 0] * v[:, 2], poro * v[:, 1] * v[:, 1],
            poro * v[:, 1] * v[:, 2], poro * v[:, 2] * v[:, 2],
        ]) / (v_norm + eps)[:, None]
        iso = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])

        out = []
        for diff in (f.diffm, f.diffo, f.diffu):
            tensor = (poro * diff - f.alphaT * v_norm)[:, None] * iso + dd
            rock = make_rock(self.g, tensor,
                             np.ones((self.g.cells.num, 1)))
            t_half = compute_trans(self.g, rock, fix_negative=True)
            t_face = 1.0 / np.bincount(
                self.g.cells.faces, weights=1.0 / t_half,
                minlength=self.g.faces.num
            )
            out.append(t_face[self.internal])
        return out[0], out[1], out[2], dpw

    # -- assembly ----------------------------------------------------------
    def assemble(self, st0: MICPState, st: MICPState, p, m, o, u, b, c,
                 bhp, dt: float):
        f = self.fluid
        pv = self.pv

        # Porosity and everything it drives are evaluated on values only
        # (MATLAB: `poro = poro.val`), so these couplings lag in Newton.
        poro = self.poro0 - value(b) - value(c)
        poro0 = self.poro0 - st0.b - st0.c
        t = self._face_trans(poro)

        dp = self.grad(p) - f.rhoWS * self.gdz
        upw = value(dp) <= 0
        vW = -(1.0 / f.muw) * t * dp

        up = sp.csr_array(
            (np.ones(self.N.shape[0]),
             (np.arange(self.N.shape[0]),
              np.where(upw, self.N[:, 0], self.N[:, 1]))),
            shape=(self.N.shape[0], self.g.cells.num),
        )
        vM = ad_matmul(up, m) * (-(1.0 / f.muw) * t * dp)
        vO = ad_matmul(up, o) * (-(1.0 / f.muw) * t * dp)
        vU = ad_matmul(up, u) * (-(1.0 / f.muw) * t * dp)

        well_cells = np.concatenate(
            [w.well.cells for w in self.wells]
        ) if self.wells else np.zeros(0, dtype=np.int64)
        d_m, d_o, d_u, dpw = self._dispersion(
            poro, self._full_flux(value(vW)),
            well_cells if self._well_on_boundary
            else np.zeros(0, dtype=np.int64),
        )

        # Monod kinetics and the detachment term (values of poro/dpW).
        growth = f.Y * f.mu * (o / (f.k_o + o))
        detach = f.k_str * (poro * dpw) ** 0.58
        urea_rate = f.mu_u * (u / (f.k_u + u))

        water = (pv / dt) * (poro - poro0) + self.div(vW)
        micro = ((pv / dt) * (poro * m - poro0 * st0.m)
                 + self.div(vM - d_m * self.grad(m))
                 - pv * (m * poro * (growth - f.k_d - f.k_a)
                         + b * f.rho_b * detach))
        oxy = ((pv / dt) * (poro * o - poro0 * st0.o)
               + self.div(vO - d_o * self.grad(o))
               + pv * (m * poro + f.rho_b * b) * f.F * f.mu
               * (o / (f.k_o + o)))
        urea = ((pv / dt) * (poro * u - poro0 * st0.u)
                + self.div(vU - d_u * self.grad(u))
                + pv * f.rho_b * b * urea_rate)
        bio = ((1.0 / dt) * (b - st0.b)
               - b * (growth - f.k_d
                      - f.rho_b * b * f.Yuc * urea_rate
                      / (f.rho_c * (poro + b))
                      - detach)
               - m * poro * f.k_a / f.rho_b)
        cal = ((1.0 / dt) * (c - st0.c)
               - f.rho_b * b * f.Yuc * urea_rate / f.rho_c)

        eqs = [water, micro, oxy, urea, bio, cal]
        ctrls = []

        # Wells: rate-controlled water injection carrying m/o/u.
        for k, mw in enumerate(self.wells):
            w = mw.well
            wc = w.cells
            wi = w.WI / f.muw
            qw = wi * (bhp[k:k + 1] - p[wc])
            inj = value(qw) > 0

            eqs[0][wc] = eqs[0][wc] - qw
            for i, (conc, cell_conc) in enumerate(
                ((mw.m, m), (mw.o, o), (mw.u, u))
            ):
                q_up = qw * (inj * conc + (~inj) * value(cell_conc[wc]))
                eqs[1 + i][wc] = eqs[1 + i][wc] - q_up
            if w.type == "rate":
                ctrls.append(ad_sum(qw) - w.val)
            elif w.type == "bhp":
                ctrls.append(bhp[k:k + 1] - w.val)
            else:
                raise ValueError(f"unsupported control '{w.type}'")

        # Pressure boundary conditions with inflow concentrations.
        if self.bc is not None and self.bc.bc.face.size > 0:
            bcf = self.bc.bc
            faces = bcf.face
            n = self.g.faces.neighbors[faces]
            cells = n.sum(axis=1) + 1
            k = f.permeability(poro)
            rock = make_rock(self.g, np.atleast_2d(k).T, poro[:, None])
            t_half = compute_trans(self.g, rock)
            t_all = 1.0 / np.bincount(self.g.cells.faces,
                                      weights=1.0 / t_half,
                                      minlength=self.g.faces.num)
            tb = t_all[faces] / f.muw
            qb = tb * (bcf.value - p[cells])
            inj = value(qb) > 0
            eqs[0][cells] = eqs[0][cells] - qb
            for i, (bc_conc, cell_conc) in enumerate((
                (self.bc.m, m), (self.bc.o, o), (self.bc.u, u)
            )):
                cval = np.zeros(faces.size) if bc_conc is None \
                    else np.asarray(bc_conc, float)
                q_up = qb * (inj * cval + (~inj) * value(cell_conc[cells]))
                eqs[1 + i][cells] = eqs[1 + i][cells] - q_up

        return eqs, ctrls, value(vW)

    @property
    def _well_on_boundary(self) -> bool:
        return getattr(self, "well_on_boundary", False)

    # -- stepping -----------------------------------------------------------
    def step(self, state: MICPState, dt: float
             ) -> Tuple[MICPState, bool, int]:
        st0 = state
        st = state.copy()
        f = self.fluid
        nc = self.g.cells.num
        nw = len(self.wells)

        for it in range(self.max_iterations):
            p, m, o, u, b, c, bhp = init_variables_adi(
                st.pressure, st.m, st.o, st.u, st.b, st.c, st.bhp
            )
            eqs, ctrls, flux = self.assemble(
                st0, st, p, m, o, u, b, c, bhp, dt
            )
            res = np.concatenate([value(e) for e in eqs]
                                 + [value(x) for x in ctrls])
            if np.max(np.abs(res)) <= self.tol:
                st.flux = self._full_flux(flux)
                return st, True, it

            all_eqs = eqs + ctrls
            eq = combine_equations(*all_eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)

            st.pressure = st.pressure + upd[:nc]
            st.m = np.clip(st.m + upd[nc:2 * nc], 0.0, f.Cmmax)
            st.o = np.clip(st.o + upd[2 * nc:3 * nc], 0.0, f.Comax)
            st.u = np.clip(st.u + upd[3 * nc:4 * nc], 0.0, f.Cumax)
            cap = self.poro0 - f.ptol
            st.b = np.clip(st.b + upd[4 * nc:5 * nc], 0.0, cap)
            st.c = np.clip(st.c + upd[5 * nc:6 * nc], 0.0, cap)
            st.bhp = st.bhp + upd[6 * nc:6 * nc + nw]

        p, m, o, u, b, c, bhp = init_variables_adi(
            st.pressure, st.m, st.o, st.u, st.b, st.c, st.bhp
        )
        eqs, ctrls, flux = self.assemble(st0, st, p, m, o, u, b, c, bhp, dt)
        res = np.concatenate([value(e) for e in eqs]
                             + [value(x) for x in ctrls])
        st.flux = self._full_flux(flux)
        return st, bool(np.max(np.abs(res)) <= self.tol), self.max_iterations

    def clogged(self, st: MICPState) -> bool:
        """True once any cell reaches the clogging porosity tolerance."""
        return bool(np.any(self.poro0 - st.b - st.c < self.fluid.ptol))


def initial_micp_state(g: Grid, p0: float, nw: int) -> MICPState:
    z = np.zeros(g.cells.num)
    return MICPState(
        pressure=np.full(g.cells.num, p0), m=z.copy(), o=z.copy(),
        u=z.copy(), b=z.copy(), c=z.copy(),
        bhp=np.full(nw, p0), flux=np.zeros(g.faces.num),
    )


if __name__ == "__main__":
    from pymrst.gridprocessing import compute_geometry, tensor_grid
    from pymrst.params.wells_and_bc import add_bc, add_well
    from pymrst.utils.units import ATM, DAY, HOUR, MINUTE

    def build_1d(nx_len=75.0):
        """The runMICP1DCase geometry, rock, fluid, well, and BC."""
        dw, dl, l_ = 0.5, 0.25, 25.0
        x = np.concatenate([
            [0.0, dw], np.arange(dw + dl, l_ + 1e-12, dl),
            nx_len * np.exp(np.arange(-1.075, 1e-12, 0.025)),
        ])
        g = compute_geometry(tensor_grid(x, np.array([0.0, 1.0]),
                                         np.array([0.0, 1.0])))
        nc = g.cells.num
        k0 = 1e-12 * np.ones(nc)
        poro = 0.2 * np.ones(nc)
        rock = make_rock(g, k0[:, None], poro[:, None])
        fluid = MICPFluid(K0=k0, poro0=poro, Comax=0.04, Cumax=60.0)

        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=2.0 / DAY, radius=0.15,
                     compi=np.array([1.0, 0.0]), sign=1)[0]
        mw = MICPWell(well=w, m=0.01, o=0.0, u=0.0)

        bd = np.any(g.faces.neighbors == -1, axis=1)
        fx = np.flatnonzero(bd & (np.abs(g.faces.normals[:, 0]) > 0)
                            & (g.faces.centroids[:, 0] > x[-2]))
        bc = add_bc(None, fx, "pressure", ATM, sat=np.array([[0.0, 0.0]]))
        micp_bc = MICPBC(bc=bc, m=np.zeros(fx.size), o=np.zeros(fx.size),
                         u=np.zeros(fx.size))

        model = MICPModelAD(g, rock, fluid, wells=[mw], bc=micp_bc)
        model.well_on_boundary = True
        return g, model

    def test_water_only_steady_state():
        """With no components, flow reaches the analytic 1D steady state."""
        g, model = build_1d()
        model.wells[0].m = 0.0
        st = initial_micp_state(g, ATM, 1)
        st, ok, its = model.step(st, 20 * MINUTE)
        assert ok, "Newton failed"
        # rate is honoured and pressure decreases away from the injector
        q = 2.0 / DAY
        assert np.all(np.diff(st.pressure) < 0)
        flux_in = st.flux[np.flatnonzero(
            np.all(g.faces.neighbors >= 0, axis=1))][0]
        assert np.isclose(flux_in, q, rtol=1e-6)
        print("ok test_water_only_steady_state")

    def test_microbe_injection_and_growth():
        """Injected microbes advect, then oxygen feeds growth and biofilm."""
        g, model = build_1d()
        st = initial_micp_state(g, ATM, 1)
        # Phase 1: inject microbes for 20 h
        for _ in range(60):
            st, ok, _ = model.step(st, 20 * MINUTE)
            assert ok
        assert st.m[0] > 1e-3, "microbes must reach the injection cell"
        assert st.m[0] <= model.fluid.Cmmax
        assert st.b[0] > 0, "attachment must create biofilm"
        assert np.all(st.c == 0), "no calcite without urea"
        front = np.flatnonzero(st.m > 1e-6).max()
        assert front < g.cells.num - 1, "microbe front must not break through"
        print("ok test_microbe_injection_and_growth")

    def test_calcite_clogs_and_reduces_permeability():
        """Urea injection with biofilm present precipitates calcite and
        lowers the permeability."""
        g, model = build_1d()
        st = initial_micp_state(g, ATM, 1)
        st.b = np.full(g.cells.num, 0.01)     # pre-existing biofilm
        model.wells[0].m = 0.0
        model.wells[0].u = 60.0
        k_before = model.fluid.permeability(model.poro0 - st.b - st.c)
        for _ in range(30):
            st, ok, _ = model.step(st, 20 * MINUTE)
            assert ok
        assert st.c[0] > 0, "calcite must precipitate"
        assert np.all(np.diff(st.c) <= 1e-12) or st.c[0] >= st.c[1]
        poro = model.poro0 - st.b - st.c
        k_after = model.fluid.permeability(poro)
        assert k_after[0] < k_before[0], "permeability must drop"
        assert st.u[0] > 0 and st.u[0] <= model.fluid.Cumax
        print("ok test_calcite_clogs_and_reduces_permeability")

    def test_water_mass_balance():
        """Incompressible steady state: every internal face carries the
        injected rate (state.flux holds internal faces only, as in MRST)."""
        g, model = build_1d()
        model.wells[0].m = 0.0
        st = initial_micp_state(g, ATM, 1)
        for _ in range(5):
            st, ok, _ = model.step(st, 20 * MINUTE)
            assert ok
        q = 2.0 / DAY
        internal = np.flatnonzero(np.all(g.faces.neighbors >= 0, axis=1))
        assert np.allclose(st.flux[internal], q, rtol=1e-6), (
            f"internal fluxes range "
            f"[{st.flux[internal].min():.3e}, {st.flux[internal].max():.3e}]"
            f" vs injected {q:.3e}"
        )
        print("ok test_water_mass_balance")

    test_water_only_steady_state()
    test_microbe_injection_and_growth()
    test_calcite_clogs_and_reduces_permeability()
    test_water_mass_balance()
    print("\nAll tests passed!")
