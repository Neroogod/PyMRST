"""Experimental Torch backend residual for live-oil black-oil systems."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from pymrst.backends import NewtonReport, get_backend, torch_jvp_newton_step
from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_blackoil.backend_blackoil_fluid import BackendBlackOilFluid
from pymrst.modules.ad_core.operators import setup_operators_tpfa
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well

ETOL = np.sqrt(np.finfo(float).eps)


@dataclass
class BackendBOState:
    pressure: np.ndarray
    s: np.ndarray
    rs: np.ndarray
    status: np.ndarray
    qWs: np.ndarray
    qOs: np.ndarray
    qGs: np.ndarray
    bhp: np.ndarray

    def copy(self) -> "BackendBOState":
        return BackendBOState(
            pressure=self.pressure.copy(),
            s=self.s.copy(),
            rs=self.rs.copy(),
            status=self.status.copy(),
            qWs=self.qWs.copy(),
            qOs=self.qOs.copy(),
            qGs=self.qGs.copy(),
            bhp=self.bhp.copy(),
        )

    @classmethod
    def from_bo_state(cls, st):
        return cls(
            pressure=np.asarray(st.pressure, dtype=float).copy(),
            s=np.asarray(st.s, dtype=float).copy(),
            rs=np.asarray(st.rs, dtype=float).copy(),
            status=np.asarray(st.status, dtype=np.int64).copy(),
            qWs=np.asarray(st.qWs, dtype=float).copy(),
            qOs=np.asarray(st.qOs, dtype=float).copy(),
            qGs=np.asarray(st.qGs, dtype=float).copy(),
            bhp=np.asarray(st.bhp, dtype=float).copy(),
        )

    @classmethod
    def from_state(cls, st):
        return cls(
            pressure=np.asarray(st["pressure"], dtype=float).copy(),
            s=np.column_stack([
                np.asarray(st["sW"], dtype=float),
                np.asarray(st.get("sO", 1.0 - st["sW"] - st["sG"]), dtype=float),
                np.asarray(st["sG"], dtype=float),
            ]),
            rs=np.asarray(st["rs"], dtype=float).copy(),
            status=np.asarray(st["status"], dtype=np.int64).copy(),
            qWs=np.asarray(st["qWs"], dtype=float).copy(),
            qOs=np.asarray(st["qOs"], dtype=float).copy(),
            qGs=np.asarray(st["qGs"], dtype=float).copy(),
            bhp=np.asarray(st["bhp"], dtype=float).copy(),
        )

    def to_state(self):
        from pymrst.modules.ad_core.statefunctions.state_function import State

        return State(
            pressure=self.pressure.copy(),
            sW=self.s[:, 0].copy(),
            sO=self.s[:, 1].copy(),
            sG=self.s[:, 2].copy(),
            rs=self.rs.copy(),
            status=self.status.copy(),
            qWs=self.qWs.copy(),
            qOs=self.qOs.copy(),
            qGs=self.qGs.copy(),
            bhp=self.bhp.copy(),
        )


class BackendBlackOilResidual:
    """Torch residual matching the current fixed-status black-oil equations."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: BackendBlackOilFluid,
        wells: Sequence[Well],
        *,
        gravity: float = 0.0,
        backend: str = "torch",
        device: str | None = None,
        dtype: str = "float64",
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.wells = list(wells)
        self.backend = get_backend(backend, device=device, dtype=dtype)
        self.ops = setup_operators_tpfa(g, rock)
        self.N = np.asarray(self.ops.N, dtype=np.int64)
        self.T = np.asarray(self.ops.T, dtype=float)
        self.pv_ref = np.asarray(pore_volume(g, rock), dtype=float)
        self.gdz = gravity * (g.cells.centroids[self.N[:, 1], 2] - g.cells.centroids[self.N[:, 0], 2])

    def pack(self, st: BackendBOState) -> np.ndarray:
        x = np.where(st.status == 1, st.rs, st.s[:, 2])
        return np.concatenate([st.pressure, st.s[:, 0], x, st.qWs, st.qOs, st.qGs, st.bhp])

    def residual(self, st0: BackendBOState, st_template: BackendBOState, dt: float):
        torch = self.backend.torch
        be = self.backend
        nc = self.g.cells.num
        nw = len(self.wells)
        n0 = be.as_int(self.N[:, 0])
        n1 = be.as_int(self.N[:, 1])
        T = be.asarray(self.T)
        gdz = be.asarray(self.gdz)
        pv_ref = be.asarray(self.pv_ref)
        status = np.asarray(st_template.status, dtype=np.int64)
        st1 = be.asarray((status == 1).astype(float))
        st3 = be.asarray((status == 3).astype(float))
        st3_bool = status == 3
        p0 = be.asarray(st0.pressure)
        s0 = be.asarray(st0.s)
        rs0 = be.asarray(st0.rs)
        st3_0 = st0.status == 3

        def fun(xvec):
            p = xvec[:nc]
            sw = xvec[nc:2 * nc]
            x = xvec[2 * nc:3 * nc]
            qws = xvec[3 * nc:3 * nc + nw]
            qos = xvec[3 * nc + nw:3 * nc + 2 * nw]
            qgs = xvec[3 * nc + 2 * nw:3 * nc + 3 * nw]
            bhp = xvec[3 * nc + 3 * nw:3 * nc + 4 * nw]

            rs_sat = self.fluid.rs_sat(p)
            rs = x * st1 + rs_sat * (1.0 - st1)
            sg = x * st3
            so = 1.0 - sw - sg

            bw = self.fluid.b_w(p)
            bo = self.fluid.b_o(p, rs, st3_bool)
            bg = self.fluid.b_g(p)
            mob_w = self.fluid.kr_w(sw) / self.fluid.mu_w(p)
            mob_o = self.fluid.kr_o(sw, so, sg) / self.fluid.mu_o(p, rs, st3_bool)
            mob_g = self.fluid.kr_g(sg) / self.fluid.mu_g(p)

            rho_ws, rho_os, rho_gs = [be.asarray(v) for v in self.fluid.rhoS]
            rho_w = bw * rho_ws
            rho_o = bo * (rho_os + rs * rho_gs)
            rho_g = bg * rho_gs

            bwvw = self._phase_surface_flux(torch, p, mob_w, rho_w, bw, T, gdz, n0, n1)
            bovo = self._phase_surface_flux(torch, p, mob_o, rho_o, bo, T, gdz, n0, n1)
            bgvg = self._phase_surface_flux(torch, p, mob_g, rho_g, bg, T, gdz, n0, n1)
            rsbovo = self._upwind(p, rs, n0, n1, rho_o, gdz) * bovo

            pv = pv_ref * self.fluid.pv_mult(p)
            pv0 = pv_ref * self.fluid.pv_mult(p0)
            bw0 = self.fluid.b_w(p0)
            bo0 = self.fluid.b_o(p0, rs0, st3_0)
            bg0 = self.fluid.b_g(p0)
            water = (pv * bw * sw - pv0 * bw0 * s0[:, 0]) / dt + _torch_div(torch, nc, n0, n1, bwvw)
            oil = (pv * bo * so - pv0 * bo0 * s0[:, 1]) / dt + _torch_div(torch, nc, n0, n1, bovo)
            gas = (
                pv * (bg * sg + rs * bo * so)
                - pv0 * (bg0 * s0[:, 2] + rs0 * bo0 * s0[:, 1])
            ) / dt + _torch_div(torch, nc, n0, n1, bgvg + rsbovo)

            well_eqs = [[], [], []]
            ctrls = []
            for k, w in enumerate(self.wells):
                cq_s, weq, ctrl = self._well(k, p, (mob_w, mob_o, mob_g), (bw, bo, bg), rs,
                                              (qws[k], qos[k], qgs[k]), bhp[k])
                wc = be.as_int(w.cells)
                water = water.clone()
                oil = oil.clone()
                gas = gas.clone()
                water[wc] = water[wc] - cq_s[0]
                oil[wc] = oil[wc] - cq_s[1]
                gas[wc] = gas[wc] - cq_s[2]
                for ph in range(3):
                    well_eqs[ph].append(weq[ph].reshape(1))
                ctrls.append(ctrl.reshape(1))
            return torch.cat([
                water, oil, gas,
                torch.cat(well_eqs[0]), torch.cat(well_eqs[1]), torch.cat(well_eqs[2]),
                torch.cat(ctrls),
            ])

        return fun

    def _phase_surface_flux(self, torch, p, mob, rho, b, T, gdz, n0, n1):
        pot = p[n1] - p[n0] - 0.5 * (rho[n0] + rho[n1]) * gdz
        flag = pot <= 0.0
        mobf = torch.where(flag, mob[n0], mob[n1])
        bf = torch.where(flag, b[n0], b[n1])
        return bf * (-T * mobf * pot)

    def _upwind(self, p, x, n0, n1, rho, gdz):
        pot = p[n1] - p[n0] - 0.5 * (rho[n0] + rho[n1]) * gdz
        return self.backend.torch.where(pot <= 0.0, x[n0], x[n1])

    def _well(self, k, p, mob, b, rs, q_s, bhp_k):
        torch = self.backend.torch
        be = self.backend
        w = self.wells[k]
        wc = be.as_int(w.cells)
        tw = be.asarray(w.WI)
        compi = be.asarray(w.compi[:3])
        dd = -bhp_k + p[wc]
        inj = dd <= 0.0
        prod = ~inj
        mob_c = [m[wc] for m in mob]
        b_c = [x[wc] for x in b]
        rs_c = rs[wc]

        cq_ps = [torch.zeros_like(dd), torch.zeros_like(dd), torch.zeros_like(dd)]
        mix = [compi[0], compi[1], compi[2]]
        if bool(torch.any(prod).detach().cpu()):
            con = prod.to(dd.dtype) * tw
            cq_p = [-con * m * dd for m in mob_c]
            cq_ps = [b_c[0] * cq_p[0], b_c[1] * cq_p[1], b_c[2] * cq_p[2] + rs_c * (b_c[1] * cq_p[1])]
            wbq = [q_s[ph] * (q_s[ph].detach() > 0.0).to(dd.dtype) - torch.sum(cq_ps[ph]) for ph in range(3)]
            wbqt = wbq[0] + wbq[1] + wbq[2]
            if abs(float(wbqt.detach().cpu())) > 0.0:
                mix = [wq / wbqt for wq in wbq]
        if bool(torch.any(inj).detach().cpu()):
            mt = mob_c[0] + mob_c[1] + mob_c[2]
            cqt_i = -(inj.to(dd.dtype) * tw) * (mt * dd)
            vol_rat = mix[0] / b_c[0] + mix[1] / b_c[1] + (mix[2] - rs_c * mix[1]) / b_c[2]
            cqt_is = cqt_i / vol_rat
            cq_s = [cq_ps[ph] + mix[ph] * cqt_is for ph in range(3)]
        else:
            cq_s = cq_ps
        weq = [q_s[ph] - torch.sum(cq_s[ph]) for ph in range(3)]
        if w.type == "bhp":
            ctrl = bhp_k - be.asarray(w.val)
        elif w.type == "rate":
            ctrl = q_s[0] + q_s[1] + q_s[2] - be.asarray(w.val)
        elif w.type == "orat":
            ctrl = q_s[1] - be.asarray(w.val)
        else:
            raise ValueError(f"unsupported control {w.type!r}")
        return cq_s, weq, ctrl


class BackendBlackOilModel:
    """Experimental Torch GenericBlackOil-style backend for live-oil systems.

    This is the GPU/tensor execution path for the same primary variables used
    by ``ThreePhaseBlackOilModelAD``: pressure, water saturation, and a
    hydrocarbon variable that switches between ``Rs`` and ``sG`` according to
    the black-oil status. Residual/Jacobian assembly is Torch-native and can
    run on CPU or CUDA. The current Newton solve uses dense Torch autograd,
    which is appropriate for SPE1-sized validation and backend development;
    large-grid sparse CUDA solving plugs in below this class.
    """

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: BackendBlackOilFluid,
        wells: Sequence[Well],
        *,
        gravity: float = 0.0,
        backend: str = "torch",
        device: str | None = None,
        dtype: str = "float64",
        max_iterations: int = 15,
        tolerance: float = 1e-8,
        damping: float = 1.0,
        ds_max: float = 0.2,
        linear_solver: str = "dense",
        jacobian_drop_tolerance: float = 0.0,
        gmres_tolerance: float = 1e-8,
        gmres_max_iterations: int = 80,
        gmres_restart: int | None = None,
        gmres_scaling: str = "none",
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.wells = list(wells)
        self.backend = get_backend(backend, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.ds_max = float(ds_max)
        self.linear_solver = linear_solver
        self.jacobian_drop_tolerance = float(jacobian_drop_tolerance)
        self.gmres_tolerance = float(gmres_tolerance)
        self.gmres_max_iterations = int(gmres_max_iterations)
        self.gmres_restart = gmres_restart
        self.gmres_scaling = gmres_scaling
        self.residual_engine = BackendBlackOilResidual(
            g,
            rock,
            fluid,
            self.wells,
            gravity=gravity,
            backend=self.backend,
            device=device,
            dtype=dtype,
        )

    @property
    def G(self) -> Grid:
        return self.g

    def pack(self, state: BackendBOState) -> np.ndarray:
        return self.residual_engine.pack(state)

    def unpack_numpy(self, x, template: BackendBOState) -> BackendBOState:
        arr = self.backend.to_numpy(x)
        nc = self.g.cells.num
        nw = len(self.wells)
        out = BackendBOState.from_bo_state(template)
        out.pressure = arr[:nc].copy()
        out.s[:, 0] = arr[nc:2 * nc]
        hydro = arr[2 * nc:3 * nc]
        out.rs = np.where(out.status == 1, hydro, out.rs)
        out.s[:, 2] = np.where(out.status == 3, hydro, out.s[:, 2])
        out.s[:, 1] = 1.0 - out.s[:, 0] - out.s[:, 2]
        off = 3 * nc
        out.qWs = arr[off:off + nw].copy()
        out.qOs = arr[off + nw:off + 2 * nw].copy()
        out.qGs = arr[off + 2 * nw:off + 3 * nw].copy()
        out.bhp = arr[off + 3 * nw:off + 4 * nw].copy()
        return out

    def residual_torch(self, previous: BackendBOState, template: BackendBOState, dt: float):
        return self.residual_engine.residual(previous, template, dt)

    def step(self, state: BackendBOState, dt: float) -> tuple[BackendBOState, NewtonReport]:
        if not self.backend.is_torch:
            raise ValueError("BackendBlackOilModel.step currently requires backend='torch'")
        torch = self.backend.torch
        st0 = state.copy()
        st = state.copy()
        report = NewtonReport(converged=False, iterations=0)
        for it in range(1, self.max_iterations + 1):
            residual = self.residual_torch(st0, st, dt)
            x = self.backend.asarray(self.pack(st)).clone().detach().requires_grad_(True)
            r = residual(x)
            norm = float(torch.linalg.norm(r.detach()).cpu())
            report.residual_norms.append(norm)
            if norm <= self.tolerance:
                report.converged = True
                report.iterations = it - 1
                self._apply_control_limits(st)
                return st, report
            if self.linear_solver.lower() in ("matrix_free_gmres", "matrix_free_gmres_fallback"):
                dx, linear_report = self._solve_newton_system_matrix_free(residual, x, -r)
                report.linear_iterations.append(linear_report.iterations)
                report.linear_residuals.append(linear_report.residual_norm)
                if not linear_report.converged:
                    if self.linear_solver.lower() != "matrix_free_gmres_fallback":
                        report.converged = False
                        return st, report
                    jac = torch.autograd.functional.jacobian(residual, x, vectorize=True)
                    dx = torch.linalg.solve(jac, -r).detach().cpu().numpy()
            else:
                jac = torch.autograd.functional.jacobian(residual, x, vectorize=True)
                dx = self._solve_newton_system(jac, -r)
            st = self._update_state(st, dx)
            report.iterations = it

        residual = self.residual_torch(st0, st, dt)
        x = self.backend.asarray(self.pack(st)).clone().detach()
        final = float(torch.linalg.norm(residual(x).detach()).cpu())
        report.residual_norms.append(final)
        report.converged = final <= self.tolerance
        if report.converged:
            self._apply_control_limits(st)
        return st, report

    def simulate(self, initial_state: BackendBOState, timesteps) -> tuple[list[BackendBOState], list[NewtonReport]]:
        states: list[BackendBOState] = []
        reports: list[NewtonReport] = []
        state = BackendBOState.from_bo_state(initial_state)
        for dt in timesteps:
            state, report = self.step(state, float(dt))
            states.append(state)
            reports.append(report)
        return states, reports

    def _solve_newton_system(self, jac, rhs) -> np.ndarray:
        solver = self.linear_solver.lower()
        if solver == "dense":
            return self.backend.torch.linalg.solve(jac, rhs).detach().cpu().numpy()
        if solver in ("scipy_sparse", "cupy_sparse"):
            jac_np = jac.detach().cpu().numpy()
            if self.jacobian_drop_tolerance > 0.0:
                jac_np[np.abs(jac_np) <= self.jacobian_drop_tolerance] = 0.0
            jac_csr = sp.csr_array(jac_np)
            be = get_backend(
                "torch",
                device=self.device,
                dtype=self.dtype,
                linear_solver="cupy" if solver == "cupy_sparse" else "scipy",
            )
            out = be.solve(jac_csr, rhs.detach().cpu().numpy())
            return be.to_numpy(out)
        raise ValueError(
            "linear_solver must be 'dense', 'scipy_sparse', 'cupy_sparse', "
            "'matrix_free_gmres', or 'matrix_free_gmres_fallback'"
        )

    def _solve_newton_system_matrix_free(self, residual, x, rhs):
        left_scale, right_scale = self._gmres_scales(x, rhs)
        dx, report = torch_jvp_newton_step(
            residual,
            x.detach(),
            rhs.detach(),
            tolerance=self.gmres_tolerance,
            max_iterations=self.gmres_max_iterations,
            restart=self.gmres_restart,
            left_scale=left_scale,
            right_scale=right_scale,
        )
        return dx.detach().cpu().numpy(), report

    def _gmres_scales(self, x, rhs):
        scaling = self.gmres_scaling.lower()
        if scaling in ("none", "off", ""):
            return None, None
        if scaling not in ("row", "variable", "block"):
            raise ValueError("gmres_scaling must be 'none', 'row', 'variable', or 'block'")
        torch = self.backend.torch
        nc = self.g.cells.num
        nw = len(self.wells)
        eps = torch.as_tensor(1e-30, dtype=x.dtype, device=x.device)

        def scalar_scale(vals, floor):
            v = torch.as_tensor(floor, dtype=x.dtype, device=x.device)
            if vals.numel():
                v = torch.maximum(v, torch.quantile(torch.abs(vals.detach()).reshape(-1), 0.5))
            return v

        p_scale = scalar_scale(x[:nc], 1e5)
        sw_scale = torch.as_tensor(1.0, dtype=x.dtype, device=x.device)
        h_scale = scalar_scale(x[2 * nc:3 * nc], 1.0)
        q_vals = x[3 * nc:3 * nc + 3 * nw]
        q_scale = scalar_scale(q_vals, 1.0)
        bhp_scale = scalar_scale(x[3 * nc + 3 * nw:3 * nc + 4 * nw], 1e5)
        right = torch.cat([
            torch.ones(nc, dtype=x.dtype, device=x.device) * p_scale,
            torch.ones(nc, dtype=x.dtype, device=x.device) * sw_scale,
            torch.ones(nc, dtype=x.dtype, device=x.device) * h_scale,
            torch.ones(nw, dtype=x.dtype, device=x.device) * q_scale,
            torch.ones(nw, dtype=x.dtype, device=x.device) * q_scale,
            torch.ones(nw, dtype=x.dtype, device=x.device) * q_scale,
            torch.ones(nw, dtype=x.dtype, device=x.device) * bhp_scale,
        ])

        row_scales = []
        for lo, hi, floor in (
            (0, nc, 1.0),
            (nc, 2 * nc, 1.0),
            (2 * nc, 3 * nc, 1.0),
            (3 * nc, 3 * nc + nw, 1.0),
            (3 * nc + nw, 3 * nc + 2 * nw, 1.0),
            (3 * nc + 2 * nw, 3 * nc + 3 * nw, 1.0),
            (3 * nc + 3 * nw, 3 * nc + 4 * nw, 1e5),
        ):
            scale = scalar_scale(rhs[lo:hi], floor)
            row_scales.append(
                torch.ones(hi - lo, dtype=x.dtype, device=x.device)
                * (1.0 / torch.maximum(scale, eps))
            )
        left = torch.cat(row_scales)
        if scaling == "row":
            return left, None
        if scaling == "variable":
            return None, right
        return left, right

    def _flash(self, st: BackendBOState, prev: BackendBOState, st1_prev: np.ndarray):
        be = self.backend
        torch = be.torch
        p = be.asarray(st.pressure)
        rs_sat = self.fluid.rs_sat(p).detach().cpu().numpy()
        rs_sat0 = self.fluid.rs_sat(be.asarray(prev.pressure)).detach().cpu().numpy()

        sw, so, sg = st.s[:, 0], st.s[:, 1], st.s[:, 2]
        rs = st.rs
        wat_only = sw > 1.0 - ETOL
        gas_present = ((sg > 0) | (rs == 0)) & ~st1_prev
        gas_present |= wat_only
        gas_present |= (sg < 0) & (prev.s[:, 2] > ETOL)
        ix2 = (rs > rs_sat * (1 + ETOL)) & st1_prev \
            & (prev.rs > rs_sat0 * (1 - ETOL))
        sg[ix2] = 0.0
        gas_present |= ix2

        ix = sg < 0
        sw[ix] /= 1.0 - sg[ix]
        so[ix] /= 1.0 - sg[ix]
        sg[ix] = 0.0
        ix = so < 0
        sw[ix] /= 1.0 - so[ix]
        sg[ix] /= 1.0 - so[ix]
        so[ix] = 0.0
        ix = sw < 0
        so[ix] /= 1.0 - sw[ix]
        sg[ix] /= 1.0 - sw[ix]
        sw[ix] = 0.0

        rs[gas_present] = rs_sat[gas_present]
        rs[~gas_present] = np.minimum(rs_sat[~gas_present], rs[~gas_present])
        st.rs = np.maximum(rs, 0.0)
        st.status = 1 + 2 * gas_present.astype(np.int64)
        s = np.column_stack([sw, so, sg])
        st.s = s / s.sum(axis=1, keepdims=True)

    def _update_state(self, st: BackendBOState, increment: np.ndarray) -> BackendBOState:
        nc = self.g.cells.num
        nw = len(self.wells)
        prev = st
        new = st.copy()
        st1 = prev.status == 1
        st3 = prev.status == 3

        dp = increment[:nc]
        dsw = increment[nc:2 * nc]
        dx = increment[2 * nc:3 * nc]
        new.pressure = prev.pressure + dp
        dsg = np.where(st3, dx, 0.0)
        new.rs = np.maximum(prev.rs + np.where(st1, dx, 0.0), 0.0)
        ds = np.column_stack([dsw, -(dsg + dsw), dsg])
        ds = np.clip(ds, -self.ds_max, self.ds_max)
        new.s = prev.s + ds

        off = 3 * nc
        new.qWs = prev.qWs + increment[off:off + nw]
        new.qOs = prev.qOs + increment[off + nw:off + 2 * nw]
        new.qGs = prev.qGs + increment[off + 2 * nw:off + 3 * nw]
        new.bhp = prev.bhp + increment[off + 3 * nw:off + 4 * nw]
        self._flash(new, prev, st1)
        return new

    def _apply_control_limits(self, st: BackendBOState):
        updated = []
        for k, w in enumerate(self.wells):
            lims = w.lims or {}
            new_w = w
            liquid = st.qWs[k] + st.qOs[k]
            total = liquid + st.qGs[k]
            if w.sign >= 0:
                if w.type != "bhp" and "bhp" in lims and st.bhp[k] > lims["bhp"]:
                    new_w = replace(w, type="bhp", val=float(lims["bhp"]))
                elif w.type == "bhp" and "rate" in lims and total > lims["rate"]:
                    new_w = replace(w, type="rate", val=float(lims["rate"]))
            else:
                prod_o = -st.qOs[k]
                prod_g = -st.qGs[k]
                prod_l = -liquid
                prod_t = -total
                if w.type != "bhp" and "bhp" in lims and st.bhp[k] < lims["bhp"]:
                    new_w = replace(w, type="bhp", val=float(lims["bhp"]))
                elif w.type == "bhp":
                    if "orat" in lims and prod_o > lims["orat"]:
                        new_w = replace(w, type="orat", val=-float(lims["orat"]))
                    elif "grat" in lims and prod_g > lims["grat"]:
                        new_w = replace(w, type="rate", val=-float(lims["grat"]))
                    elif "lrat" in lims and prod_l > lims["lrat"]:
                        new_w = replace(w, type="rate", val=-float(lims["lrat"]))
                    elif "vrat" in lims and prod_t > lims["vrat"]:
                        new_w = replace(w, type="rate", val=-float(lims["vrat"]))
            updated.append(new_w)
        if any(a is not b for a, b in zip(updated, self.wells)):
            self.wells = updated
            self.residual_engine.wells = updated


def _torch_div(torch, nc, n0, n1, face_flux):
    out = torch.zeros(nc, dtype=face_flux.dtype, device=face_flux.device)
    out.index_add_(0, n0, face_flux)
    out.index_add_(0, n1, -face_flux)
    return out
