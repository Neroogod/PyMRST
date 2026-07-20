"""Experimental Torch-autograd oil-water backend model.

This module is a research/backend path, not a replacement for the validated
MRST-parity ADI models. It gives PyMRST a real two-phase residual, dense
autograd Jacobian, and Newton loop that can run on Torch CPU/CUDA for small
cases while the sparse GPU strategy is developed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from pymrst.backends import get_backend, torch_newton
from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_core.operators import setup_operators_tpfa
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Source


@dataclass
class BackendOWState:
    pressure: np.ndarray
    sW: np.ndarray

    def copy(self):
        return BackendOWState(self.pressure.copy(), self.sW.copy())


@dataclass
class BackendOWFluid:
    mu: np.ndarray
    n: np.ndarray
    c: np.ndarray
    p_ref: float = 0.0

    @classmethod
    def simple(
        cls,
        mu=(1e-3, 5e-3),
        n=(2.0, 2.0),
        c=(0.0, 0.0),
        p_ref=0.0,
    ):
        return cls(
            mu=np.asarray(mu, dtype=float),
            n=np.asarray(n, dtype=float),
            c=np.asarray(c, dtype=float),
            p_ref=float(p_ref),
        )


class BackendOilWaterModel:
    """Small backend-aware fully implicit oil-water model.

    Supported now: TPFA interior fluxes, source terms with phase split, pressure
    anchoring, Torch autograd Newton. Wells/gravity/control switching are still
    handled in the dedicated backend TPFA slice and will be lifted here next.
    """

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: BackendOWFluid,
        *,
        src: Optional[Source] = None,
        pressure_anchor: float | None = None,
        backend: str = "torch",
        device: str | None = None,
        dtype: str = "float64",
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.src = src
        self.backend = get_backend(backend, device=device, dtype=dtype)
        self.dtype = dtype
        self.ops = setup_operators_tpfa(g, rock)
        self.pv = np.asarray(pore_volume(g, rock), dtype=float)
        self.pressure_anchor = pressure_anchor
        self.N = np.asarray(self.ops.N, dtype=np.int64)
        self.T = np.asarray(self.ops.T, dtype=float)

    def initial_state(self, p0: float, sw0: float) -> BackendOWState:
        return BackendOWState(
            pressure=np.full(self.g.cells.num, float(p0)),
            sW=np.full(self.g.cells.num, float(sw0)),
        )

    def pack(self, state: BackendOWState):
        return np.concatenate([state.pressure, state.sW])

    def unpack_numpy(self, x) -> BackendOWState:
        arr = self.backend.to_numpy(x)
        nc = self.g.cells.num
        return BackendOWState(arr[:nc].copy(), arr[nc:].copy())

    def residual_torch(self, previous: BackendOWState, dt: float):
        torch = self.backend.torch
        nc = self.g.cells.num
        n0 = self.backend.as_int(self.N[:, 0])
        n1 = self.backend.as_int(self.N[:, 1])
        T = self.backend.asarray(self.T)
        pv = self.backend.asarray(self.pv)
        p0 = self.backend.asarray(previous.pressure)
        sw0 = self.backend.asarray(previous.sW)
        mu = self.backend.asarray(self.fluid.mu)
        n = self.backend.asarray(self.fluid.n)
        c = self.backend.asarray(self.fluid.c)
        p_ref = self.backend.asarray(self.fluid.p_ref)
        q_w, q_o = self._source_terms()
        q_w = self.backend.asarray(q_w)
        q_o = self.backend.asarray(q_o)

        def residual(x):
            p = x[:nc]
            sw = torch.clamp(x[nc:], 1e-8, 1.0 - 1e-8)
            so = 1.0 - sw
            bw = torch.exp(c[0] * (p - p_ref))
            bo = torch.exp(c[1] * (p - p_ref))
            bw0 = torch.exp(c[0] * (p0 - p_ref))
            bo0 = torch.exp(c[1] * (p0 - p_ref))
            mob_w = sw**n[0] / mu[0]
            mob_o = so**n[1] / mu[1]
            dp = p[n1] - p[n0]
            upstream = dp <= 0.0
            mwf = torch.where(upstream, mob_w[n0], mob_w[n1])
            mof = torch.where(upstream, mob_o[n0], mob_o[n1])
            bwf = torch.where(upstream, bw[n0], bw[n1])
            bof = torch.where(upstream, bo[n0], bo[n1])
            vw = -T * mwf * dp
            vo = -T * mof * dp
            water = (pv * bw * sw - pv * bw0 * sw0) / dt
            oil = (pv * bo * so - pv * bo0 * (1.0 - sw0)) / dt
            water = water + _torch_div(torch, nc, n0, n1, bwf * vw) - q_w
            oil = oil + _torch_div(torch, nc, n0, n1, bof * vo) - q_o
            if self.pressure_anchor is not None:
                oil = oil.clone()
                oil[0] = p[0] - self.backend.asarray(self.pressure_anchor)
            return torch.cat([water, oil])

        return residual

    def step(
        self,
        state: BackendOWState,
        dt: float,
        *,
        max_iterations: int = 15,
        tolerance: float = 1e-9,
    ):
        x, report = torch_newton(
            self.residual_torch(state, dt),
            self.pack(state),
            backend=self.backend,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return self.unpack_numpy(x), report

    def _source_terms(self) -> tuple[np.ndarray, np.ndarray]:
        nc = self.g.cells.num
        qw = np.zeros(nc)
        qo = np.zeros(nc)
        if self.src is None or self.src.cell.size == 0:
            return qw, qo
        sat = np.asarray(self.src.sat, dtype=float)
        if sat.ndim == 1:
            sat = sat[:, None]
        sw = sat[:, 0] if sat.shape[1] > 0 else np.ones(self.src.cell.size)
        so = sat[:, 1] if sat.shape[1] > 1 else 1.0 - sw
        np.add.at(qw, self.src.cell, self.src.rate * sw)
        np.add.at(qo, self.src.cell, self.src.rate * so)
        return qw, qo


def _torch_div(torch, nc, n0, n1, face_flux):
    out = torch.zeros(nc, dtype=face_flux.dtype, device=face_flux.device)
    out.index_add_(0, n0, -face_flux)
    out.index_add_(0, n1, face_flux)
    return out
