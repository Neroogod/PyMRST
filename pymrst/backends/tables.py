"""Backend table interpolation for Torch/NumPy black-oil properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pymrst.backends import ArrayBackend, get_backend


@dataclass
class BackendPVTTable:
    key: np.ndarray
    branches: list[tuple[np.ndarray, np.ndarray]]
    sat_x: np.ndarray
    sat_f: np.ndarray

    @classmethod
    def from_numpy_table(cls, table) -> "BackendPVTTable":
        return cls(
            key=np.asarray(table.key, dtype=float),
            branches=[(np.asarray(p, dtype=float), np.asarray(f, dtype=float))
                      for p, f in table.branches],
            sat_x=np.asarray(table.sat_x, dtype=float),
            sat_f=np.asarray(table.sat_f, dtype=float),
        )


def interp_table_backend(x_tab, f_tab, x, backend: str | ArrayBackend = "torch"):
    """Linear interpolation with MRST-style linear extrapolation."""
    be = get_backend(backend) if not isinstance(backend, ArrayBackend) else backend
    if be.is_torch:
        return _torch_interp1(be, x_tab, f_tab, x)[0]
    from pymrst.utils.interp_table import interp_table

    return interp_table(np.asarray(x_tab, dtype=float), np.asarray(f_tab, dtype=float), x)


def pvt_interp_backend(
    tab: BackendPVTTable,
    p,
    rs,
    sat_flag,
    backend: str | ArrayBackend = "torch",
):
    """Backend version of MRST/PyMRST PVTO `parallel` interpolation."""
    be = get_backend(backend) if not isinstance(backend, ArrayBackend) else backend
    if not be.is_torch:
        from pymrst.modules.ad_props.utils.interp_pvt import PVTTable, pvt_interp

        np_tab = PVTTable(tab.key, tab.branches, tab.sat_x, tab.sat_f)
        return pvt_interp(np_tab, p, rs, sat_flag)

    torch = be.torch
    p = be.asarray(p)
    rs = be.asarray(rs)
    sat = torch.as_tensor(sat_flag, device=p.device, dtype=torch.bool).clone()
    key = be.asarray(tab.key)
    sat_x = be.asarray(tab.sat_x)
    sat_f = be.asarray(tab.sat_f)
    sat = torch.where(p < sat_x[0], torch.zeros_like(sat), sat)
    out = torch.zeros_like(p)

    if torch.any(sat):
        out = torch.where(sat, _torch_interp1(be, sat_x, sat_f, p)[0], out)

    usat = ~sat
    if torch.any(usat):
        pu = p[usat]
        ru = rs[usat]
        bin_, w = _torch_bins(be, ru, key)
        ps = _torch_interp1(be, key, sat_x, ru)[0]
        dp = pu - ps
        f_l = torch.zeros_like(pu)
        f_r = torch.zeros_like(pu)
        for tn, (px_np, fx_np) in enumerate(tab.branches):
            px = be.asarray(px_np)
            fx = be.asarray(fx_np)
            dpx = px - px[0]
            ixl = bin_ == (tn - 1)
            ixr = bin_ == tn
            if torch.any(ixl):
                f_l = torch.where(ixl, _torch_interp1(be, dpx, fx, dp)[0], f_l)
            if torch.any(ixr):
                f_r = torch.where(ixr, _torch_interp1(be, dpx, fx, dp)[0], f_r)
        fu = f_l * w + f_r * (1.0 - w)
        out = out.clone()
        out[usat] = fu
    return out


def _torch_interp1(be: ArrayBackend, x_tab, f_tab, x):
    torch = be.torch
    xt = be.asarray(x_tab).contiguous()
    ft = be.asarray(f_tab).contiguous()
    x = be.asarray(x)
    idx = torch.searchsorted(xt, x, right=True) - 1
    idx = torch.clamp(idx, 0, xt.numel() - 2)
    x0, x1 = xt[idx], xt[idx + 1]
    f0, f1 = ft[idx], ft[idx + 1]
    slope = (f1 - f0) / (x1 - x0)
    return f0 + slope * (x - x0), slope


def _torch_bins(be: ArrayBackend, r, key):
    torch = be.torch
    lims = key.clone()
    lims[0] = -torch.inf
    lims[-1] = torch.inf
    bin_ = torch.searchsorted(lims, r, right=True) - 1
    bin_ = torch.clamp(bin_, 0, key.numel() - 2)
    w = (r - key[bin_]) / (key[bin_ + 1] - key[bin_])
    return bin_, w
