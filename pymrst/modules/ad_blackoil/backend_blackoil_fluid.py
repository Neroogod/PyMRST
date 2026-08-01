"""Backend black-oil fluid properties for Torch/GPU experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.backends import ArrayBackend, BackendPVTTable, get_backend
from pymrst.backends.tables import interp_table_backend, pvt_interp_backend
from pymrst.modules.ad_props import extend_tab, make_pvt_table


@dataclass
class BackendBlackOilFluid:
    rhoS: np.ndarray
    p_w_ref: float
    c_w: float
    b_w_ref: float
    mu_w_ref: float
    rock_p_ref: float
    rock_c: float
    p_g: np.ndarray
    bg_tab: np.ndarray
    mug_tab: np.ndarray
    bo_tab: BackendPVTTable
    muo_tab: BackendPVTTable
    rs_t: np.ndarray
    p_t: np.ndarray
    sw_tab: np.ndarray
    krw_tab: np.ndarray
    sg_tab: np.ndarray
    krg_tab: np.ndarray
    so_w_tab: np.ndarray
    krow_tab: np.ndarray
    so_g_tab: np.ndarray
    krog_tab: np.ndarray
    swcon: float
    backend: ArrayBackend

    def b_w(self, p):
        x = self.c_w * (p - self.backend.asarray(self.p_w_ref))
        return (1.0 + x + 0.5 * x**2) / self.b_w_ref

    def mu_w(self, p):
        return p * 0.0 + self.mu_w_ref

    def pv_mult(self, p):
        x = self.rock_c * (p - self.backend.asarray(self.rock_p_ref))
        return 1.0 + x + 0.5 * x**2

    def b_g(self, p, rv=None, flag=None):
        return interp_table_backend(self.p_g, self.bg_tab, p, self.backend)

    def mu_g(self, p, rv=None, flag=None):
        return interp_table_backend(self.p_g, self.mug_tab, p, self.backend)

    def rs_sat(self, p):
        return interp_table_backend(self.p_t, self.rs_t, p, self.backend)

    def b_o(self, p, rs, gas_present):
        return pvt_interp_backend(self.bo_tab, p, rs, gas_present, self.backend)

    def mu_o(self, p, rs, gas_present):
        return pvt_interp_backend(self.muo_tab, p, rs, gas_present, self.backend)

    def kr_w(self, sw):
        return interp_table_backend(self.sw_tab, self.krw_tab, sw, self.backend)

    def kr_g(self, sg):
        return interp_table_backend(self.sg_tab, self.krg_tab, sg, self.backend)

    def kr_ow(self, so):
        return interp_table_backend(self.so_w_tab, self.krow_tab, so, self.backend)

    def kr_og(self, so):
        return interp_table_backend(self.so_g_tab, self.krog_tab, so, self.backend)

    def kr_o(self, sw, so, sg):
        be = self.backend
        swcon = be.xp.minimum(be.asarray(self.swcon), sw - 1e-5) if not be.is_torch else be.torch.minimum(be.asarray(self.swcon), sw - 1e-5)
        d = sg + sw - swcon
        ww = (sw - swcon) / d
        return (1.0 - ww) * self.kr_og(so) + ww * self.kr_ow(so)


def init_backend_blackoil_fluid(
    deck: dict,
    *,
    backend: str = "torch",
    device: str | None = None,
    dtype: str = "float64",
) -> BackendBlackOilFluid:
    be = get_backend(backend, device=device, dtype=dtype)
    w = deck["pvtw"]
    r = deck["rock_cp"]
    pvdg = deck["pvdg"]
    p_g = np.concatenate([[0.0], pvdg[:, 0]])
    bg_tab = np.concatenate([[1.0 / pvdg[0, 1]], 1.0 / pvdg[:, 1]])
    mug_tab = np.concatenate([[pvdg[0, 2]], pvdg[:, 2]])

    keys = np.array([b[0] for b in deck["pvto"]])
    bo_tab = make_pvt_table(keys, [(rows[:, 0], 1.0 / rows[:, 1]) for _, rows in deck["pvto"]])
    muo_tab = make_pvt_table(keys, [(rows[:, 0], rows[:, 2]) for _, rows in deck["pvto"]])
    p_bub = np.array([rows[0, 0] for _, rows in deck["pvto"]])
    rs_t = np.concatenate([[0.0], keys])
    p_t = np.concatenate([[0.0], p_bub])

    swof = extend_tab(deck["swof"])
    sgof = extend_tab(deck["sgof"])
    swcon = float(deck["swof"][0, 0])
    return BackendBlackOilFluid(
        rhoS=deck["rhoS"],
        p_w_ref=w["p_ref"],
        c_w=w["c"],
        b_w_ref=w["b_ref"],
        mu_w_ref=w["mu_ref"],
        rock_p_ref=r["p_ref"],
        rock_c=r["c"],
        p_g=p_g,
        bg_tab=bg_tab,
        mug_tab=mug_tab,
        bo_tab=BackendPVTTable.from_numpy_table(bo_tab),
        muo_tab=BackendPVTTable.from_numpy_table(muo_tab),
        rs_t=rs_t,
        p_t=p_t,
        sw_tab=swof[:, 0],
        krw_tab=swof[:, 1],
        sg_tab=sgof[:, 0],
        krg_tab=sgof[:, 1],
        so_w_tab=1.0 - swof[::-1, 0],
        krow_tab=swof[::-1, 2],
        so_g_tab=1.0 - (sgof[::-1, 0] + swcon),
        krog_tab=sgof[::-1, 2],
        swcon=swcon,
        backend=be,
    )
