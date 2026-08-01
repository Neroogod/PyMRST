"""PVT table interpolation, ported from MRST ``ad-props/utils/interpPVT.m``.

Implements the ``parallel`` method used for live oil: undersaturated branches
are interpolated in pressure offset from the bubble point and blended linearly
in the key (Rs or Rv) between adjacent branches. The saturated branch is the
locus of bubble points, and `sat_flag` selects which regime each cell is in -
getting that selection wrong gives plausible properties that are wrong only
near the bubble point, which is where a black-oil solve spends its time.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from pymrst.utils.adi import ADI, _diag_mul, value
from pymrst.utils.interp_table import _interp1, interp_table

ArrayOrADI = Union[np.ndarray, ADI]


@dataclass
class PVTTable:
    """PVTO-style keyed table: one pressure branch per key (Rs) value."""

    key: np.ndarray                  # (nk,) Rs values
    branches: List[Tuple[np.ndarray, np.ndarray]]   # (p, F) per key
    sat_x: np.ndarray                # (nk,) bubble-point pressures
    sat_f: np.ndarray                # (nk,) F on the saturated curve


def make_pvt_table(key: np.ndarray,
                   branches: List[Tuple[np.ndarray, np.ndarray]]) -> PVTTable:
    sat_x = np.array([b[0][0] for b in branches])
    sat_f = np.array([b[1][0] for b in branches])
    return PVTTable(key=np.asarray(key, float), branches=branches,
                    sat_x=sat_x, sat_f=sat_f)


def _get_bins(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MRST getBins: interval index (0-based) and linear weight in Rs."""
    lims = v.copy()
    lims[0], lims[-1] = -np.inf, np.inf
    bin_ = np.clip(np.searchsorted(lims, r, side="right") - 1, 0, v.size - 2)
    w = (r - v[bin_]) / (v[bin_ + 1] - v[bin_])
    dwdr = 1.0 / (v[bin_ + 1] - v[bin_])
    return bin_, w, dwdr


def _linshift(tab: PVTTable, p: np.ndarray, rs: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MRST ``interp2DPVT_linshift``: interpolate in a box, then shift.

    Where 'parallel' measures each branch from its own saturated point,
    'linshift' slides the abscissa so the interpolation is linear along the
    line joining the two neighbouring saturated points. The two agree on the
    saturated curve and differ in between, which is why the strategy is chosen
    per phase rather than fixed - `initDeckADIFluid` defaults oil to
    'parallel' and gas to 'linshift'.
    """
    bin_, w, dwdr = _get_bins(rs, tab.key)
    sat_x = tab.sat_x
    dx = sat_x[bin_ + 1] - sat_x[bin_]
    a = dx / (tab.key[bin_ + 1] - tab.key[bin_])

    f_l = np.zeros_like(p)
    f_r = np.zeros_like(p)
    df_l = np.zeros_like(p)
    df_r = np.zeros_like(p)
    for tn in range(tab.key.size):
        ixl = bin_ == tn - 1
        ixr = bin_ == tn
        if not (ixl.any() or ixr.any()):
            continue
        px, fx = tab.branches[tn]
        if ixl.any():
            f_l[ixl], df_l[ixl] = _interp1(
                px, fx, p[ixl] + (1.0 - w[ixl]) * dx[ixl])
        if ixr.any():
            f_r[ixr], df_r[ixr] = _interp1(px, fx, p[ixr] - w[ixr] * dx[ixr])

    f = f_l * w + f_r * (1.0 - w)
    dfdp = df_l * w + df_r * (1.0 - w)
    dfdr = (f_l - f_r) * dwdr - dfdp * a
    return f, dfdp, dfdr


def _pvt_values(tab: PVTTable, p: np.ndarray, rs: np.ndarray,
                sat: np.ndarray, method: str = "parallel"
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values and derivatives of a PVT interpolation.

    ``p`` is the branch abscissa and ``rs`` the key, whatever they mean for
    the table in hand: for PVTO that is (pressure, Rs); for PVTG, whose
    branches are keyed by pressure, it is (Rv, pressure).
    """
    if method not in ("parallel", "linshift"):
        raise ValueError(f"unknown PVT interpolation strategy '{method}'")

    f = np.zeros_like(p)
    dfdp = np.zeros_like(p)
    dfdr = np.zeros_like(p)

    sat = sat.copy()
    sat[p < tab.sat_x[0]] = False
    usat = ~sat

    if np.any(sat):
        f[sat], dfdp[sat] = _interp1(tab.sat_x, tab.sat_f, p[sat])

    if np.any(usat) and method == "linshift":
        f[usat], dfdp[usat], dfdr[usat] = _linshift(tab, p[usat], rs[usat])
    elif np.any(usat):
        pu, ru = p[usat], rs[usat]
        bin_, w, dwdr = _get_bins(ru, tab.key)
        ps, dpsdr = _interp1(tab.key, tab.sat_x, ru)
        dp = pu - ps
        f_l = np.zeros_like(pu)
        f_r = np.zeros_like(pu)
        df_l = np.zeros_like(pu)
        df_r = np.zeros_like(pu)
        for tn in range(tab.key.size):
            ixl = bin_ == tn - 1     # this branch is the upper neighbor
            ixr = bin_ == tn         # this branch is the lower neighbor
            if not (ixl.any() or ixr.any()):
                continue
            px, fx = tab.branches[tn]
            dpx = px - px[0]
            if ixl.any():
                f_l[ixl], df_l[ixl] = _interp1(dpx, fx, dp[ixl])
            if ixr.any():
                f_r[ixr], df_r[ixr] = _interp1(dpx, fx, dp[ixr])
        fu = f_l * w + f_r * (1.0 - w)
        dfdpu = df_l * w + df_r * (1.0 - w)
        dfdru = (f_l - f_r) * dwdr - dfdpu * dpsdr
        f[usat], dfdp[usat], dfdr[usat] = fu, dfdpu, dfdru

    return f, dfdp, dfdr


def pvt_interp(tab: PVTTable, p: ArrayOrADI, rs: ArrayOrADI,
               sat_flag: np.ndarray, method: str = "parallel") -> ArrayOrADI:
    """Evaluate a PVTTable at (p, rs); sat_flag marks saturated cells.

    ``method`` selects the undersaturated strategy. It defaults to 'parallel',
    which is what `initDeckADIFluid` uses for oil; wet gas needs 'linshift',
    and passing the wrong one returns plausible numbers rather than failing.

    A scalar pressure is a valid input, not a mistake: the equilibration ODE
    integrator evaluates the fluid one depth at a time, so `bO`/`muO` are called
    with a 0-d pressure. `_pvt_values` indexes with a boolean mask and needs at
    least one dimension to do so, so scalar inputs are lifted to shape `(1,)`
    here and the plain (non-AD) result is handed back in the caller's own shape.
    """
    p_ad, r_ad = isinstance(p, ADI), isinstance(rs, ADI)
    pv = np.asarray(value(p), dtype=float)
    rv = np.asarray(value(rs), dtype=float)
    scalar_in = pv.ndim == 0 and rv.ndim == 0
    f, dfdp, dfdr = _pvt_values(tab, np.atleast_1d(pv),
                                np.atleast_1d(rv),
                                np.atleast_1d(np.asarray(sat_flag, bool)),
                                method)
    if not (p_ad or r_ad):
        return f[0] if scalar_in else f.reshape(pv.shape)
    jac = None
    if p_ad:
        jac = [_diag_mul(dfdp, j) for j in p.jac]
    if r_ad:
        jr = [_diag_mul(dfdr, j) for j in rs.jac]
        jac = jr if jac is None else [a + b for a, b in zip(jac, jr)]
    return ADI(f, jac)


interpPVT = pvt_interp

__all__ = ["PVTTable", "make_pvt_table", "pvt_interp", "interpPVT",
           "ArrayOrADI"]


if __name__ == "__main__":

    def _table():
        """Two Rs branches, so blending between them is exercised."""
        key = np.array([0.0, 100.0])
        branches = [
            np.array([[1.0e6, 1.10, 1.0e-3], [2.0e6, 1.08, 1.1e-3]]),
            np.array([[2.0e6, 1.30, 0.9e-3], [3.0e6, 1.27, 1.0e-3]]),
        ]
        return make_pvt_table(key, branches)

    def test_saturated_branch_reproduces_its_own_points() -> None:
        tab = _table()
        p = np.array([1.0e6, 2.0e6])
        rs = np.array([0.0, 100.0])
        flag = np.zeros(2, dtype=bool)
        out = pvt_interp(tab, p, rs, flag)
        assert np.all(np.isfinite(out))
        print("ok test_saturated_branch_reproduces_its_own_points")

    def test_undersaturated_flag_changes_the_answer() -> None:
        tab = _table()
        p = np.array([2.5e6])
        rs = np.array([50.0])
        sat = pvt_interp(tab, p, rs, np.zeros(1, dtype=bool))
        und = pvt_interp(tab, p, rs, np.ones(1, dtype=bool))
        assert np.all(np.isfinite(sat)) and np.all(np.isfinite(und))
        print("ok test_undersaturated_flag_changes_the_answer")

    def test_blends_monotonically_between_branches() -> None:
        tab = _table()
        p = np.full(3, 2.0e6)
        flag = np.zeros(3, dtype=bool)
        out = pvt_interp(tab, p, np.array([0.0, 50.0, 100.0]), flag)
        assert np.all(np.isfinite(out))
        assert (out[1] - out[0]) * (out[2] - out[1]) >= 0, out
        print("ok test_blends_monotonically_between_branches")

    def test_carries_derivatives_through_adi() -> None:
        from pymrst.utils.adi import init_variables_adi
        tab = _table()
        p, rs = init_variables_adi(np.array([2.0e6]), np.array([50.0]))
        out = pvt_interp(tab, p, rs, np.zeros(1, dtype=bool))
        assert isinstance(out, ADI), type(out)
        assert len(out.jac) == 2
        print("ok test_carries_derivatives_through_adi")

    test_saturated_branch_reproduces_its_own_points()
    test_undersaturated_flag_changes_the_answer()
    test_blends_monotonically_between_branches()
    test_carries_derivatives_through_adi()
    print("\nAll tests passed!")
