"""AD-aware table interpolation, ported from MRST interpTable/interpPVT.

interp_table is linear interpolation with linear extrapolation at both
ends (MRST's interpTable / fastInterpTable + dinterpq1). PVTTable and
pvt_interp implement the 'parallel' method of interpPVT.m used for live
oil: undersaturated branches are interpolated in pressure offset from
the bubble point and blended linearly in Rs between adjacent branches.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from pymrst.utils.adi import ADI, _diag_mul, value

ArrayOrADI = Union[np.ndarray, ADI]


def _interp1(x_tab: np.ndarray, f_tab: np.ndarray,
             x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interp with linear extrapolation; returns (f, df/dx)."""
    idx = np.clip(np.searchsorted(x_tab, x, side="right") - 1,
                  0, x_tab.size - 2)
    x0, x1 = x_tab[idx], x_tab[idx + 1]
    f0, f1 = f_tab[idx], f_tab[idx + 1]
    slope = (f1 - f0) / (x1 - x0)
    return f0 + slope * (x - x0), slope


def interp_table(x_tab: np.ndarray, f_tab: np.ndarray,
                 x: ArrayOrADI) -> ArrayOrADI:
    """MRST interpTable: linear with extrapolation, ADI-transparent."""
    xv = value(x)
    f, df = _interp1(np.asarray(x_tab, float), np.asarray(f_tab, float), xv)
    if isinstance(x, ADI):
        return ADI(f, [_diag_mul(df, j) for j in x.jac])
    return f


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


def _pvt_values(tab: PVTTable, p: np.ndarray, rs: np.ndarray,
                sat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values and derivatives of the 'parallel' PVT interpolation."""
    f = np.zeros_like(p)
    dfdp = np.zeros_like(p)
    dfdr = np.zeros_like(p)

    sat = sat.copy()
    sat[p < tab.sat_x[0]] = False
    usat = ~sat

    if np.any(sat):
        f[sat], dfdp[sat] = _interp1(tab.sat_x, tab.sat_f, p[sat])

    if np.any(usat):
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
               sat_flag: np.ndarray) -> ArrayOrADI:
    """Evaluate a PVTTable at (p, rs); sat_flag marks saturated cells."""
    f, dfdp, dfdr = _pvt_values(tab, value(p).astype(float),
                                value(rs).astype(float),
                                np.asarray(sat_flag, bool))
    p_ad, r_ad = isinstance(p, ADI), isinstance(rs, ADI)
    if not (p_ad or r_ad):
        return f
    jac = None
    if p_ad:
        jac = [_diag_mul(dfdp, j) for j in p.jac]
    if r_ad:
        jr = [_diag_mul(dfdr, j) for j in rs.jac]
        jac = jr if jac is None else [a + b for a, b in zip(jac, jr)]
    return ADI(f, jac)


def extend_tab(tab: np.ndarray, pm: float = 1.0) -> np.ndarray:
    """MRST extendTab: pad flat rows at x[0]-pm and x[-1]+pm."""
    first = tab[0].copy()
    first[0] -= pm
    last = tab[-1].copy()
    last[0] += pm
    return np.vstack([first, tab, last])


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi

    def test_interp_table_and_extrapolation():
        """Linear inside, linear extrapolation outside, exact slopes."""
        x = np.array([0.0, 1.0, 3.0])
        f = np.array([0.0, 2.0, 4.0])
        q = np.array([-1.0, 0.5, 2.0, 5.0])
        out = interp_table(x, f, q)
        assert np.allclose(out, [-2.0, 1.0, 3.0, 6.0])
        (xa,) = init_variables_adi(q)
        oa = interp_table(x, f, xa)
        assert np.allclose(oa.jac[0].toarray().diagonal(), [2, 2, 1, 1])
        print("ok test_interp_table_and_extrapolation")

    def test_pvt_saturated_branch():
        """Saturated cells interpolate the bubble-point envelope."""
        tab = make_pvt_table(
            np.array([0.5, 1.0]),
            [(np.array([10.0, 20.0]), np.array([1.0, 0.9])),
             (np.array([30.0, 40.0]), np.array([1.4, 1.2]))],
        )
        p = np.array([20.0])
        f = pvt_interp(tab, p, np.array([0.75]), np.array([True]))
        assert np.isclose(f[0], 1.2)  # midway between (10,1.0) and (30,1.4)
        print("ok test_pvt_saturated_branch")

    def test_pvt_undersaturated_parallel_fd():
        """Undersaturated derivatives match finite differences."""
        tab = make_pvt_table(
            np.array([0.5, 1.0]),
            [(np.array([10.0, 20.0, 40.0]), np.array([1.00, 0.95, 0.88])),
             (np.array([30.0, 45.0, 70.0]), np.array([1.40, 1.31, 1.20]))],
        )
        p0 = np.array([50.0, 38.0])
        r0 = np.array([0.8, 0.6])
        flag = np.array([False, False])
        (p, r) = init_variables_adi(p0, r0)
        out = pvt_interp(tab, p, r, flag)
        h = 1e-6
        for k in range(2):
            e = np.zeros(2)
            e[k] = h
            fd_p = (_pvt_values(tab, p0 + e, r0, flag)[0]
                    - _pvt_values(tab, p0 - e, r0, flag)[0]) / (2 * h)
            fd_r = (_pvt_values(tab, p0, r0 + e, flag)[0]
                    - _pvt_values(tab, p0, r0 - e, flag)[0]) / (2 * h)
            assert np.allclose(out.jac[0].toarray()[:, k], fd_p, atol=1e-6)
            assert np.allclose(out.jac[1].toarray()[:, k], fd_r, atol=1e-6)
        print("ok test_pvt_undersaturated_parallel_fd")

    def test_extend_tab():
        """Flat padding rows appear outside the original range."""
        t = extend_tab(np.array([[0.1, 5.0], [0.9, 7.0]]))
        assert t.shape == (4, 2)
        assert np.allclose(t[0], [-0.9, 5.0]) and np.allclose(t[-1], [1.9, 7.0])
        print("ok test_extend_tab")

    test_interp_table_and_extrapolation()
    test_pvt_saturated_branch()
    test_pvt_undersaturated_parallel_fd()
    test_extend_tab()
    print("\nAll tests passed!")
