"""PVT table interpolation, ported from MRST ``ad-props/utils/interpPVT.m``.

The saturated branch is the locus of bubble points, and `sat_flag` selects
which regime each cell is in - getting that selection wrong gives plausible
properties that are wrong only near the bubble point, which is where a
black-oil solve spends its time.

All three of MRST's undersaturated strategies are here. They agree on the
saturated curve and differ between branches, which is why the strategy is
chosen per phase rather than fixed:

===========  =====================================================
'box'        look up both branches at the same abscissa
'parallel'   measure each branch from its own bubble point
'linshift'   slide the abscissa along the line joining the two
             neighbouring saturated points
===========  =====================================================

**One deliberate divergence.** `interpPVT`'s own default is 'box'; this port
defaults to 'parallel'. That default is nearly unreachable in MRST, because
`assignPVTO`/`assignPVTG` always pass `reg.pvtMethodOil`/`reg.pvtMethodGas`
explicitly - and `handle_regions` defaults those to 'parallel' and 'linshift'
exactly as MRST does, so every deck-built fluid agrees with MRST either way.
Adopting 'box' here would only change the handful of callers that pass no
method at all, silently, and each of those wants a specific strategy rather
than the entry point's fallback. Pass `method=` and this is moot.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from pymrst.utils.adi import ADI, _diag_mul, ad_from_blocks, is_ad, value
from pymrst.utils.interp_table import _interp1, interp_table

ArrayOrADI = Union[np.ndarray, ADI]


@dataclass
class PVTTable:
    """PVTO-style keyed table: one pressure branch per key (Rs) value."""

    key: np.ndarray                  # (nk,) Rs values
    branches: List[Tuple[np.ndarray, np.ndarray]]   # (p, F) per key
    sat_x: np.ndarray                # (nk,) bubble-point pressures
    sat_f: np.ndarray                # (nk,) F on the saturated curve
    # MRST's tables (`assignPVTO`/`assignPVTG`) interpolate as MATLAB's
    # `interp1` does. PyMRST's earlier builder (`make_pvt_table`) keeps the
    # slope form it was written with: its branches can be degenerate (two equal
    # values 1e-9 Pa apart), where the weights form cancels catastrophically.
    matlab_interp1: bool = True


def make_pvt_table(key: np.ndarray,
                   branches: List[Tuple[np.ndarray, np.ndarray]]) -> PVTTable:
    sat_x = np.array([b[0][0] for b in branches])
    sat_f = np.array([b[1][0] for b in branches])
    return PVTTable(key=np.asarray(key, float), branches=branches,
                    sat_x=sat_x, sat_f=sat_f, matlab_interp1=False)


def _interp1_slope(x_tab: np.ndarray, f_tab: np.ndarray, x: np.ndarray):
    """`f0 + slope*(x - x0)`: the interpolation of PyMRST's earlier tables."""
    idx = np.clip(np.searchsorted(x_tab, x, side="right") - 1, 0, x_tab.size - 2)
    x0, x1 = x_tab[idx], x_tab[idx + 1]
    f0, f1 = f_tab[idx], f_tab[idx + 1]
    slope = (f1 - f0)/(x1 - x0)
    return f0 + slope*(x - x0), slope


def _interp_for(tab: PVTTable):
    return _interp1 if tab.matlab_interp1 else _interp1_slope


def _get_bins(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MRST getBins: interval index (0-based) and linear weight in Rs."""
    lims = v.copy()
    lims[0], lims[-1] = -np.inf, np.inf
    bin_ = np.clip(np.searchsorted(lims, r, side="right") - 1, 0, v.size - 2)
    w = (r - v[bin_]) / (v[bin_ + 1] - v[bin_])
    dwdr = 1.0 / (v[bin_ + 1] - v[bin_])
    return bin_, w, dwdr


def _box(tab: PVTTable, p: np.ndarray, rs: np.ndarray
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MRST ``interp2DPVT_box``: look up each branch at the same abscissa.

    The plainest of the three strategies and `interpPVT`'s own default: no
    shift at all, just a lookup at `p` on both bracketing branches blended
    linearly in the key. Because the abscissa is not moved with the key, the
    key-derivative is the bare blend term - there is no `-dfdp*dpsdr` chain
    contribution as in 'parallel' and 'linshift'.

    The cost of that simplicity is extrapolation: a cell below branch `tn`'s
    own bubble point is off the low end of that branch, so `_interp1` continues
    it with its end slope. MRST does the same, on `T.expanded`.
    """
    _interp1 = _interp_for(tab)
    bin_, w, dwdr = _get_bins(rs, tab.key)

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
            f_l[ixl], df_l[ixl] = _interp1(px, fx, p[ixl])
        if ixr.any():
            f_r[ixr], df_r[ixr] = _interp1(px, fx, p[ixr])

    f = f_l * w + f_r * (1.0 - w)
    dfdp = df_l * w + df_r * (1.0 - w)
    dfdr = (f_l - f_r) * dwdr
    return f, dfdp, dfdr


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
    _interp1 = _interp_for(tab)
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


def _matlab_if_any(mask: np.ndarray) -> bool:
    """MATLAB `if any(mask)`. For a matrix, `any` gives one flag per column
    and `if` needs all of them, so a matrix with one column free of true
    entries counts as false: MRST then skips that branch for every entry,
    leaving them 0 (e.g. the saturated points of blackoilTutorialSPE9's
    (p, Rs) grids). Vectors behave as `np.any`."""
    mask = np.asarray(mask)
    if mask.ndim < 2 or min(mask.shape[:2]) == 1:
        return bool(mask.any())
    return bool(np.all(np.any(mask, axis=0)))


def _pvt_values(tab: PVTTable, p: np.ndarray, rs: np.ndarray,
                sat: np.ndarray, method: str = "parallel"
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values and derivatives of a PVT interpolation.

    ``p`` is the branch abscissa and ``rs`` the key, whatever they mean for
    the table in hand: for PVTO that is (pressure, Rs); for PVTG, whose
    branches are keyed by pressure, it is (Rv, pressure).
    """
    _interp1 = _interp_for(tab)
    if method not in ("box", "parallel", "linshift"):
        raise ValueError(f"unknown PVT interpolation strategy '{method}'")

    f = np.zeros_like(p)
    dfdp = np.zeros_like(p)
    dfdr = np.zeros_like(p)

    sat = sat.copy()
    sat[p < tab.sat_x[0]] = False
    usat = ~sat

    if _matlab_if_any(sat):
        f[sat], dfdp[sat] = _interp1(tab.sat_x, tab.sat_f, p[sat])

    has_usat = _matlab_if_any(usat)
    if has_usat and method == "linshift":
        f[usat], dfdp[usat], dfdr[usat] = _linshift(tab, p[usat], rs[usat])
    elif has_usat and method == "box":
        f[usat], dfdp[usat], dfdr[usat] = _box(tab, p[usat], rs[usat])
    elif has_usat:
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
    p_ad, r_ad = is_ad(p), is_ad(rs)
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
    return ad_from_blocks(f, jac)


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

    def _linear_table():
        """Well-formed (p, F) branches on which F is exactly p/1e6 + key/10.

        Both branches carry that same field, so 'box' - which looks each up at
        the shared abscissa - must reproduce it exactly, including where it has
        to extrapolate branch 1 below its own bubble point.
        """
        return make_pvt_table(
            np.array([0.0, 100.0]),
            [(np.array([1.0e6, 3.0e6]), np.array([1.0, 3.0])),
             (np.array([2.0e6, 4.0e6]), np.array([12.0, 14.0]))])

    def test_box_reproduces_an_exactly_linear_field() -> None:
        tab = _linear_table()
        p = np.array([2.5e6, 3.5e6])
        rs = np.array([50.0, 20.0])
        f, _, _ = _box(tab, p, rs)
        assert np.allclose(f, p / 1.0e6 + rs / 10.0), f
        print("ok test_box_reproduces_an_exactly_linear_field")

    def test_box_extrapolates_the_branch_it_is_off_the_end_of() -> None:
        # 1.5e6 is below branch 1's first point (2.0e6): MRST continues the
        # branch with its end slope rather than clamping, and clamping here
        # would return 12.0 instead of 11.5 and give f = 6.75, not 6.5.
        tab = _linear_table()
        f, dfdp, dfdr = _box(tab, np.array([1.5e6]), np.array([50.0]))
        assert np.isclose(f[0], 6.5), f
        assert np.isclose(dfdp[0], 1.0e-6), dfdp
        assert np.isclose(dfdr[0], 0.1), dfdr
        print("ok test_box_extrapolates_the_branch_it_is_off_the_end_of")

    def test_box_differs_from_parallel_when_bubble_points_differ() -> None:
        # Branch 1 is stretched so the two strategies cannot coincide: 'box'
        # reads it at 2.5e6, 'parallel' reads it 1.0e6 above its own bubble
        # point. If _box were silently routed to the parallel code this test
        # is what fails.
        tab = make_pvt_table(
            np.array([0.0, 100.0]),
            [(np.array([1.0e6, 3.0e6]), np.array([1.0, 3.0])),
             (np.array([2.0e6, 6.0e6]), np.array([12.0, 13.0]))])
        p, rs = np.array([2.5e6]), np.array([50.0])
        flag = np.zeros(1, dtype=bool)
        box = pvt_interp(tab, p, rs, flag, method="box")
        par = pvt_interp(tab, p, rs, flag, method="parallel")
        assert np.isclose(box[0], 7.3125), box     # 0.5*2.5 + 0.5*12.125
        assert np.isclose(par[0], 7.125), par      # 0.5*2.0 + 0.5*12.25
        print("ok test_box_differs_from_parallel_when_bubble_points_differ")

    def test_box_is_reachable_through_the_dispatch() -> None:
        tab = _linear_table()
        out = pvt_interp(tab, np.array([2.5e6]), np.array([50.0]),
                         np.zeros(1, dtype=bool), method="box")
        assert np.isclose(out[0], 7.5), out
        try:
            pvt_interp(tab, np.array([2.5e6]), np.array([50.0]),
                       np.zeros(1, dtype=bool), method="nonesuch")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown strategy must still be refused")
        print("ok test_box_is_reachable_through_the_dispatch")

    test_saturated_branch_reproduces_its_own_points()
    test_undersaturated_flag_changes_the_answer()
    test_blends_monotonically_between_branches()
    test_carries_derivatives_through_adi()
    test_box_reproduces_an_exactly_linear_field()
    test_box_extrapolates_the_branch_it_is_off_the_end_of()
    test_box_differs_from_parallel_when_bubble_points_differ()
    test_box_is_reachable_through_the_dispatch()

    def test_matlab_if_any_needs_every_column_of_a_matrix():
        """`if any(M)` on a matrix holds only when every column has a true
        entry; vectors of either orientation behave as `any`."""
        assert _matlab_if_any(np.array([False, True]))
        assert _matlab_if_any(np.array([[False, True, False]]))
        assert _matlab_if_any(np.array([[False], [True]]))
        assert _matlab_if_any(np.array([[True, False], [False, True]]))
        assert not _matlab_if_any(np.array([[True, False], [True, False]]))
        assert not _matlab_if_any(np.zeros(3, dtype=bool))
        print("ok test_matlab_if_any_needs_every_column_of_a_matrix")

    test_matlab_if_any_needs_every_column_of_a_matrix()
    def test_the_earlier_builder_keeps_degenerate_branches_flat() -> None:
        # Two equal values 4e-9 Pa apart, read 2e7 Pa away: the slope form
        # returns the value; MATLAB's weights form would cancel to noise.
        tab = make_pvt_table(np.array([10.0, 20.0]),
                             [(np.array([1e6, 1e6 + 3.7e-9]), np.array([0.91, 0.91])),
                              (np.array([2e6, 2e6 + 3.7e-9]), np.array([0.92, 0.92]))])
        assert not tab.matlab_interp1
        out = pvt_interp(tab, np.array([2.2e7]), np.array([15.0]), np.array([False]))
        assert abs(float(np.ravel(value(out))[0]) - 0.915) < 1e-12
        print("ok test_the_earlier_builder_keeps_degenerate_branches_flat")

    test_the_earlier_builder_keeps_degenerate_branches_flat()
    print("\nAll tests passed!")
