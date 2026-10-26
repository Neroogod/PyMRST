"""MRST ``ad-core/models/facilities/computeWellContributionsSingleWell.m``.

Inflow from the reservoir into one standard well, and the well's rate
equations. Each perforation sees the drawdown `p - (bhp + cdp)`; producing
perforations carry each phase at its own mobility, injecting ones carry the
wellbore mixture at the total mobility. The mixture `mix_s` is what is in the
wellbore: the declared injection stream plus whatever the producing
perforations put there, which is what a crossflowing perforation pushes back.

Returns `(eqs, cq_mass, mix_s, status, cstatus, cq_vol)` as MRST does: the
well equations `q_s - sum(cq_s)` per phase, the perforation mass and
reservoir-volume rates, and the numeric mixture.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_core.utils.phase_densities_to_bfactor import (
    phase_densities_to_bfactor,
)
from pymrst.utils.adi import ad_sum, value


def compute_well_contributions_single_well(wellmodel: Any, wellSol: Any,
                                           resmodel: Any, q_s: Sequence[Any],
                                           pBH: Any, packed: Any) -> Tuple:
    p, mob, rho, dissolved = (packed.pressure, packed.mob, packed.rho,
                              packed.dissolved)
    W = wellmodel.W
    num_ph = len(q_s)
    rho_s = resmodel.get_surface_densities()
    qt_s = 0.0
    if W.status:
        for ph in range(num_ph):
            qt_s = qt_s + q_s[ph]
    drawdown = _get_drawdown(wellSol, pBH, p)
    Tw, is_inj, conn_inj, cstatus = _get_well_trans(wellmodel, wellSol, qt_s,
                                                    drawdown)
    cq_vol, cq_s, mix_s_ad = _compute_perforation_rates(
        resmodel, wellmodel, conn_inj, q_s, qt_s, drawdown, Tw, is_inj, mob,
        rho, rho_s, dissolved)
    eqs = [q_s[ph] - ad_sum(cq_s[ph]) for ph in range(num_ph)]
    if not W.status:
        # MRST zeroes the derivatives of a shut well's rows here through a
        # variable (`wellStatus`) it never defines, so this branch errors in
        # MATLAB. The facility only assembles active wells, so it is not
        # reached; it refuses rather than guessing what was meant.
        raise RuntimeError(
            "computeWellContributionsSingleWell was handed a shut well; MRST's "
            "branch for that case references an undefined variable")
    mix_s = np.array([np.ravel(value(m))[0] for m in mix_s_ad], dtype=float)
    status = bool(_get(wellSol, "status"))
    cq_mass = [cq_s[i]*_linear(rho_s, i) for i in range(num_ph)]
    return eqs, cq_mass, mix_s, status, cstatus, cq_vol


def _conn2surf(v: List[Any], b: List[Any], r: Any, model: Any) -> List[Any]:
    """Reservoir-volume connection rates to surface volume, with carry-over."""
    bv = [b[ph]*v[ph] for ph in range(len(v))]
    qrho = list(bv)
    dg = bool(getattr(model, "disgas", False))
    vo = bool(getattr(model, "vapoil", False))
    if vo or dg:
        isgas = model.get_variable_field("sg")[1]
        isoil = model.get_variable_field("so")[1]
        if _is_black_oil(model):
            if dg:
                qrho[isgas] = qrho[isgas] + r[isgas][isoil]*bv[isoil]
            if vo:
                qrho[isoil] = qrho[isoil] + r[isoil][isgas]*bv[isgas]
        elif dg:
            qrho[isgas] = qrho[isgas] + r[isgas][isoil]*bv[isoil]
    return qrho


def _comp_vol_rat(cmix_s: List[Any], b: List[Any], r: Any, model: Any) -> Any:
    """Reservoir volume of one surface volume of the wellbore mixture."""
    tmp = list(cmix_s)
    dg = bool(getattr(model, "disgas", False))
    vo = bool(getattr(model, "vapoil", False))
    if vo or dg:
        isgas = model.get_variable_field("sg")[1]
        isoil = model.get_variable_field("so")[1]
        rs = r[isgas][isoil] if dg else 0.0
        rv = r[isoil][isgas] if vo else 0.0
        if _is_black_oil(model):
            d = 1 - rv*rs
            if dg:
                tmp[isgas] = (tmp[isgas] - rs*cmix_s[isoil])/d
            if vo:
                tmp[isoil] = (tmp[isoil] - rv*cmix_s[isgas])/d
        elif dg:
            tmp[isgas] = tmp[isgas] - rs*cmix_s[isoil]
    vol_rat = 0.0
    for ph in range(len(b)):
        vol_rat = vol_rat + tmp[ph]/b[ph]
    return vol_rat


def _get_drawdown(wellSol: Any, pBH: Any, p: Any) -> Any:
    return -(pBH + np.asarray(_get(wellSol, "cdp"), dtype=float)) + p


def _get_well_trans(wellmodel: Any, wellSol: Any, qt_s: Any, drawdown: Any):
    W = wellmodel.W
    Tw = np.array(W.WI, dtype=float, copy=True).ravel()
    well_status = bool(W.status)
    cstatus_w = (np.ones(Tw.size, dtype=bool) if W.cstatus is None
                 else np.asarray(W.cstatus, dtype=bool).ravel())
    perf_status = cstatus_w & well_status
    Tw[~perf_status] = 0.0
    if wellmodel.allowSignChange:
        is_inj = np.ravel(value(qt_s)) > 0
    else:
        is_inj = W.sign > 0
    conn_inj = np.asarray(value(drawdown)) < 0
    cross_flow = conn_inj != is_inj
    closed = ~np.asarray(_get(wellSol, "cstatus"), dtype=bool).ravel()
    if not wellmodel.allowCrossflow:
        closed = closed | cross_flow
    Tw[closed] = 0.0
    conn_inj = conn_inj & ~closed
    return Tw, is_inj, conn_inj, ~closed


def _compute_perforation_rates(resmodel, wellmodel, conn_inj, q_s, qt_s,
                               drawdown, Tw, is_inj, mob, rho, rho_s,
                               dissolved):
    compi = np.asarray(wellmodel.W.compi, dtype=float).ravel()
    num_ph = rho_s.shape[1]
    any_inj = bool(np.any(conn_inj))
    any_prod = bool(np.any(~conn_inj))
    b = phase_densities_to_bfactor(
        rho, [_linear(rho_s, i) for i in range(num_ph)], dissolved)
    cq_p: List[Any] = []
    cq_ps: List[Any] = []
    if any_prod:
        con_eff = (~conn_inj)*Tw
        cq_p = [-con_eff*mob[ph]*drawdown for ph in range(num_ph)]
        cq_ps = _conn2surf(cq_p, b, dissolved, resmodel)
        wbq = [q_s[ph]*(np.ravel(value(q_s[ph])) > 0) - ad_sum(cq_ps[ph])
               for ph in range(num_ph)]
        wbqt = 0.0
        for ph in range(num_ph):
            wbqt = wbqt + wbq[ph]
        dead = np.ravel(value(wbqt)) == 0
        if np.any(dead):
            wbq = [(~dead)*wbq[ph] + compi[ph]*dead for ph in range(num_ph)]
            wbqt = _assign(wbqt, dead, 1.0)
        mix_s = [wbq[ph]/wbqt for ph in range(num_ph)]
    else:
        mix_s = [compi[i] for i in range(num_ph)]
    if any_inj:
        mt = 0.0
        for ph in range(num_ph):
            mt = mt + mob[ph]
        cqt_i = -(conn_inj*Tw)*(mt*drawdown)
        vol_rat = _comp_vol_rat(mix_s, b, dissolved, resmodel)
        cqt_is = cqt_i/vol_rat
    cq_s: List[Any] = []
    cq_vol: List[Any] = []
    for ph in range(num_ph):
        if any_inj and any_prod:
            cq_vol.append(conn_inj*cqt_i*compi[ph] + (~conn_inj)*cq_p[ph])
            cq_s.append(cq_ps[ph] + mix_s[ph]*cqt_is)
        elif any_inj:
            cq_vol.append(cqt_i*compi[ph])
            cq_s.append(mix_s[ph]*cqt_is)
        else:
            cq_vol.append(cq_p[ph])
            cq_s.append(cq_ps[ph])
    return cq_vol, cq_s, mix_s


def _assign(x: Any, mask: np.ndarray, new: float) -> Any:
    """`x(mask) = new`, keeping derivatives elsewhere."""
    if hasattr(x, "jac"):
        out = x + 0.0
        out[np.flatnonzero(mask)] = new
        return out
    out = np.array(x, dtype=float, copy=True).reshape(-1)
    out[mask] = new
    return out


def _linear(matrix: np.ndarray, i: int) -> float:
    """MATLAB `rhoS(i)`: linear, column-major. One PVT region makes it the
    phase-`i` value; with several, MRST reads whatever lies at position `i`,
    and so does this."""
    return float(np.asarray(matrix).ravel(order="F")[i])


def _get(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


def _is_black_oil(model: Any) -> bool:
    from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
        ThreePhaseBlackOilModel,
    )
    return isinstance(model, ThreePhaseBlackOilModel)


computeWellContributionsSingleWell = compute_well_contributions_single_well

__all__ = ["compute_well_contributions_single_well",
           "computeWellContributionsSingleWell"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.utils.adi import init_variables_adi

    class _Model:
        disgas = vapoil = False

        def get_surface_densities(self):
            return np.array([[1000.0, 700.0]])

    def _well(sign: int, WI=(1e-12, 1e-12), compi=(1.0, 0.0)):
        W = SimpleNamespace(WI=np.array(WI), status=True, cstatus=None,
                            sign=sign, compi=np.array(compi))
        return SimpleNamespace(W=W, allowSignChange=False, allowCrossflow=True)

    def _packed(p, mob=(1e3, 2e3), b=(1.0, 1.0)):
        rho = [np.full(2, 1000.0*b[0]), np.full(2, 700.0*b[1])]
        return SimpleNamespace(pressure=p, mob=[np.full(2, m) for m in mob],
                               rho=rho, dissolved=[])

    WS = SimpleNamespace(cdp=np.zeros(2), cstatus=np.ones(2, dtype=bool),
                         status=True)

    def test_producer_perforations_carry_each_phase_at_its_mobility():
        q_w, q_o, bhp, p = init_variables_adi(
            np.array([0.0]), np.array([0.0]), np.array([90e5]),
            np.array([100e5, 110e5]))
        eqs, cq_mass, mix, status, cstatus, cq_vol = \
            compute_well_contributions_single_well(
                _well(-1), WS, _Model(), [q_w, q_o], bhp, _packed(p))
        dd = np.array([10e5, 20e5])
        np.testing.assert_allclose(value(cq_vol[0]), -1e-12*1e3*dd)
        np.testing.assert_allclose(value(cq_vol[1]), -1e-12*2e3*dd)
        np.testing.assert_allclose(value(eqs[0]), 1e-12*1e3*dd.sum())
        np.testing.assert_allclose(value(cq_mass[1]), 700.0*value(cq_vol[1]))
        assert status and np.all(cstatus)
        print("ok test_producer_perforations_carry_each_phase_at_its_mobility")

    def test_inflow_has_derivatives_in_bhp_and_cell_pressure():
        q_w, q_o, bhp, p = init_variables_adi(
            np.array([0.0]), np.array([0.0]), np.array([90e5]),
            np.array([100e5, 110e5]))
        _, _, _, _, _, cq_vol = compute_well_contributions_single_well(
            _well(-1), WS, _Model(), [q_w, q_o], bhp, _packed(p))
        # d(cq_w)/d(bhp) = +WI*mob; d(cq_w)/d(p) = -WI*mob on the diagonal.
        np.testing.assert_allclose(cq_vol[0].jac[2].toarray().ravel(), 1e-9)
        np.testing.assert_allclose(cq_vol[0].jac[3].diagonal(), -1e-9)
        print("ok test_inflow_has_derivatives_in_bhp_and_cell_pressure")

    def test_injector_injects_its_declared_stream():
        q_w, q_o, bhp, p = init_variables_adi(
            np.array([1e-3]), np.array([0.0]), np.array([120e5]),
            np.array([100e5, 110e5]))
        _, _, mix, _, _, cq_vol = compute_well_contributions_single_well(
            _well(1), WS, _Model(), [q_w, q_o], bhp, _packed(p))
        np.testing.assert_allclose(mix, [1.0, 0.0])
        mt = 3e3
        np.testing.assert_allclose(value(cq_vol[0]),
                                   1e-12*mt*np.array([20e5, 10e5]))
        np.testing.assert_allclose(value(cq_vol[1]), 0.0)
        print("ok test_injector_injects_its_declared_stream")

    test_producer_perforations_carry_each_phase_at_its_mobility()
    test_inflow_has_derivatives_in_bhp_and_cell_pressure()
    test_injector_injects_its_declared_stream()
    print("\nAll tests passed!")
