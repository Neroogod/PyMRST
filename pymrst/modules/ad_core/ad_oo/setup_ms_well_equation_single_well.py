"""MRST ``ad-core/models/facilities/setupMSWellEquationSingleWell.m``.

The equations of one multisegment well. Its nodes carry a pressure (the top
node's is the bhp) and phase mass fractions `alpha`; its segments a mixture
mass flux `vmix`. Perforations flow by Peaceman inflow against the pressure
of their node, a cross-flowing (injecting) perforation taking the mixture of
its node. Each node conserves each phase's mass (with storage, from the
mixture density of the node at the start and end of the step); the top
node's balance closes on the well's surface rates. Each segment's pressure
drop is hydrostatic (from the averaged mixture density) plus the segment
flow model `W.segments.flowModel(v, rho, mu)` (mass rate, averaged density,
upstream viscosity), scaled by the bhp. The mass fractions of a node sum to
one.

Node properties (`getNodeMix`) come from the top node's surface-rate
fractions and the other nodes' mass fractions, as surface-volume fractions,
through the fluid at the node pressures (`computeNodeProps`); dissolved gas
and vaporised oil take the saturated value or the node's gas-oil ratio,
whichever is smaller.

As in MRST, the connection pressure-drop correction is multiplied by zero,
three phases are assumed for the start-of-step mixture, and a well without
a `flowModel` function calls `wm.pressureDropModel`, which
`MultisegmentWell` does not have.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np

from pymrst.params.gravity.gravity import gravity
from pymrst.utils.adi import ad_abs, ad_matmul, ad_max, ad_min, ad_sum, value, vertcat_adi


def setup_ms_well_equation_single_well(wm: Any, model: Any, wellSol0: Any, wellSol: Any,
                                       q_s: Sequence[Any], bhp: Any, pN: Any,
                                       alpha: Sequence[Any], vmix: Any, resProps: Any,
                                       dt: float, iteration: int):
    """MRST `[eqs, eqsMS, cq_s, wellSol, mix_s, status, cstatus, cq_r] =
    setupMSWellEquationSingleWell(wm, model, wellSol0, wellSol, q_s, bhp, pN,
    alpha, vmix, resProps, dt, iteration)`."""
    op = wm.operators
    w = wm.W
    m = resProps.mob
    b = resProps.b
    r = resProps.dissolved
    pr = resProps.pressure
    numPh = len(b)
    eqs: List[Any] = [None]*numPh
    eqsMS: List[Any] = [None]*(numPh + 2)
    rho_s = np.asarray(model.get_surface_densities(), dtype=float).ravel()

    mix_s, _, rhom, mum = _get_node_mix(model, bhp, q_s, pN, alpha, rho_s)
    ws0 = wellSol0
    bhp0, q_s0, pN0 = ws0.bhp, [ws0.qWs, ws0.qOs, ws0.qGs], ws0.nodePressure
    alpha0 = [ws0.nodeComp[:, 0], ws0.nodeComp[:, 1], ws0.nodeComp[:, 2]]
    _, _, rhom0, _ = _get_node_mix(model, bhp0, q_s0, pN0, alpha0, rho_s)

    c2nT = w.cell2node.T.tocsr()
    dZ = np.asarray(w.dZ, dtype=float).ravel()
    corrDP = 0*ad_matmul(c2nT, rhom)*((np.asarray(w.connDZ, dtype=float).ravel() - dZ)
                                      + (c2nT @ (w.nodes.depth - w.refDepth) - dZ))
    drawdown = pr - ad_matmul(c2nT, vertcat_adi(bhp, pN)) + corrDP
    injInx = np.asarray(value(drawdown)) < 0

    Tw = np.array(w.WI, dtype=float).ravel()
    if not wm.allowCrossflow:
        Tw[injInx] = 0
    cq = [-Tw*m[ph]*drawdown for ph in range(numPh)]
    cq_s = _conn2surf(cq, b, r, model)
    if wm.allowCrossflow and np.any(injInx):
        cq_s_tot = cq_s[0][injInx]
        for ph in range(1, numPh):
            cq_s_tot = cq_s_tot + cq_s[ph][injInx]
        c2n_inj = w.cell2node[:, np.flatnonzero(injInx)].T.tocsr()
        for ph in range(numPh):
            cq_s[ph][injInx] = cq_s_tot*ad_matmul(c2n_inj, mix_s[ph])

    vols = np.asarray(w.nodes.vol, dtype=float).ravel()
    up = np.asarray(value(vmix)) >= 0
    ddz = op.grad(np.asarray(w.nodes.depth, dtype=float).ravel())
    rhoSeg = op.aver(rhom)
    muSeg = op.segmentUpstr(up, mum[1:])
    dph = np.linalg.norm(gravity())*rhoSeg*ddz
    for ph in range(numPh):
        ec = (op.div(op.segmentUpstr(up, alpha[ph])*vmix)
              + ad_matmul(w.cell2node, cq_s[ph])*rho_s[ph])
        ec[1:] = ec[1:] + (alpha[ph]*rhom[1:] - alpha0[ph]*rhom0[1:])*vols[1:]/dt
        eqs[ph] = q_s[ph] - ad_sum(ec)/rho_s[ph]
        eqsMS[ph] = ec[1:]

    pressures = vertcat_adi(bhp, pN)
    if not callable(w.segments.flowModel):
        drop = wm.pressureDropModel(pN, vmix, alpha, rho_s, rhoSeg)
    else:
        drop = w.segments.flowModel(vmix, rhoSeg, muSeg)
    eqsMS[numPh] = (op.grad(pressures) - dph - drop)/value(bhp)
    closure = 1
    for i in range(numPh):
        closure = closure - alpha[i]
    eqsMS[-1] = closure

    status = True
    cstatus = np.ones(np.size(value(pr)), dtype=bool)
    cq_r = [value(c) for c in cq]
    mix_s = np.column_stack([np.asarray(value(x), dtype=float).ravel() for x in mix_s])
    return eqs, eqsMS, cq_s, wellSol, mix_s, status, cstatus, cq_r


setupMSWellEquationSingleWell = setup_ms_well_equation_single_well


def _compute_node_props(model: Any, p: Any, mixs: Sequence[Any]):
    numPh = len(mixs)
    fluid = model.fluid
    b: List[Any] = [None]*3
    mu: List[Any] = [None]*3
    wat, oil, gas = (bool(x) for x in np.asarray(model.get_active_phases()).ravel()[:3])
    dg = bool(getattr(model, "disgas", False))
    vo = bool(getattr(model, "vapoil", False))
    if wat:
        _, wix = model.get_variable_field("sw")
        b[wix] = fluid.bW(p)
        mu[wix] = fluid.muW(p)
    if oil:
        _, oix = model.get_variable_field("so")
        fo = mixs[oix]
        if not dg:
            b[oix] = fluid.bO(p)
            mu[oix] = fluid.muO(p)
    if gas:
        _, gix = model.get_variable_field("sg")
        fg = mixs[gix]
        if not vo:
            b[gix] = fluid.bG(p)
            mu[gix] = fluid.muG(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        if dg:
            rsMax = fluid.rsSat(p)
            gor = ad_max(ad_abs(fg/fo), 0)
            gor = _set_where(gor, np.isnan(np.asarray(value(gor))), np.inf)
            rs = ad_min(rsMax, gor)
            saturated = np.asarray(value(rs)) >= np.asarray(value(rsMax))
            b[oix] = fluid.bO(p, rs, saturated)
            mu[oix] = fluid.muO(p, rs, saturated)
        else:
            rs = 0
        if vo:
            rvMax = fluid.rvSat(p)
            ogr = ad_max(ad_abs(fo/fg), 0)
            ogr = _set_where(ogr, np.isnan(np.asarray(value(gor))), np.inf)
            rv = ad_min(rvMax, ogr)
            saturated = np.asarray(value(rv)) >= np.asarray(value(rvMax))
            b[gix] = fluid.bG(p, rv, saturated)
            mu[gix] = fluid.muG(p, rv, saturated)
        else:
            rv = 0
    r = [rs, rv]
    d = 1 - rs*rv
    x = list(mixs)
    if dg:
        x[gix] = (x[gix] - rs*fo)/d
    if vo:
        x[oix] = (x[oix] - rv*fg)/d
    volRatio = x[0]/b[0]
    for ph in range(1, numPh):
        volRatio = volRatio + x[ph]/b[ph]
    mixr: List[Any] = [None]*3
    for ph in range(numPh):
        mixr[ph] = x[ph]/(b[ph]*volRatio)
    return b, r, mu, mixr, volRatio


def _get_node_mix(model: Any, bhp: Any, q_s: Sequence[Any], pN: Any,
                  alpha: Sequence[Any], rho_s: np.ndarray):
    numPh = len(alpha)
    tmp = [alpha[ph]/rho_s[ph] for ph in range(numPh)]
    sumtmp = tmp[0]
    for ph in range(1, numPh):
        sumtmp = sumtmp + tmp[ph]
    mix_s = [tmp[ph]/sumtmp for ph in range(numPh)]
    qt_s = ad_abs(q_s[0])
    for ph in range(1, numPh):
        qt_s = qt_s + ad_abs(q_s[ph])
    for ph in range(numPh):
        mix_s[ph] = vertcat_adi(ad_abs(q_s[ph])/qt_s, mix_s[ph])
    _, _, mu, mix_r, volRatio = _compute_node_props(model, vertcat_adi(bhp, pN), mix_s)
    tmp = mix_s[0]*rho_s[0]
    for ph in range(1, numPh):
        tmp = tmp + mix_s[ph]*rho_s[ph]
    rhom = tmp/volRatio
    mum = mix_r[0]*mu[0]
    for ph in range(1, numPh):
        mum = mum + mix_r[ph]*mu[ph]
    return mix_s, mix_r, rhom, mum


def _conn2surf(v: Sequence[Any], b: Sequence[Any], r: Any, model: Any) -> List[Any]:
    from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
        ThreePhaseBlackOilModel,
    )
    nPh = len(v)
    bv = [b[ph]*v[ph] for ph in range(nPh)]
    vs = list(bv)
    dg = bool(getattr(model, "disgas", False))
    vo = bool(getattr(model, "vapoil", False))
    if vo or dg:
        _, isgas = model.get_variable_field("sg")
        _, isoil = model.get_variable_field("so")
        if isinstance(model, ThreePhaseBlackOilModel):
            if dg:
                vs[isgas] = vs[isgas] + r[isgas][isoil]*bv[isoil]
            if vo:
                vs[isoil] = vs[isoil] + r[isoil][isgas]*bv[isgas]
        elif dg:
            vs[isgas] = vs[isgas] + r[isgas][isoil]*bv[isoil]
    return vs


def _set_where(x: Any, mask: np.ndarray, new: float) -> Any:
    """MATLAB `x(mask) = new` on a copy."""
    x = x.copy() if hasattr(x, "jac") else np.array(x, dtype=float)
    x[mask] = new
    return x


__all__ = ["setup_ms_well_equation_single_well", "setupMSWellEquationSingleWell"]


if __name__ == "__main__":
    def _fluid() -> SimpleNamespace:
        return SimpleNamespace(bW=lambda p: 0*p + 1.0, muW=lambda p: 0*p + 1e-3,
                               bO=lambda p: 0*p + 0.8, muO=lambda p: 0*p + 2e-3,
                               bG=lambda p: 0*p + 100.0, muG=lambda p: 0*p + 2e-5)

    def _model() -> Any:
        phases = {"sw": 0, "so": 1, "sg": 2}
        return SimpleNamespace(
            fluid=_fluid(), disgas=False, vapoil=False,
            get_active_phases=lambda: np.array([True, True, True]),
            get_variable_field=lambda name: ("s", phases[name]),
            get_surface_densities=lambda: np.array([[1000.0, 800.0, 1.0]]))

    def test_node_mix_is_the_surface_volume_mixture():
        alpha = [np.array([0.5]), np.array([0.5]), np.array([0.0])]
        q_s = [-1.0, -1.0, 0.0]
        mix_s, mix_r, rhom, mum = _get_node_mix(_model(), 2e7, q_s, np.array([2e7]),
                                                alpha, np.array([1000.0, 800.0, 1.0]))
        # Node 2: equal masses, surface volumes 0.5/1000 and 0.5/800.
        np.testing.assert_allclose(mix_s[0][1], (0.5/1000)/(0.5/1000 + 0.5/800))
        np.testing.assert_allclose(mix_s[0][0], 0.5)
        x = np.array([mix_s[0][1], mix_s[1][1]])
        vol = x[0]/1.0 + x[1]/0.8
        np.testing.assert_allclose(rhom[1], (x[0]*1000 + x[1]*800)/vol)
        np.testing.assert_allclose(mix_r[0][1] + mix_r[1][1], 1.0)
        print("ok test_node_mix_is_the_surface_volume_mixture")

    def test_dissolved_gas_counts_towards_the_gas_surface_rate():
        from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
            ThreePhaseBlackOilModel,
        )
        model = ThreePhaseBlackOilModel.__new__(ThreePhaseBlackOilModel)
        model.disgas, model.vapoil = True, False
        model.get_variable_field = lambda name: ("s", {"sw": 0, "so": 1, "sg": 2}[name])
        rs = np.array([50.0])
        vs = _conn2surf([np.array([1.0])]*3, [np.array([1.0])]*3,
                        [[None]*3, [None]*3, [None, rs, None]], model)
        np.testing.assert_allclose(vs[2], 51.0)
        np.testing.assert_allclose(vs[1], 1.0)
        print("ok test_dissolved_gas_counts_towards_the_gas_surface_rate")

    test_node_mix_is_the_surface_volume_mixture()
    test_dissolved_gas_counts_towards_the_gas_surface_rate()
    print("\nAll tests passed!")
