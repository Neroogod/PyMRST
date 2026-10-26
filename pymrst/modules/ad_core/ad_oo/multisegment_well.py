"""MRST ``ad-core/models/facilities/MultisegmentWell.m``.

A well resolved into nodes and segments (a well from `convert2MSWell`). On
top of a `SimpleWell`'s rates and bhp it carries, per node below the top, a
pressure `pN` and phase mass fractions `rW`/`rO`/`rG`, and per segment a
mixture mass flux `vmix`; its extra equations are per-node phase balances,
per-segment pressure drops and per-node fraction closures
(`setupMSWellEquationSingleWell`).

The segment operators come from the topology: `C` has a row per segment,
+1 at its first node and -1 at its second, so `grad(x) = -C x` and
`div(v) = C' v`; `aver` averages the two nodes; `segmentUpstr(flag, val)`
takes, for each segment, the value at its first node where `flag` holds and
at its second otherwise, from values stored without the top node.

A well solution without node pressures is initialised as MRST does: node
pressures at the bhp, segment fluxes of a uniform 10 m^3/day per
perforation, node mixtures from the perforated cells' mobilities.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.ad_oo.setup_ms_well_equation_single_well import (
    setup_ms_well_equation_single_well,
)
from pymrst.modules.ad_core.ad_oo.setup_well_control_equations_single_well import (
    setup_well_control_equations_single_well,
)
from pymrst.modules.ad_core.ad_oo.simple_well import SimpleWell
from pymrst.utils.adi import ad_matmul, value
from pymrst.utils.units import DAY

INITIAL_PERFORATION_RATE: float = -10/DAY
NODE_FIELDS = {"rw": 0, "watermassfraction": 0, "ro": 1, "oilmassfraction": 1,
               "rg": 2, "gasmassfraction": 2}


class MultisegmentWell(SimpleWell):
    """A multisegment well: node pressures and fractions, segment fluxes."""

    def __init__(self, W: Any, **options: Any):
        super().__init__(W)
        assert W.isMS
        self.signChangeChop = False
        if options:
            from pymrst.modules.ad_core.ad_oo.physical_model import merge_options
            merge_options(self, options)
        nn, ns = np.size(W.nodes.depth), np.size(W.segments.length)
        topo = np.asarray(W.segments.topo, dtype=np.int64).reshape(-1, 2)
        rows = np.repeat(np.arange(ns), 2)
        C = sp.csr_matrix((np.tile([1.0, -1.0], ns), (rows, topo.ravel())), shape=(ns, nn))
        aver = sp.csr_matrix((np.ones(2*ns), (rows, topo.ravel())), shape=(ns, nn))
        aver = sp.csr_matrix(aver.multiply(1/np.asarray(aver.sum(axis=1))))
        self.operators = SimpleNamespace(
            grad=lambda x: -ad_matmul(C, x),
            div=lambda x: ad_matmul(C.T.tocsr(), x),
            segmentUpstr=lambda flag, val: _segment_upstream_value(flag, val, topo),
            C=C,
            aver=lambda val: ad_matmul(aver, val))

    def get_variable_counts(self, fld: str) -> int:
        key = fld.lower()
        if key in ("pn", "rw", "ro", "rg"):
            return np.size(self.W.nodes.depth) - 1
        if key == "vmix":
            return np.asarray(self.W.segments.topo).reshape(-1, 2).shape[0]
        return super().get_variable_counts(fld)

    def compute_well_equations(self, wellSol0: Any, wellSol: Any, resmodel: Any,
                               q_s: Sequence[Any], bh: Any, packed: Any,
                               dt: float, iteration: int):
        names = [n.lower() for n in packed.extravars_names]
        pN = packed.extravars[names.index("pn")]
        wN = [v for v, n in zip(packed.extravars, names) if n.startswith("r")]
        vmS = packed.extravars[names.index("vmix")]
        resProps = SimpleNamespace(pressure=packed.pressure, mob=packed.mob,
                                   rho=packed.rho, dissolved=packed.dissolved,
                                   b=list(packed.rho))
        rhoS = np.asarray(resmodel.get_surface_densities(), dtype=float).ravel()
        dissolved = resProps.dissolved
        for i in range(len(resProps.b)):
            den = rhoS[i]
            for j in range(len(resProps.b)):
                if dissolved and dissolved[j][i] is not None:
                    den = den + rhoS[j]*dissolved[j][i]
            resProps.b[i] = resProps.b[i]/den
        weqs, weqsMS, qSurf, wellSol, alpha_s, status, cstatus, _ = \
            setup_ms_well_equation_single_well(self, resmodel, wellSol0, wellSol, q_s,
                                               bh, pN, wN, vmS, resProps, dt, iteration)
        extraNames, _ = self.get_extra_equation_names(resmodel)
        qMass = [qSurf[i]*rhoS[i] for i in range(len(qSurf))]
        ctrlEq = setup_well_control_equations_single_well(
            self, wellSol0, wellSol, bh, q_s, status, alpha_s, resmodel)
        wellSol.cqs = np.column_stack([np.asarray(value(q), dtype=float).ravel()
                                       for q in qSurf])
        wellSol.cstatus = cstatus
        wellSol.status = status
        return weqs, ctrlEq, weqsMS, extraNames, qMass, qSurf, wellSol

    def get_variable_field(self, name: str, throw_error: bool = True):
        key = name.lower()
        if key in ("pn", "nodepressure"):
            return "nodePressure", ":"
        if key in NODE_FIELDS:
            return "nodeComp", NODE_FIELDS[key]
        if key in ("vmix", "segmentflux"):
            return "segmentFlux", ":"
        return super().get_variable_field(name, throw_error)

    def update_well_sol(self, ws: Any, variables: Sequence[str], dx: Sequence[Any],
                        resmodel: Any) -> Any:
        variables, dx = list(variables), list(dx)
        act = [True]*len(dx)
        for i, name in enumerate(variables):
            dv = np.asarray(value(dx[i]), dtype=float)
            if name == "pN":
                dv, _ = self.limit_update_relative(dv, np.asarray(ws.bhp, dtype=float),
                                                   self.dpMaxRel)
                dv, _ = self.limit_update_absolute(dv, self.dpMaxAbs)
                ws.nodePressure = ws.nodePressure + dv
            elif name in ("rW", "rO", "rG"):
                ws, _, _ = self.update_state_from_increment(ws, dv, None, name, np.inf,
                                                            self.dsMaxAbs)
                ws = self.cap_property(ws, name, 0, 1)
            elif name == "vmix":
                v = np.array(ws.segmentFlux, dtype=float)
                if self.signChangeChop:
                    changes = (((v + dv < 0) & (v >= 0)) | ((v + dv > 0) & (v <= 0)))
                    if np.any(changes):
                        v[changes] = -np.finfo(float).eps*np.sign(v[changes])
                        dv = dv.copy()
                        dv[changes] = 0
                ws.segmentFlux = v + dv
            else:
                continue
            act[i] = False
        ws = super().update_well_sol(ws, [v for v, a in zip(variables, act) if a],
                                     [d for d, a in zip(dx, act) if a], resmodel)
        return self.ensure_well_sol_consistency(ws)

    def get_extra_primary_variable_names(self, resmodel: Any):
        names, from_res = super().get_extra_primary_variable_names(resmodel)
        active = np.asarray(resmodel.get_active_phases(), dtype=bool).ravel()[:3]
        enames = [n for n, a in zip(["rW", "rO", "rG"], active) if a]
        names = list(names) + ["pN"] + enames + ["vmix"]
        from_res = list(from_res) + [False]*(len(names) - len(from_res))
        return names, from_res

    def get_extra_equation_names(self, resmodel: Any):
        names, types = super().get_extra_equation_names(resmodel)
        active = np.asarray(resmodel.get_active_phases(), dtype=bool).ravel()[:3]
        enames = [n for n, a in zip(["waterNode", "oilNode", "gasNode"], active) if a]
        names = list(names) + enames + ["pDropSeg", "segMassClosure"]
        types = list(types) + ["node"]*len(enames) + ["seg", "alpha"]
        return names, types

    def validate_well_sol(self, resmodel: Any, wellSol: Any, state: Any = None) -> Any:
        if getattr(wellSol, "nodePressure", None) is not None \
                and np.size(wellSol.nodePressure) > 0:
            return wellSol
        W = self.W
        nn = np.size(W.nodes.depth)
        wellSol.nodePressure = np.repeat(np.atleast_1d(wellSol.bhp).astype(float), nn - 1)
        wc = np.asarray(W.cells, dtype=np.int64).ravel()
        sc = np.asarray(state["s"], dtype=float)[wc, :]
        pc = np.asarray(state["pressure"], dtype=float).ravel()[wc]
        fluid = resmodel.fluid
        krw, kro, krg = resmodel.rel_perm_wog(sc[:, 0], sc[:, 1], sc[:, 2], fluid)
        muw = fluid.muW(pc)
        bw = fluid.bW(pc)
        no_sat = np.zeros(wc.size, dtype=bool)
        if resmodel.disgas:
            rs = np.asarray(state["rs"], dtype=float).ravel()[wc]
            muo = fluid.muO(pc, rs, no_sat)
            bo = fluid.bO(pc, rs, no_sat)
        else:
            muo = fluid.muO(pc)
            bo = fluid.bO(pc)
            rs = 0
        if resmodel.vapoil:
            rv = np.asarray(state["rv"], dtype=float).ravel()[wc]
            mug = fluid.muG(pc, rv, no_sat)
            bg = fluid.bG(pc, rv, no_sat)
        else:
            mug = fluid.muG(pc)
            bg = fluid.bG(pc)
            rv = 0
        mw, mo, mg = krw/muw, kro/muo, krg/mug
        dens = np.asarray(resmodel.get_surface_densities(), dtype=float).ravel()
        fwm = dens[0]*bw*mw
        fom = dens[1]*(bo*mo + bg*rv*mg)
        fgm = dens[2]*(bg*mg + bo*rs*mo)
        mob = np.column_stack([fwm, fom, fgm])
        nodemix = W.cell2node @ (mob/(fwm + fom + fgm)[:, None])
        cq = INITIAL_PERFORATION_RATE*np.ones(wc.size)
        cn = W.cell2node @ cq
        C = self.operators.C
        M = C[:, 1:].tocsc()
        MtM = (M.T @ M).tocsc()
        wellSol.segmentFlux = -(M @ spla.spsolve(MtM, cn[1:]))
        nodemix_seg = -(M @ _solve_columns(MtM, nodemix[1:, :]))
        tp = np.asarray(W.segments.topo, dtype=np.int64).reshape(-1, 2)[:, 1] - 1
        while np.any(np.sum(nodemix, axis=1) == 0):
            nodemix[tp, :] = nodemix_seg
        nodemix = nodemix/np.sum(nodemix, axis=1)[:, None]
        nodemix = nodemix[1:, :]
        wellSol.nodeComp = nodemix
        q_top = abs(float((C[:, 0].T @ wellSol.segmentFlux).ravel()[0]))
        wellSol.qWs = -q_top*nodemix[0, 0]*dens[0]
        wellSol.qOs = -q_top*nodemix[0, 1]*dens[1]
        wellSol.qGs = -q_top*nodemix[0, 2]*dens[2]
        return wellSol

    def ensure_well_sol_consistency(self, ws: Any) -> Any:
        ws.nodeComp = ws.nodeComp/np.sum(ws.nodeComp, axis=1)[:, None]
        return ws

    getVariableCounts = get_variable_counts
    computeWellEquations = compute_well_equations
    getVariableField = get_variable_field
    updateWellSol = update_well_sol
    getExtraPrimaryVariableNames = get_extra_primary_variable_names
    getExtraEquationNames = get_extra_equation_names
    validateWellSol = validate_well_sol
    ensureWellSolConsistency = ensure_well_sol_consistency


def _segment_upstream_value(flag: np.ndarray, val: Any, topo: np.ndarray) -> Any:
    """MRST's `segmentUpstreamValue`, with 0-based nodes: the segment's first
    node where `flag`, else its second; `val` has no entry for the top node."""
    flag = np.asarray(flag, dtype=bool).ravel()
    ix = np.where(flag, topo[:, 0], topo[:, 1])
    if ix[0] == -1:
        ix[0] = topo[0, 1]
    return val[np.maximum(ix, 1) - 1]


def _solve_columns(A: sp.csc_matrix, B: np.ndarray) -> np.ndarray:
    """MATLAB `A\\B` for a sparse `A` and a dense matrix `B`."""
    return np.column_stack([spla.spsolve(A, B[:, k]) for k in range(B.shape[1])])


__all__ = ["MultisegmentWell"]


if __name__ == "__main__":
    def _ms_well() -> SimpleNamespace:
        # Top node 0, a two-node tubing 0-1-2, perforations on nodes 1 and 2.
        return SimpleNamespace(
            isMS=True, cells=np.array([0, 1]), WI=np.ones(2), dZ=np.zeros(2),
            refDepth=0.0, sign=-1, status=True, name="ms", type="rate", val=-1.0,
            compi=np.array([0.0, 1.0, 0.0]), lims=None,
            cell2node=sp.csr_matrix(np.array([[0, 0], [1, 0], [0, 1]], dtype=float)),
            nodes=SimpleNamespace(depth=np.array([0.0, 10.0, 20.0]), vol=np.ones(3)),
            segments=SimpleNamespace(topo=np.array([[0, 1], [1, 2]]),
                                     length=np.array([10.0, 10.0])))

    def _resmodel() -> SimpleNamespace:
        return SimpleNamespace(get_active_phases=lambda: np.array([True, True, True]),
                               get_extra_well_primary_variable_names=lambda: [],
                               get_extra_well_equation_names=lambda: ([], []))

    def test_segment_operators_follow_the_topology():
        wm = MultisegmentWell(_ms_well())
        x = np.array([1.0, 3.0, 6.0])
        np.testing.assert_allclose(wm.operators.grad(x), [2.0, 3.0])
        np.testing.assert_allclose(wm.operators.div(np.array([1.0, 1.0])), [1.0, 0.0, -1.0])
        np.testing.assert_allclose(wm.operators.aver(x), [2.0, 4.5])
        val = np.array([10.0, 20.0])
        np.testing.assert_allclose(wm.operators.segmentUpstr(np.array([True, True]), val),
                                   [10.0, 10.0])
        np.testing.assert_allclose(wm.operators.segmentUpstr(np.array([False, False]), val),
                                   [10.0, 20.0])
        print("ok test_segment_operators_follow_the_topology")

    def test_variables_and_equations_are_named_per_node_and_segment():
        wm = MultisegmentWell(_ms_well())
        names, from_res = wm.get_extra_primary_variable_names(_resmodel())
        assert names == ["pN", "rW", "rO", "rG", "vmix"] and not any(from_res)
        enames, etypes = wm.get_extra_equation_names(_resmodel())
        assert enames[-2:] == ["pDropSeg", "segMassClosure"]
        assert etypes == ["node", "node", "node", "seg", "alpha"]
        assert [wm.get_variable_counts(n) for n in ("pN", "rO", "vmix")] == [2, 2, 2]
        assert wm.get_variable_field("rG") == ("nodeComp", 2)
        print("ok test_variables_and_equations_are_named_per_node_and_segment")

    def test_updates_move_node_pressures_and_renormalise_fractions():
        wm = MultisegmentWell(_ms_well())
        ws = SimpleNamespace(bhp=1e7, nodePressure=np.array([1e7, 1e7]),
                             nodeComp=np.array([[0.2, 0.8, 0.0], [0.5, 0.5, 0.0]]),
                             segmentFlux=np.array([-1.0, -1.0]), name="ms")
        ws = wm.update_well_sol(ws, ["pN", "rW", "vmix"],
                                [np.array([1e5, 2e5]), np.array([0.1, -0.6]),
                                 np.array([0.5, 2.0])], _resmodel())
        np.testing.assert_allclose(ws.nodePressure, [1.01e7, 1.02e7])
        np.testing.assert_allclose(ws.nodeComp.sum(axis=1), [1.0, 1.0])
        # Node 2's water fraction is capped at zero before renormalising.
        np.testing.assert_allclose(ws.nodeComp[1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(ws.segmentFlux, [-0.5, 1.0])
        print("ok test_updates_move_node_pressures_and_renormalise_fractions")

    test_segment_operators_follow_the_topology()
    test_variables_and_equations_are_named_per_node_and_segment()
    test_updates_move_node_pressures_and_renormalise_fractions()
    print("\nAll tests passed!")
