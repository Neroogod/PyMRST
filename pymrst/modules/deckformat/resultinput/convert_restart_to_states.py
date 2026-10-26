"""MRST ``deckformat/resultinput/convertRestartToStates.m``.

MRST states from an ECLIPSE (or OPM/tNavigator) restart: per report step the
pressure, saturations (the missing phase as the remainder), time and date,
`rs`, `rv`, polymer, adsorption, phase densities and compositions where
present; the reservoir fluxes per face (`includeFluxes`, mapped by
`eclFaceToFace`, through the NNCs when the restart has them); the well
solutions (`includeWellSols`, from the restart's well arrays or, with
`wellSolsFromRestart` false, from the summary), made consistent over the
steps (`consistentWellSols`: shut wells closed, cross-flow removed, wells
never open dropped, wells that change sign optionally split); aquifers
(`includeAquifers`); and phase mobilities estimated from the fluxes
(`includeMobilities`). Options that cannot be satisfied (no fluxes, no well
arrays) are dropped with MRST's messages; an unknown option only warns, as
MRST's `merge_options` does.

`G` is a grid (or a list whose first is used); cells, faces and `steps` are
as elsewhere in PyMRST (0-based cells, 1-based report steps). States are
dicts. MRST's progress output under `mrstVerbose` is not printed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
    fix_var_name,
)
from pymrst.utils.units import BARSA, DAY, FT, METER, PSIA, STB

OPTIONS = dict(additionalFields=(), includeWellSols=True, includeFluxes=True,
               includeMobilities=True, includeAquifers=False, neighbors=None,
               wellSolsFromRestart=True, consistentWellSols=True,
               splitWellsOnSignChange=False, steps=None, restartInfo=None,
               addTrajectory=True, removeClosedWells=True, removeCrossflow=True,
               setToClosedTol=0*METER**3/DAY)
ECLIPSE_PROGRAMS = (100, 300, 500, 700)
PHASE_MASKS = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1],
                        [1, 1, 1]], dtype=bool)
PHASE_NAMES = ("WAT", "OIL", "GAS")


def convert_restart_to_states(fn: Any, G: Any, **options: Any
                              ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """MRST `[states, rstrt] = convertRestartToStates(fn, G, ...)`."""
    from pymrst.modules.deckformat.resultinput.convert_summary_to_well_sols import (
        convert_summary_to_well_sols,
    )
    from pymrst.modules.deckformat.resultinput.private.create_aqui_sol import (
        create_aqui_sol,
    )
    from pymrst.modules.deckformat.resultinput.private.create_well_sol import (
        create_well_sol,
    )
    from pymrst.modules.deckformat.resultinput.private.make_well_sols_consistent import (
        make_well_sols_consistent,
    )
    from pymrst.modules.deckformat.resultinput.read_eclipse_restart_un_fmt import (
        read_eclipse_restart_un_fmt,
    )
    opt = SimpleNamespace(**OPTIONS)
    unsupported = [k for k in options if k not in OPTIONS]
    if unsupported:
        warnings.warn("Unsupported option in convertRestartToStates\n * "
                      + "\n * ".join(unsupported))
    for k, v in options.items():
        if k in OPTIONS:
            setattr(opt, k, v)
    G = G[0] if isinstance(G, (list, tuple)) else G
    N = np.asarray(G.faces.neighbors if opt.neighbors is None else opt.neighbors)
    fn = str(fn)
    if fn.endswith(".UNRST"):
        fn = fn[:-len(".UNRST")]
    rstrt = read_eclipse_restart_un_fmt(fn, opt.restartInfo, opt.steps)
    opt, phn, unit, tr, na, is_eclipse = _check_and_process_input(fn, rstrt, opt)

    if opt.includeFluxes:
        ix = getattr(G.faces, "ix", None)
        if ix is None:
            nnc = _nnc(fn) if opt.nncPresent else None
            from pymrst.modules.deckformat.resultinput.private.ecl_face_to_face import (
                ecl_face_to_face,
            )
            ix = ecl_face_to_face(G, nnc, neighbors=opt.neighbors)
        flux_nms = _get_res_flux_nms(phn, is_eclipse)
    if opt.includeWellSols:
        if opt.wellSolsFromRestart:
            dims = np.asarray(G.cartDims, dtype=np.int64)
            glob2act = np.full(int(np.prod(dims)), -1, dtype=np.int64)
            glob2act[np.asarray(G.cells.indexMap, dtype=np.int64)] = np.arange(G.cells.num)

            def ijk2act(ijk):
                ijk = np.atleast_2d(np.asarray(ijk, dtype=np.int64)) - 1
                return glob2act[np.ravel_multi_index(ijk.T, dims, order="F")]
        else:
            well_sols, tm = convert_summary_to_well_sols(fn, unit.name)
            well_sols = [w for w, keep in zip(well_sols, _ms2rs(tm, tr)) if keep]

    satnms = [f"S{x}" for x in phn]
    satix = np.array([nm in rstrt for nm in satnms])
    assert np.count_nonzero(~satix) <= 1, "Saturation output found for less than nPh-1 phases"
    states: List[Dict[str, Any]] = []
    for k in range(tr.size):
        state: Dict[str, Any] = {"pressure": np.zeros(na), "s": np.zeros((na, len(phn))),
                                 "flux": None, "wellSol": None, "time": "-1", "date": None}
        p = rstrt["PRESSURE"][k]
        if p is not None and np.size(p):
            state["pressure"] = np.asarray(p, dtype=float)*unit.p
        for ii in np.flatnonzero(satix):
            state["s"][:, ii] = rstrt[satnms[ii]][k]
        if np.any(~satix):
            state["s"][:, ~satix] = (1 - state["s"].sum(axis=1))[:, None]
        state["time"] = tr[k]*unit.t
        state["date"] = np.asarray(rstrt["INTEHEAD"][k]).ravel()[64:67]
        if "SEQNUM" in rstrt:
            state["index"] = rstrt["SEQNUM"][k]
        if "RS" in rstrt:
            state["rs"] = np.asarray(rstrt["RS"][k], dtype=float)*(unit.qg/unit.ql)
        if "RV" in rstrt:
            state["rv"] = np.asarray(rstrt["RV"][k], dtype=float)*(unit.ql/unit.qg)
        for src, dst in (("POLYMER", "c"), ("PADS", "ads"), ("GAS_DEN", "rho_gas"),
                         ("OIL_DEN", "rho_oil"), ("WAT_DEN", "rho_water")):
            if src in rstrt:
                state[dst] = rstrt[src][k]
        for prefix, dst in (("XMF", "x"), ("YMF", "y"), ("ZMF", "components")):
            _convert_compositions(state, prefix, rstrt, k, dst)
        if opt.includeFluxes:
            f = np.zeros((N.shape[0], len(phn)))
            if rstrt[flux_nms[0].I][k] is not None and np.size(rstrt[flux_nms[0].I][k]):
                for phi, nms in enumerate(flux_nms):
                    f[ix.iI, phi] = ix.sI*np.asarray(rstrt[nms.I][k])[ix.iIe]
                    f[ix.iJ, phi] = ix.sJ*np.asarray(rstrt[nms.J][k])[ix.iJe]
                    if np.any(ix.iK):
                        f[ix.iK, phi] = ix.sK*np.asarray(rstrt[nms.K][k])[ix.iKe]
                    if getattr(ix, "iN", None) is not None and ix.iN.size:
                        f[ix.iN, phi] = ix.sN*np.asarray(rstrt[nms.N][k])[ix.iNe]
            f[~np.isfinite(f)] = 0
            state["flux"] = f*unit.qr
        if opt.includeWellSols:
            state["wellSol"] = (create_well_sol(rstrt, k + 1, ijk2act)
                                if opt.wellSolsFromRestart else well_sols[k])
        if opt.includeAquifers:
            aq = create_aqui_sol(rstrt, k + 1, ijk2act)
            if aq:
                state["aquiSol"] = aq
        states.append(state)

    emap = getattr(G.cells, "eMap", None)
    if emap is not None:
        for state in states:
            for name in ("pressure", "s", "rs", "rv"):
                if name in state:
                    state[name] = np.asarray(state[name])[emap]
    if opt.consistentWellSols and opt.includeWellSols:
        states = make_well_sols_consistent(states)
        states = _process_well_states(states, opt)
    if opt.includeMobilities:
        _add_mobilities(states, N, G.cells.num)
    return states, rstrt


convertRestartToStates = convert_restart_to_states


def _get_units(unit: str) -> SimpleNamespace:
    if unit == "metric":
        return SimpleNamespace(name=unit, len=METER, p=BARSA, ql=METER**3/DAY,
                               qg=METER**3/DAY, qr=METER**3/DAY, t=DAY)
    if unit == "field":
        return SimpleNamespace(name=unit, p=PSIA, ql=STB/DAY, qg=1000*FT**3/DAY,
                               qr=STB/DAY, t=DAY)
    raise ValueError(f"Unit {unit} not supported")


def _get_res_flux_nms(phns, is_eclipse: bool) -> List[SimpleNamespace]:
    prefix = "FLR" if is_eclipse else "FLO"
    return [SimpleNamespace(**{d: fix_var_name(f"{prefix}{ph}{d}+") for d in "IJKN"})
            for ph in phns]


def _check_and_process_input(fn: str, rstrt: Dict[str, List[Any]], opt: SimpleNamespace):
    heads = rstrt["INTEHEAD"]
    is_eclipse = int(np.ravel(heads[0])[94]) in ECLIPSE_PROGRAMS
    ih = np.ravel(next(h for h in heads if h is not None and np.size(h)))
    na = int(ih[11])
    ii = int(ih[2])
    if ii not in (1, 2, 3):
        warnings.warn(f"Unknown unit indicator {ii}, assuming metric")
        ii = 1
    unit = _get_units(("metric", "field", "lab")[ii - 1])
    ii = int(ih[14])
    if ii not in range(1, 8):
        warnings.warn(f"Unknown phase indicator {ii}, assuming three-phase")
        ii = 7
    phn = [p for p, m in zip(PHASE_NAMES, PHASE_MASKS[ii - 1]) if m]
    tr = np.array([float(np.ravel(x)[0]) for x in rstrt["DOUBHEAD"]])
    if opt.includeFluxes:
        prefix = "FLR" if is_eclipse else "FLO"
        if fix_var_name(f"{prefix}{phn[0]}I+") not in rstrt:
            opt.includeFluxes = False
        else:
            opt.nncPresent = fix_var_name(f"FLR{PHASE_NAMES[0]}N+") in rstrt
            if opt.nncPresent:
                chk = [Path(f"{fn}.{x}").exists() for x in ("EGRID", "GRID", "INIT")]
                if not ((chk[0] or chk[1]) and chk[2]):
                    warnings.warn("Unable to process NNCs, could not find EGRID/GRID and/or "
                                  "INIT-files. No flux fields will be added to states.")
                    opt.includeFluxes = False
    opt.includeMobilities = opt.includeMobilities and opt.includeFluxes
    if opt.includeWellSols:
        if opt.wellSolsFromRestart:
            req = ("IWEL", "SWEL", "XWEL", "ZWEL", "ICON", "SCON", "XCON")
            missing = [r for r in req if r not in rstrt]
            if missing:
                warnings.warn("Unable to produce wellSols from RSTRT-file. Could not find "
                              "fields: " + "".join(f"\n{r} " for r in missing))
                opt.includeWellSols = False
        else:
            missing = [x for x in ("SMSPEC", "UNSMRY") if not Path(f"{fn}.{x}").exists()]
            if missing:
                warnings.warn("Unable to produce wellSols from summary. Could not find "
                              "files: " + "".join(f"\n{fn}.{x} " for x in missing))
                opt.includeWellSols = False
    if opt.additionalFields:
        warnings.warn("Including addtional fields is not yet implemented")
    return opt, phn, unit, tr, na, is_eclipse


def _nnc(fn: str) -> np.ndarray:
    from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
        read_eclipse_output_file_un_fmt,
    )
    grd_file = f"{fn}.EGRID" if Path(f"{fn}.EGRID").exists() else f"{fn}.GRID"
    grd = read_eclipse_output_file_un_fmt(grd_file)
    if "NNC1" in grd:
        return np.column_stack([np.ravel(grd["NNC1"].values), np.ravel(grd["NNC2"].values)])
    init = read_eclipse_output_file_un_fmt(f"{fn}.INIT")
    if "NNC1" in init:
        return np.column_stack([np.ravel(init["NNC1"].values),
                                np.ravel(init["NNC2"].values)])
    raise ValueError("NNC fluxes given in restart, but indices for NNC cells neither found "
                     "in GRID or INIT files")


def _ms2rs(tm: np.ndarray, tr: np.ndarray) -> np.ndarray:
    thr = np.min(np.diff(tm))/10
    ii = np.zeros(tm.size, dtype=bool)
    ir_next = 0
    for k in range(tm.size):
        if ir_next < tr.size and tr[ir_next] <= tm[k] + thr:
            ii[k] = True
            ir_next += 1
    assert tr.size == np.count_nonzero(ii), "Ministeps not compatible with report times"
    if np.max(np.abs(tr - tm[ii])) > thr:
        warnings.warn("Mismatch observed between ministeps and reportsteps")
    return ii


def _process_well_states(states: List[Dict[str, Any]], opt: SimpleNamespace):
    import copy
    ns = len(states)
    nw = len(states[0]["wellSol"])
    if opt.splitWellsOnSignChange:
        ws = [s["wellSol"] for s in states]
        sgn = np.array([[w.sign for w in row] for row in ws])
        for k in range(nw):
            if np.any(sgn[:, k] != sgn[0, k]):
                first_sgn = sgn[0, k]
                first_nm, second_nm = f"{ws[0][k].name} (inj)", f"{ws[0][k].name} (prod)"
                if first_sgn < 0:
                    first_nm, second_nm = second_nm, first_nm
                for row in ws:
                    ncon = np.size(row[k].cells)
                    other = copy.deepcopy(row[k])
                    row.append(other)
                    for w, nm, keep in ((row[k], first_nm, row[k].sign == first_sgn),
                                        (other, second_nm, other.sign != first_sgn)):
                        w.name = nm
                        w.status = bool(w.status and keep)
                        w.sign = first_sgn if w is row[k] else -first_sgn
                        if not w.status:
                            _close(w, ncon)
        nw = len(ws[0])
        for s, row in zip(states, ws):
            s["wellSol"] = row
    always_closed = np.ones(nw, dtype=bool)
    for state in states:
        ws = state["wellSol"]
        for w in ws:
            if w.status:
                if opt.removeCrossflow:
                    cf = np.asarray(w.flux)*w.sign < 0
                    w.flux = np.where(cf, 0.0, w.flux)
                if w.resv*w.sign <= opt.setToClosedTol:
                    _close(w, np.size(w.cells))
                always_closed = always_closed & ~np.array([x.status for x in ws])
        state["wellSol"] = ws
    if opt.removeClosedWells:
        for state in states:
            state["wellSol"] = [w for w, c in zip(state["wellSol"], always_closed) if not c]
    return states


def _close(w: Any, ncon: int) -> None:
    w.status = False
    w.cstatus = np.zeros(ncon, dtype=bool)
    w.resv = 0
    w.flux = np.zeros(ncon)
    w.cqs = np.zeros((ncon, 3))


def _convert_compositions(state, prefix, rstrt, step, newfield) -> None:
    if f"{prefix}1" not in rstrt:
        return
    ncomp = 1
    while ncomp < 1000 and f"{prefix}{ncomp + 1}" in rstrt:
        ncomp += 1
    state[newfield] = np.column_stack([np.asarray(rstrt[f"{prefix}{i}"][step], dtype=float)
                                       for i in range(1, ncomp + 1)])


def _add_mobilities(states, N: np.ndarray, nc: int) -> None:
    ie = np.all(N >= 0, axis=1)
    n = N[ie]
    nn = n.shape[0]
    C = sp.csr_matrix((np.ones(2*nn), (n.ravel(), np.repeat(np.arange(nn), 2))),
                      shape=(nc, nn))
    eps = np.finfo(float).eps
    for state in states:
        qa = C @ np.abs(state["flux"][ie, :])
        ws = state.get("wellSol")
        if ws:
            wc = np.concatenate([np.ravel(w.cells) for w in ws])
            qwr = qa[wc, :]
            qw_tot = np.abs(np.concatenate([np.ravel(w.flux) for w in ws]))
            sqw = np.maximum(qwr.sum(axis=1) + qw_tot, 100*eps*qw_tot.max())
            qa[wc, :] = qwr/(1 - qw_tot/sqw)[:, None]
        sq = qa.sum(axis=1)
        sq = np.maximum(sq, 100*eps*sq.max())
        state["mob"] = qa/sq[:, None]


__all__ = ["convert_restart_to_states", "convertRestartToStates"]


if __name__ == "__main__":
    NORNE = (Path(__file__).resolve().parents[5] / "mrst-2024a-Geomech" / "examples" / "data"
             / "norne_field_model" / "opm-tests" / "norne" / "opm-simulation-reference"
             / "flow_legacy" / "NORNE_ATW2013")

    def test_mobilities_are_flux_fractions():
        N = np.array([[0, 1], [1, -1]])
        states = [{"flux": np.array([[1.0, 3.0], [0.0, 0.0]])}]
        _add_mobilities(states, N, 2)
        np.testing.assert_allclose(states[0]["mob"], [[0.25, 0.75], [0.25, 0.75]])
        print("ok test_mobilities_are_flux_fractions")

    def test_report_steps_pick_their_ministeps():
        ii = _ms2rs(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0]))
        np.testing.assert_array_equal(ii, [False, True, False, True])
        print("ok test_report_steps_pick_their_ministeps")

    def test_norne_opm_restart_converts_with_mrsts_warnings():
        if not Path(str(NORNE) + ".UNRST").exists():
            print("skip test_norne_opm_restart_converts_with_mrsts_warnings")
            return
        from pymrst.modules.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
        from pymrst.modules.deckformat.grid.init_eclipse_grid import init_eclipse_grid
        deck = read_eclipse_deck(NORNE.parents[4] / "opm-data" / "norne" / "NORNE_ATW2013.DATA")
        G = init_eclipse_grid(deck, split_disconnected=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            states, _ = convert_restart_to_states(NORNE, G, use_opm=True,
                                                  consistentWellSols=False)
        messages = " ".join(str(w.message) for w in caught)
        assert "use_opm" in messages and "wellSols from RSTRT" in messages
        assert len(states) == 65 and states[0]["s"].shape[1] == 3
        np.testing.assert_allclose(states[0]["s"].sum(axis=1), 1.0)
        print("ok test_norne_opm_restart_converts_with_mrsts_warnings")

    test_mobilities_are_flux_fractions()
    test_report_steps_pick_their_ministeps()
    test_norne_opm_restart_converts_with_mrsts_warnings()
    print("\nAll tests passed!")
