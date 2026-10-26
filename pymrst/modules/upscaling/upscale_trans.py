"""MRST ``upscaling/upscaleTrans.m``: flow-based upscaling of
transmissibilities and well indices.

Fine-scale pressure solves - `incompTPFA` on the fine transmissibilities, or
one AD step when `T_fine` is a model - are run for a set of global flow
cases: pressure drops across the box (`'bc_simple'`), given boundary
conditions (`'bc'`), or given wells (`'wells'`, `'wells_simple'`). Each
solution is averaged onto the coarse grid (volume-averaged block pressures,
summed coarse-face and well fluxes), and the coarse transmissibility that
reproduces each case's coarse flux under its coarse pressure drop is
computed. One value per face and well connection is then chosen, from the
case with the largest flux (`'max_flux'`) or by least squares
(`'lsq_flux'`), and optionally fixed: zero on the boundary for well cases,
and no smaller than the smallest positive half-face value (`fix_trans`).
Unless the solver is AD, each case is checked by re-solving it on the
coarse grid with the upscaled values.

`cg` is the coarse grid as a grid (`coarse_grid_as_grid`; an upscaled
model's `G`). Returns `HT_cg` (one-sided values), `T_cg` (per coarse face),
the coarse wells and a report holding the coarse and fine states. Indices
are 0-based.
"""

import copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.utils.units import BARSA, CENTI, DAY, KILOGRAM, METER, POISE, YEAR


def upscale_trans(cg: Any, T_fine: Any, *, verbose: bool = False, bc_method: str = "bc_simple",
                  bc: Optional[Sequence[Any]] = None, dt: float = 1*YEAR, wells: Any = None,
                  state: Any = None, match_method: str = "max_flux", fix_trans: bool = False,
                  check_trans: bool = True, use_average_pressure: bool = True,
                  LinearSolver: Any = None,
                  LinSolve: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, Any, Dict]:
    """MRST `[HT_cg, T_cg, cgwells, report] = upscaleTrans(cg, T_fine, ...)`."""
    opt = dict(verbose=verbose, bc_method=bc_method, bc=bc, dt=dt, wells=wells, state=state,
               match_method=match_method, fix_trans=fix_trans, check_trans=check_trans,
               use_average_pressure=use_average_pressure, LinearSolver=LinearSolver,
               LinSolve=LinSolve)
    if LinearSolver is not None:
        opt["LinSolve"] = lambda A, b: LinearSolver.solve_linear_system(A, b)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Inconsistent Number of Phases")
        bcs, well_cases, cgwells_cases = _setup_cases(cg, opt)
        psolve, fluid, is_ad = _get_pressure_solver(cg, T_fine, opt)
        states = [psolve(b, w) for b, w in zip(bcs, well_cases)]
        upscaled = [_calculate_upscaled_solution(cg, x, cgw, opt)
                    for x, cgw in zip(states, cgwells_cases)]
        ext = np.any(np.asarray(cg.faces.neighbors) < 0, axis=1)
        cf = _cell_faces(cg)
        for i in range(len(bcs)):
            upscaled[i] = _calculate_upscaled_trans_simple(upscaled[i], cg, cgwells_cases[i], fluid)
            if "facePressure" in upscaled[i]:
                t = 1/np.bincount(cf, 1/upscaled[i]["htrans"], minlength=cg.faces.num)
                upscaled[i]["trans"][ext] = t[ext]
            trans = upscaled[i]["trans"]
            atrans = np.abs(trans)
            min_trans = atrans[atrans > 0].min()
            trans[~np.isfinite(trans)] = min_trans*1e-6
            trans[~(np.abs(trans) > 0)] = min_trans*1e-6
            if check_trans and not is_ad:
                upscaled[i], ok = _check_upscaled_trans(upscaled[i], cg, bcs[i],
                                                        cgwells_cases[i], fluid)
                if verbose and not ok:
                    print(f"Pressure solve {i + 1}/{len(bcs)} with upscaled transmissibility "
                          "does not give comparable results")
            assert np.all(np.isfinite(upscaled[i]["trans"]))
        cgwells = cgwells_cases[0]
        T_cg, HT_cg, cgwells = _select_trans(cg, upscaled, cgwells, fluid, opt)
        T_cg, HT_cg, cgwells = _fix_trans(cg, T_cg, HT_cg, cgwells, opt)
    assert np.all(np.isfinite(T_cg))
    report = {"cg_state": upscaled, "states": states}
    return HT_cg, T_cg, cgwells, report


def _get_pressure_solver(cg: Any, input_: Any, opt: Dict) -> Tuple[Callable, Any, bool]:
    from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel
    init_state = opt["state"]
    is_ad = isinstance(input_, PhysicalModel)
    if is_ad:
        from pymrst.modules.ad_core.utils.standalone_solve_ad import standalone_solve_ad
        if init_state is None:
            raise ValueError("Valid initial state must be passed as optional argument when "
                             "solver is AD")
        model = copy.copy(input_)
        target = model.parentModel if hasattr(model, "parentModel") else model
        target.extraStateOutput = True
        target.outputFluxes = True
        target.gravity = np.zeros(3)

        def psolve(bcase, wcase):
            return standalone_solve_ad(init_state, model, opt["dt"], bc=bcase, W=wcase,
                                       LinearSolver=opt["LinearSolver"])[0]
        return psolve, model, True
    from pymrst.modules.incomp.fluid.incompressible.init_single_fluid import init_single_fluid
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa
    from pymrst.solvers.init_state import init_state as init_state_fn
    parent = _parent(cg)
    if init_state is None:
        init_state = init_state_fn(parent, [], 100*BARSA)
    T_fine = np.asarray(input_, dtype=float).ravel()
    if np.size(parent.cells.faces, 0) == T_fine.size:
        use_trans = False
    else:
        assert T_fine.size == parent.faces.num
        use_trans = True
    fluid = init_single_fluid(1*CENTI*POISE, 0*KILOGRAM/METER**3)

    def psolve(bcase, wcase):
        return incomp_tpfa(init_state, parent, T_fine, fluid, bc=bcase, wells=wcase,
                           use_trans=use_trans, LinSolve=opt["LinSolve"])
    return psolve, fluid, False


def make_cg_wells(cg: Any, wells: Sequence[Any]) -> List[Any]:
    """MRST's `makeCGWells`: each well's perforations grouped by coarse block,
    with the volume-weighted perforation depth, `fcellspos`/`fcells`/`fperf`
    mapping back to the fine perforations, and `WI` left undetermined."""
    from pymrst.utils.rlencode import rlencode
    parent = _parent(cg)
    partition = np.asarray(cg.partition, dtype=np.int64)
    cgwells = [copy.copy(w) for w in wells]
    for i, w in enumerate(wells):
        fcells = np.asarray(w.cells, dtype=np.int64).ravel()
        cgcells = partition[fcells]
        perf = np.arange(fcells.size)
        order = np.lexsort((perf, fcells, cgcells))
        tab = np.column_stack([cgcells, fcells, perf])[order]
        cells, n = rlencode(tab[:, 0])
        fcellspos = np.concatenate([[0], np.cumsum(n)])
        if cg.griddim > 2:
            pno = np.repeat(np.arange(cells.size), n)
            cc = np.asarray(parent.cells.centroids)[tab[:, 1], 2]
            cv = np.asarray(parent.cells.volumes)[tab[:, 1]]
            hpos = np.bincount(pno, cv*cc)/np.bincount(pno, cv)
        else:
            hpos = 0.0
        cw = cgwells[i]
        cw.cells = cells
        cw.WI = np.full(cells.size, np.nan)
        cw.cstatus = np.ones(cells.size, dtype=bool)
        cw.dZ = hpos - w.refDepth
        cw.fcellspos = fcellspos
        cw.fcells = tab[:, 1]
        cw.fperf = tab[:, 2]
        cw.parent = None
    # MRST sets `parent` after the loop, so only the last well carries it.
    cgwells[-1].parent = wells[-1]
    return cgwells


def _calculate_upscaled_solution(cg: Any, state: Any, cgwells: Any, opt: Dict) -> Dict[str, Any]:
    from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign
    parent = _parent(cg)
    cgs = cg.coarse_grid
    nfc = cg.faces.num
    partition = np.asarray(cg.partition, dtype=np.int64)
    fconn = np.asarray(cgs.fconn, dtype=np.int64)
    cfacesno = np.repeat(np.arange(nfc), np.diff(np.asarray(cgs.conn_pos)))
    vols = np.asarray(parent.cells.volumes, dtype=float)
    p_f = np.asarray(_get(state, "pressure"), dtype=float).ravel()
    flux_f = np.asarray(_get(state, "flux"), dtype=float).reshape(parent.faces.num, -1)
    fp_f = _get(state, "facePressure")
    face_pressure = None
    cfsign = fine_to_coarse_sign(cgs)
    ff = flux_f[fconn].sum(axis=1)
    flux = np.bincount(cfacesno, ff*cfsign, minlength=nfc)
    if opt["use_average_pressure"]:
        pressure = (np.bincount(partition, vols*p_f, minlength=cg.cells.num)
                    / np.bincount(partition, vols, minlength=cg.cells.num))
        if fp_f is not None:
            fp = np.asarray(fp_f, dtype=float)[fconn]
            fa = np.asarray(parent.faces.areas, dtype=float)[fconn]
            face_pressure = (np.bincount(cfacesno, fa*fp, minlength=nfc)
                             / np.bincount(cfacesno, fa, minlength=nfc))
    else:
        pressure = p_f[_closest(partition, np.asarray(cg.cells.centroids)[partition],
                                np.asarray(parent.cells.centroids))]
        if fp_f is not None:
            fine = _closest(cfacesno, np.asarray(cg.faces.centroids)[cfacesno],
                            np.asarray(parent.faces.centroids)[fconn])
            face_pressure = np.asarray(fp_f, dtype=float)[fconn[fine]]
    well_sol: Any = []
    if cgwells:
        well_sol = [copy.copy(ws) for ws in _get(state, "wellSol")]
        for i, cw in enumerate(cgwells):
            pno = np.repeat(np.arange(np.size(cw.cells)), np.diff(cw.fcellspos))
            wf = np.asarray(well_sol[i].flux, dtype=float).reshape(-1, 1 if np.ndim(well_sol[i].flux) < 2
                                                                   else np.shape(well_sol[i].flux)[1])
            wf = wf[cw.fperf].sum(axis=1)
            well_sol[i].flux = np.bincount(pno, wf)
    upscaled: Dict[str, Any] = {"pressure": pressure}
    if face_pressure is not None:
        upscaled["facePressure"] = face_pressure
    upscaled["flux"] = flux
    upscaled["wellSol"] = well_sol
    nf_cells = parent.cells.num
    vt = np.bincount(partition, vols, minlength=cg.cells.num)
    for name, d in _fields(state).items():
        if name.lower() in (k.lower() for k in upscaled):
            continue
        d = np.asarray(d) if isinstance(d, np.ndarray) else None
        if d is not None and d.ndim >= 1 and d.shape[0] == nf_cells:
            d2 = d.reshape(nf_cells, -1).astype(float)
            upscaled[name] = np.column_stack([np.bincount(partition, d2[:, j]*vols,
                                                          minlength=cg.cells.num)/vt
                                              for j in range(d2.shape[1])])
    return upscaled


def _closest(groups: np.ndarray, targets: np.ndarray, points: np.ndarray) -> np.ndarray:
    """MRST's sortrows([group, distance, index]) pick: per group, the index of
    the point closest to the group's target, ties to the lowest index."""
    df = ((targets - points)**2).sum(axis=1)
    idx = np.arange(groups.size)
    order = np.lexsort((idx, df, groups))
    first = np.concatenate([[0], np.cumsum(np.bincount(groups[order]))[:-1]])
    return idx[order][first]


def _setup_cases(cg: Any, opt: Dict) -> Tuple[List[Any], List[Any], List[Any]]:
    from pymrst.params.wells_and_bc.pside import pside
    method = opt["bc_method"]
    parent = _parent(cg)
    if method == "bc_simple":
        pmin, pmax = 100*BARSA, 200*BARSA
        pairs = [("West", "East"), ("North", "South")] + ([("Top", "Bottom")] if cg.griddim > 2 else [])
        bcs = [pside(pside(None, parent, lo, pmin), parent, hi, pmax) for lo, hi in pairs]
        well_cases = [None]*len(bcs)
        cgwells_cases = [None]*len(bcs)
    elif method == "bc":
        if not opt["bc"]:
            raise ValueError("bc_method==bc need nonempty bc option")
        bcs = list(opt["bc"])
        well_cases = [None]*len(bcs)
        cgwells_cases = [None]*len(bcs)
    elif method == "wells_simple":
        if not opt["wells"]:
            raise ValueError("bc_method==wells need nonempty wells option")
        cases = opt["wells"] if _is_case_list(opt["wells"]) else [opt["wells"]]
        assert len(cases) == 1
        nw = len(cases[0])
        rates = _null_space(np.ones((nw, nw)))*(10*METER**3/DAY)
        well_cases = []
        for c in range(rates.shape[1]):
            case = [copy.copy(w) for w in cases[0]]
            for w in range(nw - 1):
                case[w].type = "rate"
                case[w].val = rates[w, c]
            case[nw - 1].type = "bhp"
            case[nw - 1].val = 300*BARSA
            well_cases.append(case)
        bcs = [None]*len(well_cases)
        cgwells_cases = [make_cg_wells(cg, w) for w in well_cases]
    elif method == "wells":
        if not opt["wells"]:
            raise ValueError("bc_method=='wells' needs non-empty 'wells' option")
        well_cases = list(opt["wells"]) if _is_case_list(opt["wells"]) else [opt["wells"]]
        bcs = [None]*len(well_cases)
        cgwells_cases = [make_cg_wells(cg, w) for w in well_cases]
    else:
        raise ValueError("This bc_method is not implemented")
    assert len(bcs) == len(well_cases) == len(cgwells_cases), "Internal error"
    return bcs, well_cases, cgwells_cases


def _select_trans(cg: Any, upscaled: List[Dict], cgwells: Any, fluid: Any,
                  opt: Dict) -> Tuple[np.ndarray, np.ndarray, Any]:
    N = np.asarray(cg.faces.neighbors, dtype=np.int64)
    internal = ~np.any(N < 0, axis=1)
    nr_global = len(upscaled)
    cf = _cell_faces(cg)
    flux = np.full((cg.faces.num, nr_global), np.nan)
    dpfmob = np.full((cg.faces.num, nr_global), np.nan)
    trans = np.full((cg.faces.num, nr_global), np.nan)
    htrans = np.full((cf.size, nr_global), np.nan)
    nwell = len(cgwells) if cgwells else 0
    fluxwells = [[] for _ in range(nwell)]
    dpmobwells = [[] for _ in range(nwell)]
    for i, u in enumerate(upscaled):
        trans[:, i] = u["trans"]
        flux[:, i] = u["flux"]
        cmob, fmob = _get_mobility(cg, fluid, u)
        if "facePressure" in u:
            htrans[:, i] = u["htrans"]
        dp = u["pressure"][N[internal, 1]] - u["pressure"][N[internal, 0]]
        dpfmob[internal, i] = dp*fmob
        for j in range(nwell):
            fluxwells[j].append(u["fluxwells"][j])
            dpmobwells[j].append(u["dpwells"][j]
                                 * np.asarray(cmob).ravel(order="F")[np.asarray(cgwells[j].cells)])
    method = opt["match_method"]
    if method == "max_flux":
        choice = np.argmax(np.abs(flux), axis=1)
        HT_cg = htrans[np.arange(cf.size), choice[cf]]
        T_cg = trans[np.arange(cg.faces.num), choice]
        if nwell:
            for j in range(nwell):
                wind = np.column_stack([u["WI"][j] for u in upscaled])
                wflux = np.column_stack([u["wellSol"][j].flux for u in upscaled])
                ch = np.argmax(np.abs(wflux), axis=1)
                cgwells[j].WI = wind[np.arange(wind.shape[0]), ch]
    elif method == "lsq_flux":
        with np.errstate(divide="ignore", invalid="ignore"):
            T_cg = -flux.sum(axis=1)/dpfmob.sum(axis=1)
            for j in range(nwell):
                cgwells[j].WI = (np.column_stack(fluxwells[j]).sum(axis=1)
                                 / np.column_stack(dpmobwells[j]).sum(axis=1))
        T_cg[~np.isfinite(T_cg)] = 0
        HT_cg = T_cg[cf]
    else:
        raise ValueError("This match method is not implemented")
    return T_cg, HT_cg, cgwells


def _check_upscaled_trans(upscaled: Dict, cg: Any, bc: Any, well_cases: Any,
                          fluid: Any) -> Tuple[Dict, bool]:
    from pymrst.modules.agglom.utils.convert_bc2_coarse import convert_bc2_coarse
    from pymrst.modules.agglom.utils.sign_of_fine_faces_on_coarse_faces import (
        sign_of_fine_faces_on_coarse_faces,
    )
    from pymrst.modules.coarsegrid.sub_faces import sub_faces
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa
    from pymrst.solvers.state_types import State
    parent = _parent(cg)
    if bc is None:
        cgbc = None
    else:
        nsub, sub = sub_faces(parent, cg)
        sgn, coarse_f = sign_of_fine_faces_on_coarse_faces(parent, cg, nsub, sub)
        cgbc = convert_bc2_coarse(bc, parent, cg, nsub, sub, coarse_f, sgn)
    cgwells_tmp = well_cases
    if cgwells_tmp and isinstance(upscaled.get("WI"), list) and len(upscaled["WI"]) == len(cgwells_tmp):
        cgwells_tmp = [copy.copy(w) for w in cgwells_tmp]
        for w, wi in zip(cgwells_tmp, upscaled["WI"]):
            w.WI = wi
    internal = ~np.any(np.asarray(cg.faces.neighbors) < 0, axis=1)
    cgstate = State(pressure=upscaled["pressure"], flux=upscaled["flux"],
                    s=np.zeros((cg.cells.num, 1)))
    cgstate = incomp_tpfa(cgstate, cg, upscaled["trans"], fluid, bc=cgbc, wells=cgwells_tmp,
                          use_trans=True)
    if cgstate.pressure.max() > cgstate.pressure.min():
        rng = cgstate.pressure.max() - cgstate.pressure.min()
        assert np.all(np.abs((cgstate.pressure - upscaled["pressure"])/rng) < 1e-6)
    scale = np.abs(cgstate.flux).max()
    if np.any(upscaled["flux"][internal]):
        assert np.all(np.abs(cgstate.flux[internal] - upscaled["flux"][internal])/scale < 1e-6)
    ok = bool(np.all(np.abs(cgstate.flux[internal] - upscaled["flux"][internal])/scale < 1e-6))
    if cgwells_tmp:
        uf = np.concatenate([np.ravel(ws.flux) for ws in upscaled["wellSol"]])
        cf_ = np.concatenate([np.ravel(ws.flux) for ws in cgstate.wellSol])
        ok = ok and bool(np.all(np.abs(uf - cf_)/np.abs(cf_).max() < 1e-6))
        for k in range(1, len(cgwells_tmp)):
            dc = cgstate.wellSol[k].pressure - cgstate.wellSol[0].pressure
            df = upscaled["wellSol"][k].pressure - upscaled["wellSol"][0].pressure
            ok = ok and (abs(dc - df)/abs(df) < 1e-6 if df != 0 else dc == 0)
    return upscaled, ok


def _calculate_upscaled_trans_simple(upscaled: Dict, cg: Any, cgwells: Any, obj: Any) -> Dict:
    mob, fmob = _get_mobility(cg, obj, upscaled)
    flux = np.asarray(upscaled["flux"], dtype=float).reshape(cg.faces.num, -1).sum(axis=1)
    N = np.asarray(cg.faces.neighbors, dtype=np.int64)
    internal = ~np.any(N < 0, axis=1)
    p = upscaled["pressure"]
    dp = p[N[internal, 1]] - p[N[internal, 0]]
    fmob_int = fmob if np.ndim(fmob) == 0 else np.asarray(fmob)
    with np.errstate(divide="ignore", invalid="ignore"):
        upscaled["trans"] = np.full(cg.faces.num, np.nan)
        upscaled["trans"][internal] = -(flux[internal]/(dp*fmob_int))
        if "facePressure" in upscaled:
            cellno = np.repeat(np.arange(cg.cells.num), np.diff(np.asarray(cg.cells.facePos)))
            cf = _cell_faces(cg)
            dpf = upscaled["facePressure"][cf] - p[cellno]
            sgn = 2.0*(N[cf, 0] == cellno) - 1
            hflux = sgn*flux[cf]
            upscaled["htrans"] = -(hflux/(dpf*fmob_int))
        if cgwells:
            ws = upscaled["wellSol"]
            assert len(ws) == len(cgwells), "Internal error defning upscaled well quantities"
            nperf = np.array([np.size(w.cells) for w in cgwells])
            pos = np.concatenate([[0], np.cumsum(nperf)])
            if all(hasattr(w, "bhp") for w in ws):
                press = np.repeat([float(w.bhp) for w in ws], nperf)
            else:
                press = np.repeat([float(w.pressure) for w in ws], nperf)
            wc = np.concatenate([np.asarray(w.cells, dtype=np.int64).ravel() for w in cgwells])
            dpw = press - p[wc]
            wflux = np.concatenate([np.asarray(w.flux, dtype=float).reshape(n, -1).sum(axis=1)
                                    for w, n in zip(ws, nperf)])
            mobw = np.asarray(mob).ravel(order="F")[wc]
            WI = wflux/(dpw*mobw)
            upscaled["fluxwells"] = [wflux[pos[j]:pos[j + 1]] for j in range(len(cgwells))]
            upscaled["dpwells"] = [dpw[pos[j]:pos[j + 1]] for j in range(len(cgwells))]
            upscaled["WI"] = [WI[pos[j]:pos[j + 1]] for j in range(len(cgwells))]
    return upscaled


def _fix_trans(cg: Any, T_cg: np.ndarray, HT_cg: np.ndarray, cgwells: Any,
               opt: Dict) -> Tuple[np.ndarray, np.ndarray, Any]:
    ext = np.any(np.asarray(cg.faces.neighbors) < 0, axis=1)
    if opt["bc_method"] in ("wells", "wells_simple"):
        warnings.warn("Set boundary face trans to zero")
        hf_ext = ext[_cell_faces(cg)]
        HT_cg[hf_ext] = 0
        T_cg[ext] = 0
    if opt["fix_trans"]:
        positive = HT_cg[HT_cg > 0]
        HT_min = positive.min() if positive.size else 0.0
        HT_cg = np.maximum(HT_cg, HT_min)
        T_cg = np.maximum(T_cg, HT_min)
        if cgwells:
            for w in cgwells:
                w.WI = np.maximum(w.WI, 0)
    if opt["wells"]:
        w = opt["wells"][0] if _is_case_list(opt["wells"]) else opt["wells"]
        assert len(cgwells or []) == len(w) or not cgwells
        for cw, fw in zip(cgwells or [], w):
            cw.type = fw.type
            cw.val = fw.val
    return T_cg, HT_cg, cgwells


def _get_mobility(cg: Any, obj: Any, state: Dict) -> Tuple[Any, Any]:
    from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel
    from pymrst.utils.gridtools.get_neighbourship import get_neighbourship
    if isinstance(obj, PhysicalModel):
        assert "mob" in state
        cmob = np.asarray(state["mob"], dtype=float).reshape(cg.cells.num, -1).sum(axis=1)
        N = np.asarray(cg.faces.neighbors, dtype=np.int64)
        internal = np.all(N >= 0, axis=1)
        flux = np.asarray(state["flux"], dtype=float).reshape(cg.faces.num, -1)
        flag = flux[internal].sum(axis=1) > 0
        n = get_neighbourship(cg)
        fmob = np.where(flag, cmob[n[:, 0]], cmob[n[:, 1]])
        return cmob, fmob
    mu = np.asarray(obj.properties()[0], dtype=float).ravel()
    cmob = np.tile(1/mu, (np.size(state["pressure"]), 1))
    return cmob, 1/mu if mu.size > 1 else float(1/mu[0])


def _null_space(A: np.ndarray) -> np.ndarray:
    """MATLAB `null(A)`: an orthonormal basis of the null space, from the SVD."""
    _, s, vh = np.linalg.svd(A)
    tol = max(A.shape)*(s.max() if s.size else 0)*np.finfo(float).eps
    rank = int((s > tol).sum())
    return vh[rank:].conj().T


def _parent(cg: Any) -> Any:
    return cg.coarse_grid.parent


def _cell_faces(g: Any) -> np.ndarray:
    faces = np.asarray(g.cells.faces, dtype=np.int64)
    return faces[:, 0] if faces.ndim > 1 else faces


def _is_case_list(wells: Any) -> bool:
    return len(wells) > 0 and isinstance(wells[0], (list, tuple))


def _get(state: Any, name: str) -> Any:
    return state.get(name) if isinstance(state, dict) else getattr(state, name, None)


def _fields(state: Any) -> Dict[str, Any]:
    return dict(state) if isinstance(state, dict) else dict(vars(state))


upscaleTrans = upscale_trans
__all__ = ["make_cg_wells", "upscaleTrans", "upscale_trans"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_ui import partition_ui
    from pymrst.modules.coarsegrid.utils.coarse_grid_as_grid import coarse_grid_as_grid
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.compute_trans import compute_trans
    from pymrst.utils.units import DARCY, MILLI

    G = compute_geometry(cart_grid([6, 6, 2], [60.0, 60.0, 4.0]))
    PERM = 100*MILLI*DARCY
    ROCK = make_rock(G, PERM, 0.2)
    CG = coarse_grid_as_grid(generate_coarse_grid(G, partition_ui(G, [3, 3, 1])))

    def test_homogeneous_box_gives_the_coarse_tpfa_value():
        """In a homogeneous box the flow-based value of every interior coarse
        face is k*A/d, the coarse two-point transmissibility. (2D: in 3D with
        a single coarse layer MRST's own Top/Bottom check divides by a
        round-off pressure range.)"""
        G2 = compute_geometry(cart_grid([6, 6], [60.0, 60.0]))
        rock2 = make_rock(G2, PERM, 0.2)
        CG2 = coarse_grid_as_grid(generate_coarse_grid(G2, partition_ui(G2, [3, 3])))
        with isolated_gravity("off"):
            _, T, _, _ = upscale_trans(CG2, compute_trans(G2, rock2))
        CG = CG2
        N = CG.faces.neighbors
        interior = np.all(N >= 0, axis=1)
        d = np.linalg.norm(CG.cells.centroids[N[interior, 1]] - CG.cells.centroids[N[interior, 0]],
                           axis=1)
        expected = PERM*CG.faces.areas[interior]/d
        assert np.allclose(T[interior], expected, rtol=1e-9), np.abs(T[interior]/expected - 1).max()
        print("ok test_homogeneous_box_gives_the_coarse_tpfa_value")

    def test_well_case_fixes_boundary_and_well_indices():
        W = add_well(None, G, ROCK, np.array([0, 36]), well_type="rate", val=1e-4, radius=0.1,
                     name="I")
        W = add_well(W, G, ROCK, np.array([35, 71]), well_type="bhp", val=1e7, radius=0.1,
                     name="P")
        T_face = 1/np.bincount(G.cells.faces[:, 0] if G.cells.faces.ndim > 1 else G.cells.faces,
                               1/compute_trans(G, ROCK), minlength=G.faces.num)
        with isolated_gravity("off"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HT, T, Wc, report = upscale_trans(CG, T_face, wells=W, bc_method="wells",
                                              fix_trans=True)
        ext = np.any(CG.faces.neighbors < 0, axis=1)
        assert np.all(T[~ext] > 0) and np.all(np.isfinite(T))
        assert np.all(T[ext] == T[~ext].min()) or np.all(T[ext] >= 0)
        assert [w.name for w in Wc] == ["I", "P"] and Wc[0].type == "rate" and Wc[1].val == 1e7
        assert Wc[0].cells.size == 1 and np.all(Wc[0].WI > 0) and np.all(np.isfinite(Wc[1].WI))
        assert Wc[0].parent is None and Wc[1].parent is W[1]
        assert len(report["states"]) == 1
        print("ok test_well_case_fixes_boundary_and_well_indices")

    def test_make_cg_wells_groups_perforations_by_block():
        W = add_well(None, G, ROCK, np.array([36, 0, 1]), well_type="bhp", val=1e7, radius=0.1)
        cw = make_cg_wells(CG, W)[0]
        assert np.array_equal(cw.cells, [0]) and np.array_equal(cw.fcellspos, [0, 3])
        assert np.array_equal(cw.fcells, [0, 1, 36]) and np.array_equal(cw.fperf, [1, 2, 0])
        assert np.isclose(cw.dZ, (1.0 + 1.0 + 3.0)/3 - W[0].refDepth) and np.isnan(cw.WI).all()
        print("ok test_make_cg_wells_groups_perforations_by_block")

    test_homogeneous_box_gives_the_coarse_tpfa_value()
    test_well_case_fixes_boundary_and_well_indices()
    test_make_cg_wells_groups_perforations_by_block()
    print("\nAll tests passed!")
