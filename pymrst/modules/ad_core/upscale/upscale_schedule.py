"""MRST ``ad-core/upscale/upscaleSchedule.m``: a schedule moved onto the
coarse grid of an upscaled model.

Each well's perforations become the coarse blocks they lie in, in
first-appearance order; its well indices are recomputed on the coarse grid
(`computeWellIndex`, the default `'recompute'`) or combined from the fine
ones (`'sum'`, `'harmonic'`, `'mean'`), its depths taken from the coarse
cells, and `fperf`/`parentIndices` map the fine perforations to them.
Sources are summed per block, boundary conditions per coarse face. Indices
are 0-based.

`upscale_schedule_ad`/`upscale_well_ad`, below, are PyMRST's earlier
version, kept for their callers. Its notes follow.

Coarsen a schedule's wells onto a coarse grid (MRST `upscaleSchedule`).

A schedule points its wells at fine cells; on the coarse grid those perforations
have to be re-expressed in coarse blocks. Several fine perforations can collapse
into one coarse block, so the well's cell list becomes the *unique* coarse blocks
it touches, and the collapsed connections' well indices are combined. Parallel
flow paths into the same block add, so the default here sums the well index
(`sum`) - the conductance-preserving choice; `harmonic` and `mean` are available
too. MRST's default is `recompute` (a fresh `computeWellIndex` on the coarse
grid); that needs a coarse well-index routine for an arbitrary perforation
direction, so this port defaults to `sum`, which is well-defined from the fine
well index alone, and notes the difference.

Boundary-condition and source coarsening (the `handleBC`/`handleSRC` branches)
attach here when a consuming case needs them; the well path is what schedules in
this port use.
"""

import copy
from typing import Any, List, Tuple

import numpy as np

_METHODS = ("sum", "harmonic", "mean")


def upscale_schedule(model: Any, schedule: Any, *, wellUpscaleMethod: str = "recompute",
                     bcUpscaleMethod: str = "linear") -> Any:
    """MRST `schedule = upscaleSchedule(model, schedule, ...)`: each control's
    wells, boundary conditions and sources moved onto `model`'s coarse grid."""
    schedule = copy.deepcopy(schedule)
    for control in schedule.controls:
        W = getattr(control, "W", None)
        if W is not None:
            control.W = [_handle_well(model, w, wellUpscaleMethod) for w in W]
        bc = getattr(control, "bc", None)
        if bc is not None:
            control.bc = _handle_bc(model, bc, bcUpscaleMethod)
        src = getattr(control, "src", None)
        if src is not None:
            control.src = _handle_src(model, src)
    return schedule


def _handle_well(model: Any, W: Any, method: str) -> Any:
    from pymrst.utils.compute_well_index import compute_well_index
    from pymrst.utils.unique_stable import unique_stable
    p = np.asarray(model.G.partition, dtype=np.int64)
    Wc = copy.deepcopy(W)
    pc = p[np.asarray(W.cells, dtype=np.int64).ravel()]
    cells, first, new_map = unique_stable(pc)
    Wc.cells = cells
    nc = cells.size
    counts = np.bincount(new_map, minlength=nc).astype(float)
    dirs = np.asarray(list(W.dir) if isinstance(W.dir, str) and len(W.dir) > 1 else
                      np.atleast_1d(W.dir), dtype=object)
    dirs = np.resize(dirs, np.size(W.cells))
    Wc.dir = dirs[first] if dirs.size > 1 else dirs
    r = np.atleast_1d(W.r)
    if r.size > 1:
        Wc.r = r[first]
    if method.lower() == "recompute":
        rr = np.atleast_1d(Wc.r)
        dc = np.resize(np.atleast_1d(Wc.dir), nc)
        Wc.WI = np.zeros(nc)
        for d in np.unique(dc):
            sel = dc == d
            Wc.WI[sel] = compute_well_index(model.G, model.rock,
                                            rr[sel] if rr.size > 1 else rr[0],
                                            cells[sel], well_dir=str(d))
    else:
        WI = np.asarray(W.WI, dtype=float).ravel()
        if method.lower() == "sum":
            Wc.WI = np.bincount(new_map, WI, minlength=nc)
        elif method.lower() == "harmonic":
            Wc.WI = 1/(np.bincount(new_map, 1/WI, minlength=nc)/counts)
        elif method.lower() == "mean":
            Wc.WI = np.bincount(new_map, WI, minlength=nc)/counts
        else:
            raise ValueError(f'Unknown upscale mode: "{method}"')
    if model.G.griddim > 2:
        z = np.asarray(model.G.cells.centroids)[cells, 2]
    else:
        z = np.zeros(nc)
    Wc.dZ = z - W.refDepth
    Wc.cstatus = np.ones(nc, dtype=bool)
    Wc.fperf = new_map
    topo = getattr(W, "topo", None)
    if topo is not None:
        mp = np.concatenate([[-1], new_map])
        newtopo = np.sort(mp[np.asarray(topo) + 1], axis=1)
        newtopo = unique_stable(newtopo, rows=True)[0]
        newtopo = newtopo[newtopo[:, 0] != newtopo[:, 1]]
        Wc.topo = newtopo[np.lexsort(newtopo.T[::-1])]
    Wc.parentIndices = first
    return Wc


def _handle_bc(model: Any, bc: Any, method: str) -> Any:
    from pymrst.params.wells_and_bc import add_bc
    CG = model.G
    G = CG.coarse_grid.parent
    cg = CG.coarse_grid
    nfc = cg.num_faces
    conn_coarse = np.repeat(np.arange(nfc), np.diff(cg.conn_pos))
    faces_bc = np.asarray(bc.face, dtype=np.int64).ravel()
    is_face_bc = np.zeros(G.faces.num, dtype=bool)
    is_face_bc[faces_bc] = True
    coarse_face_no = np.full(G.faces.num, -1)
    coarse_face_no[cg.fconn] = conn_coarse
    coarse_faces_bc = np.unique(conn_coarse[is_face_bc[cg.fconn]])
    types = np.asarray(bc.type, dtype=object).ravel()
    values_all = np.asarray(bc.value, dtype=float).ravel()
    sat_all = getattr(bc, "sat", None)
    out = None
    for cf in coarse_faces_bc:
        act = coarse_face_no[faces_bc] == cf
        faces = faces_bc[act]
        areas = np.asarray(G.faces.areas)[faces]
        kinds = types[act]
        kind = str(kinds[0])
        assert all(str(k).lower() == kind.lower() for k in kinds), (
            "Mixture of boundary condition types on the same coarse face. Not possible to upscale.")
        values = values_all[act]
        sat = None
        if kind.lower() == "flux":
            if sat_all is not None and np.size(sat_all):
                sat = (np.asarray(sat_all)[act]*values[:, None]).sum(axis=0)
                sat = sat/sat.sum()
            val = values.sum()
        elif kind.lower() == "pressure":
            if sat_all is not None and np.size(sat_all):
                sat = (np.asarray(sat_all)[act]*areas[:, None]).sum(axis=0)
                sat = sat/sat.sum()
            C = (np.asarray(G.nodes.coords) if getattr(G, "nodes", None) is not None
                 else np.asarray(G.faces.centroids))
            lo, hi = C.min(axis=0), C.max(axis=0)
            xq = (np.asarray(CG.faces.centroids)[cf] - lo)/(hi - lo)
            x = (np.asarray(G.faces.centroids)[faces] - lo)/(hi - lo)
            val = values if faces.size == 1 else _interpolate_bc(x, values, areas, xq, method, G.griddim)
            val = float(np.ravel(val)[0])
        else:
            raise ValueError(f'Unable to upscale boundary condition type "{kind}"')
        extra = {} if sat is None else {"sat": sat}
        out = add_bc(out, np.array([cf]), kind, val, **extra)
    return out


def _interpolate_bc(x, values, areas, xq, method, griddim):
    from pymrst.utils.matlab_interp1 import matlab_interp1
    m = method.lower()
    if m == "idw":
        d = np.sqrt(((x - xq)**2).sum(axis=1))
        if np.any(d == 0):
            return values[d == 0][0]
        w = 1/d**2
        val = (w*values).sum()/w.sum()
    elif m == "mean":
        val = np.mean(values*areas)/areas.sum()
    elif m == "nearest":
        val = _nearest(x, values, xq)
    else:
        keep = np.ones(griddim, dtype=bool)
        for j in range(griddim):
            tol = 2*max(np.spacing(np.abs(x[:, j])).max(), np.spacing(abs(xq[j])))
            if np.all(np.abs(x[:, j] - xq[j]) <= tol):
                keep[j] = False
        if keep.sum() > 1:
            raise NotImplementedError(
                "upscaleSchedule: MRST interpolates a pressure boundary condition over a "
                "multi-dimensional coarse face with scatteredInterpolant and linear "
                "extrapolation, which is not available here; use bcUpscaleMethod 'idw', "
                "'mean' or 'nearest'.")
        if m != "linear":
            raise NotImplementedError(f"bcUpscaleMethod '{method}' is not available")
        order = np.argsort(x[:, keep].ravel())
        val = matlab_interp1(x[:, keep].ravel()[order], values[order], xq[keep])
    val = np.atleast_1d(val)
    if val.size == 0 or not np.all(np.isfinite(val)):
        val = _nearest(x, values, xq)
    return val


def _handle_src(model: Any, src: Any) -> Any:
    from pymrst.params.wells_and_bc import add_source
    cells = np.asarray(src.cell, dtype=np.int64).ravel()
    cpart = np.asarray(model.G.partition)[cells]
    rates = np.asarray(src.rate, dtype=float).ravel()
    sat_all = getattr(src, "sat", None)
    out = None
    for cc in np.unique(cpart):
        act = cpart == cc
        values = rates[act]
        assert np.all(values < 0) or np.all(values >= 0), (
            "Producing and injecting source terms in same coarse block. Unable to upscale.")
        val = values.sum()
        sat = None
        if sat_all is not None and np.size(sat_all):
            sat = (np.asarray(sat_all)[act]*values[:, None]).sum(axis=0)/values.sum()
        out = add_source(out, np.array([cc]), val, sat=sat)
    return out


def _nearest(x: np.ndarray, values: np.ndarray, xq: np.ndarray) -> float:
    return values[int(np.argmin(np.sqrt(((x - xq)**2).sum(axis=1))))]


upscaleSchedule = upscale_schedule


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

def _unique_stable(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unique values in first-appearance order, plus a map from each input to
    its position in that unique list (MRST `uniqueStable`)."""
    _, first_idx = np.unique(a, return_index=True)
    order = np.sort(first_idx)
    uniq = a[order]
    pos = {v: i for i, v in enumerate(uniq)}
    new_map = np.array([pos[v] for v in a], dtype=np.int64)
    return uniq, new_map


def upscale_well_ad(coarse_model: Any, well: Any, method: str = "sum") -> Any:
    """Remap a single well onto the coarse grid, combining its well indices."""
    if method not in _METHODS:
        raise ValueError(f"unknown wellUpscaleMethod {method!r}; "
                         f"use one of {_METHODS}")
    partition = np.asarray(coarse_model.partition, dtype=np.int64)
    pc = partition[np.asarray(well.cells, dtype=np.int64)]
    coarse_cells, new_map = _unique_stable(pc)
    nc = coarse_cells.size
    counts = np.bincount(new_map, minlength=nc)

    wi = np.asarray(well.WI, dtype=float).ravel()
    if method == "sum":
        wi_c = np.bincount(new_map, weights=wi, minlength=nc)
    elif method == "mean":
        wi_c = np.bincount(new_map, weights=wi, minlength=nc) / counts
    else:  # harmonic
        wi_c = 1.0 / (np.bincount(new_map, weights=1.0 / wi, minlength=nc) / counts)

    wc = copy.copy(well)
    wc.cells = coarse_cells
    wc.WI = wi_c
    wc.cstatus = np.ones(nc, dtype=bool)
    # first-encountered perforation direction/radius, as MRST does
    _, first_ind = np.unique(pc, return_index=True)
    first_ind = np.sort(first_ind)
    if np.ndim(well.dir) and np.size(well.dir) > 1:
        wc.dir = np.asarray(well.dir)[first_ind]
    if np.ndim(well.r) and np.size(well.r) > 1:
        wc.r = np.asarray(well.r)[first_ind]
    z = coarse_model.G.cells.centroids[coarse_cells, -1] \
        if coarse_model.G.cells.centroids.shape[1] > 2 else np.zeros(nc)
    wc.dZ = z - well.refDepth
    wc.fperf = new_map
    return wc


def upscale_schedule_ad(coarse_model: Any, schedule: Any, method: str = "sum") -> Any:
    """Return a copy of `schedule` with every control's wells coarsened."""
    coarse = copy.copy(schedule)
    controls = []
    for ctrl in schedule.controls:
        c = copy.copy(ctrl)
        if getattr(ctrl, "W", None):
            c.W = [upscale_well_ad(coarse_model, w, method) for w in ctrl.W]
        controls.append(c)
    coarse.controls = controls
    return coarse


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.

if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa_ad as upscale_model_tpfa
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.upscaling.coarsegrid.partition_cart_grid import (
        partition_cart_grid,
    )
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _coarse():
        g = compute_geometry(cart_grid([4, 4, 1], [400.0, 400.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 3.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
        model = GenericBlackOilModelAD(g, rock, fluid, [])
        part = partition_cart_grid([4, 4, 1], [2, 2, 1])       # 2x2 blocks
        return model, upscale_model_tpfa(model, part), g, rock

    def test_perforations_collapse_to_unique_coarse_blocks():
        """A well perforating cells 0,1,4,5 - all in coarse block 0 of a 4x4 ->
        2x2 partition - collapses to a single coarse perforation."""
        model, cm, g, rock = _coarse()
        w = add_well(None, g, rock, np.array([0, 1, 4, 5]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wc = upscale_well_ad(cm, w)
        assert wc.cells.size == 1, wc.cells
        assert wc.cells[0] == cm.partition[0]
        print("ok test_perforations_collapse_to_unique_coarse_blocks")

    def test_well_index_sums_over_collapsed_perforations():
        model, cm, g, rock = _coarse()
        w = add_well(None, g, rock, np.array([0, 1, 4, 5]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wc = upscale_well_ad(cm, w, method="sum")
        assert np.isclose(wc.WI[0], np.sum(w.WI)), (wc.WI, w.WI)
        # cstatus and dZ have the coarse length
        assert wc.cstatus.size == 1 and wc.dZ.size == 1
        print("ok test_well_index_sums_over_collapsed_perforations")

    def test_two_blocks_stay_separate():
        """A well crossing two coarse blocks keeps two coarse perforations."""
        model, cm, g, rock = _coarse()
        # cell 0 -> block 0, cell 2 -> block 1 (columns 0-1 vs 2-3)
        w = add_well(None, g, rock, np.array([0, 2]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wc = upscale_well_ad(cm, w)
        assert wc.cells.size == 2
        assert set(wc.cells.tolist()) == {cm.partition[0], cm.partition[2]}
        print("ok test_two_blocks_stay_separate")

    def test_harmonic_and_mean_methods():
        model, cm, g, rock = _coarse()
        w = add_well(None, g, rock, np.array([0, 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), name="P")[0]
        wi = np.asarray(w.WI, float)
        wm = upscale_well_ad(cm, w, method="mean")
        wh = upscale_well_ad(cm, w, method="harmonic")
        assert np.isclose(wm.WI[0], wi.mean())
        assert np.isclose(wh.WI[0], 1.0 / np.mean(1.0 / wi))
        print("ok test_harmonic_and_mean_methods")

    def _faithful_coarse():
        from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
        from pymrst.modules.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa as upscale_mrst
        g = compute_geometry(cart_grid([4, 4, 2], [400.0, 400.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 3e-3], rho=[1000.0, 800.0], n=[2, 2])
        model = GenericBlackOilModel(g, rock, fluid, gas=False)
        return model, upscale_mrst(model, partition_cart_grid([4, 4, 2], [2, 2, 1])), g, rock

    def test_faithful_well_maps_perforations_to_blocks():
        """Perforations 0, 16 (block 0, both layers) and 2 (block 1): two coarse
        perforations in first-appearance order, `fperf` and `parentIndices`
        mapping back, depth from the coarse cells."""
        from pymrst.modules.ad_core.utils.schedule import simple_schedule
        model, cm, g, rock = _faithful_coarse()
        w = add_well(None, g, rock, np.array([0, 16, 2]), well_type="bhp", val=150 * BARSA,
                     compi=np.array([0.0, 1.0]), name="P", ref_depth=0.0)
        sched = upscale_schedule(cm, simple_schedule(np.ones(2), W=w))
        wc = sched.controls[0].W[0]
        assert np.array_equal(wc.cells, [cm.G.partition[0], cm.G.partition[2]])
        assert np.array_equal(wc.fperf, [0, 0, 1]) and np.array_equal(wc.parentIndices, [0, 2])
        assert np.allclose(wc.dZ, cm.G.cells.centroids[wc.cells, 2]) and wc.cstatus.all()
        from pymrst.utils.compute_well_index import compute_well_index
        expected = compute_well_index(cm.G, cm.rock, 0.1, wc.cells, well_dir="z")
        assert np.allclose(wc.WI, expected, rtol=1e-12)
        assert len(w) == 1 and w[0].cells.size == 3, "the fine schedule was modified"
        print("ok test_faithful_well_maps_perforations_to_blocks")

    def test_faithful_sum_method_and_sources():
        from pymrst.modules.ad_core.utils.schedule import simple_schedule
        from pymrst.params.wells_and_bc import add_source
        model, cm, g, rock = _faithful_coarse()
        w = add_well(None, g, rock, np.array([0, 1, 16]), well_type="bhp", val=150 * BARSA,
                     compi=np.array([0.0, 1.0]), name="P")
        src = add_source(None, np.array([0, 1, 2]), np.array([1.0, 2.0, 4.0]),
                         sat=np.array([1.0, 0.0]))
        sched = simple_schedule(np.ones(1), W=w)
        sched.controls[0].src = src
        out = upscale_schedule(cm, sched, wellUpscaleMethod="sum").controls[0]
        assert np.isclose(out.W[0].WI[0], np.sum(w[0].WI))
        assert np.array_equal(out.src.cell, [cm.G.partition[0], cm.G.partition[2]])
        assert np.allclose(out.src.rate, [3.0, 4.0])
        print("ok test_faithful_sum_method_and_sources")

    test_faithful_well_maps_perforations_to_blocks()
    test_faithful_sum_method_and_sources()
    test_perforations_collapse_to_unique_coarse_blocks()
    test_well_index_sums_over_collapsed_perforations()
    test_two_blocks_stay_separate()
    test_harmonic_and_mean_methods()
    print("\nAll tests passed!")
