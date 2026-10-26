"""MRST ``ad-core/utils/makeScheduleConsistent.m``.

`make_schedule_consistent` is the port. Every control gets the same wells,
in the same order: new wells are appended sorted by name, as MATLAB's
`setdiff` returns them. Each well is perforated in the union of its cells
over all controls, and a well absent from a control, or shut in it, is
closed there. Then the perforations are reordered (by depth for vertical
wells, along the well path from the heel otherwise), defaulted reference
depths are set to the top perforation, and missing signs are filled in.

`make_schedule_consistent_ad` is PyMRST's earlier version (the well superset
only), kept for its callers.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from pymrst.modules.ad_core.utils.schedule import Control, Schedule


def _control_order(schedule: Schedule) -> list[int]:
    order = []
    for index in schedule.step_control:
        index = int(index)
        if index not in order:
            order.append(index)
    order.extend(i for i in range(len(schedule.controls)) if i not in order)
    return order


def _perf_array(values, count: int, dtype=float):
    arr = np.asarray(values, dtype=dtype).ravel()
    if arr.size == 1 and count > 1:
        arr = np.full(count, arr[0], dtype=dtype)
    return arr


def make_schedule_consistent_ad(schedule: Schedule) -> Schedule:
    """Return a schedule with stable well and completion dimensions."""
    if not isinstance(schedule, Schedule):
        raise TypeError("schedule must be a Schedule")
    if not schedule.controls:
        return deepcopy(schedule)

    order = _control_order(schedule)
    names: list[str] = []
    cells_by_name: dict[str, list[int]] = {}
    first_active = {}
    first_seen = {}
    for ci in order:
        for well in schedule.controls[ci].W or []:
            if well.name not in names:
                names.append(well.name)
                cells_by_name[well.name] = []
                first_seen[well.name] = deepcopy(well)
            for cell in np.asarray(well.cells, dtype=np.int64).ravel():
                if int(cell) not in cells_by_name[well.name]:
                    cells_by_name[well.name].append(int(cell))
            if well.status and well.name not in first_active:
                first_active[well.name] = deepcopy(well)

    closed = {}
    for name in names:
        template = deepcopy(first_active.get(name, first_seen[name]))
        cells = np.asarray(cells_by_name[name], dtype=np.int64)
        template.cells = cells
        template.WI = np.zeros(cells.size)
        template.dZ = np.zeros(cells.size)
        template.r = np.zeros(cells.size)
        template.cstatus = np.zeros(cells.size, dtype=bool)
        template.status = False
        closed[name] = template

    controls = []
    for ctrl in schedule.controls:
        present = {well.name: well for well in ctrl.W or []}
        wells = []
        for name in names:
            source = present.get(name)
            if source is None or not source.status:
                wells.append(deepcopy(closed[name]))
                continue
            target = deepcopy(source)
            union_cells = np.asarray(cells_by_name[name], dtype=np.int64)
            source_cells = np.asarray(source.cells, dtype=np.int64).ravel()
            lookup = {int(cell): i for i, cell in enumerate(source_cells)}
            target.cells = union_cells
            for field, dtype in (("WI", float), ("dZ", float), ("r", float)):
                raw = _perf_array(getattr(source, field), source_cells.size, dtype)
                expanded = np.zeros(union_cells.size, dtype=dtype)
                for ui, cell in enumerate(union_cells):
                    if int(cell) in lookup:
                        expanded[ui] = raw[lookup[int(cell)]]
                setattr(target, field, expanded)
            raw_status = (
                np.ones(source_cells.size, dtype=bool)
                if source.cstatus is None
                else _perf_array(source.cstatus, source_cells.size, bool)
            )
            target.cstatus = np.zeros(union_cells.size, dtype=bool)
            for ui, cell in enumerate(union_cells):
                if int(cell) in lookup:
                    target.cstatus[ui] = raw_status[lookup[int(cell)]]
            target.status = bool(target.status and np.any(target.cstatus))
            wells.append(target if target.status else deepcopy(closed[name]))
        controls.append(Control(W=wells, bc=deepcopy(ctrl.bc), src=deepcopy(ctrl.src)))
    return Schedule(
        step_val=schedule.step_val.copy(),
        step_control=schedule.step_control.copy(),
        controls=controls,
    )


PERFORATION_FIELDS = ("WI", "dZ", "dir", "r", "rR")


def make_schedule_consistent(schedule: Schedule, *,
                             perforationFields=PERFORATION_FIELDS,
                             DepthReorder: bool = False, ReorderStrategy=(),
                             setDepths: bool = False, G=None,
                             fixSign: bool = True) -> Schedule:
    """MRST `makeScheduleConsistent`."""
    opt = dict(perforationFields=tuple(perforationFields), DepthReorder=DepthReorder,
               ReorderStrategy=list(ReorderStrategy), setDepths=setDepths, G=G)
    if not schedule.controls:
        return schedule
    schedule = deepcopy(schedule)
    ctrl_order = _get_control_ordering(schedule)
    W_all, changed = _get_well_superset(schedule, ctrl_order, opt)
    schedule = _update_schedule(schedule, ctrl_order, W_all, changed, opt)
    schedule = _reorder_wells_perforations(schedule, opt)
    if W_all:
        W_all = schedule.controls[0].W
    return _set_reference_depths(schedule, W_all, opt)


def _perf(w, field: str) -> np.ndarray:
    """A perforation field as an array (`dir` as an array of characters)."""
    item = getattr(w, field, None)
    if item is None:
        return np.zeros(0)
    if field == "dir":
        return np.array(list(str(item)))
    return np.asarray(item)


def _set_perf(w, field: str, item: np.ndarray) -> None:
    setattr(w, field, "".join(item) if field == "dir" else item)


def _set_reference_depths(schedule, W_all, opt):
    for w_no, w in enumerate(W_all):
        defaulted = getattr(w, "defaulted", None)
        is_defaulted = isinstance(defaulted, dict) and bool(defaulted.get("refDepth", False))
        if is_defaulted or np.isnan(w.refDepth) or opt["setDepths"]:
            assert opt["G"] is not None, "Grid must be provided when refDepth is defaulted"
            z = np.asarray(opt["G"].cells.centroids)[:, 2]
            cells = np.asarray(w.cells, dtype=np.int64)
            ref_depth = z[cells[0]]
            for ctrl in schedule.controls:
                ctrl.W[w_no].refDepth = float(ref_depth)
                ctrl.W[w_no].dZ = z[cells] - ref_depth
    return schedule


def _get_control_ordering(schedule) -> list[int]:
    """Controls in order of first use, then the unused ones."""
    nctrl = len(schedule.controls)
    order: list[int] = []
    for ctrl in schedule.step_control:
        if int(ctrl) not in order:
            order.append(int(ctrl))
        if len(order) >= nctrl:
            break
    last = max(order) + 1 if order else 0
    order += list(range(last, last + nctrl - len(order)))
    return order


def _get_well_superset(schedule, ctrl_order, opt):
    W_all: list = []
    names: list[str] = []
    changed: list[bool] = []
    for i, order in enumerate(ctrl_order, start=1):
        W = schedule.controls[order].W
        if not W:
            continue
        current = [w.name for w in W]
        new_names = sorted(set(current) - set(names))
        subs = [current.index(n) for n in new_names]
        for k in (k for k in range(len(W)) if k not in subs):
            ind_all = names.index(current[k])
            c_all = np.asarray(W_all[ind_all].cells, dtype=np.int64)
            c = np.asarray(W[k].cells, dtype=np.int64)
            if c.size == c_all.size and (np.array_equal(c, c_all)
                                         or np.array_equal(np.sort(c), np.sort(c_all))):
                flag = not np.all(W[k].cstatus)
            else:
                new_cells = c[~np.isin(c, c_all)]
                _, first = np.unique(new_cells, return_index=True)
                new_cells = new_cells[np.sort(first)]
                flag = True
                W_all[ind_all].cells = np.concatenate([c_all, new_cells])
                W_all[ind_all].cell_origin = np.concatenate(
                    [W_all[ind_all].cell_origin, np.full(new_cells.size, i)])
                if getattr(W[k], "perf_map", None) is not None and getattr(W[k], "isMS", False):
                    W_all[ind_all].perf_map = W[k].perf_map
                    W_all[ind_all].cells_to_nodes = W[k].cells_to_nodes
            changed[ind_all] = changed[ind_all] or flag
        if new_names:
            names += new_names
            W_new = [deepcopy(W[s]) for s in subs]
            for w in W_new:
                w.cell_origin = np.full(np.size(w.cells), i)
            _consolidate_well_fields(W_new, W_all)
            _consolidate_well_fields(W_all, W_new)
            W_all += W_new
            changed += [not np.all(w.cstatus) for w in W_new]
    for w in W_all:
        nc = np.size(w.cells)
        for pf in opt["perforationFields"]:
            sample = _perf(w, pf)
            if sample.size == 0:
                continue
            width = sample.shape[1:] if sample.ndim > 1 else ()
            if sample.dtype.kind == "U":
                _set_perf(w, pf, np.full((nc,) + width, " "))
            else:
                _set_perf(w, pf, np.zeros((nc,) + width, dtype=sample.dtype))
        w.cstatus = np.zeros(nc, dtype=bool)
        w.status = False
    return W_all, changed


def _update_schedule(schedule, ctrl_order, W_all, changed, opt):
    perf_fields = opt["perforationFields"]
    W_closed = deepcopy(W_all)
    passed = [False]*len(W_all)
    for ctrl in ctrl_order:
        W = schedule.controls[ctrl].W
        if not W:
            continue
        active = np.zeros(len(W_all), dtype=bool)
        rest = set(vars(W[0])) - set(perf_fields) - {"cells", "cstatus"}
        for j, wa in enumerate(W_all):
            sub = [k for k, w in enumerate(W) if w.name == wa.name]
            # MRST marks the position in the control's list, not `j`.
            active[sub] = True
            if not sub:
                continue
            w = W[sub[0]]
            cells_all = np.asarray(wa.cells, dtype=np.int64)
            w_cells = np.asarray(w.cells, dtype=np.int64)
            if changed[j]:
                keep = np.isin(cells_all, w_cells)
                position = {int(c): k for k, c in enumerate(w_cells)}
                order = np.array([position[int(c)] for c in cells_all[keep]], dtype=np.int64)
                open_cells = w_cells[np.asarray(w.cstatus, dtype=bool)]
                perf_active = np.isin(cells_all, open_cells)
            else:
                keep = np.ones(cells_all.size, dtype=bool)
                perf_active = keep.copy()
                order = np.arange(cells_all.size)
            wa.cstatus = perf_active
            for pf in perf_fields:
                source = _perf(w, pf)
                if np.any(keep) and source.size:
                    target = _perf(wa, pf).copy()
                    target[keep] = source[order]
                    _set_perf(wa, pf, target)
            for fn in rest:
                if fn in ("cells_to_nodes", "perf_map"):
                    continue
                setattr(wa, fn, deepcopy(getattr(w, fn)))
            if not passed[j] and wa.status:
                W_closed[j].val, W_closed[j].type, W_closed[j].sign = w.val, w.type, w.sign
                passed[j] = True
        new_W = deepcopy(W_all)
        for k in np.flatnonzero(~active):
            new_W[k] = deepcopy(W_closed[k])
        schedule.controls[ctrl].W = new_W
    from pymrst.modules.ad_core.utils.set_well_sign import set_well_sign
    for ctrl in schedule.controls:
        W = ctrl.W
        if W:
            W = [w if w.status else deepcopy(W_closed[k]) for k, w in enumerate(W)]
            ctrl.W = set_well_sign(W)
    return schedule


def _reorder_wells_perforations(schedule, opt):
    order = list(opt["ReorderStrategy"])
    for w_no in range(len(schedule.controls[0].W)):
        schedule = _set_uniform_direction(schedule, w_no)
    W = schedule.controls[0].W
    nw = len(W)
    if not order:
        if opt["DepthReorder"]:
            order = ["depth"]
        else:
            # Case-sensitive, as in MRST: deck wells carry 'Z', while
            # `addWell`'s default 'z' takes the heel-based ordering.
            order = ["depth" if np.all(_perf(w, "dir") == "Z")
                     else "direction-ijz-heel" for w in W]
    if len(order) == 1:
        order = order*nw
    assert len(order) == nw
    for w_no, strategy in enumerate(order):
        strategy = strategy.lower()
        if "direction" in strategy:
            schedule = _direction_reorder(schedule, w_no, opt, strategy)
        elif strategy == "origin":
            schedule = _origin_reorder(schedule, w_no, opt)
        elif strategy == "depth":
            schedule = _depth_reorder(schedule, w_no, opt)
        elif strategy == "depth-origin":
            schedule = _depth_origin_reorder(schedule, w_no, opt)
        elif strategy != "none":
            raise ValueError(f"Unknown ordering strategy {strategy}")
    return schedule


def _set_uniform_direction(schedule, well_no: int):
    nc = np.size(schedule.controls[0].W[well_no].cells)
    direction = np.full(nc, " ")
    for ctrl in schedule.controls:
        d = _perf(ctrl.W[well_no], "dir")
        if d.size == 1 and nc > 1:
            d = np.full(nc, d[0])
        ok = d != " "
        direction[ok] = d[ok]
    direction[direction == " "] = "Z"
    for ctrl in schedule.controls:
        _set_perf(ctrl.W[well_no], "dir", direction.copy())
    return schedule


def _reorder_cell_fields(schedule, well_no: int, opt, sort_ix: np.ndarray):
    for ctrl in schedule.controls:
        w = ctrl.W[well_no]
        for f in tuple(opt["perforationFields"]) + ("cells", "cstatus"):
            item = _perf(w, f)
            if item.size:
                _set_perf(w, f, item[sort_ix])
    return schedule


def _depth_origin_reorder(schedule, well_no: int, opt):
    w = schedule.controls[0].W[well_no]
    if opt["G"] is None:
        return schedule
    origin = np.asarray(w.cell_origin)
    z = np.asarray(opt["G"].cells.centroids)[np.asarray(w.cells, dtype=np.int64), 2]
    uorigin = np.unique(origin)
    group_z = np.array([z[origin == o].mean() for o in uorigin])
    uorigin = uorigin[np.argsort(group_z, kind="stable")]
    sort_ix = np.concatenate([np.flatnonzero(origin == o) for o in uorigin])
    return _reorder_cell_fields(schedule, well_no, opt, sort_ix)


def _origin_reorder(schedule, well_no: int, opt):
    origin = np.asarray(schedule.controls[0].W[well_no].cell_origin)
    sort_ix = np.concatenate([np.flatnonzero(origin == o) for o in np.unique(origin)])
    return _reorder_cell_fields(schedule, well_no, opt, sort_ix)


def _depth_reorder(schedule, well_no: int, opt):
    w = schedule.controls[0].W[well_no]
    if opt["G"] is None:
        return schedule
    z = np.asarray(opt["G"].cells.centroids)[np.asarray(w.cells, dtype=np.int64), 2]
    if np.all(np.diff(z) >= 0):
        return schedule
    return _reorder_cell_fields(schedule, well_no, opt, np.argsort(z, kind="stable"))


def _direction_reorder(schedule, well_no: int, opt, strategy: str):
    """Walk the perforations from a start cell, each step to the nearest
    remaining one along the current perforation's direction."""
    from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
    G = opt["G"]
    w = schedule.controls[0].W[well_no]
    options = strategy.split("-")
    cells = np.asarray(w.cells, dtype=np.int64)
    ijk = np.column_stack(grid_logical_indices(G))[cells] + 1       # 1-based
    centroids = np.asarray(G.cells.centroids)[cells]
    direction = options[1] if len(options) > 1 else "xyz"
    if len(options) > 2:
        if options[2] == "first":
            index = None
        elif options[2] == "heel":
            heel = getattr(w, "heel", None)
            assert heel is not None and np.size(heel) > 1, "Heel I, J index must be provided."
            ij_dist = np.abs(ijk[:, :2] - np.asarray(heel, dtype=float)[:2]).sum(axis=1)
            candidates = np.flatnonzero(ij_dist == ij_dist.min())
            index = int(candidates[np.argmin(centroids[candidates, 2])])
        else:
            index = int(options[2]) - 1
    else:
        index = 0
    assert G is not None, "Grid must be provided for directional ordering"
    nc = cells.size
    sort_ix = np.zeros(nc, dtype=np.int64)
    origin = np.asarray(w.cell_origin)
    sort_ix[0] = int(np.flatnonzero(origin == origin.min())[0])
    dirs = np.char.lower(_perf(w, "dir"))
    if dirs.size == 1:
        dirs = np.full(nc, dirs[0])
    convention = "xyz"
    assert np.all(np.isin(dirs, list(convention)))
    pts = np.zeros(centroids.shape)
    unitscale = lambda x: (x - x.min())/max(x.max() - x.min(), 1e-12)
    for d in range(G.griddim):
        source = ijk[:, d] if direction[d].lower() == "ijk"[d] else centroids[:, d]
        pts[:, d] = unitscale(source.astype(float))
    if index is None:
        index = int(np.argmin(pts[:, 2]))
    pts[[0, index]] = pts[[index, 0]]
    current = pts[0].copy()
    pts[0] = np.inf
    for i in range(1, nc):
        coord = np.array([c == dirs[i - 1] for c in convention])
        dist = ((current[coord] - pts[:, coord])**2).sum(axis=1)
        pos = int(np.argmin(dist))
        sort_ix[i] = pos
        current = pts[pos].copy()
        pts[pos] = np.inf
    a, b = sort_ix == index, sort_ix == 0
    sort_ix[a], sort_ix[b] = 0, index
    return _reorder_cell_fields(schedule, well_no, opt, sort_ix)


def _consolidate_well_fields(W_new, W_all) -> None:
    """Give `W_all` the fields `W_new` has and it lacks (`isMS = False`)."""
    if not W_new or not W_all:
        return
    fields_new = set(vars(W_new[0]))
    fields_all = set(vars(W_all[0]))
    if len(fields_all) < len(fields_new):
        for w in W_all:
            w.isMS = False
            for name in fields_new - fields_all:
                setattr(w, name, None)


makeScheduleConsistent = make_schedule_consistent

__all__ = ["make_schedule_consistent", "make_schedule_consistent_ad",
           "makeScheduleConsistent"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc.add_well import add_well

    G = compute_geometry(cart_grid([3, 3, 3], [30.0, 30.0, 30.0]))
    ROCK = make_rock(G, 1e-13, 0.2)

    def _well(name, cells, **kw):
        w = add_well([], G, ROCK, cells, name=name, well_type="bhp", val=1e7,
                     sign=-1, **kw)[0]
        w.dir = "Z"*len(cells)                 # as a deck well (processWells)
        return w

    def test_wells_are_unioned_and_sorted_by_name():
        """`setdiff` returns new names sorted, and each well is perforated in
        the union of its cells. MRST then marks wells active by their position
        in the control's own list (`active(sub)`), so here - control 2 lists
        only PROD - INJ stays open and PROD is the one closed. MATLAB gives
        exactly this (cells [5 14 23], cstatus [0 0 0] for PROD)."""
        c1 = Control(W=[_well("PROD", [4, 13]), _well("INJ", [0])])
        c2 = Control(W=[_well("PROD", [4, 22])])
        out = make_schedule_consistent(Schedule([1.0, 1.0], [0, 1], [c1, c2]), G=G)
        assert [w.name for w in out.controls[0].W] == ["INJ", "PROD"]
        prod1, prod2 = out.controls[0].W[1], out.controls[1].W[1]
        assert list(prod1.cells) == [4, 13, 22] == list(prod2.cells)
        assert prod1.cstatus.tolist() == [True, True, False]
        assert out.controls[1].W[0].status and not prod2.status
        assert not np.any(prod2.cstatus)
        print("ok test_wells_are_unioned_and_sorted_by_name")

    def test_vertical_wells_are_ordered_by_depth():
        out = make_schedule_consistent(
            Schedule([1.0], [0], [Control(W=[_well("P", [22, 4, 13])])]), G=G)
        z = np.asarray(G.cells.centroids)[out.controls[0].W[0].cells, 2]
        assert np.all(np.diff(z) > 0), z
        print("ok test_vertical_wells_are_ordered_by_depth")

    def test_a_defaulted_reference_depth_is_the_top_perforation():
        w = _well("P", [13, 22])
        w.defaulted = {"refDepth": True}
        out = make_schedule_consistent(Schedule([1.0], [0], [Control(W=[w])]), G=G)
        w = out.controls[0].W[0]
        z = np.asarray(G.cells.centroids)[:, 2]
        assert np.isclose(w.refDepth, z[13]) and np.allclose(w.dZ, z[[13, 22]] - z[13])
        print("ok test_a_defaulted_reference_depth_is_the_top_perforation")

    test_wells_are_unioned_and_sorted_by_name()
    test_vertical_wells_are_ordered_by_depth()
    test_a_defaulted_reference_depth_is_the_top_perforation()
    print("\nAll tests passed!")
