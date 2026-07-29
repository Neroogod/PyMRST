"""Normalize wells and perforations across schedule control periods.

Port of the numerical/data-layout core of MRST ``makeScheduleConsistent.m``.
Every control receives the same wells in the same order and every occurrence
of a well receives the stable union of its perforated cells.
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


def make_schedule_consistent(schedule: Schedule) -> Schedule:
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


makeScheduleConsistent = make_schedule_consistent

__all__ = ["make_schedule_consistent", "makeScheduleConsistent"]
