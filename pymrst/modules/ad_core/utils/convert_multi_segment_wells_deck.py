"""Convert parsed COMPSEGS/WELSEGS data into multisegment wells."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _set(value, name, item):
    if isinstance(value, dict):
        value[name] = item
    else:
        setattr(value, name, item)


def _named_entries(entries):
    if isinstance(entries, dict):
        return entries
    return {str(row[0]): row[1] for row in entries}


def _rows(value, name):
    rows = _field(value, name)
    if rows is None:
        raise ValueError(f"WELSEGS entry is missing {name}")
    return [list(row) for row in rows]


def _logical_indices(grid):
    dims = tuple(int(item) for item in grid.cartDims)
    cart = np.asarray(grid.cells.indexMap, dtype=np.int64)
    return np.column_stack(np.unravel_index(cart, dims, order="F")) + 1


def _compseg_cells(grid, rows):
    logical = _logical_indices(grid)
    lookup = {tuple(index): cell for cell, index in enumerate(logical)}
    return np.asarray(
        [lookup.get(tuple(int(item) for item in row[:3]), -1) for row in rows],
        dtype=np.int64,
    )


def _take(values, indices):
    array = np.asarray(values)
    return array[indices].copy()


def _convert(well, grid, compseg_rows, welsegs, compact):
    header = list(_field(welsegs, "header"))
    if len(header) < 4:
        raise ValueError("WELSEGS header must contain four entries")
    top, top_tube, top_volume = map(float, header[:3])
    segment_rows = _rows(welsegs, "segments")

    segment_number = np.concatenate([
        np.arange(int(row[0]), int(row[1]) + 1) for row in segment_rows
    ])
    if segment_number.size and np.any(np.diff(segment_number) != 1):
        raise ValueError("WELSEGS segment numbers must be consecutive")
    branch_number = np.asarray([int(row[2]) for row in segment_rows])
    number_of_branches = int(branch_number.max(initial=0))
    if set(branch_number.tolist()) != set(range(1, number_of_branches + 1)):
        raise ValueError("WELSEGS branch numbers must be consecutive")

    branches = []
    for branch_id in range(1, number_of_branches + 1):
        selected = [
            row for row, branch in zip(segment_rows, branch_number)
            if branch == branch_id
        ]
        parent = int(selected[0][3])
        if parent == 1:
            current_distance, current_depth = top_tube, top
        else:
            parent_branch = next(
                branch for branch in branches if parent in branch.index
            )
            position = int(np.flatnonzero(parent_branch.index == parent)[0])
            current_distance = parent_branch.tube_depths[position]
            current_depth = parent_branch.z_depths[position]

        index = []
        volumes = []
        distances = []
        depths = []
        lengths = []
        diameters = []
        roughness = []
        for row in selected:
            start, stop = int(row[0]), int(row[1])
            delta_distance, delta_depth = float(row[4]), float(row[5])
            count = stop - start + 1
            numbers = np.arange(start, stop + 1)
            local_distances = current_distance + delta_distance * np.arange(
                1, count + 1
            )
            local_depths = current_depth + delta_depth * np.arange(1, count + 1)
            index.extend(numbers)
            distances.extend(local_distances)
            depths.extend(local_depths)
            lengths.extend([delta_distance] * count)
            diameters.extend([float(row[6])] * count)
            roughness.extend([float(row[7])] * count)
            volumes.extend([float(row[8])] * count)
            current_distance = local_distances[-1]
            current_depth = local_depths[-1]
        branches.append(SimpleNamespace(
            index=np.asarray(index, dtype=np.int64),
            volumes=np.asarray(volumes),
            parent=parent,
            tube_depths=np.asarray(distances),
            roughness=np.asarray(roughness),
            diameters=np.asarray(diameters),
            z_depths=np.asarray(depths),
            lengths=np.asarray(lengths),
        ))

    cells = _compseg_cells(grid, compseg_rows)
    active = cells >= 0
    cells = cells[active]
    compseg_rows = [row for row, keep in zip(compseg_rows, active) if keep]
    keep = []
    seen = set()
    for index, cell in enumerate(cells):
        if int(cell) not in seen:
            seen.add(int(cell))
            keep.append(index)
    cells = cells[keep]
    compseg_rows = [compseg_rows[index] for index in keep]

    mapping = []
    for branch_id, branch in enumerate(branches, start=1):
        for cell, row in zip(cells, compseg_rows):
            if int(row[3]) != branch_id:
                continue
            closest = np.argmin(np.abs(float(row[4]) - branch.tube_depths))
            mapping.append((int(cell), int(branch.index[closest])))
    mapping = np.asarray(mapping, dtype=np.int64).reshape(-1, 2)

    old_cells = np.asarray(_field(well, "cells"), dtype=np.int64).ravel()
    original = np.asarray(
        [np.flatnonzero(old_cells == cell)[0] for cell in mapping[:, 0]],
        dtype=np.int64,
    )
    for name in ("WI", "dZ", "r"):
        _set(well, name, _take(_field(well, name), original))
    _set(well, "perf_map", original)

    if compact:
        topology = []
        edge_length = []
        edge_roughness = []
        edge_diameter = []
        number_of_nodes = max(
            [1] + [int(branch.index.max()) for branch in branches]
        )
        volumes = np.zeros(number_of_nodes)
        depths = np.zeros(number_of_nodes)
        branch_ids = np.zeros(number_of_nodes, dtype=np.int64)
        volumes[0], depths[0] = top_volume, top
        for branch_id, branch in enumerate(branches, start=1):
            nodes = branch.index
            edges = np.column_stack([
                np.concatenate([[branch.parent], nodes[:-1]]),
                nodes,
            ])
            topology.extend(edges.tolist())
            edge_length.extend(branch.lengths)
            edge_roughness.extend(branch.roughness)
            edge_diameter.extend(branch.diameters)
            zero_nodes = nodes - 1
            volumes[zero_nodes] = branch.volumes
            depths[zero_nodes] = branch.z_depths
            branch_ids[zero_nodes] = branch_id

        _set(well, "topo", np.asarray(topology, dtype=np.int64) - 1)
        _set(well, "nodes", SimpleNamespace(
            depth=depths, vol=volumes, branch_id=branch_ids
        ))
        _set(well, "segments", SimpleNamespace(
            roughness=np.asarray(edge_roughness),
            diameter=np.asarray(edge_diameter),
            length=np.asarray(edge_length),
        ))
        _set(well, "cells", mapping[:, 0])
    else:
        _set(well, "branches", branches)
    zero_mapping = mapping.copy()
    zero_mapping[:, 1] -= 1
    _set(well, "cells_to_nodes", zero_mapping)
    return well


def convert_multi_segment_wells_deck(
    wells, grid, deck_control, *, compact=True
):
    """Convert wells described by parsed ``COMPSEGS`` and ``WELSEGS`` data."""
    output = deepcopy(list(wells))
    compsegs = _named_entries(_field(deck_control, "COMPSEGS"))
    welsegs = _named_entries(_field(deck_control, "WELSEGS"))
    for well in output:
        for name, empty in (
            ("isMS", False), ("topo", None), ("nodes", None),
            ("segments", None), ("perf_map", None), ("cells_to_nodes", None),
        ):
            _set(well, name, empty)

    by_name = {str(_field(well, "name")): well for well in output}
    for name, compseg_rows in compsegs.items():
        if name not in welsegs or name not in by_name:
            raise ValueError(f"incomplete multisegment specification for {name}")
        well = _convert(
            by_name[name], grid, [list(row) for row in compseg_rows],
            welsegs[name], compact,
        )
        _set(well, "isMS", True)
    return output


convertMultiSegmentWellsDeck = convert_multi_segment_wells_deck


__all__ = [
    "convert_multi_segment_wells_deck",
    "convertMultiSegmentWellsDeck",
]
