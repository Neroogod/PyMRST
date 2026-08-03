"""Insert matrix/fracture completions (`addWellShale`)."""

from typing import Any

import numpy as np

from pymrst.params.wells_and_bc.types import Well
from pymrst.modules.shale.data import ShaleGlobalGrid
from pymrst.modules.shale.other_mrst_fns.compute_wi_shale import compute_wi_shale
from pymrst.utils.compute_well_index import cell_dims, compute_well_index


def add_well_shale(
    wells: list[Well] | None,
    global_grid: ShaleGlobalGrid,
    completion_cells,
    *,
    fracture_aperture=None,
    representative_radius=None,
    fracture_cell_side_lengths=None,
    radius=0.1,
    skin=0.0,
    wi=None,
    well_type: str = "bhp",
    val: float = 0.0,
    compi=(1.0, 0.0, 0.0),
    sign: int = 0,
    name: str | None = None,
    ref_depth: float = 0.0,
    hydraulic_plane_count: int | None = None,
    lims: dict[str, float] | None = None,
    components=None,
    status: bool = True,
    metadata: dict[str, Any] | None = None,
    direction: str = "z",
    inner_product: str = "ip_tpf",
    kh=None,
    cell_dims_override=None,
    calculate_representative_radius: bool = True,
    vfp_index: int = 0,
    c=None,
) -> list[Well]:
    """Append a shale well completed in global fracture cells.

    `completion_cells` is an ``(n, 2)`` table of zero-based
    ``[global_fracture_cell, fracture_plane]`` rows, matching MRST's EDFM
    calling convention after index conversion.
    """
    wells = list(wells or [])

    def infer_planes(cell_ids):
        inferred = np.full(np.asarray(cell_ids).size, -1, dtype=np.int64)
        is_fracture = np.asarray(cell_ids) >= global_grid.n_matrix
        if np.any(is_fracture):
            inferred[is_fracture] = np.asarray(
                global_grid.fracture_plane, dtype=np.int64
            )[np.asarray(cell_ids)[is_fracture] - global_grid.n_matrix]
        return inferred

    raw = np.asarray(completion_cells)
    if raw.dtype == bool:
        if raw.size != global_grid.n_cells:
            raise ValueError("logical completion mask must have one value per global cell")
        cells = np.flatnonzero(raw.reshape(-1)).astype(np.int64)
        planes = infer_planes(cells)
    else:
        table = np.asarray(completion_cells, dtype=np.int64)
        if table.ndim == 2 and table.shape[1] == 2:
            cells, planes = table[:, 0], table[:, 1]
        else:
            cells = table.reshape(-1)
            planes = infer_planes(cells)
    if cells.size == 0:
        raise ValueError("at least one completion is required")
    if np.any(cells < 0) or np.any(cells >= global_grid.n_cells):
        raise IndexError("shale completion cell is out of range")
    fracture_completion = cells >= global_grid.n_matrix
    if np.any(planes[fracture_completion] < 0):
        raise IndexError("fracture plane indices must be non-negative")
    count = cells.size
    if (
        calculate_representative_radius
        and np.any(fracture_completion)
        and representative_radius is None
        and fracture_cell_side_lengths is None
    ):
        raise ValueError(
            "fracture_cell_side_lengths is required to calculate rR when "
            "representative_radius is not supplied"
        )

    def expand(value, name_):
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 1:
            array = np.full(count, array.item())
        if array.size != count:
            raise ValueError(f"{name_} must be scalar or per completion")
        return array

    radius_array = expand(radius, "radius")
    skin_array = expand(skin, "skin")
    aperture = np.zeros(count)
    if np.any(fracture_completion):
        if fracture_aperture is None:
            raise ValueError("fracture_aperture is required for fracture completions")
        aperture_by_plane = np.asarray(fracture_aperture, dtype=float).reshape(-1)
        if aperture_by_plane.size == 1:
            aperture[fracture_completion] = aperture_by_plane.item()
        else:
            if planes[fracture_completion].max() >= aperture_by_plane.size:
                raise IndexError("fracture plane has no aperture entry")
            aperture[fracture_completion] = aperture_by_plane[planes[fracture_completion]]
    active = (
        np.ones(count, dtype=bool)
        if hydraulic_plane_count is None
        else (~fracture_completion) | (planes < int(hydraulic_plane_count))
    )
    supplied_wi = None if wi is None else expand(wi, "wi")
    needs_wi = np.ones(count, dtype=bool) if supplied_wi is None else supplied_wi < 0
    wi_array = np.empty(count) if supplied_wi is None else supplied_wi.copy()
    if np.any(needs_wi):
        if np.any(~fracture_completion):
            dirs = direction.lower()
            if len(dirs) == 1:
                dirs *= count
            if len(dirs) != count or any(item not in "xyz" for item in dirs):
                raise ValueError("direction must contain x/y/z per completion")
            dims = None if cell_dims_override is None else np.asarray(cell_dims_override, float)
            for axis in "xyz":
                selected = needs_wi & (~fracture_completion) & np.array(
                    [item == axis for item in dirs]
                )
                if np.any(selected):
                    selected_dims = None
                    if dims is not None:
                        if dims.ndim != 2 or dims.shape[1] != 3:
                            raise ValueError("cell_dims_override must have shape (n, 3)")
                        if dims.shape[0] == count:
                            selected_dims = dims[selected]
                        elif dims.shape[0] == global_grid.n_matrix:
                            selected_dims = dims[cells[selected]]
                        else:
                            raise ValueError(
                                "cell_dims_override must have one row per completion "
                                "or matrix cell"
                            )
                    wi_array[selected] = compute_well_index(
                        global_grid.matrix, global_grid.rock, radius_array[selected],
                        cells[selected], well_dir=axis, skin=skin_array[selected],
                        kh=None if kh is None else expand(kh, "kh")[selected],
                        cell_dims_override=selected_dims,
                        inner_product=inner_product,
                    )
        if np.any(needs_wi & fracture_completion):
            selected = needs_wi & fracture_completion
            rr = (None if representative_radius is None else
                  expand(representative_radius, "representative_radius")[selected])
            sides = None if fracture_cell_side_lengths is None else np.asarray(
                fracture_cell_side_lengths, dtype=float)
            if sides is not None and sides.shape[0] == count:
                sides = sides[selected]
            wi_array[selected] = compute_wi_shale(
                global_grid.rock.perm[cells[selected], 0], aperture[selected],
                radius_array[selected], representative_radius=rr,
                cell_side_lengths=sides, skin=skin_array[selected],
                hydraulic_completion=active[selected],
            )
    representative_well_radius = None
    if calculate_representative_radius:
        re = np.empty(count)
        if np.any(~fracture_completion):
            dimensions = cell_dims(global_grid.matrix, cells[~fracture_completion])
            dirs = direction.lower() * count if len(direction) == 1 else direction.lower()
            for local, completion in enumerate(np.flatnonzero(~fracture_completion)):
                axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[dirs[completion]]
                re[completion] = np.sqrt(
                    dimensions[local, axes[0]]*dimensions[local, axes[1]]/np.pi)
        if np.any(fracture_completion):
            selected = fracture_completion
            if representative_radius is not None:
                re[selected] = expand(representative_radius, "representative_radius")[selected]
            else:
                if fracture_cell_side_lengths is None:
                    raise ValueError(
                        "fracture_cell_side_lengths is required to calculate rR "
                        "when representative_radius is not supplied"
                    )
                sides = np.asarray(fracture_cell_side_lengths, dtype=float)
                if sides.ndim == 1:
                    sides = sides.reshape(1, 2)
                if sides.ndim != 2 or sides.shape[1] != 2:
                    raise ValueError("fracture_cell_side_lengths must have shape (n, 2)")
                if sides.shape[0] == 1:
                    sides = np.repeat(sides, np.count_nonzero(selected), axis=0)
                elif sides.shape[0] == count:
                    sides = sides[selected]
                elif sides.shape[0] != np.count_nonzero(selected):
                    raise ValueError(
                        "fracture_cell_side_lengths must have one row, one row per "
                        "completion, or one row per fracture completion"
                    )
                re[selected] = 0.14*np.sqrt(np.sum(sides**2, axis=1))
        representative_well_radius = np.sqrt(re*radius_array)

    depth = np.empty(count)
    depth[fracture_completion] = global_grid.fracture_centroids[
        cells[fracture_completion]-global_grid.n_matrix, -1]
    depth[~fracture_completion] = global_grid.matrix.cells.centroids[
        cells[~fracture_completion], -1]
    if sign not in (-1, 0, 1):
        raise ValueError("sign must be -1, 0 or 1")
    if well_type.lower() == "rate" and val != 0:
        sign = int(np.sign(val))
    well = Well(
        cells=cells,
        type=well_type.lower(),
        val=float(val),
        r=radius_array,
        dir=direction * count if len(direction) == 1 else direction,
        WI=wi_array,
        dZ=depth - float(ref_depth),
        name=name or f"W{len(wells) + 1}",
        compi=np.asarray(compi, dtype=float).reshape(-1),
        refDepth=float(ref_depth),
        sign=sign,
        status=bool(status),
        cstatus=np.ones(count, dtype=bool),
        components=(
            None if components is None else np.asarray(components, dtype=float).reshape(-1)
        ),
        lims=None if lims is None else dict(lims),
        rR=representative_well_radius,
        cp={"fracture_plane": planes, **(metadata or {})},
        vfp_index=int(vfp_index),
        c=c,
    )
    wells.append(well)
    return wells


__all__ = ["add_well_shale"]
