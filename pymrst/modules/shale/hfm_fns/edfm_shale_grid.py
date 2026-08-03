"""Shale EDFM grid preprocessing (`EDFMshalegrid`)."""

from collections.abc import Sequence
import copy

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.hfm import (
    PlanarFracture,
    build_edfm_3d,
    project_planar_fracture_to_grid,
)
from pymrst.modules.shale.data import FracturePlane, ShaleGlobalGrid, ShaleRock
from pymrst.modules.shale.hfm_fns.assemble_shale_global_grid import assemble_shale_global_grid
from pymrst.modules.shale.hfm_fns.fracture_matrix_shale_nncs_3d import fracture_matrix_shale_nncs_3d


def _as_planar_fracture(plane: FracturePlane, tolerance: float) -> PlanarFracture:
    points = np.asarray(plane.points, dtype=float)
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= tolerance or singular[1] <= tolerance:
        raise ValueError("fracture polygon is degenerate")
    if singular.size > 2 and singular[2] > tolerance * singular[0]:
        raise ValueError("fracture points must be planar")
    normal = vectors[-1]
    axis1 = points[1] - points[0]
    axis1 = axis1 - normal * float(axis1 @ normal)
    if np.linalg.norm(axis1) <= tolerance:
        axis1 = vectors[0]
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = np.cross(normal, axis1)
    local = (points - center) @ np.column_stack([axis1, axis2])
    half1 = 0.5 * np.ptp(local[:, 0])
    half2 = 0.5 * np.ptp(local[:, 1])
    return PlanarFracture(
        center, normal, axis1, float(half1), float(half2),
        polygon_points=points.copy(),
    )


def edfm_shale_grid(
    grid: Grid | ShaleGlobalGrid,
    fracture_planes: Sequence[FracturePlane],
    matrix_rock: ShaleRock,
    *,
    tolerance: float = 1e-5,
    fracture_list=None,
) -> tuple[ShaleGlobalGrid, list[FracturePlane]]:
    """Preprocess rectangular shale fractures and assemble global indexing.

    Processed status and the zero-based fracture-grid index are retained in
    each plane's metadata. Repeated calls skip already processed planes, as the
    MRST owner does. New selections deterministically rebuild the fracture
    portion in stable source order; an unchanged selection is idempotent.
    """
    previous_global = grid if isinstance(grid, ShaleGlobalGrid) else None
    matrix_grid = grid.matrix if previous_global is not None else grid
    if matrix_grid.griddim != 3:
        raise ValueError("EDFM shale preprocessing requires a 3D grid")
    planes = [copy.deepcopy(plane) for plane in fracture_planes]
    selected = range(len(planes)) if fracture_list is None else fracture_list
    selected = [int(index) for index in selected]
    if any(index < 0 or index >= len(planes) for index in selected):
        raise IndexError("fracture_list contains an invalid fracture index")
    active = [index for index in selected if not planes[index].metadata.get("processed", False)]
    if not active:
        if previous_global is not None:
            return previous_global, planes
        raise ValueError("fracture_list contains no unprocessed fractures")

    # A deterministic rebuild preserves global indices for existing planes and
    # appends newly selected planes in source order. This provides incremental
    # public behavior without mutating or splicing Grid topology in place.
    build_indices = [
        index for index, plane in enumerate(planes)
        if plane.metadata.get("processed", False) or index in active
    ]
    fractures = [_as_planar_fracture(planes[index], tolerance) for index in build_indices]
    apertures = np.array([planes[index].aperture for index in build_indices], dtype=object)
    permeabilities = np.array([planes[index].permeability for index in build_indices], dtype=object)
    porosities = np.array([planes[index].porosity for index in build_indices], dtype=object)
    if any(value is None for value in apertures):
        raise ValueError("every fracture plane requires aperture")
    if any(value is None for value in permeabilities):
        raise ValueError("every fracture plane requires permeability")
    if any(value is None for value in porosities):
        raise ValueError("every fracture plane requires porosity")
    aperture = apertures.astype(float)
    projections = tuple(
        project_planar_fracture_to_grid(
            matrix_grid, fracture, aperture=float(aperture[index])
        )
        for index, fracture in enumerate(fractures)
    )
    matrix_perm = np.asarray(matrix_rock.perm, dtype=float)
    matrix_count = matrix_grid.cells.num
    if matrix_perm.shape[0] != matrix_count:
        matrix_perm = matrix_perm[:matrix_count]
    matrix_perm = matrix_perm if matrix_perm.ndim == 1 else matrix_perm[:, 0]
    matrix_poro = np.asarray(matrix_rock.poro, dtype=float).reshape(-1)[:matrix_count]
    edfm = build_edfm_3d(
        matrix_grid, fractures, aperture=aperture,
        perm_matrix=matrix_perm,
        perm_frac=permeabilities.astype(float),
        poro_matrix=matrix_poro,
        poro_frac=porosities.astype(float),
    )
    global_grid = assemble_shale_global_grid(
        edfm, projections=projections, tolerance=tolerance
    )
    global_grid.nnc = fracture_matrix_shale_nncs_3d(edfm, fractures, projections)
    global_grid.rock.shale_mechanisms = dict(matrix_rock.shale_mechanisms)
    global_grid.edfm = edfm
    global_grid.fractures = tuple(fractures)
    global_grid.projections = projections
    for fracgrid, index in enumerate(build_indices):
        planes[index].metadata["processed"] = True
        planes[index].metadata["fracgrid"] = fracgrid
        planes[index].metadata["normal"] = fractures[fracgrid].frame()[0]
    return global_grid, planes


EDFMshalegrid = edfm_shale_grid

__all__ = ["EDFMshalegrid", "edfm_shale_grid"]
