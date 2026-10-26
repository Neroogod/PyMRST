"""Well insertion, ported from MRST `addWell`."""

from typing import Dict, List, Optional, Union
import warnings

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock
from pymrst.utils.compute_well_index import cell_dims, compute_well_index
from pymrst.params.wells_and_bc.types import Well

Scalar = Union[float, np.ndarray]


def add_well(
    wells: Optional[List[Well]],
    g: Grid,
    rock: Rock,
    cells: np.ndarray,
    well_type: str = "bhp",
    val: float = 0.0,
    radius: Scalar = 0.1,
    well_dir: str = "z",
    name: Optional[str] = None,
    skin: Scalar = 0.0,
    kh: Optional[np.ndarray] = None,
    wi: Optional[np.ndarray] = None,
    compi: Optional[np.ndarray] = None,
    ref_depth: Optional[float] = None,
    sign: int = 0,
    gravity: Optional[np.ndarray] = None,
    lims: Optional[Dict[str, float]] = None,
    inner_product: str = "ip_tpf",
    calc_repr_rad: bool = True,
    cell_dims_override: Optional[np.ndarray] = None,
    line_segments: Optional[np.ndarray] = None,
    status: bool = True,
    cstatus: Optional[np.ndarray] = None,
    vfp_index: Optional[int] = None,
    cp=None,
    components: Optional[np.ndarray] = None,
    heel=None,
) -> List[Well]:
    """Append a well perforating `cells`, computing WI where not supplied."""
    wells = list(wells) if wells is not None else []
    raw_cells = np.asarray(cells)
    if raw_cells.dtype == bool:
        if raw_cells.ndim != 1 or raw_cells.size != g.cells.num:
            raise ValueError(
                "logical completion mask must have one entry per grid cell"
            )
        cells = np.flatnonzero(raw_cells)
    else:
        cells = np.asarray(raw_cells, dtype=np.int64).ravel()
    if cells.size == 0:
        warnings.warn(
            f"Empty cell selection in well {name or len(wells) + 1!r}; well ignored",
            RuntimeWarning,
            stacklevel=2,
        )
        return wells
    if cells.min() < 0 or cells.max() >= g.cells.num:
        raise IndexError("well completion cell index out of range")
    # MRST's addWell takes the control type as given: processWells passes
    # 'default' for a well whose deck control it does not support (Norne).
    well_type = well_type.lower()

    inner_product = inner_product.lower()
    if inner_product not in (
        "ip_tpf", "ip_quasitpf", "ip_rt", "ip_simple", "ip_quasirt"
    ):
        raise ValueError(f"unknown inner product {inner_product!r}")
    radius_arr = _broadcast(radius, cells.size, "radius")
    skin_arr = _broadcast(skin, cells.size, "skin")
    kh_arr = None if kh is None else _broadcast(kh, cells.size, "Kh")
    dims_arr = _cell_dims_override(
        cell_dims_override, cells, g.cells.num
    )
    wi_arr = _well_index(
        g, rock, radius_arr, cells, well_dir, skin_arr, kh_arr, wi,
        dims_arr, line_segments, inner_product,
    )
    compi_arr = np.array([1.0, 0.0, 0.0]) if compi is None else np.asarray(compi, float).ravel()
    down = _depth_direction(g, gravity)
    depth = g.cells.centroids[cells] @ down
    ref_depth = _reference_depth(g, down, gravity, ref_depth)
    sign = _well_sign(sign, well_type, val)
    name = f"W{len(wells) + 1}" if name is None else name
    directions = _directions(well_dir, cells.size)
    completion_status = _completion_status(cstatus, cells.size)
    representative_radius = (
        _representative_radius(g, cells, radius_arr, directions)
        if calc_repr_rad else np.empty(0, dtype=float)
    )

    wells.append(
        Well(
            cells=cells,
            type=well_type,
            val=float(val),
            r=radius_arr,
            dir=directions,
            WI=wi_arr,
            dZ=depth - ref_depth,
            name=name,
            compi=compi_arr,
            refDepth=float(ref_depth),
            rR=representative_radius,
            heel=heel,
            cp=cp,
            sign=sign,
            status=bool(status),
            cstatus=completion_status,
            components=(
                None if components is None
                else np.asarray(components, dtype=float).ravel()
            ),
            vfp_index=vfp_index,
            lims=dict(lims) if lims else None,
        )
    )
    return wells


def _broadcast(value: Scalar, count: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    if arr.size != count:
        raise ValueError(
            f"{name} must contain one value or one value per completion"
        )
    return arr


def _well_index(
    g: Grid,
    rock: Rock,
    radius: np.ndarray,
    cells: np.ndarray,
    well_dir: str,
    skin: Scalar,
    kh: Optional[np.ndarray],
    wi: Optional[np.ndarray],
    dims: Optional[np.ndarray],
    line_segments: Optional[np.ndarray],
    inner_product: str,
) -> np.ndarray:
    wi_arr = (
        np.full(cells.size, -1.0) if wi is None
        else _broadcast(wi, cells.size, "WI").copy()
    )
    compute = wi_arr < 0.0
    if not np.any(compute):
        return wi_arr
    segments = None if line_segments is None else np.asarray(
        line_segments, dtype=float
    )
    if segments is not None and segments.shape != (cells.size, 3):
        raise ValueError(
            f"line_segments must have shape ({cells.size}, 3)"
        )
    if segments is None:
        directions = _directions(well_dir, cells.size)
        # computeWellIndex accepts one direction per call. Group completion
        # directions so mixed-direction wells retain supplied WI entries.
        for direction in "xyz":
            selected = compute & np.asarray([d == direction for d in directions])
            if not np.any(selected):
                continue
            wi_arr[selected] = compute_well_index(
                g, rock, radius[selected], cells[selected],
                well_dir=direction, skin=skin[selected],
                kh=None if kh is None else kh[selected],
                cell_dims_override=None if dims is None else dims[selected],
                inner_product=inner_product,
            )
    else:
        used_dims = cell_dims(g, cells) if dims is None else dims
        projected = np.zeros(np.count_nonzero(compute), dtype=float)
        for axis, direction in enumerate("xyz"):
            directional = compute_well_index(
                g, rock, radius[compute], cells[compute],
                well_dir=direction, skin=skin[compute],
                kh=None if kh is None else kh[compute],
                cell_dims_override=used_dims[compute],
                inner_product=inner_product,
            )
            projected += (
                segments[compute, axis]/used_dims[compute, axis]*directional
            )**2
        wi_arr[compute] = np.sqrt(projected)
    return wi_arr


def _cell_dims_override(value, cells, grid_cell_count):
    if value is None:
        return None
    dims = np.asarray(value, dtype=float)
    if dims.shape == (grid_cell_count, 3):
        return dims[cells]
    if dims.shape == (cells.size, 3):
        return dims
    raise ValueError(
        "cell_dims_override must have one row per grid cell or completion"
    )


def _directions(value, count: int) -> str:
    raw = str(value).lower()
    if len(raw) == 1:
        raw *= count
    if len(raw) != count or any(direction not in "xyz" for direction in raw):
        raise ValueError("well_dir must contain x/y/z per completion")
    return raw


def _completion_status(value, count):
    if value is None:
        return np.ones(count, dtype=bool)
    status = np.asarray(value, dtype=bool).ravel()
    if status.size != count:
        raise ValueError("cstatus must contain one flag per completion")
    return status


def _representative_radius(g, cells, radius, directions):
    # MRST's radiusRep always derives geometry from the grid.  The optional
    # cellDims input affects WI construction only, not this polymer radius.
    dimensions = cell_dims(g, cells)
    cross_section = np.empty(cells.size, dtype=float)
    for index, direction in enumerate(directions):
        axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[direction]
        cross_section[index] = dimensions[index, axes[0]]*dimensions[index, axes[1]]
    equivalent = np.sqrt(cross_section/np.pi)
    return np.sqrt(equivalent*radius)


def _depth_direction(g: Grid, gravity: Optional[np.ndarray]) -> np.ndarray:
    dims = g.griddim
    if gravity is not None and np.linalg.norm(np.ravel(gravity)[:dims]) > 0:
        return np.ravel(gravity)[:dims] / np.linalg.norm(np.ravel(gravity)[:dims])
    down = np.zeros(dims)
    if dims > 2:
        down[-1] = 1.0
    return down


def _reference_depth(
    g: Grid,
    down: np.ndarray,
    gravity: Optional[np.ndarray],
    ref_depth: Optional[float],
) -> float:
    if ref_depth is not None:
        return float(ref_depth)
    if gravity is not None and np.linalg.norm(np.ravel(gravity)[: g.griddim]) > 0:
        nodes = getattr(getattr(g, "nodes", None), "coords", None)
        if nodes is not None and np.size(nodes):
            return float((np.asarray(nodes) @ down).min())
        # A connection-list grid - MRST's `simGridTPFA`, which a field model
        # built from ECLIPSE output runs on - has no nodes at all: its faces
        # *are* the connections. The shallowest cell centre is the best datum
        # available, and it is what the top of the model means there.
        centroids = getattr(getattr(g, "cells", None), "centroids", None)
        if centroids is not None and np.size(centroids):
            return float((np.asarray(centroids) @ down).min())
        raise ValueError(
            "the grid carries neither node coordinates nor cell centroids, "
            "so a well reference depth cannot be derived; pass ref_depth")
    return 0.0


def _well_sign(sign: int, well_type: str, val: float) -> int:
    if sign not in (-1, 0, 1):
        raise ValueError("well sign must be -1, 0, or 1")
    if well_type == "rate" and val != 0.0:
        return int(np.sign(val))
    return int(sign)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc.incomp_well_pressure_drop import (
        incomp_well_pressure_drop,
    )
    from pymrst.utils.units import DARCY, MILLI

    def test_add_well_and_pressure_drop() -> None:
        g = compute_geometry(cart_grid([5, 5, 3], [500.0, 500.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        cells = np.array([12, 37, 62])
        wells = add_well(
            None,
            g,
            rock,
            cells,
            well_type="rate",
            val=1e-2,
            gravity=np.array([0.0, 0.0, 9.80665]),
        )
        assert len(wells) == 1
        assert wells[0].sign == 1
        assert np.allclose(wells[0].dZ, [5.0, 15.0, 25.0])
        mob = np.ones((g.cells.num, 1))
        dp = incomp_well_pressure_drop(wells[0], mob, np.array([1000.0]))
        assert np.allclose(dp, 9.80665 * 1000.0 * np.array([5.0, 15.0, 25.0]))
        print("ok test_add_well_and_pressure_drop")

    test_add_well_and_pressure_drop()
    print("\nAll tests passed!")
