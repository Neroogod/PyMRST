"""Hydrostatic pressure boundary condition, ported from MRST ``psideh``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, Union

import numpy as np

from pymrst.params.gravity_const import gravity_vector
from pymrst.params.wells_and_bc.add_bc import add_bc
from pymrst.params.wells_and_bc.types import BoundaryCondition
from pymrst.utils.gridtools import boundary_face_indices

Scalar = Union[float, np.ndarray, Sequence[float]]


_SIDE_NORMAL_AXIS = {
    "left": 0,
    "west": 0,
    "xmin": 0,
    "right": 0,
    "east": 0,
    "xmax": 0,
    "back": 1,
    "south": 1,
    "ymin": 1,
    "front": 1,
    "north": 1,
    "ymax": 1,
    "top": 2,
    "upper": 2,
    "zmin": 2,
    "bottom": 2,
    "lower": 2,
    "zmax": 2,
}


def psideh(
    bc: Optional[BoundaryCondition],
    g,
    side: str,
    fluid,
    i1: Optional[Sequence[int]] = None,
    i2: Optional[Sequence[int]] = None,
    *,
    sat: Scalar = 0.0,
    search_range: Optional[Sequence[int]] = None,
    ref_position: Optional[Scalar] = None,
    ref_pressure: float = 0.0,
    gravity: Optional[Scalar] = None,
    **kwargs,
) -> BoundaryCondition:
    """Impose MRST-style hydrostatic pressure on a Cartesian grid side.

    The pressure on each selected boundary face is

    ``ref_pressure + omega * dot(face_centroid - ref_position, gravity)``,

    where ``omega`` is the mobility-weighted density
    ``sum((kr/mu)*rho) / sum(kr/mu)`` evaluated at the uniform boundary
    saturation. Logical ranges use PyMRST's 0-based indexing, matching
    :func:`boundary_face_indices`.
    """
    if "range" in kwargs:
        if search_range is not None:
            raise TypeError("use either search_range or range, not both")
        search_range = kwargs.pop("range")
    if "pn" in kwargs:
        ref_pressure = kwargs.pop("pn")
    if "ref_depth" in kwargs:
        ref_position = _ref_position_from_depth(g, kwargs.pop("ref_depth"), ref_position)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected psideh option(s): {names}")

    if getattr(g, "cartDims", None) is None:
        raise ValueError("psideh is only implemented for logically Cartesian grids")
    if getattr(g.faces, "centroids", None) is None:
        raise ValueError("psideh requires face centroids; run compute_geometry first")

    faces = boundary_face_indices(g, side, i1, i2)
    faces = _restrict_perpendicular_range(g, side, faces, search_range)
    sat_row = _uniform_saturation(sat)
    mu, rho = _fluid_properties(fluid, sat_row)
    kr = _fluid_relperm(fluid, sat_row, mu.size)
    omega = _mobility_weighted_density(mu, rho, kr)

    griddim = int(g.griddim)
    ref = _as_vector(ref_position, griddim, fill=0.0)
    grav = _as_vector(gravity, griddim, fill=None)
    if gravity is None:
        grav = gravity_vector(griddim)

    dx = np.asarray(g.faces.centroids[faces, :griddim], dtype=np.float64) - ref
    pressure = float(ref_pressure) + omega * (dx @ grav)
    return add_bc(bc, faces, "pressure", pressure, sat=sat_row)


def _restrict_perpendicular_range(g, side: str, faces: np.ndarray, search_range) -> np.ndarray:
    if search_range is None:
        return faces
    values = np.asarray(search_range, dtype=np.int64).ravel()
    if values.size == 0:
        return faces

    axis = _side_normal_axis(side)
    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    nx, ny, _ = dims
    cells = _boundary_cells(g, faces)
    gids = cells if g.cells.indexMap is None else np.asarray(g.cells.indexMap)[cells]
    ijk = np.column_stack([gids % nx, (gids // nx) % ny, gids // (nx * ny)])
    keep = np.isin(ijk[:, axis], values)
    restricted = faces[keep]
    if restricted.size == 0:
        raise ValueError("psideh range removed all boundary faces")
    return restricted


def _boundary_cells(g, faces: np.ndarray) -> np.ndarray:
    neigh = np.asarray(g.faces.neighbors[faces], dtype=np.int64)
    cells = np.where(neigh[:, 0] >= 0, neigh[:, 0], neigh[:, 1])
    if np.any(cells < 0):
        raise ValueError("boundary face without an adjacent reservoir cell")
    return cells


def _side_normal_axis(side: str) -> int:
    key = side.strip().lower()
    if key not in _SIDE_NORMAL_AXIS:
        raise ValueError(f"unknown side '{side}'")
    return _SIDE_NORMAL_AXIS[key]


def _uniform_saturation(sat: Scalar) -> np.ndarray:
    arr = np.asarray(sat, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError("psideh only supports uniform boundary saturations")


def _fluid_properties(fluid, sat_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(fluid, "properties"):
        props = fluid.properties({"s": sat_row.reshape(1, -1)})
        if isinstance(props, tuple):
            mu, rho = props[:2]
        elif isinstance(props, Mapping):
            mu = _mapping_value(props, "mu", "viscosity")
            rho = _mapping_value(props, "rho", "density", "rhoS")
        else:
            mu = _attribute_value(props, "mu", "viscosity")
            rho = _attribute_value(props, "rho", "density", "rhoS")
    else:
        mu = _value_from_fluid(fluid, "mu", "viscosity")
        rho = _value_from_fluid(fluid, "rho", "density", "rhoS")
    return _as_1d(mu, "mu"), _as_1d(rho, "rho")


def _fluid_relperm(fluid, sat_row: np.ndarray, nphase: int) -> np.ndarray:
    if hasattr(fluid, "relperm"):
        kr = fluid.relperm(sat_row.reshape(1, -1))
    else:
        kr = np.ones(nphase, dtype=np.float64)
    kr = _as_1d(kr, "kr")
    if kr.size == 1 and nphase > 1:
        kr = np.full(nphase, kr[0], dtype=np.float64)
    return kr


def _mobility_weighted_density(mu: np.ndarray, rho: np.ndarray, kr: np.ndarray) -> float:
    nphase = max(mu.size, rho.size, kr.size)
    mu = _broadcast_phase(mu, nphase, "mu")
    rho = _broadcast_phase(rho, nphase, "rho")
    kr = _broadcast_phase(kr, nphase, "kr")
    if np.any(mu <= 0):
        raise ValueError("fluid viscosities must be positive")
    mob = kr / mu
    total_mob = float(np.sum(mob))
    if not total_mob > 0:
        raise ValueError("total boundary mobility must be positive")
    return float(np.sum(mob * rho) / total_mob)


def _broadcast_phase(values: np.ndarray, nphase: int, name: str) -> np.ndarray:
    if values.size == nphase:
        return values.astype(np.float64, copy=False)
    if values.size == 1:
        return np.full(nphase, values[0], dtype=np.float64)
    raise ValueError(f"{name} must have one value or one value per phase")


def _value_from_fluid(fluid, *names: str):
    if isinstance(fluid, Mapping):
        return _mapping_value(fluid, *names)
    return _attribute_value(fluid, *names)


def _mapping_value(mapping: Mapping, *names: str):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise AttributeError(f"fluid is missing one of: {', '.join(names)}")


def _attribute_value(obj, *names: str):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"fluid is missing one of: {', '.join(names)}")


def _as_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr


def _as_vector(values, size: int, fill) -> np.ndarray:
    if values is None:
        if fill is None:
            raise ValueError("missing vector")
        return np.full(size, fill, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(size, arr[0], dtype=np.float64)
    if arr.size < size:
        raise ValueError(f"vector must have at least {size} entries")
    return arr[:size]


def _ref_position_from_depth(g, depth, current):
    if current is not None:
        raise TypeError("use either ref_depth or ref_position, not both")
    ref = np.zeros(int(g.griddim), dtype=np.float64)
    ref[-1] = float(depth)
    return ref


__all__ = ["psideh"]
