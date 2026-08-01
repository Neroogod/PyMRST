"""Partially transparent grid isosurfaces (MRST plotGridVolumes.m)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.interpolate_data import interpolate_data


@dataclass(frozen=True)
class VolumeSamples:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    values: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    source_coordinates: np.ndarray
    source_values: np.ndarray


@dataclass
class IsosurfaceBand:
    value: float
    color: Any
    opacity: float
    actor: Any = None
    mesh: Any = None
    face_count: int = 0


@dataclass
class GridVolumeResult:
    samples: VolumeSamples
    isovalues: np.ndarray
    bands: tuple[IsosurfaceBand, ...]
    plotter: Any = None


def _histogram_centers(values, count):
    data = np.asarray(values, dtype=float).ravel()
    if data.size == 0:
        raise ValueError("no values remain after min/max filtering")
    if isinstance(count, bool) or int(count) != count or int(count) <= 0:
        raise ValueError("N must be a positive integer")
    count = int(count)
    low, high = float(data.min()), float(data.max())
    if count == 1:
        return np.asarray([(low+high)/2.0])
    if np.isclose(low, high):
        width = max(abs(low), 1.0)*0.5
        low, high = low-width, high+width
    width = (high-low)/count
    return np.linspace(low+width/2.0, high-width/2.0, count)


def _volume_sources(g, values, extrude_faces):
    coordinates = np.asarray(g.cells.centroids, dtype=float).copy()
    source_values = np.asarray(values, dtype=float).ravel().copy()
    if extrude_faces:
        faces, owners = boundary_faces(g, with_cells=True)
        coordinates = np.vstack([
            coordinates, np.asarray(g.faces.centroids, dtype=float)[faces]
        ])
        source_values = np.concatenate([source_values, source_values[owners]])
    node_coordinates = np.asarray(g.nodes.coords, dtype=float)
    lower = np.maximum(coordinates.min(axis=0), node_coordinates.min(axis=0))
    upper = np.minimum(coordinates.max(axis=0), node_coordinates.max(axis=0))
    return coordinates, source_values, lower, upper


def volume_samples(
    g,
    values,
    *,
    mesh=None,
    extrude_faces: bool = True,
    interpolant=None,
) -> VolumeSamples:
    """Sample cell data on the structured mesh used for isosurfaces."""
    if g.griddim != 3:
        raise ValueError("plot_grid_volumes supports only 3-D grids")
    data = np.asarray(values, dtype=float).ravel()
    if data.size != g.cells.num:
        raise ValueError(
            f"volume data has {data.size} values; expected {g.cells.num}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("volume data must be finite")
    if mesh is None:
        if g.cartDims is None or len(g.cartDims) != 3:
            raise ValueError("mesh is required when grid.cartDims is absent")
        mesh_size = np.asarray(g.cartDims, dtype=np.int64)+(
            2 if extrude_faces else 0
        )
    else:
        raw = np.asarray(mesh, dtype=float).ravel()
        if raw.size != 3 or not np.all(raw == np.floor(raw)) or np.any(raw < 1):
            raise ValueError("mesh must contain three positive integers")
        mesh_size = raw.astype(np.int64)
    sources, source_values, lower, upper = _volume_sources(
        g, data, extrude_faces
    )
    axes = tuple(
        np.linspace(lower[axis], upper[axis], mesh_size[axis]+1)
        for axis in range(3)
    )
    x, y, z = np.meshgrid(*axes, indexing="xy")
    if interpolant is None:
        query = np.column_stack([
            x.ravel(order="F"),
            y.ravel(order="F"),
            z.ravel(order="F"),
        ])
        sampled = interpolate_data(
            g,
            source_values,
            x=sources,
            xq=query,
            interpolation="linear",
            extrapolation="linear",
        ).reshape(x.shape, order="F")
    else:
        sampled = np.asarray(interpolant, dtype=float)
        if sampled.shape != x.shape:
            raise ValueError(
                f"interpolant has shape {sampled.shape}; expected {x.shape}"
            )
        sampled = sampled.copy()
    return VolumeSamples(
        x=x,
        y=y,
        z=z,
        values=sampled,
        axes=axes,
        source_coordinates=sources,
        source_values=source_values,
    )


def plot_grid_volumes(
    g,
    values,
    *,
    N: int = 20,
    minimum=None,
    maximum=None,
    mesh=None,
    cmap: str = "jet",
    basealpha: float = 1.0,
    binc=None,
    patchn: int = 1_000_000,
    interpolant=None,
    extrude_faces: bool = True,
    plotter=None,
    show: bool = True,
    **options,
) -> GridVolumeResult:
    """Sample a 3-D field and render MRST-style transparent isosurfaces."""
    if "min" in options:
        if minimum is not None:
            raise ValueError("specify only one of minimum and min")
        minimum = options.pop("min")
    if "max" in options:
        if maximum is not None:
            raise ValueError("specify only one of maximum and max")
        maximum = options.pop("max")
    if options:
        raise TypeError(
            "unknown plot_grid_volumes options: "
            + ", ".join(sorted(options))
        )
    data = np.asarray(values, dtype=float).ravel()
    filtered = data
    if minimum is not None:
        filtered = filtered[filtered >= float(minimum)]
    if maximum is not None:
        filtered = filtered[filtered <= float(maximum)]
    if binc is None:
        isovalues = _histogram_centers(filtered, N)
    else:
        isovalues = np.asarray(binc, dtype=float).ravel()
        if isovalues.size == 0 or not np.all(np.isfinite(isovalues)):
            raise ValueError("binc must contain finite isovalues")
        N = int(isovalues.size)
    if not np.isfinite(basealpha) or basealpha < 0:
        raise ValueError("basealpha must be finite and nonnegative")
    if int(patchn) <= 0:
        raise ValueError("patchn must be positive")
    samples = volume_samples(
        g,
        data,
        mesh=mesh,
        extrude_faces=extrude_faces,
        interpolant=interpolant,
    )
    try:
        import pyvista as pv
        from matplotlib import colormaps
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "plot_grid_volumes requires PyVista and Matplotlib; "
            "install pymrst[plotting]"
        ) from error
    structured = pv.StructuredGrid(samples.x, samples.y, samples.z)
    structured.point_data["volume_data"] = samples.values.ravel(order="F")
    own_plotter = plotter is None
    target = plotter or pv.Plotter()
    if callable(cmap):
        raw_colors = np.asarray(cmap(N), dtype=float)
        if raw_colors.shape != (N, 3):
            raise ValueError("callable cmap must return an (N, 3) array")
        colors = [tuple(color) for color in raw_colors]
    else:
        mapping = colormaps.get_cmap(cmap)
        colors = [mapping(position)[:3] for position in np.linspace(0, 1, N)]
    budget = max(1, int(np.ceil(int(patchn)/N)))
    bands = []
    for index, isovalue in enumerate(isovalues):
        surface = structured.contour(
            isosurfaces=[float(isovalue)], scalars="volume_data"
        )
        if surface.n_cells > budget and surface.n_cells:
            reduction = max(0.0, 1.0-budget/surface.n_cells)
            surface = surface.decimate_pro(reduction, preserve_topology=True)
        opacity = float(basealpha)*(index+1)/(2*N)
        actor = None
        if surface.n_cells:
            actor = target.add_mesh(
                surface,
                color=colors[index],
                opacity=opacity,
                show_edges=False,
            )
        bands.append(IsosurfaceBand(
            value=float(isovalue),
            color=colors[index],
            opacity=opacity,
            actor=actor,
            mesh=surface,
            face_count=int(surface.n_cells),
        ))
    if show and own_plotter:
        target.show()
    return GridVolumeResult(samples, isovalues, tuple(bands), target)


plotGridVolumes = plot_grid_volumes

__all__ = [
    "VolumeSamples",
    "IsosurfaceBand",
    "GridVolumeResult",
    "volume_samples",
    "plot_grid_volumes",
    "plotGridVolumes",
]
