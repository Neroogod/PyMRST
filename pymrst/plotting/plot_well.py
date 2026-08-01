"""Plot well trajectories (MRST ``plotting/plotWell.m``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.params.wells_and_bc import Well


@dataclass(frozen=True)
class WellGeometry:
    """Backend-neutral geometry for one rendered well."""

    name: str
    cells: np.ndarray
    completion_points: np.ndarray
    trajectory: np.ndarray
    head: np.ndarray
    heel: np.ndarray
    label_position: np.ndarray
    color: Any
    radius: float
    vertical_radius: float
    directions: tuple[str, ...]
    surface_vertices: np.ndarray
    surface_faces: np.ndarray
    head_vertices: np.ndarray
    heel_vertices: np.ndarray
    active: bool


@dataclass
class WellPlotResult:
    """Geometry and separate MRST-equivalent well handle groups."""

    geometry: tuple[WellGeometry, ...]
    plotter: Any = None
    head_actors: tuple[Any, ...] = ()
    text_actors: tuple[Any, ...] = ()
    path_actors: tuple[Any, ...] = ()
    line_actors: tuple[Any, ...] = ()


def _as_wells(wells) -> list[Well]:
    if isinstance(wells, Well):
        return [wells]
    return list(wells)


def _well_cells(g, well) -> np.ndarray:
    cells = np.asarray(well.cells, dtype=np.int64).ravel()
    if cells.size == 0:
        raise ValueError(f"well {well.name!r} has no completion cells")
    if cells.min() < 0 or cells.max() >= g.cells.num:
        raise IndexError(f"well {well.name!r} contains invalid cell indices")
    return cells


def _radii(g, cell: int, scale: float) -> tuple[float, float]:
    faces = np.asarray(
        g.cells.faces[g.cells.facePos[cell]:g.cells.facePos[cell+1]],
        dtype=np.int64,
    )
    areas = np.asarray(g.faces.areas, dtype=float)[faces]
    max_area = float(np.max(areas))
    lateral = float(scale)*0.25*np.sqrt(max_area)
    vertical = (
        float(scale)*0.25*float(np.asarray(g.cells.volumes)[cell])/max_area
    )
    return lateral, vertical


def _directions(value: Any, count: int) -> tuple[str, ...]:
    """Normalize MRST's per-completion ``W.dir`` character sequence."""
    if count == 1:
        return ("s",)
    if isinstance(value, str):
        raw = [c.lower() for c in value if not c.isspace()]
    else:
        raw = [str(c).lower() for c in np.asarray(value).ravel()]
    if not raw:
        raw = ["z"]
    if len(raw) < count:
        raw.extend([raw[-1]]*(count-len(raw)))
    return tuple(raw[:count])


def _quad_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    rows, columns = x.shape
    vertices = np.column_stack([
        x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F"),
    ])
    faces = np.empty(((rows-1)*(columns-1), 4), dtype=np.int64)
    cursor = 0
    for column in range(columns-1):
        for row in range(rows-1):
            faces[cursor] = (
                row + column*rows,
                row + 1 + column*rows,
                row + 1 + (column+1)*rows,
                row + (column+1)*rows,
            )
            cursor += 1
    return vertices, faces


def _well_surface(points, lateral, vertical, directions, cylpts):
    theta = np.linspace(0.0, 2.0*np.pi, int(cylpts)+1)
    if points.shape[0] == 1:
        # MATLAB's ``sphere(n)`` traverses longitude from -pi to pi,
        # unlike ``cylinder``, which starts at zero.
        theta = np.linspace(-np.pi, np.pi, int(cylpts)+1)
        phi = np.linspace(-0.5*np.pi, 0.5*np.pi, int(cylpts)+1)
        cp = np.cos(phi)[:, None]
        x = points[0, 0] + lateral[0]*cp*np.cos(theta)[None, :]
        y = points[0, 1] + lateral[0]*cp*np.sin(theta)[None, :]
        z = points[0, 2] + vertical[0]*np.sin(phi)[:, None]*np.ones(
            (1, theta.size)
        )
    else:
        xrows, yrows, zrows = [], [], []
        for index in range(points.shape[0]-1):
            p0, p1 = points[index:index+2]
            rl = lateral[index]
            rv = vertical[index]
            direction = directions[index]
            if direction == "s":
                continue
            if direction == "x":
                xseg = np.vstack([np.full(theta.size, p0[0]),
                                  np.full(theta.size, p1[0])])
                yseg = np.vstack([p0[1] + rv*np.sin(theta),
                                  p1[1] + rv*np.sin(theta)])
                zseg = np.vstack([p0[2] + rv*np.cos(theta),
                                  p1[2] + rv*np.cos(theta)])
            elif direction == "y":
                xseg = np.vstack([p0[0] + rv*np.cos(theta),
                                  p1[0] + rv*np.cos(theta)])
                yseg = np.vstack([np.full(theta.size, p0[1]),
                                  np.full(theta.size, p1[1])])
                zseg = np.vstack([p0[2] + rv*np.sin(theta),
                                  p1[2] + rv*np.sin(theta)])
            elif direction == "z":
                xseg = np.vstack([p0[0] + rl*np.cos(theta),
                                  p1[0] + rl*np.cos(theta)])
                yseg = np.vstack([p0[1] + rl*np.sin(theta),
                                  p1[1] + rl*np.sin(theta)])
                zseg = np.vstack([np.full(theta.size, p0[2]),
                                  np.full(theta.size, p1[2])])
            else:  # MRST's default branch for a slanted completion.
                xseg = np.vstack([p0[0] + rl*np.cos(theta),
                                  p1[0] + rl*np.cos(theta)])
                yseg = np.vstack([p0[1] + rl*np.sin(theta),
                                  p1[1] + rl*np.sin(theta)])
                zseg = np.vstack([p0[2] - 0.5*rv*np.ones(theta.size),
                                  p1[2] + 0.5*rv*np.ones(theta.size)])
            xrows.extend(xseg)
            yrows.extend(yseg)
            zrows.extend(zseg)
        if not xrows:
            return (np.empty((0, 3)), np.empty((0, 4), dtype=np.int64),
                    np.empty((0, 3)), np.empty((0, 3)))
        x, y, z = np.asarray(xrows), np.asarray(yrows), np.asarray(zrows)
    vertices, faces = _quad_surface(x, y, z)
    head = np.column_stack([x[0], y[0], z[0]])
    heel = np.column_stack([x[-1], y[-1], z[-1]])
    return vertices, faces, head, heel


def _geometry(g, wells, *, radius, height, color, color2, zscale, cylpts):
    ztop = float(np.min(np.asarray(g.nodes.coords)[:, 2]))-float(height)
    count = len(wells)
    width = max(1, len(str(max(count, 1))))
    output = []
    for index, well in enumerate(wells, start=1):
        cells = _well_cells(g, well)
        all_directions = _directions(getattr(well, "dir", "z"), cells.size)
        cstatus = getattr(well, "cstatus", None)
        if cstatus is None:
            opened = np.ones(cells.size, dtype=bool)
        else:
            opened = np.asarray(cstatus, dtype=bool).ravel()
            if opened.size != cells.size:
                raise ValueError(
                    f"well {well.name!r} cstatus must match completion cells"
                )
        cells = cells[opened]
        filtered_directions = tuple(
            np.asarray(all_directions, dtype=object)[opened]
        )
        active = bool(getattr(well, "status", True)) and cells.size > 0
        if cells.size == 0:
            # Retain a valid inspectable location for a fully closed well.
            all_cells = _well_cells(g, well)
            cells = all_cells[:1]
            filtered_directions = (all_directions[0],)
        completions = np.asarray(g.cells.centroids, dtype=float)[cells].copy()
        head = completions[0].copy()
        head[2] = ztop
        trajectory = np.vstack([head, completions])
        chosen = (
            color2 if well.sign < 0 and color2 is not None
            else color if color is not None else "red"
        )
        name = (
            well.name if isinstance(getattr(well, "name", None), str)
            and well.name else f"W{index:0{width}d}"
        )
        scale = float(zscale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("zscale must be finite and positive")
        radii = np.asarray([_radii(g, int(cell), radius) for cell in cells])
        if not np.isclose(scale, 1.0):
            z0 = float(np.min(np.asarray(g.nodes.coords)[:, 2]))
            completions[:, 2] = z0 + scale*(completions[:, 2]-z0)
            head[2] = z0 + scale*(head[2]-z0)
            trajectory = np.vstack([head, completions])
            radii[:, 1] *= scale
        # Repeated consecutive perforations create zero-length cylinders.
        # Keep the filtered completion list inspectable, but collapse repeats
        # in the rendered path and its corresponding properties.
        keep = np.r_[True, np.any(np.diff(completions, axis=0) != 0.0, axis=1)]
        rendered = completions[keep]
        rendered_radii = radii[keep]
        rendered_directions = tuple(
            np.asarray(filtered_directions, dtype=object)[keep]
        )
        if rendered.shape[0] == 1:
            rendered_directions = ("s",)
        trajectory = np.vstack([head, rendered])
        vertices, faces, head_vertices, heel_vertices = _well_surface(
            rendered,
            rendered_radii[:, 0],
            rendered_radii[:, 1],
            rendered_directions,
            cylpts,
        )
        output.append(WellGeometry(
            name=name,
            cells=cells.copy(),
            completion_points=completions,
            trajectory=trajectory,
            head=head.copy(),
            heel=rendered[-1].copy(),
            label_position=head.copy(),
            color=chosen,
            radius=float(rendered_radii[0, 0]),
            vertical_radius=float(rendered_radii[0, 1]),
            directions=rendered_directions,
            surface_vertices=vertices,
            surface_faces=faces,
            head_vertices=head_vertices,
            heel_vertices=heel_vertices,
            active=active,
        ))
    return tuple(output)


def plot_well(
    g,
    wells,
    plotter=None,
    *,
    radius: float = 1.0,
    height: float = 5.0,
    color=None,
    color2=None,
    cylpts: int = 10,
    fontsize: float = 16,
    ambstr: float = 0.8,
    linewidth: float = 2.0,
    zscale: float = 1.0,
    show: bool = False,
) -> WellPlotResult:
    """Plot 3-D wells and return heads, labels, paths, and label lines."""
    if g.griddim != 3:
        raise ValueError("plot_well is implemented only for 3-D grids")
    if cylpts < 3:
        raise ValueError("cylpts must be at least 3")
    collection = _as_wells(wells)
    geometry = _geometry(
        g,
        collection,
        radius=radius,
        height=height,
        color=color,
        color2=color2,
        zscale=zscale,
        cylpts=cylpts,
    )
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "plot_well requires PyVista; install pymrst[plotting]"
        ) from error
    own_plotter = plotter is None
    target = plotter or pv.Plotter()
    heads, texts, paths, lines = [], [], [], []
    for item in geometry:
        if not item.active:
            heads.append(None)
            texts.append(None)
            paths.append(None)
            lines.append(None)
            continue
        face_stream = np.column_stack([
            np.full(item.surface_faces.shape[0], 4, dtype=np.int64),
            item.surface_faces,
        ]).ravel()
        tube = pv.PolyData(item.surface_vertices, faces=face_stream)
        paths.append(target.add_mesh(
            tube,
            color=item.color,
            ambient=float(ambstr),
        ))

        cap_vertices = np.vstack([item.head_vertices, item.heel_vertices])
        nhead = item.head_vertices.shape[0]
        nheel = item.heel_vertices.shape[0]
        cap_faces = np.r_[nhead, np.arange(nhead), nheel,
                          nhead + np.arange(nheel)].astype(np.int64)
        markers = pv.PolyData(cap_vertices, faces=cap_faces)
        heads.append(target.add_mesh(
            markers,
            color=item.color,
            ambient=float(ambstr),
        ))

        line_points = item.trajectory
        line_stream = np.r_[line_points.shape[0],
                            np.arange(line_points.shape[0])].astype(np.int64)
        line_mesh = pv.PolyData(line_points, lines=line_stream)
        lines.append(target.add_mesh(
            line_mesh, color=item.color, line_width=float(linewidth)
        ))

        if fontsize > 0 and hasattr(target, "add_point_labels"):
            texts.append(target.add_point_labels(
                item.label_position[None, :],
                [item.name],
                font_size=int(fontsize),
                text_color=item.color,
                show_points=False,
                always_visible=True,
            ))
        else:
            texts.append(None)
    if show and own_plotter:
        target.show()
    return WellPlotResult(
        geometry=geometry,
        plotter=target,
        head_actors=tuple(heads),
        text_actors=tuple(texts),
        path_actors=tuple(paths),
        line_actors=tuple(lines),
    )


plotWell = plot_well

__all__ = [
    "WellGeometry",
    "WellPlotResult",
    "plot_well",
    "plotWell",
]
