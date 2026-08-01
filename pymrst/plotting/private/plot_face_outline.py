"""Face collection outlines (MRST ``plotting/private/plotFaceOutline.m``)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from pymrst.plotting.private.plot_patches import build_patch_geometry
from pymrst.plotting.results import PatchGeometry, PlotResult


def face_outline_geometry(g, faces, groups=None) -> PatchGeometry:
    """Return only edges occurring once in the selected face collection."""
    patches = build_patch_geometry(g, faces)
    if patches.n_entities == 0:
        return PatchGeometry(
            vertices=patches.vertices,
            connectivity=(),
            entity_ids=patches.entity_ids,
            node_ids=patches.node_ids,
            dimension=1,
        )

    group_ids = (
        np.zeros(patches.n_entities, dtype=np.int64)
        if groups is None else np.asarray(groups, dtype=np.int64).ravel()
    )
    if group_ids.size != patches.n_entities:
        raise ValueError("outline groups must match the selected faces")
    edges: list[tuple[int, int, int]] = []
    if g.griddim == 2:
        edges = [
            (int(loop[0]), int(loop[-1]), int(group))
            for loop, group in zip(patches.connectivity, group_ids)
        ]
    else:
        for loop, group in zip(patches.connectivity, group_ids):
            edges.extend(
                (int(loop[i]), int(loop[(i+1) % loop.size]), int(group))
                for i in range(loop.size)
            )
    def key(edge):
        return (*sorted(edge[:2]), edge[2])

    counts = Counter(key(edge) for edge in edges)
    boundary = tuple(
        np.asarray(edge[:2], dtype=np.int64)
        for edge in edges if counts[key(edge)] == 1
    )
    return PatchGeometry(
        vertices=patches.vertices,
        connectivity=boundary,
        entity_ids=patches.entity_ids,
        node_ids=patches.node_ids,
        dimension=1,
        group_ids=group_ids,
    )


def plot_face_outline(
    g,
    faces,
    *,
    plotter=None,
    color="black",
    line_width: float = 1.0,
    groups=None,
    show: bool = False,
    **mesh_kwargs,
) -> PlotResult:
    """Render the boundary edges of a selected face collection."""
    geometry = face_outline_geometry(g, faces, groups=groups)
    if not geometry.connectivity:
        return PlotResult(geometry=geometry, plotter=plotter)
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting face outlines requires PyVista; "
            "install pymrst[plotting]"
        ) from error
    points = geometry.vertices
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    stream: list[int] = []
    for edge in geometry.connectivity:
        stream.extend((2, int(edge[0]), int(edge[1])))
    mesh = pv.PolyData(points, lines=np.asarray(stream, dtype=np.int64))
    own_plotter = plotter is None
    target = plotter or pv.Plotter()
    actor = target.add_mesh(
        mesh, color=color, line_width=line_width, **mesh_kwargs
    )
    if show and own_plotter:
        target.show()
    return PlotResult(geometry=geometry, actor=actor, plotter=target)


plotFaceOutline = plot_face_outline

__all__ = [
    "face_outline_geometry",
    "plot_face_outline",
    "plotFaceOutline",
]
