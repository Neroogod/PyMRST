"""Plot named grid faults (MRST ``plotting/plotFaults.m``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.results import PlotResult


@dataclass
class FaultPlotResult:
    """Fault patch, names, label coordinates, and per-face grouping."""

    faces: PlotResult
    names: tuple[str, ...]
    label_positions: np.ndarray
    face_groups: np.ndarray
    text_actor: Any = None


def _field(fault, name):
    if isinstance(fault, Mapping):
        matches = [key for key in fault if str(key).lower() == name.lower()]
        if len(matches) != 1:
            raise ValueError(f"fault mapping must contain one {name!r} field")
        return fault[matches[0]]
    if not hasattr(fault, name):
        raise ValueError(f"fault object must define {name!r}")
    return getattr(fault, name)


def _normalize_faults(g, faults):
    collection = (
        [faults]
        if isinstance(faults, Mapping) or hasattr(faults, "faces")
        else list(faults)
    )
    names, groups, face_blocks, labels = [], [], [], []
    for ordinal, fault in enumerate(collection):
        name = str(_field(fault, "name"))
        faces = np.asarray(_field(fault, "faces"), dtype=np.int64).ravel()
        if faces.size == 0:
            raise ValueError(f"fault {name!r} has no faces")
        if faces.min() < 0 or faces.max() >= g.faces.num:
            raise IndexError(f"fault {name!r} contains invalid face indices")
        names.append(name)
        face_blocks.append(faces)
        groups.append(np.full(faces.size, ordinal, dtype=np.int64))
        labels.append(np.mean(np.asarray(g.faces.centroids)[faces], axis=0))
    dimension = np.asarray(g.nodes.coords).shape[1]
    return (
        tuple(names),
        np.concatenate(face_blocks) if face_blocks else
        np.empty(0, dtype=np.int64),
        np.concatenate(groups) if groups else np.empty(0, dtype=np.int64),
        np.asarray(labels, dtype=float).reshape(-1, dimension),
    )


def plot_faults(
    g,
    faults,
    *,
    height: float = 5.0,
    fontsize: float = 16,
    color="red",
    face_color="black",
    plotter=None,
    show: bool = True,
    **mesh_kwargs,
) -> FaultPlotResult:
    """Plot fault faces and labels at mean face-centroid positions."""
    if g.griddim != 3:
        raise ValueError("plot_faults requires a 3-D grid")
    names, faces, groups, positions = _normalize_faults(g, faults)
    if positions.size:
        positions[:, 2] -= float(height)
    rendered = plot_faces(
        g,
        faces,
        face_color,
        face_color=face_color,
        plotter=plotter,
        show=False,
        **mesh_kwargs,
    )
    text_actor = None
    if names and fontsize > 0 and hasattr(rendered.plotter, "add_point_labels"):
        text_actor = rendered.plotter.add_point_labels(
            positions,
            list(names),
            font_size=int(fontsize),
            text_color=color,
            show_points=False,
            always_visible=True,
        )
    if show and plotter is None and rendered.plotter is not None:
        rendered.plotter.show()
    return FaultPlotResult(
        faces=rendered,
        names=names,
        label_positions=positions,
        face_groups=groups,
        text_actor=text_actor,
    )


plotFaults = plot_faults

__all__ = ["FaultPlotResult", "plot_faults", "plotFaults"]
