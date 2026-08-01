"""MRST-shaped plotting with lazy optional rendering backends."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pymrst_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

from pymrst.plotting.boundary_faces import boundary_faces, boundaryFaces
from pymrst.plotting.plot_faces import plot_faces, plotFaces
from pymrst.plotting.plot_face_data import plot_face_data, plotFaceData
from pymrst.plotting.plot_cell_data import plot_cell_data, plotCellData
from pymrst.plotting.plot_grid import plot_grid, plotGrid
from pymrst.plotting.plot_node_data import plot_node_data, plotNodeData
from pymrst.plotting.plot_well import (
    WellGeometry,
    WellPlotResult,
    plot_well,
    plotWell,
)
from pymrst.plotting.plot_faults import FaultPlotResult, plot_faults, plotFaults
from pymrst.plotting.outline_coarse_grid import (
    coarse_partition_faces,
    outline_coarse_grid,
    outlineCoarseGrid,
)
from pymrst.plotting.plot_block_and_neighbors import (
    BlockNeighborsResult,
    coarse_block_neighbors,
    plot_block_and_neighbors,
    plotBlockAndNeighbors,
)
from pymrst.plotting.plot_slice import (
    SlicePlotResult,
    slice_entities,
    plot_slice,
    plotSlice,
)
from pymrst.plotting.interpolate_data import interpolate_data, interpolateData
from pymrst.plotting.unstructured_surf import (
    SurfacePlotResult,
    surface_geometry,
    unstructured_surf,
    unstructuredSurf,
)
from pymrst.plotting.unstructured_contour import (
    ContourPlotResult,
    contour_grid,
    unstructured_contour,
    unstructuredContour,
)
from pymrst.plotting.plot_contours import (
    ContourBand,
    GridContourResult,
    contour_face_sets,
    plot_contours,
    plotContours,
)
from pymrst.plotting.colorbar_hist import (
    HistogramColorbarResult,
    histogram_colorbar_data,
    colorbar_hist,
    colorbarHist,
)
from pymrst.plotting.mrst_colorbar import mrst_colorbar, mrstColorbar
from pymrst.plotting.plot_grid_volumes import (
    VolumeSamples,
    IsosurfaceBand,
    GridVolumeResult,
    volume_samples,
    plot_grid_volumes,
    plotGridVolumes,
)
from pymrst.plotting.results import ColorData, PatchGeometry, PlotResult


_LAZY = {
    "to_pyvista": ("pymrst.plotting.plot_grid", "to_pyvista"),
    "plot_caprock": ("pymrst.plotting.extensions.plot_ve", "plot_caprock"),
    "plot_plume": ("pymrst.plotting.extensions.plot_ve", "plot_plume"),
    "plot_traps": ("pymrst.plotting.extensions.plot_ve", "plot_traps"),
    "plot_streamlines": (
        "pymrst.plotting.extensions.plot_streamlines", "plot_streamlines",
    ),
    "streamlines_to_pyvista": (
        "pymrst.plotting.extensions.plot_streamlines", "streamlines_to_pyvista",
    ),
    "plot_toolbar": ("pymrst.plotting.extensions.plot_toolbar", "plot_toolbar"),
    "PlotToolbar": ("pymrst.plotting.extensions.plot_toolbar", "PlotToolbar"),
    "normalize_data": ("pymrst.plotting.extensions.plot_toolbar", "normalize_data"),
    "states_to_fields": (
        "pymrst.plotting.extensions.plot_toolbar", "states_to_fields",
    ),
    "apply_transform": ("pymrst.plotting.extensions.plot_toolbar", "apply_transform"),
    "plot_trapping_inventory": (
        "pymrst.plotting.extensions.plot_inventory", "plot_trapping_inventory",
    ),
    "load_saved_state_fields": (
        "pymrst.plotting.extensions.plot_saved_states", "load_saved_state_fields",
    ),
    "saved_cell_fields": (
        "pymrst.plotting.extensions.plot_saved_states", "saved_cell_fields",
    ),
    "plot_saved_state": (
        "pymrst.plotting.extensions.plot_saved_states", "plot_saved_state",
    ),
    "plot_toolbar_saved_states": (
        "pymrst.plotting.extensions.plot_saved_states", "plot_toolbar_saved_states",
    ),
    "plot_reservoir_browser": (
        "pymrst.plotting.extensions.browser", "plot_reservoir_browser",
    ),
    "plot_saved_states_browser": (
        "pymrst.plotting.extensions.browser", "plot_saved_states_browser",
    ),
    "reservoir_browser_data": (
        "pymrst.plotting.extensions.browser", "reservoir_browser_data",
    ),
    "BrowserReservoirPayload": (
        "pymrst.plotting.extensions.browser", "BrowserReservoirPayload",
    ),
}

__all__ = [
    "boundary_faces",
    "boundaryFaces",
    "plot_faces",
    "plotFaces",
    "plot_face_data",
    "plotFaceData",
    "plot_cell_data",
    "plotCellData",
    "plot_grid",
    "plotGrid",
    "plot_node_data",
    "plotNodeData",
    "WellGeometry",
    "WellPlotResult",
    "plot_well",
    "plotWell",
    "FaultPlotResult",
    "plot_faults",
    "plotFaults",
    "coarse_partition_faces",
    "outline_coarse_grid",
    "outlineCoarseGrid",
    "BlockNeighborsResult",
    "coarse_block_neighbors",
    "plot_block_and_neighbors",
    "plotBlockAndNeighbors",
    "SlicePlotResult",
    "slice_entities",
    "plot_slice",
    "plotSlice",
    "interpolate_data",
    "interpolateData",
    "SurfacePlotResult",
    "surface_geometry",
    "unstructured_surf",
    "unstructuredSurf",
    "ContourPlotResult",
    "contour_grid",
    "unstructured_contour",
    "unstructuredContour",
    "ContourBand",
    "GridContourResult",
    "contour_face_sets",
    "plot_contours",
    "plotContours",
    "HistogramColorbarResult",
    "histogram_colorbar_data",
    "colorbar_hist",
    "colorbarHist",
    "mrst_colorbar",
    "mrstColorbar",
    "VolumeSamples",
    "IsosurfaceBand",
    "GridVolumeResult",
    "volume_samples",
    "plot_grid_volumes",
    "plotGridVolumes",
    "ColorData",
    "PatchGeometry",
    "PlotResult",
    *_LAZY,
]


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
