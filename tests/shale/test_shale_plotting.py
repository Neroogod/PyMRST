import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.shale import (
    ShaleFracturePlane,
    plot_frac_system,
    plot_frac_system_nf,
)


def _case():
    grid = compute_geometry(cart_grid([1, 1, 1], [2.0, 2.0, 2.0]))
    # MRST-shaped fracture-grid metadata used by completion-centroid plotting.
    grid.FracGrid = {
        "Frac1": {"cells": {"start": 10, "centroids": np.array([[1.0, 1.0, 1.0]])}}
    }
    planes = [
        ShaleFracturePlane(np.array([[1, 0, 0], [1, 2, 0], [1, 2, 2], [1, 0, 2]])),
        ShaleFracturePlane(np.array([[0, 1, 0], [2, 1, 0], [2, 1, 2], [0, 1, 2]])),
    ]
    well = {
        "points": np.array([[.5, .5, 0], [.5, .5, 2]]),
        "WellXFracPts": np.array([[.5, .5, 1]]),
        "XFracCellIDs": np.array([[10, 0]]),
    }
    return grid, planes, well


def test_plot_frac_system_returns_mrst_handle_groups_and_engineering_markers():
    grid, planes, well = _case()
    plotter = pv.Plotter(off_screen=True)
    result = plot_frac_system(grid, planes, [well], plotter=plotter, show=False)
    hm, hf, hw = result
    assert hm is result.matrix
    assert len(hf) == 2
    assert len(hw) == 1
    assert len(result.labels) == 2
    assert len(result.intersections) == 1
    assert len(result.completion_centroids) == 1
    plotter.close()


def test_plot_frac_system_subset_is_zero_based_and_validated():
    grid, planes, _ = _case()
    plotter = pv.Plotter(off_screen=True)
    result = plot_frac_system(
        grid, planes, fracture_list=[1], label=False, plotter=plotter, show=False
    )
    assert len(result.fractures) == 1
    assert result.labels == ()
    with pytest.raises(IndexError, match="out-of-range"):
        plot_frac_system(grid, planes, fracture_list=[2], plotter=plotter, show=False)
    plotter.close()


def test_plot_frac_system_nf_classifies_hydraulic_and_natural_fractures():
    grid, planes, well = _case()
    plotter = pv.Plotter(off_screen=True)
    result = plot_frac_system_nf(
        grid, planes, 1, [well], plotter=plotter, show=False,
        hydraulic_color="yellow", natural_color="blue",
    )
    assert len(result.fractures) == 2
    assert len(result.wells) == 1
    colors = [tuple(actor.prop.color.float_rgb) for actor in result.fractures]
    assert np.allclose(colors[0], [1, 1, 0])
    assert np.allclose(colors[1], [0, 0, 1])
    with pytest.raises(ValueError, match="between zero"):
        plot_frac_system_nf(grid, planes, 3, plotter=plotter, show=False)
    plotter.close()
