"""PyMRST counterpart of MRST ``extrudedTriangleGrid``."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay

from pymrst.gridprocessing import compute_geometry, make_layered_grid, remove_cells
from pymrst.gridprocessing.triangle_grid import triangle_grid


def extruded_triangle_grid(max_area=100.0, dual=False, query_points=None, *, seed=None):
    """Build a layered triangular grid with a curved lateral fault.

    The areal triangulation uses SciPy rather than MRST's ``mex_triangleGrid``;
    the domain, fault curve, 15-layer extrusion, deformation, and opposing
    cell removals follow the reference fixture.
    """
    spacing = max(np.sqrt(2 * float(max_area)), 2.0)
    axis = np.linspace(0, 100, max(3, int(np.ceil(100 / spacing)) + 1))
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    y_fault = np.linspace(0, 100, 20)
    spline = CubicSpline([0, 30, 60, 100], [50, 50, 40, 40])
    fault_points = np.column_stack([spline(y_fault), y_fault])
    points = np.unique(np.vstack([points, fault_points]), axis=0)
    triangles = Delaunay(points).simplices
    areal = triangle_grid(points, triangles)
    if dual:
        from pymrst.gridprocessing.pebi import pebi
        areal = pebi(areal)
    geom = compute_geometry(areal)
    left = geom.cells.centroids[:, 0] < spline(geom.cells.centroids[:, 1])
    n = areal.cells.num
    grid = make_layered_grid(areal, 15)
    rng = np.random.default_rng(seed)
    layer_nodes = areal.nodes.num
    for layer in range(16):
        ix = slice(layer * layer_nodes, (layer + 1) * layer_nodes)
        if layer == 0:
            x, y = grid.nodes.coords[ix, 0], grid.nodes.coords[ix, 1]
            grid.nodes.coords[ix, 2] = (
                40 * (-.05*np.sin(3*rng.random()*np.pi*(x/x.max()-.5))
                      -.075*np.sin(np.pi*(y/y.max()+2*x/x.max())))
            )
        else:
            grid.nodes.coords[ix, 2] = (
                grid.nodes.coords[ix.start-layer_nodes:ix.stop-layer_nodes, 2]
                + .5 + rng.random()
            )
    c = np.tile(left, 15)
    upper_left = np.concatenate([np.ones(3*n, bool), np.zeros(12*n, bool)])
    lower_right = np.concatenate([np.zeros(10*n, bool), np.ones(5*n, bool)])
    grid = remove_cells(grid, np.flatnonzero((c & upper_left) | (~c & lower_right)))
    return (grid, query_points) if query_points is not None else grid


extrudedTriangleGrid = extruded_triangle_grid
