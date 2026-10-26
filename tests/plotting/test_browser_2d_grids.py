"""A 2-D grid's faces are edges, not polygons - the browser viewer's normal
face-boundary algorithm treats every one of them as degenerate and draws
nothing. These pin the fix: 2-D grids get per-cell polygon topology instead,
and 3-D grids are untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.plotting.extensions.browser import (
    _cell_polygons,
    plot_reservoir_browser,
    reservoir_browser_data,
)


def test_cell_polygons_are_built_for_a_2d_grid():
    g = compute_geometry(cart_grid([3, 2], [30.0, 20.0]))
    cell_pos, cell_nodes = _cell_polygons(g)
    assert cell_pos.shape == (g.cells.num + 1,)
    # Every cell in a Cartesian 2-D grid is a quad: four corner nodes each.
    assert np.all(np.diff(cell_pos) == 4)
    assert cell_nodes.size == 4 * g.cells.num


def test_cell_polygon_nodes_trace_a_simple_ring():
    """The angular sort must produce a non-self-intersecting quad, not the
    face-declaration order (which can zig-zag across the cell)."""
    g = compute_geometry(cart_grid([1, 1], [10.0, 10.0]))
    cell_pos, cell_nodes = _cell_polygons(g)
    ring = g.nodes.coords[cell_nodes[cell_pos[0]:cell_pos[1]]]
    # Shoelace area of a simple quad ring equals the cell's true area;
    # a self-intersecting (bowtie) ordering would roughly halve it.
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(100.0)


def test_reservoir_browser_data_carries_2d_topology():
    g = compute_geometry(cart_grid([4, 3], [40.0, 30.0]))
    payload = reservoir_browser_data(g, np.arange(g.cells.num, dtype=float))
    assert payload.griddim == 2
    assert payload.cell_pos is not None
    assert payload.cell_nodes is not None
    assert payload.cell_pos.size == g.cells.num + 1


def test_reservoir_browser_data_leaves_3d_grids_without_cell_polygons():
    g = compute_geometry(cart_grid([3, 3, 2], [30.0, 30.0, 20.0]))
    payload = reservoir_browser_data(g, np.arange(g.cells.num, dtype=float))
    assert payload.griddim == 3
    assert payload.cell_pos is None
    assert payload.cell_nodes is None


def test_plot_reservoir_browser_embeds_nonempty_cell_topology_for_2d(tmp_path):
    g = compute_geometry(cart_grid([5, 4], [50.0, 40.0]))
    path = plot_reservoir_browser(
        g, np.arange(g.cells.num, dtype=float),
        filename=tmp_path / "one_phase.html", title="2D regression",
    )
    html = path.read_text()
    assert '"griddim":2' in html
    assert '"cellPos"' in html and '"cellNodes"' in html
    assert '"cellPos":null' not in html
