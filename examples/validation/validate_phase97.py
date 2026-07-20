"""Validate MATLAB-style vemmech plotting handles.

Phase 97 keeps the existing backend-agnostic plotting geometry while adding a
handle layer that can be updated like MATLAB graphics handles.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech import (
    create_augmented_grid,
    plot_cell_data_deformed,
    plot_face_data_deformed,
    plot_faces_2d,
    plot_grid_deformed,
    plot_node_data_deformed,
)


def _disp(g, sx=0.1, sy=-0.05):
    return np.column_stack([sx * g.nodes.coords[:, 0],
                            sy * g.nodes.coords[:, 1]])


def test_grid_handle_update():
    g = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
    u0 = np.zeros_like(g.nodes.coords)
    u1 = _disp(g)
    h = plot_grid_deformed(g, u0)
    assert h.kind == "grid_deformed"
    assert np.allclose(h.geometry.nodes.coords, g.nodes.coords)
    h.set_displacement(u1)
    assert np.allclose(h.geometry.nodes.coords, g.nodes.coords + u1)


def test_cell_face_node_handles_update():
    g = create_augmented_grid(compute_geometry(cart_grid([2, 1], [2.0, 1.0])))
    u0 = np.zeros_like(g.nodes.coords)
    u1 = _disp(g, sx=-0.03, sy=0.08)

    cell_data = np.arange(g.cells.num, dtype=float)
    hc = plot_cell_data_deformed(g, cell_data, u0)
    assert np.allclose(hc.geometry["data"], cell_data)
    hc.set_data(cell_data + 10.0).set_displacement(u1)
    assert np.allclose(hc.geometry["data"], cell_data + 10.0)
    assert np.allclose(hc.geometry["grid"].nodes.coords, g.nodes.coords + u1)

    face_data = np.arange(g.faces.num, dtype=float)
    hf = plot_face_data_deformed(g, face_data, u0)
    hf.set_data(face_data + 1.0)
    assert np.allclose(hf.geometry["data"], face_data + 1.0)

    node_data = np.arange(g.nodes.num, dtype=float)
    hn = plot_node_data_deformed(g, node_data, u0)
    old_vertices = hn.geometry["vertices"].copy()
    hn.set_displacement(u1)
    assert not np.allclose(hn.geometry["vertices"], old_vertices)
    assert hn.geometry["cdata"].shape[0] == hn.geometry["verts"].size


def test_faces_2d_handle_update():
    g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
    faces = np.arange(g.faces.num)
    h = plot_faces_2d(g, faces)
    assert h.geometry.shape == (g.faces.num, 2, 2)
    shifted = np.linspace(0.01, 0.04, g.faces.num)
    h.update(shift=shifted)
    assert h.geometry.shape == (g.faces.num, 2, 2)
    assert np.max(np.abs(h.geometry - plot_faces_2d(g, faces).geometry)) > 0


def main():
    test_grid_handle_update()
    test_cell_face_node_handles_update()
    test_faces_2d_handle_update()
    print("Phase 97 validation passed: vemmech plotting handles update.")


if __name__ == "__main__":
    main()
