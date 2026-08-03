"""MRST references for packed topology and grid-query utilities."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.utils.gridtools import (
    connected_cells,
    get_cell_no_faces,
    get_connectivity_matrix,
    get_neighbourship,
    grid_cell_faces,
    grid_cell_no,
    grid_cell_nodes,
    grid_face_nodes,
    grid_logical_indices,
    indirection_sub,
    neighbours_by_nodes,
    translate_grid,
)


REF = loadmat(Path(__file__).parents[3] /
              "scripts/matlab/ref_utils_grid_queries.mat", squeeze_me=True)


def vec(name, dtype=None):
    return np.asarray(REF[name], dtype=dtype).ravel()


def grid():
    return cart_grid([3, 2], [3.0, 2.0])


def test_grid_cell_faces_and_cell_numbers_match_mrst():
    g = grid()
    subset = vec("subset", int) - 1
    faces, pos = grid_cell_faces(g, subset)
    np.testing.assert_array_equal(faces + 1, vec("cell_faces"))
    np.testing.assert_array_equal(pos + 1, vec("cell_face_pos"))
    np.testing.assert_array_equal(grid_cell_no(g) + 1, vec("cell_no_all"))
    np.testing.assert_array_equal(grid_cell_no(g, subset) + 1,
                                  vec("cell_no_subset"))


def test_grid_cell_nodes_unique_and_raw_match_mrst():
    g, subset = grid(), vec("subset", int) - 1
    nodes, pos = grid_cell_nodes(g, subset)
    np.testing.assert_array_equal(nodes + 1, vec("cell_nodes"))
    np.testing.assert_array_equal(pos + 1, vec("cell_node_pos"))
    nodes, pos = grid_cell_nodes(g, subset, unique=False)
    np.testing.assert_array_equal(nodes + 1, vec("cell_nodes_raw"))
    np.testing.assert_array_equal(pos + 1, vec("cell_node_pos_raw"))


def test_grid_face_nodes_and_indirection_sub_match_mrst():
    g = grid()
    nodes, pos = grid_face_nodes(g, vec("face_subset", int) - 1)
    np.testing.assert_array_equal(nodes + 1, vec("face_nodes"))
    np.testing.assert_array_equal(pos + 1, vec("face_node_pos"))
    got = indirection_sub(vec("packed_items", int) - 1,
                          vec("packed_pos", int) - 1,
                          vec("packed_values"))
    np.testing.assert_array_equal(got, vec("packed_out"))


def test_grid_logical_indices_match_mrst():
    i, j = grid_logical_indices(grid(), vec("logical_cells", int) - 1)
    np.testing.assert_array_equal(i + 1, vec("logical_i"))
    np.testing.assert_array_equal(j + 1, vec("logical_j"))


def test_neighbourship_and_connectivity_match_mrst():
    g = grid()
    np.testing.assert_array_equal(get_neighbourship(g) + 1,
                                  REF["geometric_n"])
    # MATLAB uses cell zero for outside; PyMRST uses -1.
    boundary = get_neighbourship(g, include_boundary=True).copy()
    boundary = boundary + 1
    boundary[boundary == 0] = 0
    np.testing.assert_array_equal(boundary, REF["boundary_n"])
    matrix = get_connectivity_matrix(get_neighbourship(g), True,
                                     g.cells.num).toarray()
    np.testing.assert_array_equal(matrix, REF["connectivity"])
    np.testing.assert_array_equal(neighbours_by_nodes(g) + 1, REF["node_n"])


def test_connected_cells_match_mrst_components():
    groups = connected_cells(grid(), vec("component_cells", int) - 1)
    groups = sorted((np.sort(group) + 1 for group in groups), key=lambda x: x[0])
    np.testing.assert_array_equal(groups[0], vec("component_a"))
    np.testing.assert_array_equal(groups[1], vec("component_b"))


def test_nnc_queries_match_mrst():
    g = grid()
    g.nnc = SimpleNamespace(cells=np.array([[0, 5]], dtype=int))
    cell_no, faces, flag = get_cell_no_faces(g)
    np.testing.assert_array_equal(cell_no + 1, vec("nnc_cell_no"))
    np.testing.assert_array_equal(faces + 1, vec("nnc_cell_faces"))
    np.testing.assert_array_equal(flag, vec("nnc_flag", bool))
    neighbours, is_nnc = get_neighbourship(g, topological=True,
                                           with_nnc_flag=True)
    np.testing.assert_array_equal(neighbours + 1, REF["topological_n"])
    np.testing.assert_array_equal(is_nnc, vec("topological_flag", bool))


def test_translate_grid_geometry_matches_mrst():
    g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    translated = translate_grid(g, [5.0, -2.0])
    np.testing.assert_allclose(translated.nodes.coords, REF["translated_nodes"])
    np.testing.assert_allclose(translated.cells.centroids,
                               REF["translated_cells"])
    np.testing.assert_allclose(translated.faces.centroids,
                               REF["translated_faces"])
