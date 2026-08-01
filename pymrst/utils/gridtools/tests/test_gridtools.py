"""Executable counterpart of MRST ``gridtools/tests/test_gridtools.m``."""

import numpy as np
from pymrst.gridprocessing import cart_grid
from pymrst.utils.gridtools.grid_cell_faces import grid_cell_faces
from pymrst.utils.gridtools.grid_cell_no import grid_cell_no
from pymrst.utils.gridtools.grid_cell_nodes import grid_cell_nodes
from pymrst.utils.gridtools.grid_face_nodes import grid_face_nodes
from pymrst.utils.gridtools.indirection_sub import indirection_sub


def test_gridtools_case():
    grid = cart_grid([10, 5, 3])
    cells = np.array([0, 1, 8, 6, 12, 50, 13, 39, 92])
    nodes, node_pos = grid_cell_nodes(grid, cells)
    faces, face_pos = grid_cell_faces(grid, cells)
    target = 4
    assert np.array_equal(indirection_sub(target, face_pos, faces),
                          faces[face_pos[target]:face_pos[target + 1]])
    sample_faces = np.array([0, 2, 7, 8, 22, 102, 48, 192])
    face_nodes, face_node_pos = grid_face_nodes(grid, sample_faces)
    cell_no = grid_cell_no(grid)
    for cell in range(grid.cells.num):
        block = cell_no[grid.cells.facePos[cell]:grid.cells.facePos[cell + 1]]
        assert np.array_equal(np.unique(block), [cell])
    return {"nodes": nodes, "node_pos": node_pos, "faces": faces,
            "face_pos": face_pos, "face_nodes": face_nodes,
            "face_node_pos": face_node_pos}


if __name__ == "__main__":
    test_gridtools_case(); print("gridtools checks passed")
