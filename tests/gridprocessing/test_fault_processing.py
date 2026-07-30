import numpy as np

from pymrst.gridprocessing import (
    cart_grid,
    process_faults,
    remove_fault_boundary_faces,
)


def _geometry():
    return {
        "faults": np.array([
            ["F_MAIN", 2, 2, 1, 2, 1, 2, "X+"],
            ["F_MAIN", 2, 2, 3, 3, 1, 2, "X+"],
            ["BOUND", 1, 1, 1, 3, 1, 2, "X-"],
        ], dtype=object),
        "MuLtFlT": np.array([
            ["F_MAIN", 0.25],
            ["BOUND", 0.5],
        ], dtype=object),
    }


def test_process_faults_groups_records_and_applies_multipliers():
    grid = cart_grid([3, 3, 2])
    faults, ids = process_faults(grid, _geometry())
    assert [fault.name for fault in faults] == ["BOUND", "F_MAIN"]
    assert ids == {"BOUND": 0, "F_MAIN": 1}
    assert [fault.numf for fault in faults] == [6, 6]
    assert [fault.multiplier for fault in faults] == [0.5, 0.25]
    assert np.all(grid.faces.neighbors[faults[0].faces, 0] == -1)
    assert np.all(grid.faces.neighbors[faults[1].faces] >= 0)


def test_remove_fault_boundary_faces_drops_empty_fault():
    grid = cart_grid([3, 3, 2])
    faults, _ = process_faults(grid, _geometry())
    interior, keep = remove_fault_boundary_faces(faults, grid)
    assert keep.tolist() == [False, True]
    assert [fault.name for fault in interior] == ["F_MAIN"]
    assert interior[0].numf == 6


def test_inactive_cells_are_ignored_and_default_multiplier_is_one():
    grid = cart_grid([3, 2, 1])
    grid.cells.indexMap = np.array([0, 1, 2, 3, 5])
    # This lightweight topology mutation only tests the Cartesian inverse map:
    # cell 4 is inactive, while cell 5 maps to active cell 4.
    geometry = {
        "FAULTS": [["F", 3, 3, 2, 2, 1, 1, "X+"]],
    }
    faults, _ = process_faults(grid, geometry)
    assert len(faults) == 1 and faults[0].multiplier == 1.0


def test_missing_fault_keyword_returns_empty_result():
    faults, ids = process_faults(cart_grid([2, 2, 1]), {})
    assert faults == [] and ids == {}
