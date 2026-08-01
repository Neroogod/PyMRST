"""Run the large-deck stress scenarios at a size the test suite can afford.

The full harness lives in `examples/validation/stress_gridprocessing.py` and
scales to 200k cells; these run its `small` configuration so the invariants are
checked on every commit rather than only when someone remembers to. The
invariant checker itself is tested too - a checker that cannot fail is worth
nothing.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "validation"))

from stress_gridprocessing import (  # noqa: E402
    SIZES,
    InvariantError,
    check_grid,
    stress_dense_nnc,
    stress_disconnected_regions,
    stress_faulted_deck,
    stress_repeated_slicing,
)

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402

SMALL = SIZES["small"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ------------------------------------------------------------- the scenarios


def test_faulted_corner_point_deck():
    summary = stress_faulted_deck(SMALL["fault"])
    assert summary["cells"] == int(np.prod(SMALL["fault"]))
    assert summary["closure"] < 1e-9


def test_many_disconnected_regions():
    summary = stress_disconnected_regions(SMALL["islands"])
    nx, ny, _ = SMALL["islands"]
    assert summary["components"] == -(-nx // 3) * -(-ny // 3)


def test_dense_nnc_network():
    summary = stress_dense_nnc(SMALL["nnc"])
    assert summary["resolved NNCs"] > 0
    assert summary["resolved NNCs"] < summary["requested NNCs"], "filters ran"


def test_repeated_slicing():
    summary = stress_repeated_slicing(SMALL["slice"], SMALL["cuts"])
    nx, ny, nz = SMALL["slice"]
    assert np.isclose(summary["volume"], nx * ny * nz)
    assert summary["cells"] > nx * ny * nz, "cuts must have split cells"


# ----------------------------------------------------- the checker itself


def _valid_grid():
    return compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))


def test_checker_accepts_a_valid_grid():
    summary = check_grid(_valid_grid(), name="ok", expect_volume=18.0)
    assert summary["cells"] == 18
    assert summary["closure"] < 1e-9


def test_checker_catches_a_wrong_volume():
    with pytest.raises(InvariantError, match="volume"):
        check_grid(_valid_grid(), name="bad", expect_volume=99.0)


def test_checker_catches_an_orphan_face():
    grid = _valid_grid()
    grid.faces.neighbors[0] = [-1, -1]
    with pytest.raises(InvariantError, match="orphan"):
        check_grid(grid, name="bad")


def test_checker_catches_a_face_missing_from_cells_faces():
    grid = _valid_grid()
    boundary = int(np.flatnonzero(np.any(grid.faces.neighbors == -1, axis=1))[0])
    other = int(grid.faces.neighbors[boundary].max())
    # claim a second neighbour that never lists this face
    grid.faces.neighbors[boundary] = [other, (other + 1) % grid.cells.num]
    with pytest.raises(InvariantError, match="missing from cells.faces"):
        check_grid(grid, name="bad")


def test_checker_catches_a_non_positive_volume():
    grid = _valid_grid()
    grid.cells.volumes[2] = 0.0
    with pytest.raises(InvariantError, match="non-positive volume"):
        check_grid(grid, name="bad")


def test_checker_catches_unclosed_cells():
    grid = _valid_grid()
    grid.faces.normals[3] *= 2.5
    with pytest.raises(InvariantError, match="not closed"):
        check_grid(grid, name="bad")


def test_checker_catches_an_out_of_range_neighbour():
    grid = _valid_grid()
    grid.faces.neighbors[1, 0] = grid.cells.num + 5
    with pytest.raises(InvariantError, match="outside the grid"):
        check_grid(grid, name="bad")
