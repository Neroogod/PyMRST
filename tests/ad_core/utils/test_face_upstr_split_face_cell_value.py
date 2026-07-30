"""Reference tests for MRST ``faceUpstr`` and ``splitFaceCellValue``."""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_core.utils.split_face_cell_value import (
    split_face_cell_value,
)
from pymrst.params.rock import make_rock
from pymrst.utils.adi import init_variables_adi, value
from pymrst.utils.units import DARCY, MILLI


def _fixture():
    grid = compute_geometry(cart_grid([4, 1, 1], [4.0, 1.0, 1.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    return setup_operators_tpfa(grid, rock), 3, 4


def test_face_upstr_numeric_scalar_flags_and_ad_jacobian():
    operators, _, _ = _fixture()
    cell = np.array([10.0, 20.0, 30.0, 40.0])
    assert np.array_equal(
        operators.upw(np.array([True, False, True]), cell),
        [10.0, 30.0, 30.0],
    )
    assert np.array_equal(operators.upw(True, cell), [10.0, 20.0, 30.0])
    assert np.array_equal(operators.upw(False, cell), [20.0, 30.0, 40.0])

    (cell_ad,) = init_variables_adi(cell)
    selected = operators.upw(np.array([True, False, True]), cell_ad)
    assert np.array_equal(value(selected), [10.0, 30.0, 30.0])
    assert np.array_equal(
        selected.jac[0].toarray(),
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    )


def test_split_face_cell_value_supports_all_mrst_layouts():
    operators, nf, nc = _fixture()
    flag = np.array([True, False, True])

    cell = np.array([10.0, 20.0, 30.0, 40.0])
    face, returned_cell = split_face_cell_value(
        operators, flag, cell, (nf, nc)
    )
    assert np.array_equal(face, [10.0, 30.0, 30.0])
    assert returned_cell is cell

    face_cell = np.arange(1.0, nf + nc + 1)
    face, returned_cell = split_face_cell_value(
        operators, flag, face_cell, (nf, nc)
    )
    assert np.array_equal(face, [1.0, 2.0, 3.0])
    assert np.array_equal(returned_cell, [4.0, 5.0, 6.0, 7.0])

    half_face_cell = np.arange(1.0, 2 * nf + nc + 1)
    face, returned_cell = split_face_cell_value(
        operators, flag, half_face_cell, (nf, nc)
    )
    assert np.array_equal(face, [1.0, 5.0, 3.0])
    assert np.array_equal(returned_cell, [7.0, 8.0, 9.0, 10.0])


@pytest.mark.parametrize("length", [3, 6])
def test_split_face_cell_value_rejects_mrst_unimplemented_layouts(length):
    operators, nf, nc = _fixture()
    with pytest.raises(NotImplementedError):
        split_face_cell_value(
            operators, np.ones(nf, dtype=bool), np.ones(length), (nf, nc)
        )
