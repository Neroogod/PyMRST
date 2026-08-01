import numpy as np
import pytest

from pymrst.params import add_bc, add_source


def test_add_bc_accepts_per_face_types_and_broadcasts_values_and_sat():
    bc = add_bc(
        None, [1, 3], ["Pressure", "FLUX"], 2.0, sat=[0.2, 0.8]
    )
    assert bc.type == ["pressure", "flux"]
    np.testing.assert_allclose(bc.value, [2.0, 2.0])
    np.testing.assert_allclose(bc.sat, [[0.2, 0.8], [0.2, 0.8]])


def test_add_bc_empty_faces_return_original_and_warn():
    bc = add_bc(None, [1], "pressure", 1.0)
    with pytest.warns(RuntimeWarning, match="Empty list"):
        unchanged = add_bc(bc, [], "flux", 0.0)
    assert unchanged is bc
    with pytest.warns(RuntimeWarning, match="Empty list"):
        assert add_bc(None, [], "flux", 0.0) is None


def test_add_bc_rejects_shapes_and_overlap_without_mutating():
    bc = add_bc(None, [1], "pressure", 1.0, sat=[1.0, 0.0])
    original = (bc.face.copy(), list(bc.type), bc.value.copy(), bc.sat.copy())
    with pytest.raises(ValueError, match="saturation columns"):
        add_bc(bc, [2], "flux", 1.0, sat=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="overlaps"):
        add_bc(bc, [1], "flux", 1.0, sat=[1.0, 0.0])
    with pytest.raises(ValueError, match="one string per face"):
        add_bc(bc, [2, 3], ["flux"], 1.0, sat=[1.0, 0.0])
    np.testing.assert_array_equal(bc.face, original[0])
    assert bc.type == original[1]
    np.testing.assert_array_equal(bc.value, original[2])
    np.testing.assert_array_equal(bc.sat, original[3])


def test_add_source_default_saturation_is_mrst_empty():
    src = add_source(None, [0, 2], 3.0)
    assert src.sat.shape == (0, 0)
    np.testing.assert_allclose(src.rate, [3.0, 3.0])


def test_add_source_broadcast_and_append_semantics():
    src = add_source(None, [0, 2], [1.0, -1.0], sat=[0.25, 0.75])
    src = add_source(src, [4], 2.0, sat=[[0.6, 0.4]])
    np.testing.assert_allclose(src.sat, [[.25, .75], [.25, .75], [.6, .4]])
    with pytest.raises(ValueError, match="saturation columns"):
        add_source(src, [5], 1.0)


def test_add_source_validates_before_mutating_and_extensions_align():
    src = add_source(None, [0], 1.0, sat=[1.0, 0.0], components=[0.2, 0.8])
    original_cells = src.cell.copy()
    with pytest.raises(ValueError, match="conflict"):
        add_source(src, [0], 2.0, sat=[1.0, 0.0], components=[0.2, 0.8])
    with pytest.raises(ValueError, match="component columns"):
        add_source(src, [1], 2.0, sat=[1.0, 0.0], components=[1, 0, 0])
    np.testing.assert_array_equal(src.cell, original_cells)


def test_add_source_empty_cells_are_a_noop():
    src = add_source(None, [0], 1.0)
    result = add_source(src, [], [])
    assert result is src
    np.testing.assert_array_equal(result.cell, [0])
