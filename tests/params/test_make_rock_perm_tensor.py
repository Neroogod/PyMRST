from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid
from pymrst.params import make_rock, perm_tensor


@pytest.mark.parametrize("dims, uniform, columns", [
    ([3], [[2.0]], 1),
    ([2, 2], [[2.0, 3.0]], 2),
    ([2, 2], [[2.0, .1, 3.0]], 3),
    ([2, 1, 2], [[2.0, 3.0, 4.0]], 3),
    ([2, 1, 2], [[2.0, .1, .2, 3.0, .3, 4.0]], 6),
])
def test_make_rock_accepts_all_mrst_tensor_layouts(dims, uniform, columns):
    grid = cart_grid(dims)
    rock = make_rock(grid, uniform, .2)
    assert rock.perm.shape == (grid.cells.num, columns)


def test_one_dimensional_numpy_arrays_mean_per_cell_fields():
    grid = cart_grid([4, 1, 1])
    rock = make_rock(grid, [1, 2, 3, 4], [.1, .2, .3, .4], ntg=[.9, .8, .7, .6])
    assert rock.perm.shape == (4, 1)
    np.testing.assert_allclose(rock.perm.ravel(), [1, 2, 3, 4])
    np.testing.assert_allclose(rock.poro, [.1, .2, .3, .4])


def test_make_rock_warns_for_nonpositive_porosity():
    grid = cart_grid([2, 1])
    with pytest.warns(UserWarning, match="negative porosity"):
        rock = make_rock(grid, 1.0, [0.0, -.1])
    np.testing.assert_allclose(rock.poro, [0.0, -.1])


def test_make_rock_rejects_invalid_shapes():
    grid = cart_grid([2, 2, 1])
    with pytest.raises(ValueError, match="permeability"):
        make_rock(grid, [[1, 2]], .2)
    with pytest.raises(ValueError, match="porosity"):
        make_rock(grid, 1.0, [[.2, .3]])
    with pytest.raises(ValueError, match="net-to-gross"):
        make_rock(grid, 1.0, .2, ntg=[[.8, .9]])
    with pytest.raises(ValueError, match="scalar, vector, or matrix"):
        make_rock(grid, np.ones((1, 1, 1)), .2)


@pytest.mark.parametrize("dim, compact, expected", [
    (1, [[2]], [[2]]),
    (2, [[2]], [[2, 0, 0, 2]]),
    (2, [[2, 3]], [[2, 0, 0, 3]]),
    (2, [[2, .5, 3]], [[2, .5, .5, 3]]),
    (3, [[2]], [[2, 0, 0, 0, 2, 0, 0, 0, 2]]),
    (3, [[2, 3, 4]], [[2, 0, 0, 0, 3, 0, 0, 0, 4]]),
    (3, [[2, .1, .2, 3, .3, 4]], [[2, .1, .2, .1, 3, .3, .2, .3, 4]]),
])
def test_perm_tensor_expands_every_compact_layout(dim, compact, expected):
    rock = SimpleNamespace(perm=np.asarray(compact, dtype=float))
    values, rows, cols = perm_tensor(rock, dim)
    np.testing.assert_allclose(values, expected)
    np.testing.assert_array_equal(rows, np.repeat(np.arange(dim), dim))
    np.testing.assert_array_equal(cols, np.tile(np.arange(dim), dim))


def test_perm_tensor_validates_rock_dimension_and_component_count():
    with pytest.raises(ValueError, match="contain"):
        perm_tensor(None, 3)
    with pytest.raises(ValueError, match="unsupported in 2D"):
        perm_tensor(SimpleNamespace(perm=np.ones((2, 6))), 2)
    with pytest.raises(ValueError, match="unsupported in 3D"):
        perm_tensor(SimpleNamespace(perm=np.ones((2, 2))), 3)
    with pytest.raises(ValueError, match="space dimensions"):
        perm_tensor(SimpleNamespace(perm=np.ones((2, 1))), 4)
