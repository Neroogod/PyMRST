"""Permeability tensor expansion, ported from MRST `permTensor`."""

from typing import Tuple

import numpy as np

from pymrst.params.rock.rock import Rock


def perm_tensor(rock: Rock, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand `rock.perm` to row-wise full tensor entries and indices."""
    nc, nk = rock.perm.shape
    if dim == 1:
        assert nk == 1, "1D domains support scalar permeability only"
        return rock.perm, np.array([0]), np.array([0])
    if dim == 2:
        tensor = _symmetric_2d(rock.perm, nk, nc)
        return (
            tensor[:, [0, 1, 1, 2]],
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
        )
    if dim == 3:
        tensor = _symmetric_3d(rock.perm, nk, nc)
        return (
            tensor[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]],
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        )
    raise ValueError(f"{dim} space dimensions unsupported")


def _symmetric_2d(perm: np.ndarray, nk: int, nc: int) -> np.ndarray:
    if nk == 1:
        return perm * np.array([1.0, 0.0, 1.0])
    if nk == 2:
        return np.column_stack([perm[:, 0], np.zeros(nc), perm[:, 1]])
    assert nk == 3
    return perm


def _symmetric_3d(perm: np.ndarray, nk: int, nc: int) -> np.ndarray:
    if nk == 1:
        return perm * np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    if nk == 3:
        zeros = np.zeros(nc)
        return np.column_stack([perm[:, 0], zeros, zeros, perm[:, 1], zeros, perm[:, 2]])
    assert nk == 6
    return perm


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid
    from pymrst.params.rock.make_rock import make_rock

    def test_perm_tensor_scalar_3d() -> None:
        g = cart_grid([2, 2, 2])
        rock = make_rock(g, 5.0, 0.2)
        tensor, row, col = perm_tensor(rock, 3)
        assert tensor.shape == (8, 9)
        assert np.allclose(tensor[:, row == col], 5.0)
        assert np.allclose(tensor[:, row != col], 0.0)
        print("ok test_perm_tensor_scalar_3d")

    def test_perm_tensor_diagonal_3d() -> None:
        g = cart_grid([2, 1, 1])
        rock = make_rock(g, np.array([[1.0, 2.0, 3.0]]), 0.2)
        tensor, _, _ = perm_tensor(rock, 3)
        assert np.allclose(tensor[0, [0, 4, 8]], [1.0, 2.0, 3.0])
        assert np.allclose(np.delete(tensor[0], [0, 4, 8]), 0.0)
        print("ok test_perm_tensor_diagonal_3d")

    test_perm_tensor_scalar_3d()
    test_perm_tensor_diagonal_3d()
    print("\nAll tests passed!")

