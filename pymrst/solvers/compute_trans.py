"""Half-face transmissibilities, ported from MRST `computeTrans`."""

import warnings

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock, perm_tensor


def compute_trans(g: Grid, rock: Rock, fix_negative: bool = True) -> np.ndarray:
    """Return one-sided transmissibilities, one value per half-face."""
    assert g.cells.centroids is not None, "run compute_geometry first"
    cell_no = g.cell_no()
    cell_faces = g.cells.faces
    c = g.faces.centroids[cell_faces] - g.cells.centroids[cell_no]
    sign = np.where(g.faces.neighbors[cell_faces, 0] == cell_no, 1.0, -1.0)
    normals = sign[:, None] * g.faces.normals[cell_faces]
    tensor, row, col = perm_tensor(rock, g.griddim)
    assert tensor.shape[0] == g.cells.num, (
        "permeability must be defined in active cells only"
    )

    trans = np.zeros(cell_faces.size)
    for index in range(row.size):
        trans += c[:, row[index]] * tensor[cell_no, index] * normals[:, col[index]]
    trans /= np.sum(c * c, axis=1)
    trans = _fix_negative(trans, fix_negative)
    return _apply_ntg(g, rock, trans, cell_no)


def _fix_negative(trans: np.ndarray, fix_negative: bool) -> np.ndarray:
    if not fix_negative:
        return trans
    negative = trans < 0
    if np.any(negative):
        warnings.warn(
            f"{int(negative.sum())} negative transmissibilities; "
            "replaced by absolute value"
        )
        return np.abs(trans)
    return trans


def _apply_ntg(
    g: Grid,
    rock: Rock,
    trans: np.ndarray,
    cell_no: np.ndarray,
) -> np.ndarray:
    if rock.ntg is None or g.cells.tags is None:
        return trans
    multiplier = rock.ntg[cell_no].copy()
    multiplier[g.cells.tags // 2 == 2] = 1.0
    return multiplier * trans


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    def test_uniform_grid_analytic_trans() -> None:
        g = compute_geometry(cart_grid([3, 3, 3], [30.0, 30.0, 30.0]))
        perm = 100 * MILLI * DARCY
        rock = make_rock(g, perm, 0.2)
        trans = compute_trans(g, rock)
        assert trans.shape == (g.cells.faces.size,)
        assert np.allclose(trans, perm * 100.0 / 5.0, rtol=1e-12)
        print("ok test_uniform_grid_analytic_trans")

    def test_anisotropic_trans() -> None:
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        rock = make_rock(g, np.array([[1.0, 2.0, 4.0]]), 0.2)
        trans = compute_trans(g, rock)
        tags = g.cells.tags
        assert np.allclose(trans[tags <= 1], 2.0)
        assert np.allclose(trans[(tags == 2) | (tags == 3)], 4.0)
        assert np.allclose(trans[tags >= 4], 8.0)
        print("ok test_anisotropic_trans")

    def test_ntg_scales_lateral_only() -> None:
        g = compute_geometry(cart_grid([2, 2, 2]))
        rock = make_rock(g, 1.0, 0.2, ntg=0.5)
        base = make_rock(g, 1.0, 0.2)
        trans = compute_trans(g, rock)
        base_trans = compute_trans(g, base)
        lateral = g.cells.tags <= 3
        assert np.allclose(trans[lateral], 0.5 * base_trans[lateral])
        assert np.allclose(trans[~lateral], base_trans[~lateral])
        print("ok test_ntg_scales_lateral_only")

    test_uniform_grid_analytic_trans()
    test_anisotropic_trans()
    test_ntg_scales_lateral_only()
    print("\nAll tests passed!")

