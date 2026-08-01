"""Peaceman well-index calculation, ported from MRST `computeWellIndex`."""

from typing import Optional, Union

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock
from pymrst.utils.mcolon import mcolon

PEACEMAN_TPF_CONSTANT: float = 0.14
_MIMETIC_WELL_CONSTANT = np.array([
    [1, .292], [2, .278], [3, .262], [4, .252], [5, .244],
    [8, .231], [9, .229], [16, .220], [17, .219], [32, .213],
    [33, .213], [64, .210], [65, .210],
])
Scalar = Union[float, np.ndarray]


def cell_dims(g: Grid, cells: np.ndarray) -> np.ndarray:
    """Return bounding-box extents `(dx, dy, dz)` for cells."""
    cells = np.asarray(cells, dtype=np.int64).ravel()
    out = np.zeros((cells.size, 3))
    for index, cell in enumerate(cells):
        faces = g.cells.faces[g.cells.facePos[cell] : g.cells.facePos[cell + 1]]
        edges = mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
        coords = g.nodes.coords[np.unique(g.faces.nodes[edges])]
        extents = coords.max(axis=0) - coords.min(axis=0)
        dx = extents[0]
        dy = extents[1] if coords.shape[1] > 1 else 1.0
        dz = g.cells.volumes[cell] / (dx * dy) if coords.shape[1] > 2 else 1.0
        out[index] = [dx, dy, dz]
    return out


def compute_well_index(
    g: Grid,
    rock: Rock,
    radius: Scalar,
    cells: np.ndarray,
    well_dir: str = "z",
    skin: Scalar = 0.0,
    kh: Optional[np.ndarray] = None,
    cell_dims_override: Optional[np.ndarray] = None,
    inner_product: str = "ip_tpf",
) -> np.ndarray:
    """Return Peaceman well index per perforated cell."""
    cells = np.asarray(cells, dtype=np.int64).ravel()
    radius_arr = _broadcast(radius, cells.size)
    skin_arr = _broadcast(skin, cells.size)
    if cell_dims_override is None:
        dims = cell_dims(g, cells)
    else:
        dims = np.asarray(cell_dims_override, dtype=float)
        if dims.shape != (cells.size, 3):
            raise ValueError(
                f"cell_dims_override must have shape ({cells.size}, 3)"
            )
    perm = _perm_diag(g, rock, cells)

    axis = {"x": (1, 2, 0), "y": (0, 2, 1), "z": (0, 1, 2)}[well_dir.lower()]
    d1, d2, ell = dims[:, axis[0]], dims[:, axis[1]], dims[:, axis[2]]
    k1, k2 = perm[:, axis[0]], perm[:, axis[1]]
    radius_eq = _equivalent_radius(
        d1, d2, k1, k2, _well_constant(d1, d2, inner_product)
    )
    kh_arr = _kh(kh, cells.size, ell, np.sqrt(k1 * k2), g.griddim)

    wi = 2 * np.pi * kh_arr / (np.log(radius_eq / radius_arr) + skin_arr)
    assert np.all(wi >= 0), (
        "negative well index: equivalent radius below well radius or large negative skin"
    )
    return wi


def _broadcast(value: Scalar, count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    assert arr.size == count, "one value per perforation or one scalar required"
    return arr


def _equivalent_radius(
    d1: np.ndarray,
    d2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    well_constant: Union[float, np.ndarray] = PEACEMAN_TPF_CONSTANT,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        k21 = np.where(k1 > 0, k2 / k1, 0.0)
        k12 = np.where(k2 > 0, k1 / k2, 0.0)
    numerator = 2 * well_constant * np.sqrt(
        d1**2 * np.sqrt(k21) + d2**2 * np.sqrt(k12)
    )
    radius = numerator / (k21**0.25 + k12**0.25)
    radius[~np.isfinite(radius)] = 0.0
    return radius


def _well_constant(d1, d2, inner_product):
    method = str(inner_product).lower()
    if method in ("ip_tpf", "ip_quasitpf"):
        return PEACEMAN_TPF_CONSTANT
    if method not in ("ip_rt", "ip_simple", "ip_quasirt"):
        raise ValueError(f"unknown inner product {inner_product!r}")
    ratio = np.maximum(
        np.floor(d1/d2 + 0.5), np.floor(d2/d1 + 0.5)
    )
    x, y = _MIMETIC_WELL_CONSTANT.T
    result = np.interp(ratio, x, y)
    below = ratio < x[0]
    above = ratio > x[-1]
    result[below] = y[0] + (ratio[below]-x[0])*(y[1]-y[0])/(x[1]-x[0])
    result[above] = y[-1] + (ratio[above]-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2])
    return result


def _kh(
    kh: Optional[np.ndarray],
    count: int,
    ell: np.ndarray,
    ke: np.ndarray,
    griddim: int,
) -> np.ndarray:
    if kh is None:
        kh_arr = np.full(count, -1.0)
    else:
        kh_arr = np.asarray(kh, dtype=np.float64).ravel().copy()
    default = kh_arr < 0
    kh_arr[default] = ell[default] * ke[default] if griddim > 2 else ke[default]
    return kh_arr


def _perm_diag(g: Grid, rock: Rock, cells: np.ndarray) -> np.ndarray:
    nk = rock.perm.shape[1]
    if g.griddim == 3:
        cols = {1: [0, 0, 0], 3: [0, 1, 2], 6: [0, 3, 5]}[nk]
        return rock.perm[np.ix_(cells, cols)]
    if g.griddim == 2:
        cols = {1: [0, 0], 2: [0, 1], 3: [0, 2]}[nk]
        perm2 = rock.perm[np.ix_(cells, cols)]
        return np.column_stack([perm2, perm2[:, 0]])
    return rock.perm[cells][:, [0, 0, 0]]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    def test_cell_dims() -> None:
        g = compute_geometry(cart_grid([5, 5, 3], [500.0, 500.0, 30.0]))
        dims = cell_dims(g, np.array([0, 12]))
        assert np.allclose(dims, [100.0, 100.0, 10.0])
        print("ok test_cell_dims")

    def test_peaceman_wi_isotropic() -> None:
        g = compute_geometry(cart_grid([5, 5, 3], [500.0, 500.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        wi = compute_well_index(g, rock, 0.1, np.array([0]))
        k = 100 * MILLI * DARCY
        re = 0.14 * np.sqrt(100.0**2 + 100.0**2)
        expect = 2 * np.pi * 10.0 * k / np.log(re / 0.1)
        assert np.allclose(wi, expect, rtol=1e-12)
        print("ok test_peaceman_wi_isotropic")

    test_cell_dims()
    test_peaceman_wi_isotropic()
    print("\nAll tests passed!")
