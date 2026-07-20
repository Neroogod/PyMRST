"""Well insertion, ported from MRST `addWell`."""

from typing import Dict, List, Optional, Union

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc.compute_well_index import compute_well_index
from pymrst.params.wells_and_bc.types import Well

Scalar = Union[float, np.ndarray]


def add_well(
    wells: Optional[List[Well]],
    g: Grid,
    rock: Rock,
    cells: np.ndarray,
    well_type: str = "bhp",
    val: float = 0.0,
    radius: Scalar = 0.1,
    well_dir: str = "z",
    name: Optional[str] = None,
    skin: Scalar = 0.0,
    kh: Optional[np.ndarray] = None,
    wi: Optional[np.ndarray] = None,
    compi: Optional[np.ndarray] = None,
    ref_depth: Optional[float] = None,
    sign: int = 0,
    gravity: Optional[np.ndarray] = None,
    lims: Optional[Dict[str, float]] = None,
) -> List[Well]:
    """Append a well perforating `cells`, computing WI where not supplied."""
    wells = list(wells) if wells is not None else []
    cells = np.asarray(cells, dtype=np.int64).ravel()
    assert cells.size > 0, "empty cell index range"
    well_type = well_type.lower()
    assert well_type in ("bhp", "rate", "vrat", "resv_history", "orat", "wrat", "grat", "lrat"), (
        "well type must be one of bhp/rate/vrat/resv_history/orat/wrat/grat/lrat"
    )

    radius_arr = _radius_array(radius, cells.size)
    wi_arr = _well_index(g, rock, radius_arr, cells, well_dir, skin, kh, wi)
    compi_arr = np.array([1.0, 0.0, 0.0]) if compi is None else np.asarray(compi, float).ravel()
    down = _depth_direction(g, gravity)
    depth = g.cells.centroids[cells] @ down
    ref_depth = _reference_depth(g, down, gravity, ref_depth)
    sign = _well_sign(sign, well_type, val)
    name = f"W{len(wells) + 1}" if name is None else name

    wells.append(
        Well(
            cells=cells,
            type=well_type,
            val=float(val),
            r=radius_arr,
            dir=well_dir.lower(),
            WI=wi_arr,
            dZ=depth - ref_depth,
            name=name,
            compi=compi_arr,
            refDepth=float(ref_depth),
            sign=sign,
            lims=dict(lims) if lims else None,
        )
    )
    return wells


def _radius_array(radius: Scalar, count: int) -> np.ndarray:
    arr = np.asarray(radius, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    assert arr.size == count, "one radius per perforation or one scalar required"
    return arr


def _well_index(
    g: Grid,
    rock: Rock,
    radius: np.ndarray,
    cells: np.ndarray,
    well_dir: str,
    skin: Scalar,
    kh: Optional[np.ndarray],
    wi: Optional[np.ndarray],
) -> np.ndarray:
    if wi is None:
        return compute_well_index(g, rock, radius, cells, well_dir=well_dir, skin=skin, kh=kh)
    wi_arr = np.asarray(wi, dtype=np.float64).ravel()
    assert wi_arr.size == cells.size, "one WI entry per perforated cell required"
    return wi_arr


def _depth_direction(g: Grid, gravity: Optional[np.ndarray]) -> np.ndarray:
    dims = g.griddim
    if gravity is not None and np.linalg.norm(np.ravel(gravity)[:dims]) > 0:
        return np.ravel(gravity)[:dims] / np.linalg.norm(np.ravel(gravity)[:dims])
    down = np.zeros(dims)
    if dims > 2:
        down[-1] = 1.0
    return down


def _reference_depth(
    g: Grid,
    down: np.ndarray,
    gravity: Optional[np.ndarray],
    ref_depth: Optional[float],
) -> float:
    if ref_depth is not None:
        return float(ref_depth)
    if gravity is not None and np.linalg.norm(np.ravel(gravity)[: g.griddim]) > 0:
        return float((g.nodes.coords @ down).min())
    return 0.0


def _well_sign(sign: int, well_type: str, val: float) -> int:
    if sign == 0 and well_type in ("rate", "vrat", "resv_history", "orat", "wrat", "grat", "lrat") and val != 0.0:
        return int(np.sign(val))
    return sign


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc.incomp_well_pressure_drop import (
        incomp_well_pressure_drop,
    )
    from pymrst.utils.units import DARCY, MILLI

    def test_add_well_and_pressure_drop() -> None:
        g = compute_geometry(cart_grid([5, 5, 3], [500.0, 500.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        cells = np.array([12, 37, 62])
        wells = add_well(
            None,
            g,
            rock,
            cells,
            well_type="rate",
            val=1e-2,
            gravity=np.array([0.0, 0.0, 9.80665]),
        )
        assert len(wells) == 1
        assert wells[0].sign == 1
        assert np.allclose(wells[0].dZ, [5.0, 15.0, 25.0])
        mob = np.ones((g.cells.num, 1))
        dp = incomp_well_pressure_drop(wells[0], mob, np.array([1000.0]))
        assert np.allclose(dp, 9.80665 * 1000.0 * np.array([5.0, 15.0, 25.0]))
        print("ok test_add_well_and_pressure_drop")

    test_add_well_and_pressure_drop()
    print("\nAll tests passed!")
