"""Source-term insertion, ported from MRST `addSource`."""

from typing import Optional, Union

import numpy as np

from pymrst.params.wells_and_bc.types import Source

Scalar = Union[float, np.ndarray]


def add_source(
    src: Optional[Source],
    cells: np.ndarray,
    rates: Scalar,
    sat: Scalar = 1.0,
    components: Optional[Scalar] = None,
) -> Source:
    """Append volumetric source terms in the given cells."""
    if src is None:
        src = Source()
    cells = np.asarray(cells, dtype=np.int64).ravel()
    rates_arr = _broadcast_rates(rates, cells.size)
    assert not np.isin(cells, src.cell).any(), (
        "new source terms conflict with existing source terms"
    )
    sat_arr = _broadcast_saturation(sat, cells.size)
    src.cell = np.concatenate([src.cell, cells])
    src.rate = np.concatenate([src.rate, rates_arr])
    src.sat = sat_arr if src.sat.size == 0 else np.vstack([src.sat, sat_arr])
    if components is not None:
        comp_arr = _broadcast_saturation(components, cells.size)
        src.components = comp_arr if src.components is None else np.vstack([src.components, comp_arr])
    elif src.components is not None:
        src.components = np.vstack([src.components, np.zeros((cells.size, src.components.shape[1]))])
    return src


def _broadcast_rates(rates: Scalar, count: int) -> np.ndarray:
    arr = np.asarray(rates, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    assert arr.size == count, "one rate per cell or one scalar rate required"
    return arr


def _broadcast_saturation(sat: Scalar, count: int) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(sat, dtype=np.float64))
    if arr.shape[0] == 1:
        arr = np.repeat(arr, count, axis=0)
    return arr


if __name__ == "__main__":

    def test_add_source() -> None:
        src = add_source(None, np.array([0, 3]), 1e-3)
        assert np.allclose(src.rate, 1e-3)
        assert src.cell.size == 2
        try:
            add_source(src, np.array([3]), 0.0)
        except AssertionError:
            print("ok test_add_source")
            return
        raise AssertionError("expected conflict rejection")

    test_add_source()
    print("\nAll tests passed!")
