"""Source-term insertion, ported from MRST `addSource`."""

from typing import Optional, Union

import numpy as np

from pymrst.params.wells_and_bc.types import Source

Scalar = Union[float, np.ndarray]


def add_source(
    src: Optional[Source],
    cells: np.ndarray,
    rates: Scalar,
    sat: Optional[Scalar] = None,
    components: Optional[Scalar] = None,
) -> Source:
    """Append volumetric source terms in the given cells."""
    cells = np.asarray(cells, dtype=np.int64).ravel()
    rates_arr = _broadcast_rates(rates, cells.size)
    sat_arr = _broadcast_saturation(sat, cells.size)
    if src is None:
        src = Source()
    if src.sat.shape[1] and src.sat.shape[1] != sat_arr.shape[1]:
        raise ValueError(
            "number of saturation columns does not match existing sources"
        )
    if np.isin(cells, src.cell).any():
        raise ValueError("new source terms conflict with existing source terms")
    comp_arr = None
    if components is not None:
        comp_arr = _broadcast_saturation(components, cells.size)
        if src.components is not None and src.components.shape[1] != comp_arr.shape[1]:
            raise ValueError("component columns do not match existing sources")
    src.cell = np.concatenate([src.cell, cells])
    src.rate = np.concatenate([src.rate, rates_arr])
    src.sat = sat_arr if src.sat.size == 0 else np.vstack([src.sat, sat_arr])
    if comp_arr is not None:
        src.components = comp_arr if src.components is None else np.vstack([src.components, comp_arr])
    elif src.components is not None:
        src.components = np.vstack([src.components, np.zeros((cells.size, src.components.shape[1]))])
    return src


def _broadcast_rates(rates: Scalar, count: int) -> np.ndarray:
    arr = np.asarray(rates, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(count, arr[0])
    if arr.size != count:
        raise ValueError("one rate per cell or one scalar rate required")
    return arr


def _broadcast_saturation(sat: Optional[Scalar], count: int) -> np.ndarray:
    if sat is None:
        return np.empty((0, 0), dtype=float)
    arr = np.atleast_2d(np.asarray(sat, dtype=np.float64))
    if arr.shape[0] == 1:
        arr = np.repeat(arr, count, axis=0)
    if arr.shape[0] not in (0, count):
        raise ValueError("saturation must have one row or one row per cell")
    return arr


if __name__ == "__main__":

    def test_add_source() -> None:
        src = add_source(None, np.array([0, 3]), 1e-3)
        assert np.allclose(src.rate, 1e-3)
        assert src.cell.size == 2
        # `add_source` raises ValueError, not AssertionError. Catching the
        # wrong one let the raise escape and the self-test failed on a working
        # rejection.
        try:
            add_source(src, np.array([3]), 0.0)
        except ValueError as exc:
            assert "conflict" in str(exc), exc
            print("ok test_add_source")
            return
        raise AssertionError("expected conflict rejection")

    test_add_source()
    print("\nAll tests passed!")
