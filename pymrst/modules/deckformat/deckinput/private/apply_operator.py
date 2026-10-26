"""MRST ``deckformat/deckinput/private/applyOperator.m``.

The box operators of a deck section - `ADD`, `COPY`, `EQUALS`, `MAXVALUE`,
`MINVALUE`, `MULTIPLY` - applied to that section's arrays. Each record is
`target value [I1 I2 J1 J2 K1 K2]` (`source target [box]` for `COPY`); the
box is 1-based and inclusive, and entries left out keep the previous
record's box (the first record's default is the active `BOX`, else the whole
grid).

An operator may name an array the section does not have yet: `ADD` starts it
at zeros, `MULTIPLY` at ones, `EQUALS` at NaN (ones for a `MULT*` array), and
`COPY` makes its target zeros. As in MRST, it is the record's first field
that is checked, so a `COPY` from a missing source only warns.

`apply_operator_simple` (MRST's `deckinput_simple/applyOperatorSimple.m`) is
the other, simpler reader's version.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record import (
    expand_defaults,
    strip_quotes,
)

OPERATORS = ("ADD", "COPY", "EQUALS", "MAXVALUE", "MINVALUE", "MULTIPLY")


def apply_operator(sect: Dict[str, Any], records: Sequence[Sequence[str]], kw: str,
                   cart_dims: Sequence[int],
                   grid_box: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """MRST `sect = applyOperator(sect, fid, kw)`, on the operator's records."""
    assert kw in OPERATORS, (
        f"Keyword '{kw}' not amongst supported operators.\nMust be one of\n\t"
        "* ADD\n\t* COPY\n\t* EQUALS\n\t* MAXVALUE\n\t* MINVALUE\n\t* MULTIPLY.")
    dims = [int(d) for d in cart_dims]
    m = int(np.prod(dims))
    box = list(grid_box) if grid_box is not None else [1, dims[0], 1, dims[1], 1, dims[2]]
    sect = dict(sect)
    for fields in records:
        tmpl = ["1", "1"] + [str(int(b)) for b in box]
        rec = expand_defaults([str(f) for f in fields], tmpl)
        if rec == tmpl:
            break
        box = [int(float(v)) for v in rec[2:8]]
        target = strip_quotes(rec[0])
        if target not in sect:
            if kw == "ADD":
                sect[target] = np.zeros(m)
            elif kw == "MULTIPLY":
                sect[target] = np.ones(m)
            elif kw == "EQUALS":
                sect[target] = (np.ones(m) if re.search(r"MULT\w+", target, re.IGNORECASE)
                                else np.full(m, np.nan))
            else:
                warnings.warn(f"No reasonable initial value for operator '{kw}' applied "
                              f"to undefined array '{target}'.")
        idx = box_indices(dims, box)
        sect = _OPS[kw](sect, target, strip_quotes(rec[1]), idx, m)
    return sect


applyOperator = apply_operator


def box_indices(dims: Sequence[int], b: Sequence[int]) -> np.ndarray:
    """0-based linear indices of the 1-based, inclusive box `b`."""
    i, j, k = np.meshgrid(np.arange(b[0], b[1] + 1), np.arange(b[2], b[3] + 1),
                          np.arange(b[4], b[5] + 1), indexing="ij")
    return np.ravel_multi_index((i.ravel(order="F") - 1, j.ravel(order="F") - 1,
                                 k.ravel(order="F") - 1), dims, order="F")


def _array(sect: Dict[str, Any], x: str) -> np.ndarray:
    return np.array(sect[x], dtype=float).ravel()


def _add(sect, x, val, ix, m):
    assert x in sect and np.size(sect[x]) >= m
    a = _array(sect, x)
    assert not np.any(np.isnan(a[ix])), f"Cannot add value to undefined array '{x}'."
    a[ix] = a[ix] + _to_double(val)
    sect[x] = a
    return sect


def _copy(sect, x, y, ix, m):
    assert x in sect and np.size(sect[x]) >= m
    if y not in sect:
        sect[y] = np.zeros(max(np.size(sect[x]), m))
    a = _array(sect, y)
    a[ix] = _array(sect, x)[ix]
    sect[y] = a
    return sect


def _equals(sect, x, val, ix, m):
    assert x in sect
    a = _array(sect, x)
    a[ix] = _to_double(val)
    sect[x] = a
    return sect


def _maxvalue(sect, x, val, ix, m):
    if x in sect:
        a = _array(sect, x)
        v = _to_double(val)
        a[ix] = np.fmin(a[ix], v)
        sect[x] = a
    return sect


def _minvalue(sect, x, val, ix, m):
    if x in sect:
        a = _array(sect, x)
        v = _to_double(val)
        a[ix] = np.fmax(a[ix], v)
        sect[x] = a
    return sect


def _multiply(sect, x, val, ix, m):
    assert x in sect and np.size(sect[x]) >= m
    a = _array(sect, x)
    assert not np.any(np.isnan(a[ix])), f"Cannot multiply undefined array '{x}'."
    a[ix] = a[ix]*_to_double(val)
    sect[x] = a
    return sect


def _to_double(val: str) -> float:
    return float(str(val).replace("D", "e").replace("d", "e"))


_OPS = {"ADD": _add, "COPY": _copy, "EQUALS": _equals, "MAXVALUE": _maxvalue,
        "MINVALUE": _minvalue, "MULTIPLY": _multiply}

__all__ = ["apply_operator", "applyOperator", "box_indices"]


if __name__ == "__main__":
    DIMS = (2, 1, 3)

    def test_equals_creates_an_array_over_its_box():
        out = apply_operator({}, [["'SATNUM'", "2", "1", "2", "1", "1", "2", "3"]],
                             "EQUALS", DIMS)
        np.testing.assert_array_equal(np.isnan(out["SATNUM"]),
                                      [True, True, False, False, False, False])
        assert out["SATNUM"][2] == 2
        out = apply_operator({}, [["MULTZ", "0.5", "1", "1", "1", "1", "1", "1"]],
                             "EQUALS", DIMS)
        np.testing.assert_array_equal(out["MULTZ"], [0.5, 1, 1, 1, 1, 1])
        print("ok test_equals_creates_an_array_over_its_box")

    def test_copy_starts_its_target_at_zero_and_copies_forward():
        sect = {"SWL": np.arange(6.0)}
        out = apply_operator(sect, [["SWL", "SWCR", "1", "2", "1", "1", "3", "3"]],
                             "COPY", DIMS)
        np.testing.assert_array_equal(out["SWCR"], [0, 0, 0, 0, 4, 5])
        assert "SWCR" not in sect
        print("ok test_copy_starts_its_target_at_zero_and_copies_forward")

    def test_an_omitted_box_keeps_the_previous_records():
        sect = {"PERMX": np.ones(6)}
        out = apply_operator(sect, [["PERMX", "2", "1", "1", "1", "1", "2", "3"],
                                    ["PERMX", "3"]], "MULTIPLY", DIMS)
        np.testing.assert_array_equal(out["PERMX"], [1, 1, 6, 1, 6, 1])
        print("ok test_an_omitted_box_keeps_the_previous_records")

    def test_add_and_max_and_min():
        sect = {"A": np.array([0.0, 1, 2, 3, 4, 5])}
        out = apply_operator(sect, [["A", "1"]], "ADD", DIMS)
        out = apply_operator(out, [["A", "4"]], "MAXVALUE", DIMS)
        out = apply_operator(out, [["A", "2"]], "MINVALUE", DIMS)
        np.testing.assert_array_equal(out["A"], [2, 2, 3, 4, 4, 4])
        print("ok test_add_and_max_and_min")

    test_equals_creates_an_array_over_its_box()
    test_copy_starts_its_target_at_zero_and_copies_forward()
    test_an_omitted_box_keeps_the_previous_records()
    test_add_and_max_and_min()
    print("\nAll tests passed!")
