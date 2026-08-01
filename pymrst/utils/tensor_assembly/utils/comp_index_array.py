"""Compare two index arrays (MRST ``tensorAssembly/utils/compIndexArray.m``).

True when both describe the same set of multi-indices - same index names, same
rows, in any order and any column order.

**MRST's version cannot run, and would be wrong if it could.** Two separate
defects, both verified against MATLAB rather than read off the source:

1. It opens with `if tbl1.isvirtual || tbl2.isvirtual`. The `IndexArray`
   constructor only assigns `isvirtual` when the array *is* virtual, so for any
   ordinary one the property is `[]`, and `[] || []` raises *"Operands to the
   logical and (&&) and or (||) operators must be convertible to logical scalar
   values"*. Every call fails on that line.
2. Forced past it, the comparison reads

       mat1 = sortrows(tbl1.inds);
       mat2 = sortrows(tbl1.inds(:, fdind2));

   - `tbl1` twice. It compares the first array against a column-permutation of
   itself and never looks at the second, so it returns **true for any two
   arrays sharing field names**. Confirmed: two arrays with completely
   different contents compare equal.

This implements the evident intent. It also returns `None` rather than `NaN`
for the virtual case, since "unknown" is not a number.
"""

from typing import Optional

import numpy as np


def comp_index_array(tbl1, tbl2) -> Optional[bool]:
    """Do ``tbl1`` and ``tbl2`` hold the same multi-indices? ``None`` if virtual."""
    if tbl1.isvirtual or tbl2.isvirtual:
        return None

    names1, names2 = list(tbl1.fdnames), list(tbl2.fdnames)
    if sorted(names1) != sorted(names2):
        return False
    if tbl1.num != tbl2.num:
        return False
    if tbl1.num == 0:
        return True

    # Line the second array's columns up with the first's, then compare the
    # row sets - neither row order nor column order is meaningful.
    left = np.asarray(tbl1.inds, dtype=np.int64)
    right = tbl2.gets(names1)
    return bool(np.array_equal(_sorted_rows(left), _sorted_rows(right)))


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort(tuple(a[:, i] for i in reversed(range(a.shape[1]))))]


compIndexArray = comp_index_array


if __name__ == "__main__":
    from pymrst.utils.tensor_assembly.index_array import IndexArray

    def table(cells=(0, 1, 2), faces=(3, 4, 5)):
        return IndexArray({"cells": list(cells), "faces": list(faces)})

    def test_an_array_equals_itself() -> None:
        assert comp_index_array(table(), table()) is True
        print("ok test_an_array_equals_itself")

    def test_different_contents_are_not_equal() -> None:
        """MRST returns true here - verified in MATLAB. See the docstring."""
        assert comp_index_array(table(), table(cells=(9, 9, 9),
                                               faces=(9, 9, 9))) is False
        print("ok test_different_contents_are_not_equal")

    def test_row_order_does_not_matter() -> None:
        a = table(cells=(0, 1, 2), faces=(3, 4, 5))
        b = table(cells=(2, 0, 1), faces=(5, 3, 4))
        assert comp_index_array(a, b) is True
        print("ok test_row_order_does_not_matter")

    def test_column_order_does_not_matter() -> None:
        a = IndexArray({"cells": [0, 1], "faces": [3, 4]})
        b = IndexArray({"faces": [3, 4], "cells": [0, 1]})
        assert comp_index_array(a, b) is True
        print("ok test_column_order_does_not_matter")

    def test_a_permuted_pairing_is_not_equal() -> None:
        """Same values per column, different pairing - must not compare equal."""
        a = IndexArray({"cells": [0, 1], "faces": [3, 4]})
        b = IndexArray({"cells": [0, 1], "faces": [4, 3]})
        assert comp_index_array(a, b) is False
        print("ok test_a_permuted_pairing_is_not_equal")

    def test_different_names_are_not_equal() -> None:
        a = IndexArray({"cells": [0], "faces": [3]})
        b = IndexArray({"cells": [0], "nodes": [3]})
        assert comp_index_array(a, b) is False
        print("ok test_different_names_are_not_equal")

    def test_different_lengths_are_not_equal() -> None:
        assert comp_index_array(table(), table(cells=(0, 1), faces=(3, 4))) is False
        print("ok test_different_lengths_are_not_equal")

    def test_two_empty_arrays_are_equal() -> None:
        a = IndexArray(fdnames=["cells"], inds=np.zeros((0, 1), dtype=np.int64))
        b = IndexArray(fdnames=["cells"], inds=np.zeros((0, 1), dtype=np.int64))
        assert comp_index_array(a, b) is True
        print("ok test_two_empty_arrays_are_equal")

    def test_a_virtual_array_gives_none() -> None:
        """MRST returns NaN; 'unknown' is not a number."""
        v = IndexArray(fdnames=["cells"], isvirtual=True, num=4)
        assert comp_index_array(v, table()) is None
        print("ok test_a_virtual_array_gives_none")

    test_an_array_equals_itself()
    test_different_contents_are_not_equal()
    test_row_order_does_not_matter()
    test_column_order_does_not_matter()
    test_a_permuted_pairing_is_not_equal()
    test_different_names_are_not_equal()
    test_different_lengths_are_not_equal()
    test_two_empty_arrays_are_equal()
    test_a_virtual_array_gives_none()
    print("\nAll tests passed!")
