"""Expand a compressed PVT table (MRST ``utils/preprocessTablePVT.m``).

A deck stores a PVTO/PVTG table in packed form: one flat `data` array, a `key`
per branch and a `pos` array marking where each branch starts. This expands
that into one table per branch, plus the saturated table formed from the first
row of each branch - which is the locus of bubble points.

Branches stored with decreasing `x` are reversed, because the interpolators
assume increasing abscissae. PVTG decks routinely store them the other way
round, so skipping this gives silently wrong gas properties.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Branch:
    """One branch of a PVT table: abscissa ``x`` and its property columns."""

    x: np.ndarray
    F: np.ndarray


@dataclass
class PackedPVT:
    """A deck's packed PVT table, before expansion."""

    key: np.ndarray
    pos: np.ndarray
    data: np.ndarray
    expanded: List[Branch] = field(default_factory=list)
    sat: Branch = None


def preprocess_table_pvt(tab: PackedPVT) -> PackedPVT:
    """Fill ``tab.expanded`` and ``tab.sat`` from its packed form."""
    key = np.asarray(tab.key, dtype=float).ravel()
    pos = np.asarray(tab.pos, dtype=np.int64).ravel()
    data = np.asarray(tab.data, dtype=float)

    expanded: List[Branch] = []
    for index in range(key.size):
        block = data[pos[index]:pos[index + 1], :]
        if block.shape[0] > 1 and block[0, 0] > block[1, 0]:
            block = block[::-1]                  # deck stored it decreasing
        expanded.append(Branch(x=block[:, 0].copy(), F=block[:, 1:].copy()))

    saturated = data[pos[:-1], :]
    tab.expanded = expanded
    tab.sat = Branch(x=saturated[:, 0].copy(), F=saturated[:, 1:].copy())
    return tab


preprocessTablePVT = preprocess_table_pvt


if __name__ == "__main__":

    def _packed():
        """Two branches of two rows each, the second stored decreasing."""
        data = np.array([
            [1.0, 1.10, 1.0e-3],      # branch 0, increasing
            [2.0, 1.08, 1.1e-3],
            [3.0, 1.27, 1.0e-3],      # branch 1, decreasing in x
            [2.0, 1.30, 0.9e-3],
        ])
        return PackedPVT(key=np.array([0.0, 100.0]),
                         pos=np.array([0, 2, 4]), data=data)

    def test_expands_one_branch_per_key() -> None:
        tab = preprocess_table_pvt(_packed())
        assert len(tab.expanded) == 2
        assert tab.expanded[0].F.shape == (2, 2)
        print("ok test_expands_one_branch_per_key")

    def test_decreasing_branches_are_reversed() -> None:
        tab = preprocess_table_pvt(_packed())
        for index, branch in enumerate(tab.expanded):
            assert np.all(np.diff(branch.x) > 0), (index, branch.x)
        # the reversed branch keeps its rows paired with their x
        assert np.allclose(tab.expanded[1].x, [2.0, 3.0])
        assert np.allclose(tab.expanded[1].F[:, 0], [1.30, 1.27])
        print("ok test_decreasing_branches_are_reversed")

    def test_saturated_table_is_the_first_row_of_each_branch() -> None:
        tab = preprocess_table_pvt(_packed())
        # taken from the packed data, before any reversal, as MRST does
        assert np.allclose(tab.sat.x, [1.0, 3.0]), tab.sat.x
        assert np.allclose(tab.sat.F[:, 0], [1.10, 1.27])
        print("ok test_saturated_table_is_the_first_row_of_each_branch")

    def test_already_increasing_branches_are_untouched() -> None:
        data = np.array([[1.0, 5.0], [2.0, 6.0]])
        tab = preprocess_table_pvt(
            PackedPVT(key=np.array([0.0]), pos=np.array([0, 2]), data=data))
        assert np.allclose(tab.expanded[0].x, [1.0, 2.0])
        assert np.allclose(tab.expanded[0].F[:, 0], [5.0, 6.0])
        print("ok test_already_increasing_branches_are_untouched")

    def test_single_row_branches_survive() -> None:
        data = np.array([[1.0, 5.0], [2.0, 6.0]])
        tab = preprocess_table_pvt(
            PackedPVT(key=np.array([0.0, 1.0]), pos=np.array([0, 1, 2]),
                      data=data))
        assert len(tab.expanded) == 2
        assert tab.expanded[0].x.size == 1
        print("ok test_single_row_branches_survive")

    test_expands_one_branch_per_key()
    test_decreasing_branches_are_reversed()
    test_saturated_table_is_the_first_row_of_each_branch()
    test_already_increasing_branches_are_untouched()
    test_single_row_branches_survive()
    print("\nAll tests passed!")
