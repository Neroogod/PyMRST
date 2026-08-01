"""Connected components of a cell subset (MRST ``connectedCells.m``).

Splits the named cells into groups that are reachable from one another through
shared faces *without leaving the subset*. A fault that fully severs a region,
or a subset selected by a property threshold, produces more than one group -
and a flow solve on a disconnected region needs a pressure datum in each.

**MRST reaches this through `dmperm`**, MATLAB's Dulmage-Mendelsohn
decomposition, which finds the strongly connected components of a bipartite
matching as a side effect. That is a general-purpose tool used here for a
special case; `scipy.sparse.csgraph.connected_components` answers the actual
question directly and is what this uses. It is the one place the plan for this
module said to reach for SciPy rather than port a MATLAB graph routine, and
the reason is that `dmperm` lives in the vendored Boost Graph Library binding
that is out of scope.

Groups come back sorted by their smallest member, so the order is
deterministic - `connected_components` does not otherwise promise one.
"""

from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def connected_cells(g, cells=None) -> List[np.ndarray]:
    """Return the connected groups within ``cells``, as arrays of cell indices."""
    if cells is None:
        cells = np.arange(g.cells.num)
    else:
        cells = np.asarray(cells)
        if cells.dtype == bool:
            if cells.size != g.cells.num:
                raise ValueError(
                    f"a logical subset needs {g.cells.num} entries, "
                    f"got {cells.size}")
            cells = np.flatnonzero(cells)
        else:
            cells = np.atleast_1d(cells.astype(np.int64)).ravel()
    if cells.size == 0:
        return []

    # Renumber the subset to 0..n-1 so the graph has no gaps.
    local = np.full(g.cells.num, -1, dtype=np.int64)
    local[cells] = np.arange(cells.size)

    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    interior = neighbours[np.all(neighbours >= 0, axis=1)]
    left, right = local[interior[:, 0]], local[interior[:, 1]]
    both = (left >= 0) & (right >= 0)
    left, right = left[both], right[both]

    graph = sp.coo_array(
        (np.ones(left.size), (left, right)),
        shape=(cells.size, cells.size)).tocsr()
    count, labels = connected_components(graph, directed=False)

    groups = [cells[labels == label] for label in range(count)]
    return sorted(groups, key=lambda block: block.min())


connectedCells = connected_cells


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    G = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))

    def test_a_whole_grid_is_one_component() -> None:
        groups = connected_cells(G)
        assert len(groups) == 1 and groups[0].size == G.cells.num
        print("ok test_a_whole_grid_is_one_component")

    def test_a_split_subset_gives_two_components() -> None:
        """Columns 0 and 3 of a 4x3 grid share no face."""
        left = [0, 4, 8]
        right = [3, 7, 11]
        groups = connected_cells(G, left + right)
        assert len(groups) == 2, [b.tolist() for b in groups]
        assert set(groups[0].tolist()) == set(left)
        assert set(groups[1].tolist()) == set(right)
        print("ok test_a_split_subset_gives_two_components")

    def test_a_contiguous_subset_stays_one() -> None:
        groups = connected_cells(G, [0, 1, 2, 3])
        assert len(groups) == 1, [b.tolist() for b in groups]
        print("ok test_a_contiguous_subset_stays_one")

    def test_every_cell_appears_exactly_once() -> None:
        groups = connected_cells(G, [0, 1, 5, 7, 11])
        merged = np.concatenate(groups)
        assert merged.size == len(set(merged.tolist())) == 5
        print("ok test_every_cell_appears_exactly_once")

    def test_an_isolated_cell_is_its_own_group() -> None:
        groups = connected_cells(G, [0, 11])
        assert len(groups) == 2 and all(b.size == 1 for b in groups)
        print("ok test_an_isolated_cell_is_its_own_group")

    def test_the_groups_are_ordered_by_their_smallest_member() -> None:
        groups = connected_cells(G, [11, 7, 3, 0, 4, 8])
        firsts = [int(b.min()) for b in groups]
        assert firsts == sorted(firsts), firsts
        print("ok test_the_groups_are_ordered_by_their_smallest_member")

    def test_connectivity_is_judged_within_the_subset() -> None:
        """Cells 0 and 2 are joined through cell 1 - but only if 1 is in."""
        assert len(connected_cells(G, [0, 1, 2])) == 1
        assert len(connected_cells(G, [0, 2])) == 2
        print("ok test_connectivity_is_judged_within_the_subset")

    def test_a_logical_mask_is_accepted() -> None:
        mask = np.zeros(G.cells.num, dtype=bool)
        mask[[0, 1, 2]] = True
        assert len(connected_cells(G, mask)) == 1
        print("ok test_a_logical_mask_is_accepted")

    def test_an_empty_subset_has_no_groups() -> None:
        assert connected_cells(G, []) == []
        print("ok test_an_empty_subset_has_no_groups")

    def test_a_mis_sized_mask_is_rejected() -> None:
        try:
            connected_cells(G, np.ones(3, dtype=bool))
        except ValueError as error:
            assert "logical subset needs" in str(error)
            print("ok test_a_mis_sized_mask_is_rejected")
            return
        raise AssertionError("a mis-sized mask must be rejected")

    test_a_whole_grid_is_one_component()
    test_a_split_subset_gives_two_components()
    test_a_contiguous_subset_stays_one()
    test_every_cell_appears_exactly_once()
    test_an_isolated_cell_is_its_own_group()
    test_the_groups_are_ordered_by_their_smallest_member()
    test_connectivity_is_judged_within_the_subset()
    test_a_logical_mask_is_accepted()
    test_an_empty_subset_has_no_groups()
    test_a_mis_sized_mask_is_rejected()
    print("\nAll tests passed!")
