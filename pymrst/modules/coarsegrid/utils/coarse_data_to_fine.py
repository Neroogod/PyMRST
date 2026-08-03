"""Expand coarse values onto the fine grid (MRST ``coarseDataToFine.m``).

One value per block becomes one value per cell, by giving every cell its
block's value. That is the whole operation: ``data[partition]``. What the MRST
file spends its length on is *finding* the arrays to expand, because callers
pass whole states and lists of states rather than bare vectors.

MRST recurses through numeric arrays, struct arrays and cell arrays, expanding
any array whose first dimension is the block count. The port does the same for
the Python shapes those map to: arrays, objects with attributes, dicts, lists
and tuples.

**The length test is the whole safety mechanism, and it is a guess.** An array
is expanded when its leading dimension equals the number of blocks, so an array
that happens to have that many rows for an unrelated reason is expanded too. On
a coarse grid whose block count equals its well count, a well array would be
silently rewritten. MRST has the same exposure. `strict=True` refuses the
ambiguous case instead of guessing.
"""

from __future__ import annotations

from copy import copy

import numpy as np


def coarse_data_to_fine(cg, data, strict: bool = False):
    """Expand every block-sized array in `data` onto the fine cells.

    Containers are copied rather than modified, so the coarse data stays
    usable. Arrays that are not block-sized are passed through untouched.
    """
    partition = np.asarray(cg.partition, dtype=np.int64).ravel()
    n_blocks = int(cg.n_blocks)
    return _expand(data, partition, n_blocks, strict)


def _expand(data, partition, n_blocks, strict):
    if data is None:
        return data
    if isinstance(data, np.ndarray):
        return _expand_array(data, partition, n_blocks, strict)
    if isinstance(data, dict):
        return {k: _expand(v, partition, n_blocks, strict) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        out = [_expand(v, partition, n_blocks, strict) for v in data]
        return type(data)(out) if isinstance(data, tuple) else out
    if isinstance(data, (int, float, complex, bool, str, bytes)):
        return data
    if hasattr(data, "__dict__"):
        out = copy(data)
        for name, value in vars(data).items():
            setattr(out, name, _expand(value, partition, n_blocks, strict))
        return out
    return data


def _expand_array(a: np.ndarray, partition, n_blocks, strict) -> np.ndarray:
    if a.size == 0 or a.ndim == 0:
        return a
    if a.ndim == 1:
        if a.shape[0] != n_blocks:
            return a
        if strict and n_blocks == partition.size:
            raise ValueError(
                "cannot tell a block array from a cell array: the grid has "
                f"{n_blocks} blocks and {partition.size} cells")
        return a[partition]
    if a.shape[0] == n_blocks:
        return a[partition, ...]
    # MRST also expands a 1-by-nblocks row vector, which is how a state field
    # transposed on the way in still arrives correctly.
    if a.ndim == 2 and a.shape == (1, n_blocks):
        return a[0, partition]
    return a


__all__ = ["coarse_data_to_fine"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(4, 4, 1), part=(2, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        p = partition_cart_grid(list(dims), list(part))
        return g, p, generate_coarse_grid(g, p)

    def test_every_cell_gets_its_own_block_value():
        """The defining property, checked cell by cell."""
        g, p, cg = _case()
        block = np.arange(cg.n_blocks, dtype=float) * 10.0
        fine = coarse_data_to_fine(cg, block)
        assert fine.shape == (g.cells.num,), fine.shape
        for c in range(g.cells.num):
            assert fine[c] == block[p[c]], c
        print("ok test_every_cell_gets_its_own_block_value")

    def test_averaging_back_recovers_the_coarse_values():
        """The round trip, which is what makes the expansion checkable.

        A wrong index array still produces a full-length result with plausible
        values; only mapping it back reveals it.
        """
        _, p, cg = _case()
        block = np.array([1.0, 2.5, -3.0, 7.25])
        fine = coarse_data_to_fine(cg, block)
        back = np.array([fine[p == b].mean() for b in range(cg.n_blocks)])
        assert np.allclose(back, block), (back, block)
        print("ok test_averaging_back_recovers_the_coarse_values")

    def test_columns_are_carried_across():
        _, p, cg = _case()
        block = np.arange(cg.n_blocks * 3, dtype=float).reshape(cg.n_blocks, 3)
        fine = coarse_data_to_fine(cg, block)
        assert fine.shape == (p.size, 3), fine.shape
        assert np.array_equal(fine, block[p])
        print("ok test_columns_are_carried_across")

    def test_a_row_vector_is_expanded_too():
        _, p, cg = _case()
        block = np.arange(cg.n_blocks, dtype=float).reshape(1, cg.n_blocks)
        fine = coarse_data_to_fine(cg, block)
        assert fine.shape == (p.size,), fine.shape
        print("ok test_a_row_vector_is_expanded_too")

    def test_arrays_of_other_lengths_are_left_alone():
        """Expanding an unrelated array would corrupt it silently."""
        _, _, cg = _case()
        other = np.arange(7, dtype=float)
        assert np.array_equal(coarse_data_to_fine(cg, other), other)
        print("ok test_arrays_of_other_lengths_are_left_alone")

    def test_a_state_like_object_is_walked():
        class State:
            def __init__(self, n):
                self.pressure = np.arange(n, dtype=float)
                self.s = np.column_stack([np.ones(n), np.zeros(n)])
                self.name = "coarse"

        _, p, cg = _case()
        st = State(cg.n_blocks)
        out = coarse_data_to_fine(cg, st)
        assert out.pressure.shape == (p.size,)
        assert out.s.shape == (p.size, 2)
        assert out.name == "coarse", "a string was mangled"
        assert st.pressure.shape == (cg.n_blocks,), "the input was modified"
        print("ok test_a_state_like_object_is_walked")

    def test_dicts_and_lists_are_walked():
        _, p, cg = _case()
        data = {"p": np.arange(cg.n_blocks, dtype=float),
                "wells": [np.arange(cg.n_blocks, dtype=float), "name"]}
        out = coarse_data_to_fine(cg, data)
        assert out["p"].shape == (p.size,)
        assert out["wells"][0].shape == (p.size,)
        assert out["wells"][1] == "name"
        print("ok test_dicts_and_lists_are_walked")

    def test_a_list_of_states_is_walked():
        _, p, cg = _case()
        states = [{"pressure": np.full(cg.n_blocks, float(i))} for i in range(3)]
        out = coarse_data_to_fine(cg, states)
        assert len(out) == 3
        for i, s in enumerate(out):
            assert s["pressure"].shape == (p.size,)
            assert np.all(s["pressure"] == float(i))
        print("ok test_a_list_of_states_is_walked")

    def test_empty_input_is_returned():
        _, _, cg = _case()
        assert coarse_data_to_fine(cg, None) is None
        empty = np.array([])
        assert coarse_data_to_fine(cg, empty).size == 0
        print("ok test_empty_input_is_returned")

    def test_the_ambiguous_case_can_be_refused():
        """One cell per block makes the length test meaningless.

        Nothing distinguishes a block array from a cell array then, and the
        expansion is a permutation rather than a no-op, so the values move.
        """
        _, _, cg = _case(dims=(2, 2, 1), part=(2, 2, 1))
        assert cg.n_blocks == 4
        block = np.arange(4, dtype=float)
        assert coarse_data_to_fine(cg, block).size == 4
        try:
            coarse_data_to_fine(cg, block, strict=True)
        except ValueError as exc:
            assert "cannot tell" in str(exc), exc
        else:
            raise AssertionError("expected strict mode to refuse the ambiguity")
        print("ok test_the_ambiguous_case_can_be_refused")

    test_every_cell_gets_its_own_block_value()
    test_averaging_back_recovers_the_coarse_values()
    test_columns_are_carried_across()
    test_a_row_vector_is_expanded_too()
    test_arrays_of_other_lengths_are_left_alone()
    test_a_state_like_object_is_walked()
    test_dicts_and_lists_are_walked()
    test_a_list_of_states_is_walked()
    test_empty_input_is_returned()
    test_the_ambiguous_case_can_be_refused()
    print("\nAll tests passed!")
