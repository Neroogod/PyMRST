"""Dispatch along the first index (MRST ``tensorAssembly/utils/tbldispatch1.m``).

`u[indstruct[0].inds]` - scatter a vector defined on the first of two index
arrays onto the rows of their join. `tbldispatch2` does the same for the
second. They are one line each in MRST and kept separate here for the same
reason: at a call site, `tbldispatch1(u, s)` says which side `u` lives on, where
an inline index expression would not.
"""

import numpy as np


def _dispatch(u, indstruct, side: int):
    if len(indstruct) <= side:
        raise ValueError(
            f"the index structure has {len(indstruct)} entries; side "
            f"{side + 1} was asked for")
    entry = indstruct[side]
    # `cross_index_array` returns dicts; MRST's structs arrive as objects with
    # an `.inds` attribute; a bare index array is also accepted.
    if isinstance(entry, dict):
        entry = entry["inds"]
    else:
        entry = getattr(entry, "inds", entry)
    inds = np.asarray(entry, dtype=np.int64).reshape(-1)
    u = np.asarray(u)
    if inds.size and (inds.max() >= u.shape[0] or inds.min() < 0):
        raise ValueError(
            f"dispatch index range [{inds.min()}, {inds.max()}] does not fit a "
            f"vector of length {u.shape[0]}")
    return u[inds]


def tbldispatch1(u, indstruct):
    """Scatter ``u`` from the first index array onto the join."""
    return _dispatch(u, indstruct, 0)


if __name__ == "__main__":
    from types import SimpleNamespace

    def struct(a, b):
        return [SimpleNamespace(inds=np.asarray(a)), SimpleNamespace(inds=np.asarray(b))]

    def test_it_scatters_along_the_first_index() -> None:
        got = tbldispatch1(np.array([10.0, 20.0, 30.0]), struct([0, 0, 2], [1, 0, 1]))
        assert got.tolist() == [10.0, 10.0, 30.0]
        print("ok test_it_scatters_along_the_first_index")

    def test_it_ignores_the_second_index() -> None:
        a = tbldispatch1(np.arange(3.0), struct([2, 1], [0, 0]))
        b = tbldispatch1(np.arange(3.0), struct([2, 1], [9, 9]))
        assert a.tolist() == b.tolist()
        print("ok test_it_ignores_the_second_index")

    def test_it_works_on_a_raw_array() -> None:
        assert tbldispatch1(np.arange(4.0), [np.array([3, 0])]).tolist() == [3.0, 0.0]
        print("ok test_it_works_on_a_raw_array")

    def test_it_accepts_the_dicts_cross_index_array_returns() -> None:
        """The form actually produced by this package, as opposed to MRST's
        structs - missing it made the helper unusable with its own join."""
        got = tbldispatch1(np.array([10.0, 20.0, 30.0]),
                           [{"inds": np.array([2, 0]), "num": 3},
                            {"inds": np.array([0, 0]), "num": 3}])
        assert got.tolist() == [30.0, 10.0]
        print("ok test_it_accepts_the_dicts_cross_index_array_returns")

    def test_an_out_of_range_index_is_rejected() -> None:
        try:
            tbldispatch1(np.arange(2.0), struct([5], [0]))
        except ValueError as error:
            assert "does not fit" in str(error)
            print("ok test_an_out_of_range_index_is_rejected")
            return
        raise AssertionError("an out-of-range dispatch must be rejected")

    def test_a_missing_side_is_rejected() -> None:
        try:
            tbldispatch1(np.arange(2.0), [])
        except ValueError as error:
            assert "side 1 was asked for" in str(error)
            print("ok test_a_missing_side_is_rejected")
            return
        raise AssertionError("an empty index structure must be rejected")

    test_it_scatters_along_the_first_index()
    test_it_ignores_the_second_index()
    test_it_works_on_a_raw_array()
    test_it_accepts_the_dicts_cross_index_array_returns()
    test_an_out_of_range_index_is_rejected()
    test_a_missing_side_is_rejected()
    print("\nAll tests passed!")
