"""In-place arithmetic on a subset (MRST ``utils/subsetPlus.m``, ``subsetMinus.m``).

`u[subs] += v` and `u[subs] -= v`, written as functions so an AD object can
override them - the ADI class needs to update a slice of a value *and* the
matching rows of every Jacobian, which a bare `+=` cannot express.

**The two are not symmetric in MRST, and that is reproduced here.**
`subsetPlus` indexes `u(subs, :)`, so it works on a matrix and updates whole
rows; `subsetMinus` indexes `u(subs)`, so it is one-dimensional. Making them
match would be a change in behaviour for any caller passing a matrix to the
minus form, which currently addresses it in column-major linear order.
"""

import numpy as np


def subset_plus(u, v, subs):
    """``u[subs, :] += v``, returning ``u``."""
    u = np.array(u, copy=True)
    # An empty list is float64 by default, which numpy refuses as an index.
    index = np.asarray(subs, dtype=np.int64)
    if u.ndim > 1:
        u[index, :] = u[index, :] + v
    else:
        u[index] = u[index] + np.asarray(v).ravel()
    return u


def subset_minus(u, v, subs):
    """``u[subs] -= v``, returning ``u``.

    One-dimensional, unlike `subset_plus`; see the module docstring.
    """
    u = np.array(u, copy=True)
    index = np.asarray(subs, dtype=np.int64)
    flat = u.reshape(-1, order="F") if u.ndim > 1 else u
    flat[index] = flat[index] - np.asarray(v).ravel()
    return flat.reshape(u.shape, order="F") if u.ndim > 1 else flat


subsetPlus = subset_plus
subsetMinus = subset_minus


if __name__ == "__main__":

    def test_plus_adds_to_the_named_rows() -> None:
        u = np.zeros(5)
        assert np.allclose(subset_plus(u, np.array([1.0, 2.0]), [1, 3]),
                           [0, 1, 0, 2, 0])
        print("ok test_plus_adds_to_the_named_rows")

    def test_minus_subtracts_from_the_named_entries() -> None:
        u = np.ones(4)
        assert np.allclose(subset_minus(u, np.array([1.0, 3.0]), [0, 2]),
                           [0, 1, -2, 1])
        print("ok test_minus_subtracts_from_the_named_entries")

    def test_the_input_is_not_modified() -> None:
        u = np.zeros(3)
        subset_plus(u, np.array([1.0]), [0])
        subset_minus(u, np.array([1.0]), [0])
        assert np.allclose(u, 0.0), "the caller's array was mutated"
        print("ok test_the_input_is_not_modified")

    def test_plus_updates_whole_rows_of_a_matrix() -> None:
        u = np.zeros((3, 2))
        out = subset_plus(u, np.array([[1.0, 2.0]]), [1])
        assert np.allclose(out[1], [1.0, 2.0]) and np.allclose(out[0], 0.0)
        print("ok test_plus_updates_whole_rows_of_a_matrix")

    def test_minus_is_one_dimensional_on_a_matrix() -> None:
        """MRST's asymmetry: plus takes rows, minus takes linear indices."""
        u = np.zeros((3, 2))
        out = subset_minus(u, np.array([1.0]), [1])
        assert out[1, 0] == -1.0, out
        assert np.count_nonzero(out) == 1, out
        print("ok test_minus_is_one_dimensional_on_a_matrix")

    def test_repeated_indices_take_the_last_write() -> None:
        """Plain assignment semantics, not accumulation."""
        out = subset_plus(np.zeros(3), np.array([1.0, 2.0]), [0, 0])
        assert out[0] == 2.0, out
        print("ok test_repeated_indices_take_the_last_write")

    def test_an_empty_subset_changes_nothing() -> None:
        u = np.arange(4.0)
        assert np.allclose(subset_plus(u, np.zeros(0), []), u)
        print("ok test_an_empty_subset_changes_nothing")

    test_plus_adds_to_the_named_rows()
    test_minus_subtracts_from_the_named_entries()
    test_the_input_is_not_modified()
    test_plus_updates_whole_rows_of_a_matrix()
    test_minus_is_one_dimensional_on_a_matrix()
    test_repeated_indices_take_the_last_write()
    test_an_empty_subset_changes_nothing()
    print("\nAll tests passed!")
