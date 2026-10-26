"""Merge two ordered index sets from a common superset (MRST
`mergeOrderedArrays`).

Two arrays list elements of the same superset in the same order, but each may be
missing elements the other has - the classic case is a well's perforation list
changing between control steps, where MRST must reconcile the old and new cell
orders into one consistent index vector. Two pointers walk the arrays in step:
where they agree the element is emitted once; where they differ, whichever
element reappears later in the *other* array is deferred and the other is taken
now. If neither can be placed the merge is ambiguous and it falls back to the
sorted union (with a warning), exactly as MRST does.

The 1-based pointer bookkeeping is kept as in the MATLAB source so the ported
behaviour - including its end-of-array shortcuts - matches number for number.

MRST's routine is fragile on inputs its callers never send: the end-of-array
shortcuts compare a pointer *index* against an element *value*, and its
subset early-return does `all(new == old)` on possibly-different sizes. Both can
raise inside MATLAB (verified: `mergeOrderedArrays([1,2,3,4],[1,3,4])` and
`([1,2,4],[1,3,4])` both error). This port reproduces the supported "new is old
with element(s) inserted" case number for number; on the degenerate inputs it
raises rather than returning a silently-wrong ordering, which is what the
callers rely on anyway.
"""

import warnings

import numpy as np


def _search_forward(d: np.ndarray, val) -> float:
    """1-based offset of the first `val` in `d`, or +inf if absent."""
    pos = np.where(d == val)[0]
    return float(pos[0] + 1) if pos.size else np.inf


def merge_ordered_arrays(old, new) -> np.ndarray:
    """Merge `old` and `new` into one consistent ordering of their union."""
    old = np.asarray(old).ravel()
    new = np.asarray(new).ravel()
    combined = np.unique(np.concatenate([old, new]))
    n_total = combined.size

    if old.size == 0:
        return new.copy()
    if new.size == 0:
        return old.copy()
    if np.setdiff1d(new, old).size == 0:
        if not (new.size == old.size and np.all(new == old)):
            warnings.warn("Index sets are permuted versions of each other. "
                          "Taking old ordering uncritically.")
        return old.copy()

    ind = np.full(n_total, np.nan)
    i_old = 1
    i_new = 1
    nn = new.size

    for i in range(1, n_total + 1):
        nv = new[i_new - 1]
        no = old[i_old - 1]
        if nv == no:
            ind[i - 1] = nv
            i_new += 1
            i_old += 1
        else:
            o_old = _search_forward(old[i_old:], nv)
            o_new = _search_forward(new[i_new:], no)
            if np.isinf(o_new):
                ind[i - 1] = no
                i_old += 1
            elif np.isinf(o_old):
                ind[i - 1] = nv
                i_new += 1
            else:
                warnings.warn("Unable to correctly reassign indices based on "
                              "given data. Consider sorting them first.")
                return np.unique(np.concatenate([old, new]))
            assert not (np.isinf(o_old) and np.isinf(o_new))

        if i_new > nn and i_old < no:
            ind[i:] = old[i_old - 1:]
            break
        if i_new < nn and i_old > no:
            ind[i:] = new[i_new - 1:]
            break
    return ind


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
mergeOrderedArrays = merge_ordered_arrays

if __name__ == "__main__":
    def test_identical_arrays_pass_through():
        a = np.array([1, 2, 3, 4])
        assert np.array_equal(merge_ordered_arrays(a, a), a)
        print("ok test_identical_arrays_pass_through")

    def test_empty_returns_the_other():
        a = np.array([5, 7, 9])
        assert np.array_equal(merge_ordered_arrays(np.array([]), a), a)
        assert np.array_equal(merge_ordered_arrays(a, np.array([])), a)
        print("ok test_empty_returns_the_other")

    def test_new_inserts_one_element_before_the_tail():
        """The supported case: `new` is `old` with 4 inserted before 3. MRST
        returns [1, 2, 4, 3] on this input; so must the port."""
        out = merge_ordered_arrays(np.array([1, 2, 3]), np.array([1, 2, 4, 3]))
        assert np.array_equal(out, [1, 2, 4, 3]), out
        print("ok test_new_inserts_one_element_before_the_tail")

    def test_new_inserts_one_element_in_the_middle():
        out = merge_ordered_arrays(np.array([10, 20, 30]),
                                   np.array([10, 15, 20, 30]))
        assert np.array_equal(out, [10, 15, 20, 30]), out
        print("ok test_new_inserts_one_element_in_the_middle")

    def test_degenerate_input_raises_like_mrst():
        """Inputs MRST itself errors on (its index/value shortcut misfires)
        raise here too, rather than returning a silently-wrong ordering."""
        try:
            merge_ordered_arrays(np.array([1, 2, 4]), np.array([1, 3, 4]))
        except (ValueError, AssertionError):
            print("ok test_degenerate_input_raises_like_mrst")
            return
        raise AssertionError("degenerate merge should raise, not mis-order")

    test_identical_arrays_pass_through()
    test_empty_returns_the_other()
    test_new_inserts_one_element_before_the_tail()
    test_new_inserts_one_element_in_the_middle()
    test_degenerate_input_raises_like_mrst()
    print("\nAll tests passed!")
