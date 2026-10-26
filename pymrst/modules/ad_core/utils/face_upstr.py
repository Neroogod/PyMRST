"""MRST ``ad-core/utils/faceUpstr.m``: single-point upstream weighting.

For each interface the value of the first cell of `N` is taken where `flag`
is true and the second otherwise. A scalar flag applies to every interface.
Numeric values are indexed row-wise (one row per cell, any number of
columns); AD values are multiplied by the selection matrix, as in MRST.
"""

from typing import Any, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp


def face_upstr(flag: Any, x: Any, N: np.ndarray, sz: Sequence[int],
               nargout: int = 1) -> Union[Any, Tuple[Any, Any]]:
    """MRST `[xu, xc] = faceUpstr(flag, x, N, sz)`; `nargout=2` also returns `x`."""
    from pymrst.utils.adi import ad_matmul
    N = np.asarray(N, dtype=np.int64)
    flag = np.asarray(flag)
    if flag.size == 1:
        flag = np.repeat(flag.ravel(), N.shape[0])
    assert flag.size == N.shape[0] and flag.dtype == bool, (
        "One logical upstream flag must' be supplied per interface.")
    flag = flag.ravel()
    up_cell = N[:, 1].copy()
    up_cell[flag] = N[flag, 0]
    if isinstance(x, np.ndarray) or np.isscalar(x):
        xu = np.asarray(x)[up_cell]
    else:
        nf, nc = int(sz[0]), int(sz[1])
        select = sp.csr_array((np.ones(nf), (np.arange(nf), up_cell)), shape=(nf, nc))
        xu = ad_matmul(select, x)
    return (xu, x) if nargout > 1 else xu


faceUpstr = face_upstr
__all__ = ["faceUpstr", "face_upstr"]


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi, value

    N = np.array([[0, 1], [1, 2]])

    def test_numeric_values_pick_the_flagged_side():
        x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        xu = face_upstr(np.array([True, False]), x, N, (2, 3))
        assert np.array_equal(xu, [[1.0, 10.0], [3.0, 30.0]])
        print("ok test_numeric_values_pick_the_flagged_side")

    def test_scalar_flag_and_second_output():
        x = np.array([1.0, 2.0, 3.0])
        xu, xc = face_upstr(False, x, N, (2, 3), nargout=2)
        assert np.array_equal(xu, [2.0, 3.0]) and xc is x
        print("ok test_scalar_flag_and_second_output")

    def test_ad_values_carry_the_selection_jacobian():
        (p,) = init_variables_adi(np.array([1.0, 2.0, 3.0]))
        u = face_upstr(np.array([True, False]), p, N, (2, 3))
        assert np.allclose(value(u), [1.0, 3.0])
        assert np.allclose(u.jac[0].toarray(), [[1, 0, 0], [0, 0, 1]])
        print("ok test_ad_values_carry_the_selection_jacobian")

    def test_flags_must_be_logical_one_per_interface():
        try:
            face_upstr(np.array([1, 0]), np.ones(3), N, (2, 3))
        except AssertionError:
            print("ok test_flags_must_be_logical_one_per_interface")
            return
        raise AssertionError("expected an AssertionError")

    test_numeric_values_pick_the_flagged_side()
    test_scalar_flag_and_second_output()
    test_ad_values_carry_the_selection_jacobian()
    test_flags_must_be_logical_one_per_interface()
    print("\nAll tests passed!")
