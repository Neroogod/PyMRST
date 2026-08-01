"""Dispatch along the second index (MRST ``tensorAssembly/utils/tbldispatch2.m``).

`u[indstruct[1].inds]`. See `tbldispatch1`, which is the same for the other
side; the pair exists so a call site names which index array its vector lives
on.
"""

from pymrst.utils.tensor_assembly.utils.tbldispatch1 import _dispatch


def tbldispatch2(u, indstruct):
    """Scatter ``u`` from the second index array onto the join."""
    return _dispatch(u, indstruct, 1)


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    from pymrst.utils.tensor_assembly.utils.tbldispatch1 import tbldispatch1

    def struct(a, b):
        return [SimpleNamespace(inds=np.asarray(a)), SimpleNamespace(inds=np.asarray(b))]

    def test_it_scatters_along_the_second_index() -> None:
        got = tbldispatch2(np.array([10.0, 20.0, 30.0]), struct([0, 0, 2], [1, 0, 1]))
        assert got.tolist() == [20.0, 10.0, 20.0]
        print("ok test_it_scatters_along_the_second_index")

    def test_it_is_the_mirror_of_tbldispatch1() -> None:
        s = struct([2, 0, 1], [1, 1, 0])
        u = np.array([5.0, 6.0, 7.0])
        assert tbldispatch2(u, s).tolist() == tbldispatch1(u, s[::-1]).tolist()
        print("ok test_it_is_the_mirror_of_tbldispatch1")

    def test_a_missing_second_side_is_rejected() -> None:
        try:
            tbldispatch2(np.arange(2.0), [np.array([0])])
        except ValueError as error:
            assert "side 2 was asked for" in str(error)
            print("ok test_a_missing_second_side_is_rejected")
            return
        raise AssertionError("a one-sided index structure must be rejected")

    test_it_scatters_along_the_second_index()
    test_it_is_the_mirror_of_tbldispatch1()
    test_a_missing_second_side_is_rejected()
    print("\nAll tests passed!")
