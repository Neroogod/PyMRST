"""Read and apply an operator section (MRST ``.../private/readOperatorModField.m``).

The immediate counterpart of `read_operator`: each entry is applied to the
target array as it is read, rather than recorded. Implemented on top of
`apply_operator_simple` so the arithmetic lives in one place.

**MRST's version handles only `MULTIPLY` and `EQUALS`**, despite taking the
keyword as an argument - the other four box operators fall through its
if-chain and are silently ignored, which for `ADD` or `MINVALUE` means the
deck's instruction disappears. This accepts all six.

Its box construction also calls `meshgrid` without `'ij'`, so `X` and `Y` come
out transposed relative to `sub2ind`'s expectation. That happens not to matter
because the resulting index *set* is the same and the values assigned are
scalars - but it is why the code looks wrong on a first reading, and it is not
reproduced here.
"""

from typing import Dict, Iterable, Optional

from pymrst.utils.inout.eclipse.deckinput_simple.apply_operator_simple import (
    apply_operator_simple,
)
from pymrst.utils.inout.eclipse.deckinput_simple.private.read_operator import (
    read_operator,
)


def read_operator_mod_field(lines: Iterable[str], grdecl: Dict,
                            kw: str = "MULTIPLY",
                            cart_dims: Optional[Iterable[int]] = None) -> Dict:
    """Read one operator section and apply it to ``grdecl`` immediately."""
    dims = cart_dims if cart_dims is not None else grdecl.get("cartDims")
    if dims is None:
        raise ValueError(
            f"{kw} needs the grid dimensions to resolve its box; pass "
            f"cart_dims or set grdecl['cartDims']")

    staged = read_operator(lines, {}, kw=kw)
    for target, entry in staged.get(kw.upper(), {}).items():
        records = [[target, value, *region] for value, region in
                   zip(entry["value"], entry["region"].tolist())]
        grdecl = apply_operator_simple(grdecl, records, kw, dims)
    return grdecl


readOperatorModField = read_operator_mod_field


if __name__ == "__main__":
    import numpy as np

    def deck():
        return {"cartDims": (2, 2, 2), "PERMX": np.full(8, 100.0)}

    def test_multiply_is_applied_at_once() -> None:
        got = read_operator_mod_field(["'PERMX' 0.1 1 2 1 2 1 2 /"], deck())
        assert got["PERMX"].tolist() == [10.0] * 8
        print("ok test_multiply_is_applied_at_once")

    def test_a_box_limits_the_change() -> None:
        got = read_operator_mod_field(["'PERMX' 0.1 1 2 1 2 2 2 /"], deck())
        assert got["PERMX"][:4].tolist() == [100.0] * 4
        assert got["PERMX"][4:].tolist() == [10.0] * 4
        print("ok test_a_box_limits_the_change")

    def test_equals_sets_the_value() -> None:
        got = read_operator_mod_field(["'PERMX' 7.0 1 2 1 2 1 2 /"], deck(),
                                      kw="EQUALS")
        assert got["PERMX"].tolist() == [7.0] * 8
        print("ok test_equals_sets_the_value")

    def test_add_is_handled_where_mrst_ignores_it() -> None:
        """MRST's if-chain covers only MULTIPLY and EQUALS."""
        got = read_operator_mod_field(["'PERMX' 5.0 1 2 1 2 1 2 /"], deck(),
                                      kw="ADD")
        assert got["PERMX"].tolist() == [105.0] * 8
        print("ok test_add_is_handled_where_mrst_ignores_it")

    def test_minvalue_is_handled_too() -> None:
        got = read_operator_mod_field(["'PERMX' 500.0 1 2 1 2 1 2 /"], deck(),
                                      kw="MINVALUE")
        assert got["PERMX"].tolist() == [500.0] * 8
        print("ok test_minvalue_is_handled_too")

    def test_several_entries_apply_in_order() -> None:
        got = read_operator_mod_field(["'PERMX' 0.1 1 2 1 2 1 2",
                                       "'PERMX' 3.0 1 2 1 2 1 2", "/"], deck())
        assert got["PERMX"].tolist() == [30.0] * 8
        print("ok test_several_entries_apply_in_order")

    def test_it_agrees_with_the_deferred_route() -> None:
        from pymrst.utils.inout.eclipse.deckinput_simple.private.read_operator import (
            read_operator,
        )
        immediate = read_operator_mod_field(["'PERMX' 0.1 1 2 1 2 1 2 /"], deck())
        staged = read_operator(["'PERMX' 0.1 1 2 1 2 1 2 /"], kw="MULTIPLY")
        entry = staged["MULTIPLY"]["PERMX"]
        later = apply_operator_simple(
            deck(), [["PERMX", entry["value"][0], *entry["region"][0].tolist()]],
            "MULTIPLY", (2, 2, 2))
        assert immediate["PERMX"].tolist() == later["PERMX"].tolist()
        print("ok test_it_agrees_with_the_deferred_route")

    def test_missing_dimensions_are_rejected() -> None:
        try:
            read_operator_mod_field(["'PERMX' 0.1 1 2 1 2 1 2 /"],
                                    {"PERMX": np.ones(8)})
        except ValueError as error:
            assert "needs the grid dimensions" in str(error)
            print("ok test_missing_dimensions_are_rejected")
            return
        raise AssertionError("a deck without cartDims must be rejected")

    test_multiply_is_applied_at_once()
    test_a_box_limits_the_change()
    test_equals_sets_the_value()
    test_add_is_handled_where_mrst_ignores_it()
    test_minvalue_is_handled_too()
    test_several_entries_apply_in_order()
    test_it_agrees_with_the_deferred_route()
    test_missing_dimensions_are_rejected()
    print("\nAll tests passed!")
