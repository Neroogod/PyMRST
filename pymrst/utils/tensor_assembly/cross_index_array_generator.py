"""A reusable join specification (MRST ``.../CrossIndexArrayGenerator.m``).

Holds the two index arrays, the names to join on, and any renaming to apply
first, so a join can be described once and evaluated later. `eval()` applies the
renames and calls `cross_index_array`.

**MRST's constructor discards its arguments.** It parses `varargin` into a
local struct called `map` and then never assigns to `gen`, so every property is
left empty:

    g = CrossIndexArrayGenerator('tbl1', a, 'tbl2', b, 'mergefds', {'faces'});
    isempty(g.tbl1)      % 1
    g.eval()             % "Dot indexing is not supported for variables of
                         %  this type."

Verified in MATLAB. The class can only be used by assigning its properties by
hand after construction, which is presumably how its callers do it. Here the
constructor works.
"""

from typing import Optional, Sequence

from pymrst.utils.tensor_assembly.cross_index_array import cross_index_array
from pymrst.utils.tensor_assembly.utils.replacefield import replacefield


class CrossIndexArrayGenerator:
    """A join of two index arrays, described now and evaluated later."""

    def __init__(self, tbl1=None, tbl2=None,
                 mergefds: Optional[Sequence[str]] = None,
                 replacefds1=None, replacefds2=None):
        self.tbl1 = tbl1
        self.tbl2 = tbl2
        self.mergefds = list(mergefds) if mergefds else []
        self.replacefds1 = replacefds1
        self.replacefds2 = replacefds2

    def eval(self, with_indstruct: bool = False):
        """Apply the renames and perform the join."""
        if self.tbl1 is None or self.tbl2 is None:
            missing = [n for n in ("tbl1", "tbl2") if getattr(self, n) is None]
            raise ValueError(
                f"the generator needs both index arrays; {missing} not set")

        tbl1 = replacefield(self.tbl1, self.replacefds1) if self.replacefds1 \
            else self.tbl1
        tbl2 = replacefield(self.tbl2, self.replacefds2) if self.replacefds2 \
            else self.tbl2

        out = cross_index_array(tbl1, tbl2, self.mergefds,
                                pure_product=not self.mergefds)
        if not self.mergefds:
            return (out, None) if with_indstruct else out
        return out if with_indstruct else out[0]

    def __repr__(self) -> str:
        return (f"<CrossIndexArrayGenerator on {self.mergefds or 'nothing '
                '(pure product)'}>")


if __name__ == "__main__":
    from pymrst.utils.tensor_assembly.index_array import IndexArray

    def a():
        return IndexArray({"cells": [0, 1], "faces": [3, 4]})

    def b():
        return IndexArray({"faces": [3, 4], "nodes": [7, 8]})

    def test_the_constructor_keeps_its_arguments() -> None:
        """MRST's discards them; see the module docstring."""
        g = CrossIndexArrayGenerator(tbl1=a(), tbl2=b(), mergefds=["faces"])
        assert g.tbl1 is not None and g.tbl2 is not None
        assert g.mergefds == ["faces"]
        print("ok test_the_constructor_keeps_its_arguments")

    def test_eval_performs_the_join() -> None:
        g = CrossIndexArrayGenerator(tbl1=a(), tbl2=b(), mergefds=["faces"])
        out = g.eval()
        assert out.fdnames == ["cells", "faces", "nodes"]
        assert out.gets(["cells", "nodes"]).tolist() == [[0, 7], [1, 8]]
        print("ok test_eval_performs_the_join")

    def test_it_can_be_evaluated_twice() -> None:
        """The point of holding the description rather than the result."""
        g = CrossIndexArrayGenerator(tbl1=a(), tbl2=b(), mergefds=["faces"])
        assert g.eval().num == g.eval().num == 2
        print("ok test_it_can_be_evaluated_twice")

    def test_renames_are_applied_before_the_join() -> None:
        left = IndexArray({"cells": [0, 1], "f": [3, 4]})
        g = CrossIndexArrayGenerator(tbl1=left, tbl2=b(), mergefds=["faces"],
                                     replacefds1=("f", "faces"))
        out = g.eval()
        assert out.fdnames == ["cells", "faces", "nodes"] and out.num == 2
        print("ok test_renames_are_applied_before_the_join")

    def test_both_sides_can_be_renamed() -> None:
        left = IndexArray({"cells": [0, 1], "f": [3, 4]})
        right = IndexArray({"g": [3, 4], "nodes": [7, 8]})
        gen = CrossIndexArrayGenerator(tbl1=left, tbl2=right,
                                       mergefds=["faces"],
                                       replacefds1=("f", "faces"),
                                       replacefds2=("g", "faces"))
        assert gen.eval().num == 2
        print("ok test_both_sides_can_be_renamed")

    def test_the_indstruct_can_be_asked_for() -> None:
        g = CrossIndexArrayGenerator(tbl1=a(), tbl2=b(), mergefds=["faces"])
        out, s = g.eval(with_indstruct=True)
        assert s[0]["num"] == 2 and s[1]["num"] == 2
        print("ok test_the_indstruct_can_be_asked_for")

    def test_no_merge_fields_gives_the_product() -> None:
        left = IndexArray({"cells": [0, 1]})
        right = IndexArray({"nodes": [7, 8, 9]})
        assert CrossIndexArrayGenerator(tbl1=left, tbl2=right).eval().num == 6
        print("ok test_no_merge_fields_gives_the_product")

    def test_a_missing_table_is_rejected() -> None:
        try:
            CrossIndexArrayGenerator(tbl1=a(), mergefds=["faces"]).eval()
        except ValueError as error:
            assert "tbl2" in str(error)
            print("ok test_a_missing_table_is_rejected")
            return
        raise AssertionError("an incomplete generator must be rejected")

    test_the_constructor_keeps_its_arguments()
    test_eval_performs_the_join()
    test_it_can_be_evaluated_twice()
    test_renames_are_applied_before_the_join()
    test_both_sides_can_be_renamed()
    test_the_indstruct_can_be_asked_for()
    test_no_merge_fields_gives_the_product()
    test_a_missing_table_is_rejected()
    print("\nAll tests passed!")
