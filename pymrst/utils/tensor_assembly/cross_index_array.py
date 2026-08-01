"""Join two index arrays (MRST ``utils/tensorAssembly/crossIndexArray.m``).

An **inner join** on the index names the two arrays share. Given cell-face
pairs and face-node pairs, joining on `faces` gives cell-face-node triples: one
row for every way a row of the first lines up with a row of the second.

    tbl1: (cells, faces)      tbl2: (faces, nodes)
    join on faces         ->  (cells, faces, nodes)

The result carries all of `tbl1`'s columns plus `tbl2`'s *other* columns; the
join columns are not repeated. It also returns `indstruct`, the pair of maps
saying which row of each input produced each output row - that is what
`tbldispatch1`/`tbldispatch2` use to carry values through the join.

With **no shared names** it is the full Cartesian product, which is what
`pure_product=True` asserts and `virtual=True` records without materialising
(names and size only, for when the caller manages the numbering itself).

`crossextend` renames a shared name apart in the two inputs, so it stops being
a join key and becomes an ordinary column on each side - the way a
cell-against-cell pairing is set up before one side is contracted away.

**Ordering matches MRST**: groups in ascending order of their join values, and
within a group the pairs vary the second input fastest, both inputs in their
original row order. MRST reaches that by sorting a tagged stack and walking the
1-to-2 transitions; this groups directly, which is the same result with the
reasoning visible.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.utils.mcolon import mcolon
from pymrst.utils.tensor_assembly.index_array import IndexArray
from pymrst.utils.tensor_assembly.utils.crossindconstruct import crossindconstruct


def _join_names(crossfields) -> Tuple[List[str], List[str]]:
    """Split each join spec into its name in the first and second array."""
    left, right = [], []
    for field in crossfields or []:
        if isinstance(field, str):
            left.append(field)
            right.append(field)
        else:
            left.append(field[0])
            right.append(field[1])
    return left, right


def cross_index_array(tbl1, tbl2, crossfields: Optional[Sequence] = None, *,
                      crossextend: Optional[Sequence] = None,
                      pure_product: bool = False, virtual: bool = False):
    """Return the join of ``tbl1`` and ``tbl2``, or ``(join, indstruct)``."""
    names1, names2 = list(tbl1.fdnames), list(tbl2.fdnames)

    if crossextend:
        for old, (new1, new2) in crossextend:
            if old not in names1 or old not in names2:
                raise ValueError(
                    f"crossextend needs {old!r} in both arrays; they have "
                    f"{names1} and {names2}")
            names1[names1.index(old)] = new1
            names2[names2.index(old)] = new2
        tbl1 = tbl1._copy(names1, tbl1.inds)
        tbl2 = tbl2._copy(names2, tbl2.inds)

    left, right = _join_names(crossfields)
    for name, names in ((left, names1), (right, names2)):
        unknown = [n for n in name if n not in names]
        if unknown:
            raise ValueError(f"join names {unknown} are not in {names}")

    other2 = [n for n in names2 if n not in right]
    allnames = names1 + other2

    # ------------------------------------------------------ Cartesian product
    if not crossfields:
        if not (pure_product or virtual):
            raise ValueError(
                "the arrays share no join names, so this would be their full "
                "Cartesian product; pass pure_product=True to say you meant it")
        shared = set(names1) & set(names2)
        if shared:
            raise ValueError(
                f"a pure product needs disjoint index names, but both arrays "
                f"have {sorted(shared)}")
        if virtual:
            return IndexArray(fdnames=allnames, isvirtual=True,
                              num=tbl1.num * tbl2.num)
        repeated = np.repeat(tbl1.inds, tbl2.num, axis=0)
        tiled = np.tile(tbl2.inds, (tbl1.num, 1))
        return IndexArray(fdnames=allnames,
                          inds=np.hstack([repeated, tiled]))

    # ------------------------------------------------------------- inner join
    key1, key2 = tbl1.gets(left), tbl2.gets(right)
    unique, inverse = np.unique(np.vstack([key1, key2]), axis=0,
                                return_inverse=True)
    inverse = inverse.ravel()
    group1, group2 = inverse[:tbl1.num], inverse[tbl1.num:]

    counts1 = np.bincount(group1, minlength=unique.shape[0])
    counts2 = np.bincount(group2, minlength=unique.shape[0])
    matched = np.flatnonzero((counts1 > 0) & (counts2 > 0))

    # Rows of each input, gathered per group, in their original order.
    order1 = np.argsort(group1, kind="stable")
    order2 = np.argsort(group2, kind="stable")
    starts1 = np.concatenate([[0], np.cumsum(counts1)[:-1]])
    starts2 = np.concatenate([[0], np.cumsum(counts2)[:-1]])
    nb1, nb2 = counts1[matched], counts2[matched]

    rows1 = order1[mcolon(starts1[matched], starts1[matched] + nb1)]
    rows2 = order2[mcolon(starts2[matched], starts2[matched] + nb2)]

    crossind1, crossind2 = crossindconstruct(nb1, nb2)
    linind1, linind2 = rows1[crossind1], rows2[crossind2]

    inds = np.hstack([tbl1.inds[linind1],
                      tbl2.gets(other2)[linind2]]) if other2 \
        else tbl1.inds[linind1]

    out = IndexArray(fdnames=allnames, inds=inds)
    out.parents = {"tbl1": tbl1, "tbl2": tbl2}
    indstruct = [{"inds": linind1, "num": tbl1.num},
                 {"inds": linind2, "num": tbl2.num}]
    return out, indstruct


crossIndexArray = cross_index_array


if __name__ == "__main__":

    def cell_face():
        return IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})

    def face_node():
        return IndexArray({"faces": [3, 7, 7], "nodes": [10, 20, 30]})

    def test_it_joins_on_the_shared_name() -> None:
        out, _ = cross_index_array(cell_face(), face_node(), ["faces"])
        assert out.fdnames == ["cells", "faces", "nodes"]
        # cell 0/face 3 meets node 10; cell 0/face 7 and cell 1/face 7 each
        # meet nodes 20 and 30.
        assert out.num == 5, out.num
        print("ok test_it_joins_on_the_shared_name")

    def test_every_row_is_a_real_pairing() -> None:
        out, _ = cross_index_array(cell_face(), face_node(), ["faces"])
        left = {tuple(r) for r in cell_face().gets(["cells", "faces"]).tolist()}
        right = {tuple(r) for r in face_node().gets(["faces", "nodes"]).tolist()}
        for c, f, n in out.gets(["cells", "faces", "nodes"]).tolist():
            assert (c, f) in left and (f, n) in right
        print("ok test_every_row_is_a_real_pairing")

    def test_the_join_is_complete() -> None:
        """Every pairing that should exist does."""
        out, _ = cross_index_array(cell_face(), face_node(), ["faces"])
        got = {tuple(r) for r in out.gets(["cells", "faces", "nodes"]).tolist()}
        want = {(c, f, n)
                for c, f in cell_face().gets(["cells", "faces"]).tolist()
                for f2, n in face_node().gets(["faces", "nodes"]).tolist()
                if f == f2}
        assert got == want, got ^ want
        print("ok test_the_join_is_complete")

    def test_the_index_struct_maps_back_to_the_inputs() -> None:
        out, s = cross_index_array(cell_face(), face_node(), ["faces"])
        assert cell_face().get("cells")[s[0]["inds"]].tolist() == \
            out.get("cells").tolist()
        assert face_node().get("nodes")[s[1]["inds"]].tolist() == \
            out.get("nodes").tolist()
        assert s[0]["num"] == 3 and s[1]["num"] == 3
        print("ok test_the_index_struct_maps_back_to_the_inputs")

    def test_a_non_matching_key_drops_out() -> None:
        left = IndexArray({"faces": [1, 2]})
        right = IndexArray({"faces": [2, 3], "nodes": [7, 8]})
        out, _ = cross_index_array(left, right, ["faces"])
        assert out.get("faces").tolist() == [2]
        print("ok test_a_non_matching_key_drops_out")

    def test_no_overlap_gives_an_empty_join() -> None:
        left = IndexArray({"faces": [1]})
        right = IndexArray({"faces": [2], "nodes": [7]})
        out, s = cross_index_array(left, right, ["faces"])
        assert out.num == 0 and s[0]["inds"].size == 0
        print("ok test_no_overlap_gives_an_empty_join")

    def test_joining_on_two_names() -> None:
        left = IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})
        right = IndexArray({"cells": [0, 1], "faces": [7, 7], "v": [5, 6]})
        out, _ = cross_index_array(left, right, ["cells", "faces"])
        assert out.num == 2
        assert out.gets(["cells", "faces", "v"]).tolist() == [[0, 7, 5], [1, 7, 6]]
        print("ok test_joining_on_two_names")

    def test_a_renamed_join_key() -> None:
        left = IndexArray({"cells": [0, 1], "f": [3, 7]})
        right = IndexArray({"g": [3, 7], "nodes": [10, 20]})
        out, _ = cross_index_array(left, right, [("f", "g")])
        assert out.fdnames == ["cells", "f", "nodes"]
        assert out.gets(["cells", "nodes"]).tolist() == [[0, 10], [1, 20]]
        print("ok test_a_renamed_join_key")

    def test_the_pure_product_multiplies_the_rows() -> None:
        left = IndexArray({"cells": [0, 1]})
        right = IndexArray({"nodes": [7, 8, 9]})
        out = cross_index_array(left, right, [], pure_product=True)
        assert out.num == 6 and out.fdnames == ["cells", "nodes"]
        assert out.get("cells").tolist() == [0, 0, 0, 1, 1, 1]
        assert out.get("nodes").tolist() == [7, 8, 9, 7, 8, 9]
        print("ok test_the_pure_product_multiplies_the_rows")

    def test_a_virtual_product_records_size_only() -> None:
        left = IndexArray({"cells": [0, 1]})
        right = IndexArray({"nodes": [7, 8, 9]})
        out = cross_index_array(left, right, [], virtual=True)
        assert out.isvirtual and out.num == 6
        print("ok test_a_virtual_product_records_size_only")

    def test_an_accidental_product_is_rejected() -> None:
        """Two arrays with no shared name would silently explode."""
        try:
            cross_index_array(IndexArray({"cells": [0]}),
                              IndexArray({"nodes": [7]}), [])
        except ValueError as error:
            assert "pass pure_product=True" in str(error)
            print("ok test_an_accidental_product_is_rejected")
            return
        raise AssertionError("an unrequested Cartesian product must be rejected")

    def test_a_product_of_overlapping_names_is_rejected() -> None:
        try:
            cross_index_array(IndexArray({"cells": [0]}),
                              IndexArray({"cells": [7]}), [], pure_product=True)
        except ValueError as error:
            assert "disjoint index names" in str(error)
            print("ok test_a_product_of_overlapping_names_is_rejected")
            return
        raise AssertionError("a product of overlapping names must be rejected")

    def test_crossextend_splits_a_shared_name() -> None:
        left = IndexArray({"cells": [0, 1], "faces": [3, 3]})
        right = IndexArray({"cells": [0, 1], "faces": [3, 3]})
        out, _ = cross_index_array(left, right, ["faces"],
                                   crossextend=[("cells", ("c1", "c2"))])
        assert out.fdnames == ["c1", "faces", "c2"]
        assert out.num == 4, "cells no longer joins, so it pairs every way"
        print("ok test_crossextend_splits_a_shared_name")

    def test_an_unknown_join_name_is_rejected() -> None:
        try:
            cross_index_array(cell_face(), face_node(), ["nodes"])
        except ValueError as error:
            assert "are not in" in str(error)
            print("ok test_an_unknown_join_name_is_rejected")
            return
        raise AssertionError("an unknown join name must be rejected")

    test_it_joins_on_the_shared_name()
    test_every_row_is_a_real_pairing()
    test_the_join_is_complete()
    test_the_index_struct_maps_back_to_the_inputs()
    test_a_non_matching_key_drops_out()
    test_no_overlap_gives_an_empty_join()
    test_joining_on_two_names()
    test_a_renamed_join_key()
    test_the_pure_product_multiplies_the_rows()
    test_a_virtual_product_records_size_only()
    test_an_accidental_product_is_rejected()
    test_a_product_of_overlapping_names_is_rejected()
    test_crossextend_splits_a_shared_name()
    test_an_unknown_join_name_is_rejected()
    print("\nAll tests passed!")
