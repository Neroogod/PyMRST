"""Project onto a subset of indices (MRST ``utils/tensorAssembly/projIndexArray.m``).

Reduces an index array to the named columns and removes the duplicate rows that
projection creates, returning both the projected array and the map from the
original rows to it.

That map is the point. Projecting (cell, face) pairs onto `cells` gives each
cell once, and `indstruct["inds"][k]` says which of those the original row `k`
fell into - which is exactly the scatter needed to **sum a quantity over the
discarded index**. Without it the projection would only tell you the cells, not
how to accumulate onto them.

Returns `(projtbl, indstruct)` where `indstruct` is a dict with `inds` and
`num`, matching what `TensorMap` and `TensorProd` consume.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from pymrst.utils.tensor_assembly.index_array import IndexArray


def proj_index_array(tbl, fds: Sequence[str]) -> Tuple[IndexArray, Dict]:
    """Return ``(projected, indstruct)`` for the projection onto ``fds``."""
    fds = list(fds)
    inds = tbl.gets(fds)
    if inds.shape[0] == 0:
        projtbl = IndexArray(fdnames=fds,
                             inds=np.zeros((0, len(fds)), dtype=np.int64))
        return projtbl, {"inds": np.zeros(0, dtype=np.int64), "num": 0}

    unique, inverse = np.unique(inds, axis=0, return_inverse=True)
    projtbl = IndexArray(fdnames=fds, inds=unique)
    return projtbl, {"inds": inverse.ravel().astype(np.int64),
                     "num": projtbl.num}


projIndexArray = proj_index_array


if __name__ == "__main__":

    def table():
        # Three faces, but only two distinct cells.
        return IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})

    def test_it_removes_duplicate_rows() -> None:
        proj, _ = proj_index_array(table(), ["cells"])
        assert proj.get("cells").tolist() == [0, 1]
        print("ok test_it_removes_duplicate_rows")

    def test_the_map_points_each_original_row_at_its_image() -> None:
        proj, s = proj_index_array(table(), ["cells"])
        assert s["inds"].tolist() == [0, 0, 1]
        assert s["num"] == 2
        print("ok test_the_map_points_each_original_row_at_its_image")

    def test_the_map_sums_over_the_discarded_index() -> None:
        """What the second return value is for."""
        proj, s = proj_index_array(table(), ["cells"])
        per_face = np.array([1.0, 10.0, 100.0])
        per_cell = np.bincount(s["inds"], weights=per_face, minlength=s["num"])
        assert per_cell.tolist() == [11.0, 100.0]
        print("ok test_the_map_sums_over_the_discarded_index")

    def test_projecting_onto_everything_is_the_identity() -> None:
        proj, s = proj_index_array(table(), ["cells", "faces"])
        assert proj.num == 3 and sorted(s["inds"].tolist()) == [0, 1, 2]
        print("ok test_projecting_onto_everything_is_the_identity")

    def test_the_projection_is_sorted() -> None:
        t = IndexArray({"cells": [5, 1, 5, 3]})
        proj, _ = proj_index_array(t, ["cells"])
        assert proj.get("cells").tolist() == [1, 3, 5]
        print("ok test_the_projection_is_sorted")

    def test_the_map_survives_that_sorting() -> None:
        """The map must point at the sorted positions, not the original order."""
        t = IndexArray({"cells": [5, 1, 5, 3]})
        proj, s = proj_index_array(t, ["cells"])
        assert proj.get("cells")[s["inds"]].tolist() == [5, 1, 5, 3]
        print("ok test_the_map_survives_that_sorting")

    def test_projecting_onto_the_other_index() -> None:
        proj, s = proj_index_array(table(), ["faces"])
        assert proj.get("faces").tolist() == [3, 7]
        assert s["inds"].tolist() == [0, 1, 1]
        print("ok test_projecting_onto_the_other_index")

    def test_an_empty_array_projects_to_nothing() -> None:
        empty = IndexArray(fdnames=["cells"], inds=np.zeros((0, 1), dtype=np.int64))
        proj, s = proj_index_array(empty, ["cells"])
        assert proj.num == 0 and s["num"] == 0 and s["inds"].size == 0
        print("ok test_an_empty_array_projects_to_nothing")

    def test_an_unknown_name_is_rejected() -> None:
        try:
            proj_index_array(table(), ["nodes"])
        except ValueError as error:
            assert "no index named" in str(error)
            print("ok test_an_unknown_name_is_rejected")
            return
        raise AssertionError("an unknown index name must be rejected")

    test_it_removes_duplicate_rows()
    test_the_map_points_each_original_row_at_its_image()
    test_the_map_sums_over_the_discarded_index()
    test_projecting_onto_everything_is_the_identity()
    test_the_projection_is_sorted()
    test_the_map_survives_that_sorting()
    test_projecting_onto_the_other_index()
    test_an_empty_array_projects_to_nothing()
    test_an_unknown_name_is_rejected()
    print("\nAll tests passed!")
