"""Accumulate directional transmissibility multipliers onto faces (MRST
`accumulateCartesianMultipliers`).

An ECLIPSE deck can scale flow across cell faces by direction - `MULTX` multiplies
the +x face of each cell, `MULTZ` the +z face, and so on. Those multipliers are
given *per cell and per direction*; a solver needs them *per face*. This walks
each half-face, picks up its cell's multiplier when the half-face points in that
multiplier's direction, and forms each face's total as the product of the matched
multipliers on both sides (an unmatched face keeps 1). It is how directional
`MULT*` keywords reach the transmissibility.

The half-face direction is read from the grid's `cells.tags` (0-5: W E S N B T),
the 0-based form of MRST's `cells.faces(:,2)` (1-6).
"""

import re
from typing import Dict, Optional

import numpy as np

# field name (non-word chars -> '_', lower-cased) -> MRST direction 1..6
_DIR_ID = {"x_": 1, "i_": 1, "x": 2, "i": 2,
           "y_": 3, "j_": 3, "y": 4, "j": 4,
           "z_": 5, "k_": 5, "z": 6, "k": 6}


def accumulate_cartesian_multipliers(g, mult: Optional[Dict]) -> Optional[np.ndarray]:
    """Return a per-face multiplier array from the per-cell direction multipliers
    `mult` (e.g. `{"x": MULTX, "z": MULTZ}`), or `None` if none apply."""
    if not mult:
        return None

    ncells = g.cells.num
    totcells = int(np.prod(g.cartDims))
    index_map = np.asarray(g.cells.indexMap, dtype=np.int64)

    dirs, cols = [], []
    for name, value in mult.items():
        key = re.sub(r"\W", "_", name).lower()
        if key not in _DIR_ID:
            continue                                     # unknown direction, ignore
        value = np.asarray(value, dtype=float).ravel()
        if value.size == ncells:
            v = value
        elif value.size == totcells:
            v = value[index_map]
        else:
            continue                                     # incompatible size, ignore
        dirs.append(_DIR_ID[key])
        cols.append(v)

    if not dirs:
        return None

    m_mult = np.column_stack(cols)                       # (ncells, k)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_no = np.repeat(np.arange(ncells), np.diff(face_pos))
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    facedir = np.asarray(g.cells.tags, dtype=np.int64).ravel() + 1   # 1..6

    out = np.ones(g.faces.num)
    for col, d in enumerate(dirs):
        sel = facedir == d
        np.multiply.at(out, cell_faces[sel], m_mult[cell_no[sel], col])
    return out


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_multx_scales_the_east_face_of_each_cell():
        """MULTX applies to each cell's +x (east) face; a face between two cells
        gets the west cell's MULTX."""
        g = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        multx = np.array([2.0, 3.0, 5.0])                # per cell
        m = accumulate_cartesian_multipliers(g, {"x": multx})
        # x-faces: outer-west (cell0 west, no east match) -> 1; the two interior
        # faces get the west cell's MULTX; outer-east gets cell2's MULTX
        # identify x-faces by tag: east tag is 1 (0-based)
        neigh = np.asarray(g.faces.neighbors)
        # interior face between cell 0 and 1 -> cell 0's east face -> MULTX[0]=2
        f01 = np.flatnonzero((np.sort(neigh, axis=1) == [0, 1]).all(axis=1))[0]
        f12 = np.flatnonzero((np.sort(neigh, axis=1) == [1, 2]).all(axis=1))[0]
        assert np.isclose(m[f01], 2.0), m[f01]
        assert np.isclose(m[f12], 3.0), m[f12]
        print("ok test_multx_scales_the_east_face_of_each_cell")

    def test_no_multipliers_returns_none():
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        assert accumulate_cartesian_multipliers(g, {}) is None
        assert accumulate_cartesian_multipliers(g, None) is None
        print("ok test_no_multipliers_returns_none")

    def test_unmatched_faces_stay_one():
        """With only MULTZ on a single z-layer, the x/y faces keep multiplier 1."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        m = accumulate_cartesian_multipliers(g, {"z": np.full(4, 0.1)})
        # x/y faces (tags 0..3) untouched -> 1
        tags = np.asarray(g.cells.tags)
        cell_faces = np.asarray(g.cells.faces)
        xy_faces = cell_faces[tags < 4]
        assert np.allclose(m[np.unique(xy_faces)], 1.0)
        print("ok test_unmatched_faces_stay_one")

    def test_two_directions_multiply_independently():
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        m = accumulate_cartesian_multipliers(
            g, {"x": np.array([2.0, 4.0]), "z": np.array([0.5, 0.5])})
        assert np.all(m > 0)
        print("ok test_two_directions_multiply_independently")

    test_multx_scales_the_east_face_of_each_cell()
    test_no_multipliers_returns_none()
    test_unmatched_faces_stay_one()
    test_two_directions_multiply_independently()
    print("\nAll tests passed!")
